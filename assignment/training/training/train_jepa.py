"""Stage 2. Train a LeWM- or SMWM-style JEPA world model on the frame cache (DDP via torchrun).

Example (2 GPUs):
  torchrun --nproc_per_node 2 -m assignment.training.training.train_jepa \
    --cache cache-4cube-20k --out smwm-c3 --method smwm --context 3 --batch 256 --steps 145000
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from assignment.training.training.jepa import (
    JEPAConfig, StackJEPA, WindowStream, Windows, amp_dtype, autocast, save_model,
)

HORIZONS = (1, 2, 4, 8, 16)
PROPRIO_GROUPS = dict(qpos=slice(0, 25), qvel=slice(25, 50), target=slice(50, 75))


def eval_windows(cache, length, count, proprio=False, seed=0):
    windows = Windows(cache, 'val', length, proprio)
    return windows.sample(np.random.default_rng(seed), count)


def r2(pred, target):
    """Per-group R^2 over all samples: 1 - SSE/SST, dimensions pooled within a group."""
    out = {}
    for name, s in PROPRIO_GROUPS.items():
        t = target[..., s].flatten(0, -2)
        sse = (pred[..., s].flatten(0, -2)-t).square().sum(0)
        sst = (t-t.mean(0)).square().sum(0)
        out[name] = float(1-sse.sum()/sst.sum())
    return out


@torch.no_grad()
def evaluate(model, frames, actions, proprio, eval_context, horizon, device, chunk=8):
    """Shared val windows of eval_context+horizon frames; the model sees its last `context`
    real frames before position eval_context, then rolls out `horizon` steps open loop.
    With proprio, the detached probe reads proprioception from real and rolled-out latents."""
    model.eval()
    ctx = model.cfg.context
    eval_context = max(eval_context, ctx)
    dtype = amp_dtype(training=True)
    emb_all, err_tf, err_roll, extra, probe_real, probe_roll = [], [], [], [], [], []
    for i in range(0, len(frames), chunk):
        f = frames[i:i+chunk].to(device, non_blocking=True)
        a = actions[i:i+chunk].to(device, non_blocking=True)
        p = proprio[i:i+chunk].to(device, non_blocking=True) if proprio is not None else None
        with autocast(dtype):
            emb = model.encode(f, p)
            act = model.action_encoder(a)
            first = eval_context-ctx
            pred = model.predict(emb[:, first:eval_context], act[:, first:eval_context])
            roll = model.rollout(emb[:, first:eval_context], act[:, first:], horizon)
            if model.cfg.method == 'smwm':
                pair = torch.cat([emb[:, :-1], emb[:, 1:]], -1)
                extra.append((model.inverse(pair).float()-a[:, :-1]).square().mean(-1).mean(-1))
            if p is not None:
                probe_real.append(model.probe(emb[:, :eval_context]).float())
                probe_roll.append(model.probe(roll).float())
        emb_all.append(emb.float())
        err_tf.append((pred.float()-emb[:, first+1:eval_context+1].float()).square().mean(-1))
        err_roll.append((roll.float()-emb[:, eval_context:eval_context+horizon].float()).square().mean(-1))
    emb = torch.cat(emb_all)
    var = emb.flatten(0, 1).var(0).mean()
    tf, roll = torch.cat(err_tf)/var, torch.cat(err_roll)/var
    flat = emb.flatten(0, 1)
    s = torch.linalg.svdvals(flat-flat.mean(0))
    p = s/s.sum()
    out = dict(latent_var=float(var), effective_rank=float(torch.exp(-(p*torch.log(p+1e-12)).sum())),
               tf_last=float(tf[:, -1].mean()),
               rollout={str(h): float(roll[:, h-1].mean()) for h in HORIZONS if h <= horizon})
    if ctx > 3:
        bins = [(1, 1), (2, 3), (4, 16), (17, ctx)]
        out['tf_by_context'] = {f'{lo}-{hi}': float(tf[:, lo-1:hi].mean()) for lo, hi in bins if lo <= ctx}
    if extra:
        out['inverse'] = float(torch.cat(extra).mean())
    if probe_real:
        target = proprio.to(device).float()
        real, rolled = torch.cat(probe_real), torch.cat(probe_roll)
        out['probe_r2_real'] = r2(real, target[:, :eval_context])
        out['probe_r2_rollout'] = {str(h): r2(rolled[:, h-1], target[:, eval_context+h-1])
                                   for h in HORIZONS if h <= horizon}
    model.train()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--method', choices=['lewm', 'smwm'], required=True)
    p.add_argument('--context', type=int, default=3)
    p.add_argument('--lam', type=float, default=None, help='Default: 0.09 (lewm SIGReg), 1.0 (smwm inverse)')
    p.add_argument('--sigreg-group', type=int, default=0,
                   help='SIGReg over random groups of this many window x batch embeddings (0: per time step)')
    p.add_argument('--patch', type=int, default=8)
    p.add_argument('--proprio', action='store_true', help='Experimental: proprio token in the encoder + detached proprio probe. Needs a cache with proprio '
                        'statistics (data.cache.add_proprio); encode, on-policy collection and planning do not support it')
    p.add_argument('--probe-lr', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=256, help='Global windows per step')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-3)
    p.add_argument('--steps', type=int, default=145000)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--eval-every', type=int, default=2500)
    p.add_argument('--save-every', type=int, default=10000)
    p.add_argument('--eval-windows', type=int, default=256)
    p.add_argument('--eval-context', type=int, default=80)
    p.add_argument('--eval-horizon', type=int, default=16)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    distributed = 'RANK' in os.environ
    if distributed:
        dist.init_process_group('nccl')
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.manual_seed(args.seed+rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.batch % world:
        raise ValueError('--batch must be divisible by the number of processes')
    if args.eval_context < args.context:
        args.eval_context = args.context
    lam = args.lam if args.lam is not None else (.09 if args.method == 'lewm' else 1.)
    index = json.loads((args.cache/'index.json').read_text())
    cfg = JEPAConfig(method=args.method, context=args.context, image=tuple(index['frame_shape'][:2]),
                     patch=args.patch, action_dim=index['action_dim'], lam=lam, sigreg_group=args.sigreg_group,
                     proprio_dim=index['proprio_dim'] if args.proprio else 0)
    model = StackJEPA(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local]) if distributed else model
    step_fn = torch.compile(net) if args.compile else net
    probe = [p for n, p in model.named_parameters() if n.startswith('probe.')]
    rest = [p for n, p in model.named_parameters() if not n.startswith('probe.')]
    groups = [dict(params=rest)]+([dict(params=probe, lr=args.probe_lr, weight_decay=0.)] if probe else [])
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s+1)/args.warmup)*0.5*(
        1+math.cos(math.pi*min(1., s/args.steps))))
    stream = WindowStream(args.cache, 'train', args.context+1, args.batch//world, [args.seed, rank], args.proprio)
    loader = iter(torch.utils.data.DataLoader(stream, batch_size=None, num_workers=args.workers, pin_memory=True,
                                              prefetch_factor=4 if args.workers else None,
                                              persistent_workers=args.workers > 0))
    dtype = amp_dtype(training=True)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        info = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    config=asdict(cfg), params=params, world=world, cache_frames=index['frames'],
                    train_windows=len(Windows(args.cache, 'train', args.context+1)))
        (args.out/'run.json').write_text(json.dumps(info, indent=1))
        print(json.dumps(info), flush=True)
        val = eval_windows(args.cache, args.eval_context+args.eval_horizon, args.eval_windows, args.proprio)
        val_frames, val_actions, val_proprio = val if args.proprio else (*val, None)
        log = open(args.out/'train-log.jsonl', 'a')
    t0, seen, sums = time.time(), 0, {}
    for step in range(1, args.steps+1):
        batch = [x.to(device, non_blocking=True) for x in next(loader)]
        frames = batch[0]
        with autocast(dtype):
            out = step_fn(*batch)
        opt.zero_grad(set_to_none=True)
        out['loss'].backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        schedule.step()
        seen += frames.shape[0]*frames.shape[1]*world
        for k, v in out.items():
            sums[k] = sums.get(k, 0.)+float(v.detach())
        sums['grad_norm'] = sums.get('grad_norm', 0.)+float(grad)
        if step % 100 == 0 and rank == 0:
            elapsed = time.time()-t0
            row = dict(step=step, lr=schedule.get_last_lr()[0], frames_per_s=round(seen/elapsed),
                       hours_left=round((args.steps-step)*elapsed/step/3600, 2),
                       **{k: v/100 for k, v in sums.items()})
            print(json.dumps(row), file=log, flush=True)
        if step % 100 == 0:
            sums = {}
        if rank == 0 and (step % args.eval_every == 0 or step == args.steps):
            row = dict(step=step, eval=evaluate(model, val_frames, val_actions, val_proprio, args.eval_context,
                                                args.eval_horizon, device))
            print(json.dumps(row), file=log, flush=True)
        if rank == 0 and (step % args.save_every == 0 or step == args.steps):
            meta = dict(step=step, cache=str(args.cache), action_mean=index['action_mean'],
                        action_std=index['action_std'], proprio_mean=index.get('proprio_mean'),
                        proprio_std=index.get('proprio_std'))
            save_model(args.out/f'step-{step:06d}.pt', model, meta)
            save_model(args.out/'model.pt', model, meta)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
