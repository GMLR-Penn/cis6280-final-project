"""Stage 3. Train the diffusion-forcing latent policy (single GPU, data resident on the GPU).

Checkpoints hold EMA weights plus everything the closed-loop agent needs
(encoder path, latent mean/std, and action mean/std).
"""
import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

from assignment.training.training.jepa import amp_dtype, autocast
from assignment.training.training.policy import DFPolicy, LatentData, PolicyConfig, save_policy


@torch.no_grad()
def evaluate(model, val, generator, batch=4096):
    z, actions, mask = val.sample(generator, batch)
    dtype = amp_dtype(training=True)
    with autocast(dtype):
        loss = float(model.loss(z, actions, mask, generator))
        out = dict(val_loss=loss)
        for h in sorted({1, 4, model.cfg.horizon}):
            for lag in (0, 1) if h > 1 else (0,):
                pred = model.sample(z, h, lag=lag, generator=generator)
                err = (pred-actions[:, :h]).square().mean(-1)
                m = mask[:, :h]
                out[f'mse_h{h}_lag{lag}_first'] = float(err[:, 0][m[:, 0]].mean())
                out[f'mse_h{h}_lag{lag}_all'] = float((err*m).sum()/m.sum())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--latents', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--context', type=int, default=1)
    p.add_argument('--horizon', type=int, default=16)
    p.add_argument('--dim', type=int, default=512)
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-2)
    p.add_argument('--steps', type=int, default=200000)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--ema', type=float, default=.999)
    p.add_argument('--eval-every', type=int, default=5000)
    p.add_argument('--save-every', type=int, default=20000)
    p.add_argument('--max-sigma', type=float, default=None, help='Train only on episodes with action noise <= this')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = 'cuda'
    dtype = amp_dtype(training=True)
    train = LatentData(args.cache, args.latents, 'train', args.context, args.horizon, device, args.max_sigma)
    val = LatentData(args.cache, args.latents, 'val', args.context, args.horizon, device, args.max_sigma)
    stats = json.loads((args.latents/'latents.json').read_text())
    cfg = PolicyConfig(latent_dim=stats['dim'], context=args.context, horizon=args.horizon, dim=args.dim,
                       depth=args.depth, heads=args.heads, mlp=4*args.dim)
    model = DFPolicy(cfg).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    decay_params, other_params = [], []
    for name, prm in model.named_parameters():
        (decay_params if prm.ndim >= 2 and name != 'pos' else other_params).append(prm)
    opt = torch.optim.AdamW([dict(params=decay_params, weight_decay=args.weight_decay),
                             dict(params=other_params, weight_decay=0.)],
                            lr=args.lr, betas=(.9, .95), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s+1)/args.warmup)*.5*(1+math.cos(
        math.pi*min(1., s/args.steps))))
    info = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, encoder=stats['model'],
                latents=str(args.latents), z_mean=stats['mean'], z_std=stats['std'],
                action_mean=train.a_mean.tolist(), action_std=train.a_std.tolist(),
                train_anchors=len(train.anchors), val_anchors=len(val.anchors),
                parameters=sum(prm.numel() for prm in model.parameters()))
    (args.out/'run.json').write_text(json.dumps(dict(config=asdict(cfg), **info), indent=1))
    print(json.dumps(dict(stage='start', train_anchors=len(train.anchors), val_anchors=len(val.anchors),
                          parameters=info['parameters'])), flush=True)
    gen = torch.Generator(device).manual_seed(args.seed)
    val_gen = torch.Generator(device)
    log = open(args.out/'train-log.jsonl', 'a')
    t0, running = time.time(), []
    for step in range(1, args.steps+1):
        z, actions, mask = train.sample(gen, args.batch)
        with autocast(dtype):
            loss = model.loss(z, actions, mask, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        opt.step()
        sched.step()
        with torch.no_grad():
            ema_decay = min(args.ema, (1+step)/(10+step))
            for e, m in zip(ema.parameters(), model.parameters()):
                e.lerp_(m, 1-ema_decay)
        running.append((loss.item(), grad))
        if step % 100 == 0 or step == args.steps:
            loss_avg, grad_avg = [sum(v)/len(v) for v in zip(*running)]
            row = dict(step=step, loss=loss_avg, grad_norm=grad_avg, lr=sched.get_last_lr()[0],
                       steps_per_s=len(running)/(time.time()-t0))
            running, t0 = [], time.time()
            if step % args.eval_every == 0 or step == args.steps:
                val_gen.manual_seed(1234)
                row['eval'] = evaluate(ema, val, val_gen)
                print(json.dumps(row), flush=True)
            log.write(json.dumps(row)+'\n')
            log.flush()
        if step % args.save_every == 0 or step == args.steps:
            save_policy(args.out/f'step-{step:06d}.pt', ema, dict(info, step=step))
    save_policy(args.out/'policy.pt', ema, dict(info, step=args.steps))


if __name__ == '__main__':
    main()
