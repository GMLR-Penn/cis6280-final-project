#!/usr/bin/env python3
"""Train a chunked MLP BC policy on cached scripted (DART) data.

The whole cache lives on the GPU; chunks are gathered from row offsets and
masked at episode ends. A scene-level holdout gives the validation loss and the
per-dimension action error used as the next DART noise level (dart_noise.json).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assignment.training.data.teacher import ChunkMLP, load_cache, sample_rows, save_bc  # noqa: E402


def holdout(seed, fraction):
    return int(hashlib.sha256(f'bc-holdout/{seed}'.encode()).hexdigest(), 16) % 1000 < fraction*1000


def gather(actions, rows, ends, chunk):
    idx = rows[:, None]+torch.arange(chunk, device=rows.device)
    mask = idx < ends[:, None]
    return actions[idx.clamp(max=len(actions)-1)], mask


@torch.no_grad()
def errors(model, obs, actions, rows, ends, chunk, execute, batch=65536):
    """Mean squared error over all chunk steps, and per-dimension error std over the executed steps."""
    total, count, sq, n = 0., 0., torch.zeros(25, device=obs.device), 0.
    for i in range(0, len(rows), batch):
        r, e = rows[i:i+batch], ends[i:i+batch]
        target, mask = gather(actions, r, e, chunk)
        diff = (model(obs[r])-target)*mask[..., None]
        total += float((diff**2).sum()); count += float(mask.sum())*25
        head = diff[:, :execute]
        sq += (head**2).sum((0, 1)); n += float(mask[:, :execute].sum())
    return total/count, (sq/n).sqrt().cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--chunk', type=int, default=10)
    p.add_argument('--execute', type=int, default=5)
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.)
    p.add_argument('--max-placements', type=int, default=None,
                   help='Train only up to shortly after this many placements of each episode')
    p.add_argument('--conditions', nargs='*', default=None)
    p.add_argument('--scene-limit', type=int, default=None, help='Use only the first N training scenes')
    p.add_argument('--holdout', type=float, default=.05)
    p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--obs-noise', type=float, default=0., help='Gaussian noise on normalized inputs')
    p.add_argument('--save-every', type=int, default=20000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    obs_parts, act_parts, rows_train, rows_val, offset = [], [], [], [], 0
    scenes_used = set()
    for path in args.cache:
        cache = load_cache(path)
        index = cache['index']
        chosen = [i for i, e in enumerate(index['episodes']) if not args.conditions or e['condition'] in args.conditions]
        seeds = sorted({index['episodes'][i]['scene_seed'] for i in chosen})
        train_seeds = [s for s in seeds if not holdout(s, args.holdout)]
        if args.scene_limit:
            train_seeds = train_seeds[:args.scene_limit]
        train_seeds = set(train_seeds)
        scenes_used |= train_seeds
        val = [i for i in chosen if holdout(index['episodes'][i]['scene_seed'], args.holdout)]
        train = [i for i in chosen if index['episodes'][i]['scene_seed'] in train_seeds]
        for target, episodes in ((rows_train, train), (rows_val, val)):
            if episodes:
                r, e = sample_rows(index, episodes, args.max_placements)
                target.append((r+offset, e+offset))
        obs_parts.append(np.asarray(cache['obs'])); act_parts.append(np.asarray(cache['actions']))
        offset += len(obs_parts[-1])
    obs = torch.from_numpy(np.concatenate(obs_parts)).to(device)
    actions = torch.from_numpy(np.concatenate(act_parts)).to(device)
    cat = lambda parts, k: torch.from_numpy(np.concatenate([x[k] for x in parts])).to(device)  # noqa: E731
    train_rows, train_ends = cat(rows_train, 0), cat(rows_train, 1)
    val_rows, val_ends = cat(rows_val, 0), cat(rows_val, 1)

    model = ChunkMLP(obs.shape[1], args.chunk, args.width, args.depth, args.dropout).to(device)
    model.set_normalization(obs[train_rows].mean(0), obs[train_rows].std(0))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = 1000
    schedule = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s+1)/warmup)*.5*(1+math.cos(math.pi*min(1., s/args.steps))))
    info = dict(vars(args), cache=[str(c) for c in args.cache], out=str(args.out), train_samples=len(train_rows),
                val_samples=len(val_rows), train_scenes=len(scenes_used),
                parameters=sum(p.numel() for p in model.parameters()))
    print(json.dumps({k: info[k] for k in ('train_samples', 'val_samples', 'train_scenes', 'parameters')}), flush=True)
    log, start = [], time.time()
    for step in range(1, args.steps+1):
        pick = torch.randint(len(train_rows), (args.batch,), device=device)
        r, e = train_rows[pick], train_ends[pick]
        target, mask = gather(actions, r, e, args.chunk)
        x = obs[r]
        if args.obs_noise:
            x = x+torch.randn_like(x)*args.obs_noise*model.obs_std
        with torch.autocast('cuda', torch.bfloat16, enabled=device.type == 'cuda'):
            pred = model(x)
        loss = (((pred.float()-target)**2)*mask[..., None]).sum()/(mask.sum()*25)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); schedule.step()
        if step % 2000 == 0 or step == args.steps:
            model.eval()
            val_mse, _ = errors(model, obs, actions, val_rows, val_ends, args.chunk, args.execute)
            model.train()
            entry = dict(step=step, train_mse=float(loss.detach()), val_mse=val_mse, lr=schedule.get_last_lr()[0],
                         seconds=round(time.time()-start, 1))
            log.append(entry)
            print(json.dumps(entry), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_bc(args.out/f'step-{step:06d}.pt', model, dict(info, step=step))
    model.eval()
    val_mse, sigma = errors(model, obs, actions, val_rows, val_ends, args.chunk, args.execute)
    (args.out/'dart_noise.json').write_text(json.dumps(sigma.round(5).tolist()))
    (args.out/'train-log.json').write_text(json.dumps(dict(info=info, log=log, final_val_mse=val_mse,
                                                           dart_sigma=sigma.tolist()), indent=1))
    save_bc(args.out/'model.pt', model, dict(info, step=args.steps))
    print(json.dumps(dict(saved=str(args.out/'model.pt'), val_mse=val_mse, dart_sigma_mean=float(sigma.mean()),
                          dart_sigma_wrist=sigma[:3].round(4).tolist())), flush=True)


if __name__ == '__main__':
    main()
