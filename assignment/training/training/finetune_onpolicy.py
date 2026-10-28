"""Stage 4. Finetune a world model's predictor on on-policy chunk rollouts.

The encoder and projector stay frozen so the 192-D latent space, the pre-encoded latents,
the policy trained on them, and the subgoal latents remain unchanged; only the predictor,
its output projection (with BatchNorm layers kept in eval mode), and the action encoder
are updated. The training objective is the 16-block open-loop autoregressive rollout MSE
(backpropagated through all 16 rollout steps) between the predicted latents and the true
encoder latents after each executed block of a sampled candidate chunk.

Every --eval-every steps, validation states are scored on per-step rollout MSE, top-1
agreement with the true closest-to-subgoal candidate (agree), and normalized true-cost
ranking regret (regret). latest.pt is written at every evaluation, and model.pt at the start and whenever
regret improves. Train shards are re-read every --reload-every steps so training can run
either on a completed dataset or concurrently while collection writes shards.
"""
import argparse
import glob
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from assignment.training.training.jepa import amp_dtype, autocast, load_model, save_model

KEYS = ('seed', 'step', 'goal', 'past', 'chunks', 'ctx', 'tgt')


def load_shards(root, split):
    files = sorted(glob.glob(str(Path(root) / f'{split}-r*.npz')))
    parts, loaded = {}, []
    for f in files:
        try:
            with np.load(f) as d:
                item = {k: d[k] for k in KEYS}
        except Exception as e:
            print(f'skipping unreadable shard {f}: {e}', flush=True)
            continue
        loaded.append(f)
        for k, v in item.items():
            parts.setdefault(k, []).append(v)
    return ({k: np.concatenate(v) for k, v in parts.items()} if parts else {}), loaded


def rollout(model, ctx, acts, steps, grad=False):
    """Autoregressive rollout, differentiable when grad=True. ctx (B,C,D), acts (B,C+steps-1,A)."""
    c = model.cfg.context
    dtype = amp_dtype(training=grad)
    with torch.set_grad_enabled(grad):
        with autocast(dtype):
            act_emb = model.action_encoder(acts)
        hist, preds = ctx, []
        for _ in range(steps):
            window = hist[:, -c:]
            first = hist.size(1) - window.size(1)
            with autocast(dtype):
                pred = model.predict(window, act_emb[:, first:first + window.size(1)])[:, -1:]
            preds.append(pred.float())
            hist = torch.cat([hist, preds[-1]], 1)
    return torch.cat(preds, 1)


def freeze_for_training(model):
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(('predictor.', 'pred_proj.', 'action_encoder.')))
    model.eval()
    model.predictor.train()
    model.action_encoder.train()
    for module in model.pred_proj.modules():
        if isinstance(module, torch.nn.Linear):
            module.train()


def goal_latents(model, goals, seeds, device):
    out = {}
    dtype = amp_dtype()
    with np.load(goals) as f:
        row = {s: i for i, s in enumerate(f['seeds'].tolist())}
        sub_frames, sub_count = f['sub_frames'], f['sub_count']
        for s in sorted(set(int(x) for x in seeds)):
            i = row[s]
            frames = torch.from_numpy(sub_frames[i, :sub_count[i]]).to(device)[:, None]
            with torch.no_grad(), autocast(dtype):
                out[s] = model.encode(frames)[:, 0].float()
    return out


@torch.no_grad()
def evaluate(model, val, goal, true, device, chunk=256):
    model.eval()
    s, k = val['chunks'].shape[:2]
    finals, err = [], []
    for i in range(0, s, chunk):
        sl = slice(i, min(i + chunk, s))
        n = sl.stop - sl.start
        get = lambda key: torch.from_numpy(val[key][sl]).to(device).float()  # noqa: E731
        ctx, past, chunks, tgt = get('ctx'), get('past'), get('chunks'), get('tgt')
        acts = torch.cat([past[:, None].expand(-1, k, -1, -1), chunks], 2).flatten(0, 1)
        pred = rollout(model, ctx[:, None].expand(-1, k, -1, -1).flatten(0, 1), acts, chunks.size(2))
        pred = pred.view(n, k, *pred.shape[1:])
        finals.append(pred[:, :, -1])
        err.append((pred - tgt).square().mean(-1).flatten(0, 1))
    final, err = torch.cat(finals), torch.cat(err)
    pick = (final - goal[:, None]).square().sum(-1).argmin(1)
    agree = float((pick == true.argmin(1)).float().mean())
    regret = float(((true.gather(1, pick[:, None])[:, 0] - true.min(1).values) / (true.std(1) + 1e-8)).mean())
    return dict(h1=float(err[:, 0].mean()), h4=float(err[:, 3].mean()), h16=float(err[:, -1].mean()),
                agree=round(agree, 4), regret=round(regret, 4))


class Train:
    def __init__(self, root, device):
        self.root, self.device, self.files, self.n = root, device, None, 0
        self.reload()

    def reload(self):
        data, files = load_shards(self.root, 'train')
        if files == self.files or not data:
            return False
        self.files = files
        self.ctx, self.past, self.chunks, self.tgt = (torch.from_numpy(data[k]).to(self.device)
                                                      for k in ('ctx', 'past', 'chunks', 'tgt'))
        self.n, self.k = self.chunks.shape[:2]
        return True

    def sample(self, batch, gen):
        s = torch.randint(self.n, (batch,), device=self.device, generator=gen)
        c = torch.randint(self.k, (batch,), device=self.device, generator=gen)
        acts = torch.cat([self.past[s], self.chunks[s, c]], 1).float()
        return self.ctx[s].float(), acts, self.tgt[s, c].float()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--init', type=Path, required=True, help='World model to start from (the one used to collect)')
    p.add_argument('--goals', type=Path, required=True, help='Subgoal file covering the val shards')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--reload-every', type=int, default=2000)
    p.add_argument('--min-states', type=int, default=5000, help='Wait until this many train states exist')
    args = p.parse_args()
    device = 'cuda'
    args.out.mkdir(parents=True, exist_ok=True)
    model, info = load_model(args.init, device)
    freeze_for_training(model)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1., (s + 1) / args.warmup) * 0.5 * (
        1 + math.cos(math.pi * min(1., s / args.steps))))
    train = Train(args.data, device)
    while train.n < args.min_states:
        print(json.dumps(dict(waiting=True, states=train.n)), flush=True)
        time.sleep(300)
        train.reload()
    val, _ = load_shards(args.data, 'val')
    if not val:
        raise SystemExit(f'no readable val-*.npz shards in {args.data}; collect them with --val-workers >= 1')
    goals = goal_latents(model, args.goals, val['seed'], device)
    goal = torch.stack([goals[int(s)][int(i)] for s, i in zip(val['seed'], val['goal'])])
    true = (torch.from_numpy(val['tgt'][:, :, -1]).to(device).float() - goal[:, None]).square().sum(-1)
    meta = dict(info, onpolicy=str(args.data), init=str(args.init))
    log = open(args.out / 'train-log.jsonl', 'a')
    row = dict(step=0, train_states=train.n, val_states=len(val['seed']), eval=evaluate(model, val, goal, true, device))
    freeze_for_training(model)
    print(json.dumps(row), flush=True)
    log.write(json.dumps(row) + '\n')
    best = row['eval']['regret']
    save_model(args.out / 'model.pt', model, dict(meta, step=0))
    gen = torch.Generator(device).manual_seed(0)
    t0, sums = time.time(), {}
    for step in range(1, args.steps + 1):
        ctx, acts, tgt = train.sample(args.batch, gen)
        per = (rollout(model, ctx, acts, tgt.size(1), grad=True) - tgt).square().mean(-1).mean(0)
        loss = per.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(params, 1.))
        opt.step()
        sched.step()
        for key, v in (('loss', loss.item()), ('h1', per[0].item()), ('h16', per[-1].item()), ('grad', grad)):
            sums[key] = sums.get(key, 0.) + v
        if step % 200 == 0:
            row = dict(step=step, lr=sched.get_last_lr()[0], train_states=train.n,
                       sec_per_step=round((time.time() - t0) / 200, 3), **{k: v / 200 for k, v in sums.items()})
            t0, sums = time.time(), {}
            if step % args.eval_every == 0 or step == args.steps:
                row['eval'] = evaluate(model, val, goal, true, device)
                freeze_for_training(model)
                save_model(args.out / 'latest.pt', model, dict(meta, step=step))
                if row['eval']['regret'] < best:
                    best = row['eval']['regret']
                    save_model(args.out / 'model.pt', model, dict(meta, step=step))
                    row['saved'] = True
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + '\n')
            log.flush()
        if step % args.reload_every == 0 and train.reload():
            print(json.dumps(dict(reloaded=True, train_states=train.n)), flush=True)


if __name__ == '__main__':
    main()
