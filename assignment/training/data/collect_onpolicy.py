"""Stage 4. Collect on-policy chunk rollouts for finetuning the world-model predictor.

The latent policy drives each scene just like the single-sample controller, executing the
first action block of a freshly sampled 16-block chunk at every block boundary. At a random
--keep fraction of block boundaries that already have a full world-model context (C=3 frames),
the policy also draws --candidates chunks of 16 blocks; each candidate chunk is executed in
the simulator from the saved state, and every resulting frame is encoded with the frozen
world-model encoder. Each recorded state stores:

  seed, step   scene seed and control step (int32)
  goal         active subgoal index under the threshold rule on val scenes, else 0 (int16)
  past         (context-1, 125) normalized action blocks executed between the context frames (float16)
  chunks       (K, 16, 125) normalized candidate action blocks (float16)
  ctx          (context, 192) unnormalized encoder latents of the context frames (float16)
  tgt          (K, 16, 192) unnormalized encoder latents after each executed candidate block (float16)

States where any candidate triggers a simulator exception are skipped. Shards of --shard
states are written as <split>-r<rank>-<n>.npz.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback


def worker(rank, args, gpu, seeds, split):
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    try:
        import numpy as np
        import torch
        import mujoco
        torch.set_num_threads(1)
        from assignment.training.data.cache import STRIDE
        from assignment.training.data.recording import STATE_SPEC
        from assignment.training.data.task import StackTaskEnv, TaskConfig
        from assignment.training.training.jepa import amp_dtype, autocast, load_model
        from assignment.training.training.policy import load_policy
        policy, info = load_policy(args.policy, 'cuda')
        encoder = load_model(args.world_model or info['encoder'], 'cuda')[0].eval()
        context = encoder.cfg.context
        z_mean = torch.tensor(info['z_mean'], device='cuda')
        z_std = torch.tensor(info['z_std'], device='cuda')
        a_mean = np.asarray(info['action_mean'], np.float32)
        a_std = np.asarray(info['action_std'], np.float32)
        gen = torch.Generator(device='cuda')
        dtype = amp_dtype()
        k = args.candidates
        goals = None
        if split == 'val':
            with np.load(args.goals) as f:
                sub_frames, sub_count, init_frames = f['sub_frames'], f['sub_count'], f['init_frames']
                goals = {s: (sub_frames[i, :sub_count[i]], init_frames[i])
                         for i, s in enumerate(f['seeds'].tolist()) if s in set(seeds)}

        def encode(frames):
            x = torch.from_numpy(np.ascontiguousarray(frames)).cuda()[:, None]
            with torch.no_grad(), autocast(dtype):
                return encoder.encode(x)[:, 0].float()

        def save(sim):
            state = np.empty(mujoco.mj_stateSize(sim.model, STATE_SPEC), np.float64)
            mujoco.mj_getState(sim.model, sim.data, state, STATE_SPEC)
            return dict(state=state, targets=sim.targets.copy(), elapsed=sim.elapsed_steps, status=sim.status,
                        stop=sim.stop_reason, fin=sim.finalization_steps, hlen=len(sim.history))

        def restore(sim, snap):
            mujoco.mj_setState(sim.model, sim.data, snap['state'], STATE_SPEC)
            mujoco.mj_forward(sim.model, sim.data)
            sim.targets = snap['targets'].copy()
            sim.data.ctrl[:] = sim.targets
            sim.elapsed_steps = snap['elapsed']
            sim.status, sim.stop_reason = snap['status'], snap['stop']
            sim.finalization_steps = snap['fin']
            while len(sim.history) > snap['hlen']:
                sim.history.pop()
            sim.data.warning.number[:] = 0

        def tile(sim):
            return np.concatenate(list(sim.render()), axis=1)

        buf, shard_id, total = [], 0, 0
        out = Path(args.out)

        def flush():
            nonlocal buf, shard_id
            if not buf:
                return
            name = f'{split}-r{rank:03d}-{shard_id:05d}.npz'
            with open(out / f'.tmp-{name}', 'wb') as f:
                np.savez(f, **{key: np.stack([b[key] for b in buf]) for key in buf[0]})
            os.replace(out / f'.tmp-{name}', out / name)
            buf, shard_id = [], shard_id + 1

        for seed in seeds:
            gen.manual_seed(int(seed))
            env = StackTaskEnv(TaskConfig(target_count=4, horizon=args.horizon, terminate_on_ground=False), seed)
            env.reset(('init', seed))
            env.env.resolution = 128
            sim = env.env
            frames, executed = [], []
            gi, gz, gap = 0, None, None
            if goals is not None:
                gframes, ginit = goals[seed]
                gz = encode(gframes)
                gap = (gz - torch.cat([encode(ginit[None]), gz[:-1]])).square().sum(-1)
            block = None
            for step in range(args.horizon):
                if step % STRIDE == 0:
                    z = encode(tile(sim)[None])[0]
                    frames = (frames + [z])[-context:]
                    if gz is not None:
                        while gi < len(gz) - 1 and (z - gz[gi]).square().sum() < args.tau * gap[gi]:
                            gi += 1
                    zp = (z - z_mean) / z_std
                    blocks = policy.sample(zp[None, None].expand(k, -1, -1), 16, 10, 0, gen).float()
                    raw = np.clip(blocks.cpu().numpy() * a_std + a_mean, -1, 1).reshape(k, 16, STRIDE, -1)
                    if len(frames) == context and np.random.default_rng([seed, step]).random() < args.keep:
                        snap = save(sim)
                        shots = np.zeros((k, 16, 128, 256, 3), np.uint8)
                        ok = True
                        for c in range(k):
                            restore(sim, snap)
                            try:
                                for b in range(16):
                                    for a in raw[c, b]:
                                        sim.step(a)
                                    shots[c, b] = tile(sim)
                            except RuntimeError:
                                ok = False
                                break
                        restore(sim, snap)
                        if ok:
                            tz = encode(shots.reshape(-1, 128, 256, 3)).reshape(k, 16, -1)
                            past = torch.stack(executed[len(executed) - context + 1:]) if context > 1 else \
                                torch.zeros(0, blocks.size(-1))
                            buf.append(dict(seed=np.int32(seed), step=np.int32(step), goal=np.int16(gi),
                                            past=past.cpu().numpy().astype(np.float16),
                                            chunks=blocks.cpu().numpy().astype(np.float16),
                                            ctx=torch.stack(frames).cpu().numpy().astype(np.float16),
                                            tgt=tz.cpu().numpy().astype(np.float16)))
                            total += 1
                            if len(buf) >= args.shard:
                                flush()
                    block = raw[0, 0]
                    executed.append(torch.from_numpy((block.reshape(-1) - a_mean) / a_std).cuda())
                _, _, terminated, truncated, _ = env.step(block[step % STRIDE])
                if terminated or truncated:
                    break
            env.close()
        flush()
        print(json.dumps(dict(rank=rank, split=split, states=total, done=True)), flush=True)
    except Exception:
        print(json.dumps(dict(rank=rank, error=traceback.format_exc())), flush=True)
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--world-model', type=Path, default=None, help="Defaults to the policy's info['encoder']")
    p.add_argument('--goals', type=Path, required=True, help='Subgoal file; its first --val-scenes seeds are val')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpus', default='0')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--val-workers', type=int, default=2)
    p.add_argument('--train-scenes', type=int, default=4000)
    p.add_argument('--train-offset', type=int, default=0, help='Skip this many train scenes')
    p.add_argument('--val-scenes', type=int, default=300)
    p.add_argument('--candidates', type=int, default=16)
    p.add_argument('--horizon', type=int, default=1795)
    p.add_argument('--keep', type=float, default=0.25, help='Fraction of block boundaries recorded')
    p.add_argument('--tau', type=float, default=0.25, help='Subgoal threshold for the val goal index')
    p.add_argument('--shard', type=int, default=64)
    p.add_argument('--rank-offset', type=int, default=0, help='Added to worker ranks in shard names')
    args = p.parse_args()
    if args.train_scenes > 0 and not 1 <= args.val_workers < args.workers:
        p.error('need 1 <= --val-workers < --workers so that both train and val shards are written')
    import numpy as np
    from assignment.training.data.signals import scene_split
    args.out.mkdir(parents=True, exist_ok=True)
    train, s = [], 0
    while len(train) < args.train_offset + args.train_scenes:
        if scene_split(s) == 'train':
            train.append(s)
        s += 1
    train = train[args.train_offset:]
    with np.load(args.goals) as f:
        val = f['seeds'][:args.val_scenes].tolist()
    gpus = [int(g) for g in args.gpus.split(',')]
    val_workers = args.val_workers if args.train_scenes > 0 else args.workers
    n_train = max(0, args.workers - val_workers)
    o = args.rank_offset
    jobs = [(o + r, train[r::n_train], 'train') for r in range(n_train)]
    jobs += [(o + n_train + r, val[r::val_workers], 'val') for r in range(val_workers)]
    meta = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out / f'meta-{o}.json').write_text(json.dumps(meta, indent=1))
    ctx = mp.get_context('spawn')
    procs = [ctx.Process(target=worker, args=(r, args, gpus[r % len(gpus)], seeds, split)) for r, seeds, split in jobs]
    t0 = time.time()
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
        if proc.exitcode:
            raise SystemExit(f'on-policy collector worker failed with exit code {proc.exitcode}')
    print(json.dumps(dict(done=True, wall_s=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
