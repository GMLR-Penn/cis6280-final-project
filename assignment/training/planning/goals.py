"""Stage 5. Render per-scene goal and subgoal images with the scripted reference controller.

Runs the scripted controller (with recovery) on each validation scene and renders the tiled
front|side frame (128x256 uint8) at the first step where the goal-zone structure holds k
placed boxes (k = 1..--stages). With --every N (e.g. --every 80), also records init_frames
and dense subgoals sub_frames every N control steps up to the final stage. If the scripted run
never reaches a stage, that stage keeps step -1 and its frame is the run's last frame; the dense
subgoals then simply stop where the run stops.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np


def init_worker(counter, gpus):
    with counter.get_lock():
        rank = counter.value
        counter.value += 1
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpus[rank % len(gpus)])


def scene_goals(job):
    seed, stages, horizon, resolution, every = job
    from assignment.training.data.scripted import ScriptedStackPolicy
    from assignment.training.data.task import StackTaskEnv, TaskConfig
    env = StackTaskEnv(TaskConfig(target_count=stages, horizon=horizon, terminate_on_ground=False), seed)
    env.reset(('init', seed))
    env.env.resolution = resolution

    def render():
        return np.concatenate(list(env.env.render()), axis=1)
    policy = ScriptedStackPolicy(env.env, 'nominal', seed, recovery=True)
    frames = np.zeros((stages, resolution, 2*resolution, 3), np.uint8)
    steps = np.full(stages, -1)
    init, sub, sub_steps = render(), [], []
    step = 0
    for step in range(horizon):
        action, _ = policy.action(env.env, env.env.elapsed_steps)
        if action is None:
            break
        _, _, terminated, truncated, info = env.step(action)
        for k in range(1, min(info['placed'], stages)+1):
            if steps[k-1] < 0:
                steps[k-1] = step+1
                frames[k-1] = render()
        if every and steps[-1] < 0 and (step+1) % every == 0:
            sub.append(render())
            sub_steps.append(step+1)
        if every and steps[-1] >= 0:
            sub.append(frames[-1])
            sub_steps.append(steps[-1])
            break
        if terminated or truncated:
            break
    if every and not sub:
        sub.append(render())
        sub_steps.append(step+1)
    frames[steps < 0] = render()
    env.close()
    return seed, frames, steps, init, sub, sub_steps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--split', default='val')
    p.add_argument('--start-seed', type=int, default=100000)
    p.add_argument('--scenes', type=int, default=100)
    p.add_argument('--stages', type=int, default=4)
    p.add_argument('--horizon', type=int, default=1795)
    p.add_argument('--resolution', type=int, default=128)
    p.add_argument('--every', type=int, default=80,
                   help='Store subgoal frames every N control steps until the last stage (0 for stage goals only)')
    p.add_argument('--gpus', default='0')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    from assignment.training.data.signals import scene_split
    seeds, seed = [], args.start_seed
    while len(seeds) < args.scenes:
        if scene_split(seed) == args.split:
            seeds.append(seed)
        seed += 1
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=init_worker,
                             initargs=(ctx.Value('i', 0), [int(g) for g in args.gpus.split(',')])) as pool:
        results = list(pool.map(scene_goals, [(s, args.stages, args.horizon, args.resolution, args.every)
                                              for s in seeds]))
    args.out.mkdir(parents=True, exist_ok=True)
    frames = np.stack([r[1] for r in results])
    steps = np.stack([r[2] for r in results])
    extra = {}
    if args.every:
        count = np.array([len(r[4]) for r in results])
        sub = np.zeros((len(seeds), count.max(), *frames.shape[2:]), np.uint8)
        sub_steps = np.full((len(seeds), count.max()), -1)
        for i, r in enumerate(results):
            sub[i, :count[i]], sub_steps[i, :count[i]] = r[4], r[5]
        extra = dict(init_frames=np.stack([r[3] for r in results]), sub_frames=sub, sub_steps=sub_steps,
                     sub_count=count)
    np.savez_compressed(args.out/'goals.npz', seeds=np.array(seeds), frames=frames, steps=steps, **extra)
    import imageio.v2 as imageio
    imageio.imwrite(args.out/'preview.png', np.concatenate([np.concatenate(list(f), axis=1) for f in frames[:6]], axis=0))
    reached = {f'stage{k+1}': float((steps[:, k] >= 0).mean()) for k in range(args.stages)}
    meta = dict(source='scripted nominal + recovery', split=args.split, seeds=[seeds[0], seeds[-1]], scenes=len(seeds),
                stages=args.stages, horizon=args.horizon, goal='first step with k placed cubes', reached=reached,
                every=args.every)
    if args.every:
        meta['subgoals_per_scene'] = dict(mean=float(extra['sub_count'].mean()), min=int(extra['sub_count'].min()),
                                          max=int(extra['sub_count'].max()))
    (args.out/'goals.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
