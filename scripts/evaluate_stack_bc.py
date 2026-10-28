#!/usr/bin/env python3
"""Closed-loop evaluation of chunked BC policies from the original initial state.

Success follows StackTaskEnv: `target_count` placements in the goal-zone
structure, then hands free for `success_hold_steps`. Scenes are the first
`--scenes` seeds of the requested split from `--start-seed`. `--scripted`
evaluates the scripted controller (with recovery) on the same scenes.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_policies = {}


def run_episode(args):
    checkpoint, seed, task_kwargs, execute, noise_text = args
    import torch
    from assignment.training.data.task import StackTaskEnv, TaskConfig
    from assignment.training.data.signals import ActionNoise, NoiseSpec, row_hand_parts, row_lifted
    torch.set_num_threads(1)
    noise = ActionNoise(NoiseSpec.parse(noise_text), seed)
    task = TaskConfig(**task_kwargs)
    env = StackTaskEnv(task, seed)
    env.reset(('init', seed))
    if checkpoint == 'scripted':
        from assignment.training.data.scripted import ScriptedStackPolicy
        policy = ScriptedStackPolicy(env.env, 'nominal', seed, recovery=True)
    else:
        from assignment.training.data.teacher import BCPolicy
        policy = BCPolicy(env.env, dict(checkpoint=checkpoint, execute=execute))
    max_placed, first_placement, ground, info, success = 0, None, 0, {}, False
    touched = lifted = 0
    start = time.time()
    for step in range(task.horizon):
        action, _ = policy.action(env.env, env.env.elapsed_steps)
        if action is None:
            break
        action, _ = noise(action)
        _, _, terminated, truncated, info = env.step(action)
        ground += info['ground']
        row = env.env.history[-1]
        touched += bool((row_hand_parts(row)[0] >= 2).any())
        lifted += bool((row_lifted(row) & ~env.members).any())
        if info['placed'] > max_placed and first_placement is None:
            first_placement = step+1
        max_placed = max(max_placed, info['placed'])
        if terminated or truncated:
            success = bool(info['success'])
            break
    env.close()
    return dict(scene_seed=seed, success=success, steps=step+1, max_placed=int(max_placed),
                final_count=int(info.get('count', 0)), first_placement_step=first_placement, ground_falls=int(ground),
                grasp_steps=int(touched), lifted_steps=int(lifted),
                seconds=round(time.time()-start, 1))


def summarize(results, target):
    n = len(results)
    hist = Counter(min(r['max_placed'], target) for r in results)
    return dict(episodes=n, success=sum(r['success'] for r in results)/n,
                reached={f'placed>={k}': sum(r['max_placed'] >= k for r in results)/n for k in range(1, target+1)},
                max_placed_hist={str(k): hist[k] for k in sorted(hist)},
                grasped=sum(r['grasp_steps'] > 0 for r in results)/n,
                lifted=sum(r['lifted_steps'] >= 5 for r in results)/n,
                mean_steps_success=float(np.mean([r['steps'] for r in results if r['success']] or [np.nan])),
                ground_fall_episodes=sum(r['ground_falls'] > 0 for r in results)/n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoints', nargs='*', type=Path)
    p.add_argument('--scripted', action='store_true')
    p.add_argument('--split', default='val')
    p.add_argument('--start-seed', type=int, default=100000)
    p.add_argument('--scenes', type=int, default=100)
    p.add_argument('--target-count', type=int, default=2)
    p.add_argument('--horizon', type=int, default=1500)
    p.add_argument('--execute', type=int, default=None, help='Executed actions per chunk (default: training value)')
    p.add_argument('--noise', default='nominal', help='Action noise on executed actions (NoiseSpec.parse syntax)')
    p.add_argument('--workers', type=int, default=100)
    p.add_argument('--out', type=Path, default=None, help='Defaults to <checkpoint>.eval-<split>.json')
    args = p.parse_args()
    from assignment.training.data.signals import scene_split
    seeds, seed = [], args.start_seed
    while len(seeds) < args.scenes:
        if scene_split(seed) == args.split:
            seeds.append(seed)
        seed += 1
    task = dict(target_count=args.target_count, horizon=args.horizon, terminate_on_ground=False)
    targets = (['scripted'] if args.scripted else [])+[str(c) for c in args.checkpoints]
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
        for target in targets:
            t0 = time.time()
            results = list(pool.map(run_episode, [(target, s, task, args.execute, args.noise) for s in seeds]))
            report = dict(policy=target, split=args.split, seeds=[seeds[0], seeds[-1]], task=task, execute=args.execute,
                          noise=args.noise,
                          summary=summarize(results, args.target_count), wall_s=round(time.time()-t0, 1),
                          episodes=results)
            out = args.out or (Path(f'{target}.eval-{args.split}-t{args.target_count}.json') if target != 'scripted'
                               else Path(f'scripted.eval-{args.split}-t{args.target_count}.json'))
            out.write_text(json.dumps(report, indent=1))
            print(json.dumps(dict(policy=target, **report['summary'], wall_s=report['wall_s'])), flush=True)


if __name__ == '__main__':
    main()
