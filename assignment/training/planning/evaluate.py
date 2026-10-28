"""Stage 5. Closed-loop evaluation of the latent policy and goal-conditioned planner.

Every STRIDE=5 control steps, the front and side wrist cameras are rendered at 128 px
with EGL, tiled horizontally into a (128, 256, 3) frame, and passed to the agent, which
returns the next 5 raw actions. Each --configs entry is a JSON object of agent keyword
arguments; every (checkpoint, config, sample_seed) tuple writes its own JSON report.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time

import numpy as np

_worker = {}
_goals = {}


def summarize(results, target):
    n = len(results)
    hist = Counter(min(r['max_placed'], target) for r in results)
    return dict(episodes=n, success=sum(r['success'] for r in results)/n,
                reached={f'placed>={k}': sum(r['max_placed'] >= k for r in results)/n for k in range(1, target+1)},
                max_placed_hist={str(k): hist[k] for k in sorted(hist)},
                grasped=sum(r['grasp_steps'] > 0 for r in results)/n,
                lifted=sum(r['lifted_steps'] >= 5 for r in results)/n,
                mean_steps_success=(float(np.mean([r['steps'] for r in results if r['success']]))
                                    if any(r['success'] for r in results) else None),
                ground_fall_episodes=sum(r['ground_falls'] > 0 for r in results)/n)


def code_version():
    """Git commit of this checkout, with a -dirty suffix if tracked files are modified."""
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ('-dirty' if dirty else '')


def init_worker(counter, gpus):
    with counter.get_lock():
        rank = counter.value
        counter.value += 1
    gpu = gpus[rank % len(gpus)]
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch
    torch.set_num_threads(1)


def make_agent(kind, checkpoint, world_model, config):
    key = (kind, checkpoint, world_model, json.dumps(config, sort_keys=True))
    if key not in _worker:
        _worker.clear()
        from assignment.training.planning.agent import GoalPlanningAgent, LatentPolicyAgent
        kwargs = dict(config)
        if world_model:
            kwargs.setdefault('world_model', world_model)
        if kind == 'df':
            _worker[key] = LatentPolicyAgent(checkpoint, 'cuda', **kwargs)
        elif kind == 'goal':
            _worker[key] = GoalPlanningAgent(checkpoint, 'cuda', **kwargs)
        else:
            raise ValueError(kind)
    return _worker[key]


def scene_goal_frames(path, seed):
    """The scene's subgoal sequence and its initial frame."""
    if _goals.get('path') != path:
        with np.load(path) as f:
            _goals.clear()
            _goals.update(path=path, data={k: f[k] for k in f.files})
            _goals['row'] = {s: i for i, s in enumerate(_goals['data']['seeds'].tolist())}
    d, i = _goals['data'], _goals['row'][seed]
    if 'sub_frames' not in d:
        raise ValueError(f"{path} has no dense subgoals; regenerate it with goals.py --every N")
    return d['sub_frames'][i, :d['sub_count'][i]], d['init_frames'][i]


def run_episode(job):
    kind, checkpoint, world_model, config, seed, task_kwargs, resolution, video, goals, sample_seed = job
    from assignment.training.data.cache import STRIDE
    from assignment.training.data.signals import row_hand_parts, row_lifted
    from assignment.training.data.task import StackTaskEnv, TaskConfig
    agent = make_agent(kind, checkpoint, world_model, config)
    episode_seed = seed + 1_000_000*sample_seed
    goal_frames = None
    if kind == 'goal':
        goal_frames, init_frame = scene_goal_frames(goals, seed)
        agent.reset(goal_frames, init_frame, episode_seed)
    else:
        agent.reset(episode_seed)
    task = TaskConfig(**task_kwargs)
    env = StackTaskEnv(task, seed)
    env.reset(('init', seed))
    env.env.resolution = resolution
    max_placed, first_placement, ground, info, success = 0, None, 0, {}, False
    touched = lifted = 0
    stage_steps = []
    frames, start, agent_s = [], time.time(), 0.
    for step in range(task.horizon):
        if step % STRIDE == 0:
            frame = np.concatenate(list(env.env.render()), axis=1)
            t0 = time.time()
            block = agent.step(frame)
            agent_s += time.time()-t0
            if video:
                frames.append(frame if goal_frames is None else
                              np.concatenate([frame, goal_frames[min(agent.index, len(goal_frames)-1)]], axis=0))
        _, _, terminated, truncated, info = env.step(block[step % STRIDE])
        while len(stage_steps) < info['placed']:
            stage_steps.append(step+1)
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
    if video:
        from assignment.training.data.recording import write_video
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        write_video(Path(video), [f[None] for f in frames], STRIDE)
    goal_info = dict(goal_index=agent.index, goals=len(goal_frames)) if kind == 'goal' else {}
    return dict(scene_seed=seed, success=success, steps=step+1, max_placed=int(max_placed), **goal_info,
                final_count=int(info.get('count', 0)), first_placement_step=first_placement, ground_falls=int(ground),
                grasp_steps=int(touched), lifted_steps=int(lifted), stage_steps=stage_steps, agent_s=round(agent_s, 1),
                seconds=round(time.time()-start, 1))


def tag(config):
    return '_'.join(f'{k}{v}' for k, v in sorted(config.items())) or 'default'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoints', nargs='+', type=Path)
    p.add_argument('--world-model', type=Path, default=None,
                   help="Optional world-model checkpoint path (overrides info['encoder'] in the policy checkpoint)")
    p.add_argument('--kind', choices=('df', 'goal'), default='df')
    p.add_argument('--goals', type=Path, default=None, help='goals.npz from assignment.training.planning.goals (--kind goal)')
    p.add_argument('--configs', nargs='+', default=['{}'], help='JSON agent kwargs, e.g. {"generate": 16, "execute": 1}')
    p.add_argument('--split', default='val')
    p.add_argument('--start-seed', type=int, default=100000)
    p.add_argument('--scenes', type=int, default=100)
    p.add_argument('--target-count', type=int, default=4)
    p.add_argument('--horizon', type=int, default=1795)
    p.add_argument('--resolution', type=int, default=128)
    p.add_argument('--videos', type=int, default=0, help='Write preview videos for the first N scenes')
    p.add_argument('--sample-seeds', type=int, nargs='+', default=[0],
                   help='Policy sampler seeds; each episode is seeded with scene_seed + 1e6 * sample_seed, '
                        'and every sample seed gets its own report')
    p.add_argument('--gpus', default='0',
                   help='Comma-separated physical GPU ids; each worker sets CUDA_VISIBLE_DEVICES to one of them')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out-dir', type=Path, default=None, help='Defaults to <checkpoint dir>/eval')
    args = p.parse_args()
    if args.kind == 'goal' and args.goals is None:
        p.error('--kind goal requires --goals')
    from assignment.training.data.signals import scene_split
    seeds, seed = [], args.start_seed
    while len(seeds) < args.scenes:
        if scene_split(seed) == args.split:
            seeds.append(seed)
        seed += 1
    task = dict(target_count=args.target_count, horizon=args.horizon, terminate_on_ground=False)
    ctx = mp.get_context('spawn')
    gpus = [int(g) for g in args.gpus.split(',')]
    wm = str(args.world_model) if args.world_model else None
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=init_worker,
                             initargs=(ctx.Value('i', 0), gpus)) as pool:
        for checkpoint in args.checkpoints:
            for config in map(json.loads, args.configs):
                for sample_seed in args.sample_seeds:
                    out_dir = args.out_dir or checkpoint.parent/'eval'
                    name = f'{checkpoint.parent.name}.{checkpoint.stem}.{args.kind}.{tag(config)}.{args.split}-t{args.target_count}'
                    if args.world_model:
                        name += f'.wm-{args.world_model.parent.name}-{args.world_model.stem}'
                    if args.goals:
                        name += f'.{args.goals.parent.name}'
                    name += f'.seed{args.start_seed}-n{args.scenes}-h{args.horizon}'
                    if sample_seed:
                        name += f'.sample{sample_seed}'
                    t0 = time.time()
                    jobs = [(args.kind, str(checkpoint), wm, config, s, task, args.resolution,
                             str(out_dir/'videos'/f'{name}.s{s}.mp4') if i < args.videos else None,
                             args.goals and str(args.goals), sample_seed)
                            for i, s in enumerate(seeds)]
                    results = list(pool.map(run_episode, jobs))
                    summary = summarize(results, args.target_count)
                    summary['mean_max_placed'] = float(np.mean([r['max_placed'] for r in results]))
                    report = dict(code=code_version(), kind=args.kind, policy=str(checkpoint), world_model=wm, config=config, split=args.split,
                                  goals=args.goals and str(args.goals), sample_seed=sample_seed,
                                  seeds=[seeds[0], seeds[-1]], task=task, summary=summary,
                                  wall_s=round(time.time()-t0, 1), episodes=results)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir/f'{name}.json').write_text(json.dumps(report, indent=1))
                    print(json.dumps(dict(policy=str(checkpoint), config=config, sample_seed=sample_seed,
                                          **summary, wall_s=report['wall_s'])), flush=True)


if __name__ == '__main__':
    main()
