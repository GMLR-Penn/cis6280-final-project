"""Stage 1. Record stacking episodes in the v0.3 trajectory format.

Conditions (scripted or BC): nominal | iid:SIGMA[-MAX] | ou:SIGMA[-MAX]:RHO, optionally
suffixed with @MODE (VariedBoxController intervention), +recover and +vary.
Example:
  python -m assignment.training.data.collect --out bc-noise-4cube-20k --source bc \
    --checkpoints checkpoints/bc/step-050000.pt --execute 5 --stop-count 4 --max-steps 1795 \
    --conditions 'iid:0.005-0.2,log,z=0.1' --scenes 20000 --start-seed 400000
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import time

_counter = None


def init_worker(counter, gpus):
    global _counter
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpus[index % len(gpus)])


def parse_condition(text):
    from assignment.training.data.signals import NoiseSpec
    name, flags = text, set()
    while text.endswith(('+recover', '+vary', '+sweep')):
        text, _, flag = text.rpartition('+')
        flags.add(flag)
    noise_text, _, mode = text.partition('@')
    return name, dict(kind='scripted', mode=mode or 'nominal', recovery='recover' in flags,
                      variation=('vary' in flags) or None, sweep='sweep' in flags), asdict(NoiseSpec.parse(noise_text))


def slug(name):
    name = re.sub(r'v=([^,+@]+)', lambda m: 'v'+Path(m.group(1)).stem, name)
    return (name.replace(':', '-').replace('@', '-at-').replace('+', '-').replace('.', 'p')
            .replace(',', '_').replace('=', ''))


def bc_policy(path, args):
    """BC policy spec with the recorder stop rule and provenance."""
    import torch
    path = Path(path).resolve()
    info = torch.load(path, map_location='cpu', weights_only=False)['info']
    evaluations = {p.name: json.loads(p.read_text()) for p in path.parent.glob(f'{path.name}.eval-*.json')}
    return dict(kind='bc', checkpoint=str(path), execute=args.execute, stop_count=args.stop_count,
                stop_hold=args.stop_hold, max_steps=args.max_steps), dict(
        tag=f'{path.parent.name}-{path.stem}', checkpoint=str(path), training=info,
        evaluation={k: {m: v[m] for m in ('success', 'reached', 'task', 'execute') if m in v}
                    for k, v in evaluations.items()})


def run_record(root, job, render):
    from assignment.training.data.recording import record_episode
    return record_episode(root, job, render=render)


def run_validate(root, record):
    import traceback
    from assignment.training.data.recording import validate_episode
    try:
        return dict(validate_episode(root, record), passed=True)
    except Exception as exc:
        return dict(episode_id=record['episode_id'], passed=False, error=repr(exc), traceback=traceback.format_exc())


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[r['condition']].append(r)
    out = {}
    for name, rs in groups.items():
        ok = [r for r in rs if r['execution_status'] == 'ok']
        events = Counter()
        for r in ok:
            events.update({k: v for k, v in r['summary']['event_counts'].items() if v})
        out[name] = dict(episodes=len(rs), errors=len(rs)-len(ok),
            outcome_class=dict(Counter(r['outcome_class'] for r in rs)),
            success_rate=sum(bool(r['success']) for r in ok)/max(1, len(ok)),
            mean_final_height_m=sum(r['outcome']['final_stable_height_m'] for r in ok)/max(1, len(ok)),
            final_count=dict(Counter(r['outcome']['scored_object_count'] for r in ok)),
            episodes_with_event={k: sum(r['summary']['event_counts'][k] > 0 for r in ok) for k in events},
            recovered=sum(r['summary']['recovered'] for r in ok),
            mean_transitions=sum(r['transitions'] for r in rs)/len(rs))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', default='scripted', choices=['scripted', 'bc'])
    parser.add_argument('--checkpoints', type=Path, nargs='*', default=[], help='BC teacher checkpoints')
    parser.add_argument('--execute', type=int, default=None, help='BC: chunk steps executed per replan')
    parser.add_argument('--stop-count', type=int, default=None, help='BC: stop after this many placements')
    parser.add_argument('--stop-hold', type=int, default=10, help='BC: hands-free steps required before stopping')
    parser.add_argument('--max-steps', type=int, default=None, help='BC: stop after this many control steps')
    parser.add_argument('--scenes', type=int, default=4)
    parser.add_argument('--start-seed', type=int, default=3000)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default=None,
                        help='Keep only scene seeds of this split')
    parser.add_argument('--conditions', nargs='+', default=['nominal', 'iid:0.1', 'ou:0.1:0.9'])
    parser.add_argument('--repeats', type=int, default=1, help='Noise draws per scene and noisy condition')
    parser.add_argument('--curriculum-starts', type=int, default=0,
                        help='Extra episodes per policy starting from recorded scripted states')
    parser.add_argument('--start-data', type=Path, nargs='*', default=[])
    parser.add_argument('--max-start-placements', type=int, default=None)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--rgb-every', type=int, default=5,
                        help='Render RGB every k-th observation (plus the last); states stay full-rate')
    parser.add_argument('--time-limit', type=float, default=180.)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--validate', type=float, default=1., help='Fraction of episodes to replay-validate')
    args = parser.parse_args()
    from assignment.training.data.recording import SCHEMA, source_hashes, write_json
    from assignment.training.data.signals import LABELER_VERSION, scene_split

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    policies = [None] if args.source == 'scripted' else [bc_policy(c, args) for c in args.checkpoints]
    for seed in range(args.start_seed, args.start_seed+args.scenes):
        if args.split and scene_split(seed) != args.split:
            continue
        for text in args.conditions:
            name, policy, noise = parse_condition(text)
            for teacher in policies:
                stochastic = noise['kind'] != 'none' or policy['variation'] or policy['sweep']
                for repeat in range(args.repeats if stochastic else 1):
                    prefix = args.source if teacher is None else f"bc-{teacher[1]['tag']}"
                    jobs.append(dict(episode_id=f'{prefix}-s{seed:06d}-{slug(name)}-r{repeat}', source=args.source,
                                     condition=name, scene_seed=seed, policy=policy if teacher is None else teacher[0],
                                     noise=noise, resolution=args.resolution, rgb_every=args.rgb_every,
                                     time_limit=args.time_limit,
                                     provenance=dict(repeat=repeat, **({} if teacher is None else teacher[1]))))
    if args.curriculum_starts:
        from assignment.training.data.task import start_pool
        import numpy as np
        pool = start_pool(args.start_data, args.max_start_placements, split=None)
        chosen = [pool[i] for i in np.random.default_rng(0).choice(len(pool), args.curriculum_starts, replace=False)]
        for root, privileged, metadata, step, done in chosen:
            seed = json.loads((Path(root)/metadata).read_text())['scene_seed']
            start = dict(kind='dataset', root=str(root), privileged=privileged, metadata=metadata, step=int(step),
                         placements_done=int(done))
            for text in args.conditions:
                name, policy, noise = parse_condition(text)
                for teacher in policies:
                    stochastic = noise['kind'] != 'none'
                    for repeat in range(args.repeats if stochastic else 1):
                        prefix = args.source if teacher is None else f"bc-{teacher[1]['tag']}"
                        jobs.append(dict(episode_id=f'{prefix}-s{seed:06d}-t{step:04d}-{slug(name)}-r{repeat}',
                                         source=args.source, condition=name, scene_seed=seed,
                                         policy=policy if teacher is None else teacher[0], noise=noise, start=start,
                                         resolution=args.resolution, rgb_every=args.rgb_every,
                                         time_limit=args.time_limit,
                                         provenance=dict(repeat=repeat, **({} if teacher is None else teacher[1]))))
    settings = dict(source=args.source, scenes=args.scenes, start_seed=args.start_seed, split=args.split,
                    conditions=args.conditions, curriculum_starts=args.curriculum_starts,
                    start_data=[str(p) for p in args.start_data], repeats=args.repeats, resolution=args.resolution,
                    rgb_every=args.rgb_every, time_limit=args.time_limit, rendered=not args.no_render,
                    checkpoints=[str(c) for c in args.checkpoints],
                    bc=dict(execute=args.execute, stop_count=args.stop_count, stop_hold=args.stop_hold,
                            max_steps=args.max_steps) if args.source == 'bc' else None)
    log = open(args.out/'collect.log', 'a')
    start = time.time()
    ctx = mp.get_context('spawn')
    counter = ctx.Value('i', 0)
    records = []
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=init_worker,
                             initargs=(counter, args.gpus)) as pool:
        futures = [pool.submit(run_record, args.out, job, not args.no_render) for job in jobs]
        for future in as_completed(futures):
            r = future.result()
            records.append(r)
            print(json.dumps(dict(stage='record', done=len(records), total=len(jobs), episode=r['episode_id'],
                                  outcome=r['outcome_class'],
                                  height=r['outcome'] and r['outcome']['final_stable_height_m'],
                                  seconds=round(r['timing']['total_s'], 1))), file=log, flush=True)
        records.sort(key=lambda r: r['episode_id'])
        collect_s = time.time()-start
        ok = [r for r in records if r['execution_status'] == 'ok']
        step = max(1, round(1/args.validate)) if args.validate > 0 else None
        targets = ok[::step] if step else []
        validations = []
        t0 = time.time()
        for future in as_completed([pool.submit(run_validate, args.out, r) for r in targets]):
            validations.append(future.result())
        validate_s = time.time()-t0
    splits = {}
    for r in ok:
        splits.setdefault(scene_split(r['scene_seed']), []).append(r['episode_id'])
    timing = {k: sum(r['timing'].get(k, 0) for r in records) for k in ('simulate_s', 'labels_s', 'render_s', 'write_s')}
    manifest = dict(schema=SCHEMA, source=args.source, settings=settings, source_sha256=source_hashes(),
                    labeler_version=LABELER_VERSION,
                    episodes=ok, errors=[r for r in records if r['execution_status'] != 'ok'], splits=splits,
                    summary=summarize(records),
                    validation=dict(episodes=len(validations), all_passed=all(v['passed'] for v in validations),
                                    failures=[v for v in validations if not v['passed']]),
                    resources=dict(workers=args.workers, gpus=args.gpus, wall_collect_s=collect_s,
                                   wall_validate_s=validate_s, cpu_seconds=timing,
                                   transitions=sum(r['transitions'] for r in records),
                                   bytes=sum(sum(r['bytes'].values()) for r in records)))
    write_json(args.out/'manifest.json', manifest)
    print(json.dumps(dict(summary=manifest['summary'], validation=manifest['validation'],
                          resources=manifest['resources'], splits={k: len(v) for k, v in splits.items()}), indent=1))


if __name__ == '__main__':
    main()
