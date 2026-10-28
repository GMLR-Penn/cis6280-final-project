"""Collect every paired behaviour, including failures; split by scene, not episode."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np

from cube_pick.varied_boxes import MODES
from cube_pick.varied_box_data import (SCHEMA, record_attempt, render_episode,
                                      validate_episode, source_hashes, write_json)


def collect_job(job, root, resolution, time_limit):
    return record_attempt(root, *job, resolution=resolution, time_limit=time_limit)


def finalize_splits(manifest):
    seeds = sorted({r['scene_seed'] for r in manifest['episodes']},
                   key=lambda s: hashlib.sha256(f'varied-box-split/{s}'.encode()).hexdigest())
    # At least one validation and test scene when three or more scenes exist.
    n = len(seeds)
    holdout = max(1, round(.1*n)) if n >= 3 else 0
    groups = dict(train=seeds[:n-2*holdout], val=seeds[n-2*holdout:n-holdout],
                  test=seeds[n-holdout:] if holdout else [])
    manifest['scene_splits'] = groups
    manifest['splits'] = {k: [r['episode_id'] for r in manifest['episodes'] if r['scene_seed'] in ss]
                          for k, ss in groups.items()}


def verify_pairs(root, records):
    """Same-scene modes share initial physics and RGB; measure nominal prefixes."""
    root = Path(root)
    results = []
    for seed in sorted({r['scene_seed'] for r in records}):
        group = [r for r in records if r['scene_seed'] == seed]
        base = next((r for r in group if r['mode'] == 'nominal'), group[0])
        with np.load(root/base['observations']) as data, np.load(root/base['privileged']) as truth:
            initial_rgb, initial_state = data['rgb'][0], truth['integration_state'][0]
            actions, states = data['actions'], truth['integration_state']
            for record in group:
                with np.load(root/record['observations']) as other, np.load(root/record['privileged']) as ot:
                    np.testing.assert_array_equal(other['rgb'][0], initial_rgb)
                    np.testing.assert_array_equal(ot['integration_state'][0], initial_state)
                    n = min(len(actions), len(other['actions']))
                    different = np.flatnonzero(np.any(actions[:n] != other['actions'][:n], axis=1))
                    prefix = int(different[0]) if len(different) else n
                    np.testing.assert_array_equal(ot['integration_state'][:prefix+1], states[:prefix+1])
                    results.append(dict(scene_seed=seed, episode_id=record['episode_id'],
                        reference=base['episode_id'], shared_action_prefix_steps=prefix,
                        initial_state_and_rgb='exact'))
    return results


def summarize(records):
    return {mode: dict(episodes=len(rows), full_stacks=sum(r['success'] for r in rows),
                      partial_stacks=sum(r['outcome_class'] == 'partial_stack' for r in rows),
                      no_stable_height=sum(r['outcome_class'] == 'no_stable_height' for r in rows),
                      interventions_applied=sum(r['intervention_applied'] for r in rows),
                      mean_final_height_m=float(np.mean([r['outcome']['final_stable_height_m'] for r in rows])))
            for mode in MODES if (rows := [r for r in records if r['mode'] == mode])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scenes', type=int, default=6)
    p.add_argument('--start-seed', type=int, default=2000)
    p.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    p.add_argument('--resolution', type=int, choices=(64, 96, 128, 192, 256), default=128)
    p.add_argument('--time-limit', type=float, default=180.)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--render-workers', type=int, default=2)
    p.add_argument('--stage', choices=('collect', 'render', 'validate', 'all'), default='all')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if (args.scenes < 1 or args.start_seed < 0 or not 1 <= args.workers <= 8
            or not 1 <= args.render_workers <= 4 or len(set(args.modes)) != len(args.modes)
            or not np.isfinite(args.time_limit) or not .1 <= args.time_limit <= 600
            or not np.isclose(args.time_limit/.1, round(args.time_limit/.1))):
        p.error('Invalid scene/worker/time count or duplicate modes')
    settings = dict(scenes=args.scenes, start_seed=args.start_seed, modes=args.modes,
                    resolution=args.resolution, time_limit=args.time_limit, source_sha256=source_hashes())
    path = args.out/'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        if not args.resume or manifest['settings'] != settings or manifest['schema'] != SCHEMA:
            p.error('Use --resume with identical sources/settings, or a fresh output directory')
    else:
        if args.stage != 'all' and args.stage != 'collect':
            p.error('Collect first, then render/validate with --resume')
        if args.out.exists() and any(args.out.iterdir()):
            p.error('New output directory must be empty')
        args.out.mkdir(parents=True, exist_ok=True)
        manifest = dict(schema=SCHEMA, settings=settings, status='collecting', episodes=[], attempts=[],
            selection_rule='Every scene/mode is attempted once. Physical failures are rendered; execution errors are retained only for audit.',
            split_note='All behaviours from a scene stay in the same split; small public development pilot.')
        write_json(path, manifest)
    if args.stage in ('collect', 'all'):
        complete = {(r['scene_seed'], r['mode']) for r in manifest['attempts']}
        jobs = [(s, m) for s in range(args.start_seed, args.start_seed+args.scenes)
                for m in args.modes if (s, m) not in complete]
        worker = partial(collect_job, root=str(args.out), resolution=args.resolution, time_limit=args.time_limit)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for record in pool.map(worker, jobs):
                manifest['attempts'].append(record)
                if record['execution_status'] == 'ok':
                    manifest['episodes'].append(record)
                write_json(path, manifest)
                print(json.dumps(dict(stage='collect', episode=record['episode_id'],
                    success=record['success'], height=(record['outcome'] or {}).get('final_stable_height_m'),
                    completed=len(manifest['attempts']))), flush=True)
    if len(manifest['attempts']) != args.scenes*len(args.modes):
        p.error('Finish numerical collection first')
    manifest['execution_errors'] = [r['episode_id'] for r in manifest['attempts'] if r['execution_status'] == 'error']
    finalize_splits(manifest)
    manifest['summary'] = summarize(manifest['episodes'])
    if manifest['status'] not in ('complete', 'complete_with_errors'):
        manifest['status'] = 'collected'
    write_json(path, manifest)
    if args.stage in ('render', 'all'):
        with ProcessPoolExecutor(max_workers=args.render_workers) as pool:
            for record in pool.map(partial(render_episode, str(args.out)), manifest['episodes']):
                manifest['episodes'] = [record if r['episode_id'] == record['episode_id'] else r
                                        for r in manifest['episodes']]
                manifest['attempts'] = [record if r['episode_id'] == record['episode_id'] else r for r in manifest['attempts']]
                write_json(path, manifest)
                print(json.dumps(dict(stage='render', episode=record['episode_id'])), flush=True)
        manifest['status'] = 'rendered'
        write_json(path, manifest)
    if args.stage in ('validate', 'all'):
        if not all(r['rendered'] for r in manifest['episodes']):
            p.error('Render all episodes first')
        manifest['status'] = 'validating'
        write_json(path, manifest)
        validation = dict(schema=SCHEMA, all_passed=False, episodes=[])
        with ProcessPoolExecutor(max_workers=args.render_workers) as pool:
            for result in pool.map(partial(validate_episode, str(args.out)), manifest['episodes']):
                validation['episodes'].append(result)
                write_json(args.out/'validation.json', validation)
                print(json.dumps(dict(stage='validate', episode=result['episode_id'])), flush=True)
        validation['paired_checks'] = verify_pairs(args.out, manifest['episodes'])
        validation['episodes_passed'] = True
        validation['execution_errors'] = manifest['execution_errors']
        validation['all_passed'] = not manifest['execution_errors']
        write_json(args.out/'validation.json', validation)
        manifest['status'] = 'complete_with_errors' if manifest['execution_errors'] else 'complete'
        write_json(path, manifest)
    print(json.dumps(dict(status=manifest['status'], summary=manifest['summary'])), flush=True)


if __name__ == '__main__':
    main()
