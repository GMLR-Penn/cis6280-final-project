"""Collect a reproducible success-filtered box dataset, render it, then replay it.

The manifest includes every attempted seed, including numerical failed rollouts.
Use --resume to continue after interruption; configuration/source changes reject
resumption. Graphics are needed only in render/validate/all stages.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path

from cube_pick.box_data import (SCHEMA, record_attempt, render_episode, validate_episode,
                                source_hashes, write_json)


def finalize_splits(manifest):
    ids = sorted([r['episode_id'] for r in manifest['episodes']],
                 key=lambda s: hashlib.sha256(('box-v1-split/'+s).encode()).hexdigest())
    n = len(ids)
    manifest['splits'] = dict(train=ids[:int(.8*n)], val=ids[int(.8*n):int(.9*n)], test=ids[int(.9*n):])
    manifest['split_note'] = 'Seed-disjoint success-filtered development splits; not a hidden evaluation benchmark.'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--successes', type=int, default=100)
    p.add_argument('--start-seed', type=int, default=1000)
    p.add_argument('--max-attempts', type=int, default=1000)
    p.add_argument('--resolution', type=int, default=128)
    p.add_argument('--time-limit', type=float, default=180.)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--render-workers', type=int, default=2)
    p.add_argument('--stage', choices=('collect', 'render', 'validate', 'all'), default='all')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if (args.successes < 1 or args.max_attempts < args.successes or args.start_seed < 0
            or not 1 <= args.workers <= 8 or not 1 <= args.render_workers <= 4
            or args.resolution not in (64, 96, 128, 192, 256)):
        p.error('Invalid count/seed/workers; resolution must be 64, 96, 128, 192 or 256')
    settings = dict(target_successes=args.successes, start_seed=args.start_seed,
                    resolution=args.resolution, time_limit=args.time_limit, source_sha256=source_hashes())
    path = args.out/'manifest.json'
    if path.exists():
        if not args.resume:
            p.error('Output exists. Use --resume with the same configuration or a fresh directory')
        manifest = json.loads(path.read_text())
        if manifest['settings'] != settings or manifest['schema'] != SCHEMA:
            p.error('Cannot resume after changing configuration or source; choose a fresh directory')
    else:
        if args.out.exists() and any(args.out.iterdir()):
            p.error('New output directory must be empty')
        args.out.mkdir(parents=True, exist_ok=True)
        manifest = dict(schema=SCHEMA, settings=settings, status='collecting', episodes=[], attempts=[],
            description=f'{args.successes}-success pilot: privileged scripted demonstrations, success-filtered, not optimal-action labels.',
            selection_rule='First target_successes successful seeds in ascending order; all batch attempts are retained.')
        write_json(path, manifest)

    if args.stage in ('collect', 'all') and len(manifest['episodes']) < args.successes:
        worker = partial(record_attempt, str(args.out), resolution=args.resolution, time_limit=args.time_limit)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            while len(manifest['episodes']) < args.successes and len(manifest['attempts']) < args.max_attempts:
                start = args.start_seed+len(manifest['attempts'])
                count = min(args.workers, args.max_attempts-len(manifest['attempts']))
                for record in pool.map(worker, range(start, start+count)):
                    manifest['attempts'].append(record)
                    if record['success'] and len(manifest['episodes']) < args.successes:
                        manifest['episodes'].append(record)
                    write_json(path, manifest)
                    print(json.dumps(dict(stage='collect', seed=record['scene_seed'], success=record['success'],
                        accepted=len(manifest['episodes']), attempts=len(manifest['attempts']))), flush=True)
        if len(manifest['episodes']) < args.successes:
            raise RuntimeError('Attempt limit reached; retained progress. Increase --max-attempts and --resume.')
    if len(manifest['episodes']) != args.successes:
        p.error('Complete the collect stage before rendering or validation')
    manifest['execution_errors'] = [r['episode_id'] for r in manifest['attempts'] if r.get('execution_status') == 'error']
    finalize_splits(manifest)
    if manifest['status'] != 'complete':
        manifest['status'] = 'collected' if not all(r['rendered'] for r in manifest['episodes']) else 'rendered'
    write_json(path, manifest)

    if args.stage in ('render', 'all'):
        with ProcessPoolExecutor(max_workers=args.render_workers) as pool:
            for record in pool.map(partial(render_episode, str(args.out)), manifest['episodes']):
                for key in ('episodes', 'attempts'):
                    manifest[key] = [record if r['episode_id'] == record['episode_id'] else r for r in manifest[key]]
                write_json(path, manifest)
                print(json.dumps(dict(stage='render', episode=record['episode_id'],
                    rendered=sum(r['rendered'] for r in manifest['episodes']))), flush=True)
        manifest['status'] = 'rendered'
        write_json(path, manifest)

    if args.stage in ('validate', 'all'):
        if not all(r['rendered'] for r in manifest['episodes']):
            p.error('Complete rendering before validation')
        manifest['status'] = 'validating'
        write_json(path, manifest)
        validation = dict(schema=SCHEMA, all_passed=False, episodes=[])
        with ProcessPoolExecutor(max_workers=args.render_workers) as pool:
            for result in pool.map(partial(validate_episode, str(args.out)), manifest['episodes']):
                validation['episodes'].append(result)
                write_json(args.out/'validation.json', validation)
                print(json.dumps(dict(stage='validate', episode=result['episode_id'],
                                     validated=len(validation['episodes']))), flush=True)
        validation['all_passed'] = True
        validation['execution_errors_excluded'] = manifest['execution_errors']
        write_json(args.out/'validation.json', validation)
        manifest['status'] = 'complete'
        write_json(path, manifest)
    print(json.dumps(dict(status=manifest['status'], accepted=len(manifest['episodes']),
                         attempts=len(manifest['attempts']))), flush=True)


if __name__ == '__main__':
    main()
