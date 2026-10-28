"""Frame and action cache built from recorded stacking episodes."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path
import time

import numpy as np

CACHE_SCHEMA = 'stack-jepa-cache-v0.1'
STRIDE = 5
RAW_ACTION_DIM = 25
BLOCK_ACTION_DIM = STRIDE*RAW_ACTION_DIM
ACTION_DIM = BLOCK_ACTION_DIM
PROPRIO_DIM = 75
PROPRIO_CLIP = 10.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- cache

def _arrays(out, total, shape, mode):
    frames = np.memmap(out/'frames.u8', np.uint8, mode, shape=(total, *shape))
    actions = np.memmap(out/'actions.f32', np.float32, mode, shape=(total, ACTION_DIM))
    return frames, actions


def _write_episode(job):
    out, total, shape, root, record, offset, count = job
    frames, actions = _arrays(Path(out), total, shape, 'r+')
    with np.load(Path(root)/record['observations']) as f:
        rgb, raw = f['rgb'], f['actions']
        steps = f['rgb_steps'] if 'rgb_steps' in f.files else np.arange(len(rgb))
    keep = steps % STRIDE == 0
    if not np.array_equal(steps[keep], np.arange(count)*STRIDE):
        raise ValueError(f'Unexpected RGB steps in {record["episode_id"]}')
    rgb = rgb[keep]
    frames[offset:offset+count] = np.concatenate(list(rgb.transpose(1, 0, 2, 3, 4)), axis=2)
    block = np.zeros((count, ACTION_DIM), np.float32)
    block[:count-1] = raw[:(count-1)*STRIDE].reshape(count-1, ACTION_DIM)
    actions[offset:offset+count] = block
    frames.flush()
    actions.flush()
    return record['episode_id'], raw[:(count-1)*STRIDE].astype(np.float64).sum(0), \
        np.square(raw[:(count-1)*STRIDE].astype(np.float64)).sum(0), (count-1)*STRIDE


def build_cache(roots, out, workers=16):
    """Concatenate episodes into uint8 frame / float32 action memmaps plus index.json.

    Frame k of an episode is raw observation k*STRIDE; trailing raw steps after the
    last such observation are dropped. The final frame's action is a zero placeholder
    (never used: windows only feed actions of non-final frames to the predictor).
    Action statistics come from train-split episodes only.
    """
    roots = [Path(r) for r in roots]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    episodes, offset, shape = [], 0, None
    for root in roots:
        manifest = json.loads((root/'manifest.json').read_text())
        for r in manifest['episodes']:
            meta = json.loads((root/r['metadata']).read_text())
            if shape is None:
                res = meta['resolution']
                shape = (res, res*len(meta['camera_names']), 3)
            count = r['transitions']//STRIDE+1
            episodes.append(dict(episode_id=r['episode_id'], root=str(root), split=r['split'],
                                 scene_seed=r['scene_seed'], start=offset, length=count,
                                 control_steps=r['control_steps'],
                                 sigma=meta['noise'].get('sampled_sigma'), outcome=r['outcome_class'],
                                 placed=meta['outcome']['scored_object_count'], record=r))
            offset += count
    _arrays(out, offset, shape, 'w+')
    jobs = [(str(out), offset, shape, e['root'], e.pop('record'), e['start'], e['length']) for e in episodes]
    split = {e['episode_id']: e['split'] for e in episodes}
    total, square, n = np.zeros(RAW_ACTION_DIM), np.zeros(RAW_ACTION_DIM), 0
    all_total, all_square, all_n = np.zeros(RAW_ACTION_DIM), np.zeros(RAW_ACTION_DIM), 0
    with ProcessPoolExecutor(workers) as pool:
        for i, (episode, s, s2, k) in enumerate(pool.map(_write_episode, jobs, chunksize=8)):
            all_total, all_square, all_n = all_total+s, all_square+s2, all_n+k
            if split[episode] == 'train':
                total, square, n = total+s, square+s2, n+k
            if i % 1000 == 0:
                print(json.dumps(dict(stage='cache', done=i, total=len(jobs))), flush=True)
    if n == 0:
        total, square, n = all_total, all_square, max(1, all_n)
    mean = total/n
    std = np.sqrt(np.maximum(square/n-mean**2, 1e-12))
    index = dict(schema=CACHE_SCHEMA, roots=[str(r) for r in roots], stride=STRIDE, frames=offset,
                 frame_shape=list(shape), views='front|side tiled along width', action_dim=ACTION_DIM,
                 action_mean=mean.tolist(), action_std=std.tolist(), episodes=episodes)
    (out/'index.json').write_text(json.dumps(index))
    return index


def _proprio_array(out, total, mode):
    return np.memmap(Path(out)/'proprio.f32', np.float32, mode, shape=(total, PROPRIO_DIM))


def _write_proprio(job):
    out, total, root, record, offset, count = job
    proprio = _proprio_array(out, total, 'r+')
    with np.load(Path(root)/record['observations']) as f:
        p = f['proprio'][::STRIDE][:count]
    if len(p) != count:
        raise ValueError(f'Short proprio in {record["episode_id"]}')
    proprio[offset:offset+count] = p
    proprio.flush()
    return p.astype(np.float64)


def add_proprio(path, workers=64):
    """Add frame-aligned proprioception (raw observation k*STRIDE) and train-split statistics."""
    path = Path(path)
    index = json.loads((path/'index.json').read_text())
    _proprio_array(path, index['frames'], 'w+')
    records = {}
    for root in index['roots']:
        for r in json.loads((Path(root)/'manifest.json').read_text())['episodes']:
            records[r['episode_id']] = r
    eps = index['episodes']
    jobs = [(str(path), index['frames'], e['root'], records[e['episode_id']], e['start'], e['length']) for e in eps]
    total, square, n = np.zeros(PROPRIO_DIM), np.zeros(PROPRIO_DIM), 0
    with ProcessPoolExecutor(workers) as pool:
        for e, p in zip(eps, pool.map(_write_proprio, jobs, chunksize=8)):
            if e['split'] == 'train':
                total, square, n = total+p.sum(0), square+np.square(p).sum(0), n+len(p)
    mean = total/n
    index.update(proprio_dim=PROPRIO_DIM, proprio_mean=mean.tolist(),
                 proprio_std=np.sqrt(np.maximum(square/n-mean**2, 1e-12)).tolist(),
                 proprio_layout='25 joint positions, 25 joint velocities, 25 actuator targets')
    (path/'index.json').write_text(json.dumps(index))
    return index


def load_cache(path):
    path = Path(path)
    index = json.loads((path/'index.json').read_text())
    frames, actions = _arrays(path, index['frames'], tuple(index['frame_shape']), 'r')
    return index, frames, actions



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    start = time.time()
    index = build_cache(args.data, args.out, args.workers)
    splits = {}
    for e in index['episodes']:
        s = splits.setdefault(e['split'], dict(episodes=0, frames=0))
        s['episodes'] += 1
        s['frames'] += e['length']
    print(json.dumps(dict(frames=index['frames'], splits=splits, seconds=round(time.time()-start)), indent=1))


if __name__ == '__main__':
    main()
