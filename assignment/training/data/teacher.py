"""Chunked MLP behavior cloning for stacking with DART-style labels.

Inputs are the privileged features of `assignment.training.data.task.features`. Targets are the
scripted controller's clean actions (`policy_actions`) for the next `chunk`
steps along the executed, possibly noise-perturbed trajectory, so the learner
sees corrective labels in off-nominal states. At run time the first `execute`
actions of each chunk are applied before re-planning (receding horizon).
"""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from assignment.training.data.task import FEATURES, features, static_features
from assignment.training.data.signals import hand_point, structure

ACTION_DIM = 25


def _episode(args):
    root, record = args
    from assignment.training.data.recording import load_episode
    from assignment.training.data.task import episode_features
    data = load_episode(root, record, include_privileged=True)
    meta = json.loads((Path(root)/record['metadata']).read_text())
    n = meta['control_steps']
    obs = episode_features(data['privileged'], data, meta)[:n]
    placements = sorted(e['step'] for e in meta['events'] if e['event'] == 'placement')
    return dict(obs=obs, actions=data['policy_actions'][:n], executed=data['actions'][:n],
                episode_id=record['episode_id'], scene_seed=record['scene_seed'], condition=record['condition'],
                placements=placements, final_count=record['outcome']['scored_object_count'])


def build_cache(roots, out, split='train', workers=64):
    """Concatenated per-step arrays of all `split` episodes, saved as .npy files plus index.json."""
    from assignment.training.data.recording import load_manifest
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(root), r) for root in roots for r in load_manifest(root)['episodes'] if r['split'] == split]
    with ProcessPoolExecutor(workers) as pool:
        episodes = list(pool.map(_episode, jobs, chunksize=4))
    lengths = np.array([len(e['obs']) for e in episodes])
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    np.save(out/'obs.npy', np.concatenate([e['obs'] for e in episodes]).astype(np.float32))
    np.save(out/'actions.npy', np.concatenate([e['actions'] for e in episodes]).astype(np.float32))
    np.save(out/'executed.npy', np.concatenate([e['executed'] for e in episodes]).astype(np.float32))
    index = dict(roots=[str(r) for r in roots], split=split, starts=starts.tolist(), lengths=lengths.tolist(),
                 episodes=[{k: e[k] for k in ('episode_id', 'scene_seed', 'condition', 'placements', 'final_count')}
                           for e in episodes])
    (out/'index.json').write_text(json.dumps(index))
    return index


def load_cache(path):
    path = Path(path)
    index = json.loads((path/'index.json').read_text())
    return dict(index=index, obs=np.load(path/'obs.npy', mmap_mode='r'),
                actions=np.load(path/'actions.npy', mmap_mode='r'), executed=np.load(path/'executed.npy', mmap_mode='r'))


def sample_rows(index, episodes, max_placements=None, after_steps=50):
    """Row ids, episode start and end for training samples of the chosen episodes.

    max_placements keeps each episode up to `after_steps` past its k-th placement
    (all of it if that placement never happened).
    """
    rows, ends = [], []
    for i in episodes:
        start, length = index['starts'][i], index['lengths'][i]
        placed = index['episodes'][i]['placements']
        limit = length
        if max_placements and len(placed) >= max_placements:
            limit = min(length, placed[max_placements-1]+after_steps)
        rows.append(np.arange(start, start+limit))
        ends.append(np.full(limit, start+length))
    return np.concatenate(rows), np.concatenate(ends)


class Block(nn.Module):
    def __init__(self, width, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2*width), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(2*width, width))

    def forward(self, x):
        return x+self.net(x)


class ChunkMLP(nn.Module):
    """obs -> (chunk, 25) actions. Normalization statistics are fixed buffers set from data."""
    def __init__(self, obs_dim=FEATURES, chunk=10, width=1024, depth=4, dropout=0.):
        super().__init__()
        self.config = dict(obs_dim=obs_dim, chunk=chunk, width=width, depth=depth, dropout=dropout)
        self.chunk = chunk
        self.register_buffer('obs_mean', torch.zeros(obs_dim))
        self.register_buffer('obs_std', torch.ones(obs_dim))
        self.inp = nn.Linear(obs_dim, width)
        self.blocks = nn.Sequential(*[Block(width, dropout) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, chunk*ACTION_DIM))

    def set_normalization(self, mean, std):
        self.obs_mean.copy_(torch.as_tensor(mean))
        self.obs_std.copy_(torch.as_tensor(std).clamp_min(1e-3))

    def forward(self, obs):
        x = ((obs-self.obs_mean)/self.obs_std).clamp(-10, 10)
        return self.out(self.blocks(self.inp(x))).view(len(obs), self.chunk, ACTION_DIM)


def save_bc(path, model, info):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(dict(state_dict=model.state_dict(), config=model.config, info=info), temp)
    temp.replace(path)


def load_bc(path, device='cpu'):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = ChunkMLP(**blob['config']).to(device)
    model.load_state_dict(blob['state_dict'])
    return model.eval(), blob['info']


class BCPolicy:
    """Receding-horizon chunk executor; works with StackTaskEnv and the recorder (kind='bc').

    Optional recorder stop rule (spec keys): stop_count placements beyond the
    starting structure held hands-free for stop_hold steps (StackTaskEnv's
    success test), or max_steps control steps. Either returns None (STOP);
    stop_reason tells the recorder to mark the step cap as truncation.
    """
    def __init__(self, env, spec, policy_seed=0):
        torch.set_num_threads(1)
        self.model, self.info = load_bc(spec['checkpoint'])
        self.execute = int(spec.get('execute') or self.info.get('execute') or self.model.chunk)
        self.static = static_features(env.config)
        self.goal = (env.config.goal_x, env.config.goal_y)
        self.half = env.config.goal_half_width
        self.queue = []
        self.order, self.events = None, []
        self.stop_count, self.max_steps = spec.get('stop_count'), spec.get('max_steps')
        self.stop_hold = int(spec.get('stop_hold', 10))
        self.start_count, self.hold, self.stop_reason = None, 0, None
        self.parameters = dict(checkpoint=str(spec['checkpoint']), execute=self.execute, chunk=self.model.chunk,
                               stop_count=self.stop_count, stop_hold=self.stop_hold, max_steps=self.max_steps)

    def done(self, env, t):
        row = env.history[-1]
        count = int(structure(row, self.goal, self.half)[0].sum())
        if self.start_count is None:
            self.start_count = count
        if self.stop_count is not None:
            released = row['object_hand_contacts'].sum() == 0
            self.hold = self.hold+1 if count >= self.start_count+self.stop_count and released else 0
            if self.hold >= self.stop_hold:
                self.events.append(dict(step=t, phase='stop', cause='placed', count=count))
                return True
        if self.max_steps is not None and t >= self.max_steps:
            self.events.append(dict(step=t, phase='stop', cause='step_cap', count=count))
            self.stop_reason = 'time_limit'
            return True
        return False

    def observe(self, env):
        row = env.history[-1]
        members, _ = structure(row, self.goal, self.half)
        return features(row, env.proprio(), hand_point(env), env.time_remaining/env.config.time_limit,
                        self.goal, self.half, self.static, members)

    @torch.no_grad()
    def action(self, env, t):
        if (self.stop_count is not None or self.max_steps is not None) and self.done(env, t):
            return None, 'stop'
        if not self.queue:
            chunk = self.model(torch.from_numpy(self.observe(env))[None])[0].numpy()
            self.queue = list(np.clip(chunk[:self.execute], -1, 1).astype(np.float32))
        return self.queue.pop(0), 'bc'
