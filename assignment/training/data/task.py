"""RL interface for stacking: privileged-state features, curriculum starts, rewards.

The policy acts with the same raw 25-dim normalized actions as every other
source. Training rewards add potential-based shaping to the shared task reward;
training early termination is separate from data-collection termination.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

import mujoco
import numpy as np

from cube_pick.mixed_stacking import MATERIALS
from cube_pick.recording import scene_config
from assignment.training.data.signals import hand_point, row_hand_parts, shaping_levels, structure, tower
from cube_pick.varied_boxes import VariedBoxEnv

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
OBJECT_FEATURES = 27
FEATURES = 82+4*OBJECT_FEATURES


def static_features(config):
    sizes = np.array([o.size for o in config.objects])
    mass = np.array([o.mass for o in config.objects])
    friction = np.array([MATERIALS[o.material]['friction'] for o in config.objects])
    return np.concatenate([sizes*10, mass[:, None]*5, friction[:, None]], axis=1)


def features(row, proprio, hand, time_fraction, goal_xy, half_width, static, members=None):
    """Fixed-size privileged observation. Also computable from recorded episodes."""
    if members is None:
        members, height = structure(row, goal_xy, half_width)
    else:
        height = float(row['object_bounds'][members, 1, 2].max()) if members.any() else 0.
    goal = np.r_[goal_xy, 0.]
    pose = row['object_pose']
    quat = pose[:, 3:]*np.where(pose[:, 3:4] < 0, -1, 1)
    parts, _ = row_hand_parts(row)
    per_object = np.concatenate([
        (pose[:, :3]-goal)*10, (pose[:, :3]-hand)*10, quat, row['object_velocity'][:, 3:], row['object_velocity'][:, :3]*.1,
        static, parts[:, None]/3, row['object_hand_normal_force'].sum(-1, keepdims=True)*.05,
        row['object_table_contacts'][:, None] > 0, row['object_ground_contacts'][:, None] > 0,
        members[:, None], row['object_bounds'][:, 1, 2:3]*10], axis=1)
    out = np.concatenate([proprio[:25], proprio[25:50]*.1, proprio[50:], (hand-goal)*10,
                          [height*10, time_fraction], np.asarray(goal_xy)*10, per_object.ravel()])
    return out.astype(np.float32)


def episode_features(truth, obs, meta):
    """Features for every observation of a recorded episode (for BC and analysis)."""
    goal = tuple(obs['goal_xy'].tolist())
    half = meta['scene']['goal_half_width']
    static = static_features(scene_config(meta))
    limit = meta['time_limit']/meta['action_dt']
    rows = [{k: v[t] for k, v in truth.items() if k.startswith('object_')} for t in range(len(obs['proprio']))]
    return np.stack([features(row, obs['proprio'][t], truth['hand_point'][t],
                              obs['time_remaining'][t]*10/limit, goal, half, static)
                     for t, row in enumerate(rows)])


@dataclass
class TaskConfig:
    """stage target_count: placements needed for an early success termination."""
    target_count: int = 1
    horizon: int = 400
    time_limit: float = 180.
    task_scale: float = 20.
    shaping: tuple = (2., 1., 5., 2.)  # reach, grasp, lift, carry potential weights
    gamma: float = .995
    ground_penalty: float = 1.
    action_cost: float = .01
    terminate_on_ground: bool = True
    success_hold_steps: int = 10
    init_probability: float = .3
    seeds: tuple = tuple(range(1000))
    start_pools: list = field(default_factory=list)


def start_pool(roots, max_placements=None, stride=10, condition='nominal', split='train'):
    """Curriculum starts: states along scripted episodes, indexed by completed placements.

    Returns entries (root, privileged path, metadata path, step, placements_done).
    """
    pool = []
    for root in [roots] if isinstance(roots, (str, Path)) else roots:
        manifest = json.loads((Path(root)/'manifest.json').read_text())
        for r in manifest['episodes']:
            if r['condition'] != condition or (split and r['split'] != split):
                continue
            meta = json.loads((Path(root)/r['metadata']).read_text())
            placed = sorted(e['step'] for e in meta['events'] if e['event'] == 'placement')
            agent_steps = r['control_steps']
            for t in range(31, agent_steps, stride):
                done = sum(s <= t for s in placed)
                if max_placements is None or done <= max_placements:
                    pool.append((str(root), r['privileged'], r['metadata'], t, done))
    return pool


def restore_env(env, states, step):
    """Put a freshly reset env at recorded observation `step` (physics, targets, clock, score history)."""
    env.reset()
    env.history.clear()
    for t in range(max(0, step-30), step+1):
        mujoco.mj_setState(env.model, env.data, states[t], STATE_SPEC)
        mujoco.mj_forward(env.model, env.data)
        env.history.append(env.privileged())
    env.targets = env.data.ctrl.copy()
    env.elapsed_steps = step
    env.start_time = float(env.data.time)-step*.1


class StackTaskEnv:
    """Single-episode RL wrapper around VariedBoxEnv (not a Gym API)."""
    def __init__(self, task, seed=0):
        self.task = task
        self.rng = np.random.default_rng(seed)
        self.env = None
        self.scene_seed = None
        self._states = {}

    def _make(self, scene_seed, config=None):
        if self.env is not None and self.scene_seed == scene_seed:
            return
        if self.env is not None:
            self.env.close()
        self.env = (VariedBoxEnv(config=config, resolution=64) if config is not None
                    else VariedBoxEnv(seed=scene_seed, resolution=64, time_limit=self.task.time_limit))
        self.scene_seed = scene_seed
        self.static = static_features(self.env.config)
        self.goal = (self.env.config.goal_x, self.env.config.goal_y)
        self.half = self.env.config.goal_half_width

    def restore(self, root, privileged, metadata, step):
        key = (root, privileged)
        if key not in self._states:
            if len(self._states) > 64:
                self._states.clear()
            with np.load(Path(root)/privileged) as f:
                meta = json.loads((Path(root)/metadata).read_text())
                self._states[key] = (f['integration_state'], meta)
        states, meta = self._states[key]
        self._make(meta["scene_seed"], scene_config(meta))
        restore_env(self.env, states, step)
        return meta['scene_seed']

    def reset(self, start=None):
        """start: None (sample), 'init', a pool entry, or a recorded self.start dict (replay)."""
        task = self.task
        if start is None:
            use_init = not task.start_pools or self.rng.random() < task.init_probability
            start = 'init' if use_init else task.start_pools[self.rng.integers(len(task.start_pools))]
        if isinstance(start, dict):
            start = ('init', start['scene_seed']) if start['kind'] == 'init' else (
                start['root'], start['privileged'], start['metadata'], start['step'], start['placements_done'])
        if start == 'init' or start[0] == 'init':
            seed = int(self.rng.choice(task.seeds)) if start == 'init' else int(start[1])
            self._make(seed)
            self.env.reset()
            self.start = dict(kind='init', scene_seed=seed, placements_done=0)
        else:
            root, privileged, metadata, step, done = start
            seed = self.restore(root, privileged, metadata, step)
            self.start = dict(kind='dataset', scene_seed=seed, root=root, privileged=privileged, metadata=metadata,
                              step=int(step), placements_done=int(done))
        state = np.empty(mujoco.mj_stateSize(self.env.model, STATE_SPEC))
        mujoco.mj_getState(self.env.model, self.env.data, state, STATE_SPEC)
        self.start_state = state
        self.steps = 0
        self.hold = 0
        row = self.env.history[-1]
        self.members, self.phi = structure(row, self.goal, self.half)
        self.start_count = int(tower(row, self.goal, self.half)[0].sum())
        self.levels = shaping_levels(row, hand_point(self.env), self.goal, self.half, self.members)
        self.ground = row['object_ground_contacts'] > 0
        return self.observe(row)

    def observe(self, row):
        env = self.env
        return features(row, env.proprio(), hand_point(env), env.time_remaining/self.task.time_limit,
                        self.goal, self.half, self.static, self.members)

    def step(self, action):
        task, env = self.task, self.env
        env.step(action)
        self.steps += 1
        row = env.history[-1]
        members, phi = structure(row, self.goal, self.half)
        hand = hand_point(env)
        levels = shaping_levels(row, hand, self.goal, self.half, members)
        weights = np.asarray(task.shaping)
        ground = row['object_ground_contacts'] > 0
        new_ground = int((ground & ~self.ground).sum())
        task_reward = phi-self.phi
        shaped = task.gamma*levels@weights-self.levels@weights
        reward = (task.task_scale*task_reward+shaped-task.ground_penalty*new_ground
                  - task.action_cost*float(np.mean(np.square(action))))
        self.members, self.phi, self.levels, self.ground = members, phi, levels, ground
        count = int(tower(row, self.goal, self.half)[0].sum())
        released = row['object_hand_contacts'].sum() == 0
        self.hold = self.hold+1 if count >= self.start_count+task.target_count and released else 0
        success = self.hold >= task.success_hold_steps
        failed = task.terminate_on_ground and new_ground > 0
        terminated = success or failed
        truncated = not terminated and (self.steps >= task.horizon or env.status != 'running')
        info = dict(task_reward=task_reward, count=count, success=success, ground=new_ground,
                    height=phi, placed=count-self.start_count)
        return self.observe(row), float(reward), terminated, truncated, info

    def close(self):
        if self.env is not None:
            self.env.close()
