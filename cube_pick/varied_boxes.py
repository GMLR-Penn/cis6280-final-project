"""Variable square footprints and paired actuator-only intervention policies.

The released fixed-footprint environment and 100-success data remain unchanged.
Behaviour names describe interventions, never ground-truth outcome labels.
"""
from dataclasses import replace

import numpy as np

from .box_stacking import BoxStackEnv, BoxStackController, box_scene
from .mixed_stacking import ObjectSpec


MODES = ('nominal', 'shuffled_order', 'offset', 'weak_grip', 'early_release')


def varied_box_scene(seed=0, time_limit=180.):
    config = box_scene(seed, time_limit)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 162026]))
    edges = rng.permutation([.050, .055, .060, float(rng.choice([.050, .055, .060]))])
    objects = []
    for old, edge in zip(config.objects, edges):
        height = old.size[2]
        shape = 'cube' if edge == height else 'cuboid'
        objects.append(ObjectSpec(shape, (float(edge), float(edge), height),
                                  old.material, old.xy, old.yaw))
    return replace(config, objects=objects)


class VariedBoxEnv(BoxStackEnv):
    def __init__(self, seed=0, resolution=128, time_limit=180., config=None):
        super().__init__(seed=seed, resolution=resolution,
                         config=config or varied_box_scene(seed, time_limit))


class VariedBoxController(BoxStackController):
    """Perturb the second placement, retaining the nominal first-placement prefix.

All changes reach the simulator as bounded actuator-target increments. No direct
state writes, force injection, reset, or fabricated success/failure flags.
"""
    def __init__(self, env, mode='nominal', seed=0):
        if mode not in MODES:
            raise ValueError(f'Unknown mode: {mode}')
        super().__init__(env)
        self.mode = mode
        rng = np.random.default_rng(np.random.SeedSequence([seed, 162027]))
        angle = rng.uniform(0, 2*np.pi)
        self.offset_xy = rng.uniform(.035, .055)*np.array([np.cos(angle), np.sin(angle)])
        self.grip_fraction = float(rng.uniform(.45, .65))
        self.release_distance = float(rng.uniform(.025, .050))
        self.intervention_step = None
        if mode == 'shuffled_order':
            old = self.order.copy()
            self.order = [int(i) for i in rng.permutation(old)]
            if self.order == old:
                self.order = self.order[1:]+self.order[:1]
            self.profiles.order = self.order.copy()
        self.parameters = dict(mode=mode, target_placement=2, order=self.order.copy(),
            offset_xy_m=self.offset_xy.tolist() if mode == 'offset' else [0., 0.],
            grip_fraction=self.grip_fraction if mode == 'weak_grip' else 1.,
            release_distance_m=self.release_distance if mode == 'early_release' else None)

    def mark_intervention(self, env):
        if self.intervention_step is None:
            self.intervention_step = env.elapsed_steps
            self.events.append(dict(step=env.elapsed_steps, phase='intervention',
                                    object_index=self.index, **self.parameters))

    def select(self, env):
        super().select(env)
        if self.done:
            return
        # The 60 mm calibration needs a larger closure margin for these masses.
        # Calibrated on isolated 50/60 mm boxes, both heights and all materials.
        if self.spec.size[0] >= .060:
            previous_bias = .2 if self.spec.material == 'purple' else .1
            for j, name in enumerate(env.joint_names[3:]):
                if name.endswith(('PIP', 'DIP', 'IP', 'MCP_FE')):
                    self.closed[j] += .3-previous_bias
            self.closed = np.clip(self.closed, env.low[3:], env.high[3:])
        if self.cursor == 1 and self.mode == 'weak_grip':
            self.closed = self.open+self.grip_fraction*(self.closed-self.open)
        if self.cursor == 1 and self.mode == 'offset':
            self.goal = self.goal+self.offset_xy

    def action(self, env, t):
        if (not self.done and self.cursor == 1 and self.mode == 'early_release'
                and self.phase == 'transport'):
            position = env.privileged()['object_pose'][self.index, :3]
            if np.linalg.norm(position[:2]-self.goal) < self.release_distance:
                self.release_wrist = env.data.qpos[env.qids[:3]].copy()
                self.mark_intervention(env)
                self.transition(env, 'release', reason='early_release_intervention')
        action, label = super().action(env, t)
        if action is None:
            return action, label
        phase = label.rsplit('/', 1)[-1]
        if self.mode == 'shuffled_order':
            self.mark_intervention(env)
        if self.cursor == 1 and self.mode == 'weak_grip' and phase == 'close':
            self.mark_intervention(env)
        if self.cursor == 1 and self.mode == 'offset' and phase in ('transport', 'lower'):
            self.mark_intervention(env)
            if phase == 'lower':
                # The base controller follows the moving support centre while
                # lowering. Apply the offset to that goal via commands only.
                tr = env.privileged()
                position = tr['object_pose'][self.index, :2]
                goal = (tr['object_pose'][self.support, :2] if self.support is not None
                        else np.array([env.config.goal_x, env.config.goal_y]))
                wrist = env.data.qpos[env.qids[:2]]
                desired = wrist+.6*(goal+self.offset_xy-position)
                action[:2] = np.clip(desired-env.targets[:2], -.0015, .0015)/env.action_scale[:2]
        return np.clip(action, -1, 1).astype(np.float32), label
