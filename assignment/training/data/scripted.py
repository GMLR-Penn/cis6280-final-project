"""Scripted stacking policies with optional recovery and plan variation.

Extends VariedBoxController without changing it. Recovery continues after a
failed grasp, drop or collapse instead of stopping; plan variation perturbs
waypoints/grasp pose/speed of the plan and is distinct from action noise.
"""
from collections import Counter

import numpy as np

from cube_pick.varied_boxes import VariedBoxController

VARIATION = dict(speed=(.75, 1.25), grasp_offset_m=.004, extra_clearance_m=(0., .04), place_offset_m=.004)


class ScriptedStackPolicy(VariedBoxController):
    def __init__(self, env, mode='nominal', seed=0, recovery=False, variation=None,
                 max_retries=3, max_object_retries=2, policy_seed=0, sweep=False):
        # Intervention parameters follow the scene seed (paired with nominal);
        # plan variation follows the per-episode policy seed.
        super().__init__(env, mode, seed)
        self.recovery, self.max_retries, self.max_object_retries = recovery, max_retries, max_object_retries
        self.retries = 0
        self.object_retries = Counter()
        self.rank = {i: r for r, i in enumerate(self.order)}
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, policy_seed, 162028]))
        self.variation = None
        if variation:
            ranges = dict(VARIATION, **({} if variation is True else variation))
            self.variation = dict(ranges=ranges, speed=float(self.rng.uniform(*ranges['speed'])), placements=[])
        self.sweep = dict(depth_m=float(self.rng.uniform(.015, .04)), step=None) if sweep else None
        self.parameters.update(recovery=recovery, plan_variation=self.variation, sweep=self.sweep)

    def transition(self, env, phase, **extra):
        if phase == 'stop' and self.recovery and self.recover(env):
            return
        super().transition(env, phase, **extra)

    def recover(self, env):
        if self.retries >= self.max_retries:
            return False
        truth = env.privileged()
        stable = sorted(env.score()['object_indices'], key=lambda i: truth['object_bounds'][i, 1, 2])
        self.object_retries[self.index] += 1
        candidates = sorted((i for i in range(len(self.order))
                             if i not in stable and truth['object_ground_contacts'][i] == 0
                             and self.object_retries[i] <= self.max_object_retries),
                            key=self.rank.get)
        if not candidates:
            return False
        self.retries += 1
        self.events.append(dict(step=env.elapsed_steps, phase='recovery', retry=self.retries,
                                failed_object=int(self.index), stable=[int(i) for i in stable],
                                candidates=[int(i) for i in candidates]))
        self.order = [int(i) for i in stable+candidates]
        self.profiles.order = self.order.copy()
        self.profiles.active_cycle = -1
        self.cursor = len(stable)
        self.done = False
        # Scripted interventions target one placement; retries use the nominal plan.
        self.mode = 'nominal'
        super().transition(env, 'select')
        return True

    def select(self, env):
        super().select(env)
        if not self.done and self.sweep and self.sweep['step'] is None and self.cursor >= 2:
            # Path failure: carry the object below the tower top so it strikes the tower once.
            self.clearance = self.support_height-self.sweep['depth_m']
            self.sweep['step'] = env.elapsed_steps
            self.events.append(dict(step=env.elapsed_steps, phase='sweep_intervention', object_index=int(self.index),
                                    clearance_m=float(self.clearance)))
        if self.done or not self.variation:
            return
        ranges = self.variation['ranges']
        grasp = self.rng.uniform(-1, 1, 3)*ranges['grasp_offset_m']
        clearance = float(self.rng.uniform(*ranges['extra_clearance_m']))
        place = self.rng.uniform(-1, 1, 2)*ranges['place_offset_m']
        self.center = self.center+grasp
        self.clearance += clearance
        self.goal = self.goal+place
        self.variation['placements'].append(dict(step=env.elapsed_steps, object=int(self.index),
            grasp_offset_m=grasp.tolist(), extra_clearance_m=clearance, place_offset_m=place.tolist()))

    def action(self, env, t):
        action, label = super().action(env, t)
        if action is not None and self.variation:
            action[:3] = np.clip(action[:3]*self.variation['speed'], -1, 1)
        return action, label
