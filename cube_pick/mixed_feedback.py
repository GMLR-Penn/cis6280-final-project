"""Privileged feedback reference for the fixed-wrist mixed-shape task.

Only normalized actuator commands affect the live environment. This controller
uses measured object poses/contacts; it is not a learned or visual-only policy.
"""
import numpy as np

from .mixed_stacking import MixedStackCollector


class FeedbackStackController:
    """Pose/contact feedback with a cylinder-first, finite-inventory strategy.

    The rounded object settles on the table before it bears the other objects.
    This avoids placing a tilted, fixed-wrist cylinder on a narrow tower. It is
    a reference for the default upright scenes, not arbitrary-orientation IK.
    """
    def __init__(self, env):
        self.profiles = MixedStackCollector(env, 'stack')
        rank = {'cylinder': 0, 'cube': 1, 'cuboid': 2, 'rhombic_prism': 3}
        self.order = sorted(range(len(env.config.objects)),
                            key=lambda i: rank[env.config.objects[i].shape])
        self.profiles.order = self.order.copy()
        self.events = []
        self.cursor = 0
        self.phase = 'select'
        self.age = 0
        self.support = None
        self.done = False
        self.contact_loss_steps = 0

    def transition(self, env, phase, **extra):
        self.events.append(dict(step=env.elapsed_steps, object_index=getattr(self, 'index', None),
                                phase=phase, **extra))
        self.phase, self.age = phase, 0

    def select(self, env):
        if self.cursor >= len(self.order):
            self.done = True
            return
        self.index = self.order[self.cursor]
        self.spec = env.config.objects[self.index]
        # Reuse the released calibrations without changing their files or rules.
        self.profiles.action(env, self.cursor*self.profiles.cycle_steps)
        self.center = self.profiles.center.copy()
        self.open = self.profiles.open.copy()
        self.closed = self.profiles.closed.copy()
        if self.spec.shape == 'rhombic_prism':
            # Lower the grasp to keep the diamond footprint between the pads.
            self.center[0] -= .015
        truth = env.privileged()
        self.source = truth['object_pose'][self.index, :3].copy()
        score = env.score(window=21)
        self.support = (max(score['object_indices'], key=lambda i: truth['object_bounds'][i, 1, 2])
                        if score['object_indices'] else None)
        self.goal = (truth['object_pose'][self.support, :2].copy() if self.support is not None
                     else np.array([env.config.goal_x, env.config.goal_y]))
        self.support_height = score['height_m']
        self.previous_scored = set(score['object_indices'])
        self.clearance = max(.18, self.support_height+self.spec.size[2]+.06)
        self.lift_start = self.source[2]
        self.contact_loss_steps = 0
        self.transition(env, 'clear')

    def action(self, env, t):
        if self.done:
            return None, 'stop'
        if self.phase == 'select':
            self.select(env)
            if self.done:
                return None, 'stop'
        tr = env.privileged()
        i = self.index
        position = tr['object_pose'][i, :3]
        wrist = env.data.qpos[env.qids[:3]]
        fingers = self.open.copy()
        desired = wrist.copy()
        limits = np.array([.010, .010, .008])
        offset = env.rotation@self.center
        self.age += 1
        phase = self.phase
        if phase == 'clear':
            desired[2] = self.clearance-offset[2]
            if abs(desired[2]-wrist[2]) < .003:
                self.transition(env, 'approach')
        elif phase == 'approach':
            desired = np.r_[self.source[:2], self.clearance]-offset
            if np.linalg.norm(desired-wrist) < .003:
                self.transition(env, 'descend')
        elif phase == 'descend':
            desired = self.source-offset
            limits[2] = .004
            # Servo compliance and table contact leave a few millimetres of
            # residual error even after the commanded pose has been reached.
            if (np.linalg.norm(desired-wrist) < .006
                    or (self.age >= 40 and np.linalg.norm(desired-env.targets[:3]) < .002)):
                self.transition(env, 'close')
        elif phase == 'close':
            desired = self.source-offset
            fingers = self.open+(self.closed-self.open)*min(1., self.age/15)
            if self.age >= 25:
                self.transition(env, 'lift')
        elif phase in ('lift', 'transport', 'lower'):
            fingers = self.closed.copy()
            if phase == 'lift':
                target = np.r_[self.source[:2], self.clearance]
                desired = wrist+target-position
                limits[:] = [.003, .003, .006]
                if position[2] >= self.clearance-.003:
                    self.transition(env, 'transport')
                elif self.age > 70 and position[2] < self.lift_start+.02:
                    self.transition(env, 'failed_grasp')
            elif phase == 'transport':
                desired = wrist+np.r_[self.goal, self.clearance]-position
                limits[:] = [.005, .005, .003]
                if np.linalg.norm(position[:2]-self.goal) < .002 and abs(position[2]-self.clearance) < .004:
                    self.transition(env, 'lower')
            else:
                if self.support is not None:
                    self.goal = tr['object_pose'][self.support, :2].copy()
                    self.support_height = tr['object_bounds'][self.support, 1, 2]
                gap = tr['object_bounds'][i, 0, 2]-self.support_height
                desired[:2] = wrist[:2]+.6*(self.goal-position[:2])
                desired[2] = wrist[2]-max(.0002, min(.002, gap*.4))
                limits[:] = [.0015, .0015, .002]
                touching = (tr['object_contacts'][i, self.support] > 0 if self.support is not None
                            else tr['object_table_contacts'][i] > 0)
                if touching:
                    self.release_wrist = wrist.copy()
                    self.transition(env, 'release')
        elif phase == 'release':
            desired = self.release_wrist.copy()
            fingers = self.closed+(self.open-self.closed)*min(1., self.age/25)
            if self.age >= 30:
                self.transition(env, 'retreat')
        elif phase in ('retreat', 'failed_grasp'):
            desired[2] = self.clearance-offset[2]+.04
            if abs(desired[2]-wrist[2]) < .004 or self.age >= 70:
                self.transition(env, 'settle')
        elif phase == 'settle':
            if self.age >= 31:
                score = env.score()
                self.events.append(dict(step=env.elapsed_steps, object_index=i, phase='placement_result', **score))
                if not (self.previous_scored | {i}).issubset(score['object_indices']):
                    self.done = True
                    self.transition(env, 'stop', reason='placement_not_stable')
                else:
                    self.cursor += 1
                    self.transition(env, 'select')
        if phase == 'transport':
            self.contact_loss_steps = (self.contact_loss_steps+1
                                       if tr['object_hand_contacts'][i] == 0 else 0)
            if self.contact_loss_steps >= 5 or tr['object_ground_contacts'][i] > 0:
                self.transition(env, 'failed_grasp', reason='lost_object')
        if self.age > 180 and self.phase not in ('settle', 'failed_grasp'):
            self.transition(env, 'failed_grasp', reason='phase_timeout')
        delta = np.clip(desired-env.targets[:3], -limits, limits)
        action = np.r_[delta/env.action_scale[:3], (fingers-env.targets[3:])/env.action_scale[3:]]
        return np.clip(action, -1, 1).astype(np.float32), f'object_{i}/{phase}'
