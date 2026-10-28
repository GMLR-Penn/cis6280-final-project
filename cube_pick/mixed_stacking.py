"""Finite-inventory stacking prototype: rigid shapes, material cues, terminal height.

This is a separate task, not a change to the released cube datasets. The example
controller is privileged and scripted. Physics never welds or teleports objects.
"""
from collections import deque
from dataclasses import asdict, dataclass, field
import json
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .env import DT, SceneConfig
from .sharpa import ROOT
from .whole_hand import SharpaWholeHandEnv
from .multicube import PLACEMENT_STAGES


# Synthetic rigid materials, not calibrated wood/rubber/metal models.
MATERIALS = {
    'blue': dict(rgb=(.20, .55, .90), density=600., friction=.45),
    'orange': dict(rgb=(.95, .48, .12), density=950., friction=.90),
    'purple': dict(rgb=(.64, .30, .80), density=1300., friction=.28),
}
SHAPES = ('cube', 'cuboid', 'cylinder', 'rhombic_prism')


@dataclass(frozen=True)
class ObjectSpec:
    shape: str
    size: tuple  # Full XYZ lengths; cylinder XY diameters, prism XY diagonals.
    material: str
    xy: tuple
    yaw: float = 0.

    def __post_init__(self):
        if self.shape not in SHAPES or self.material not in MATERIALS:
            raise ValueError('Unknown shape or material')
        size, xy = np.asarray(self.size), np.asarray(self.xy)
        if (size.shape != (3,) or not np.isfinite(size).all()
                or np.any(size < .025) or np.any(size > .12)
                or xy.shape != (2,) or not np.isfinite(xy).all()
                or not np.isfinite(self.yaw)):
            raise ValueError('Expected finite object size (25–120 mm), XY and yaw')
        if self.shape == 'cylinder' and not np.isclose(size[0], size[1]):
            raise ValueError('Cylinder requires equal XY diameters')
        if self.shape == 'cube' and not np.allclose(size, size[0]):
            raise ValueError('Cube requires equal dimensions')

    @property
    def volume(self):
        factor = math.pi/4 if self.shape == 'cylinder' else .5 if self.shape == 'rhombic_prism' else 1.
        return float(np.prod(self.size)*factor)

    @property
    def mass(self):
        return self.volume*MATERIALS[self.material]['density']

    def vertices(self):
        x, y, z = np.asarray(self.size)/2
        if self.shape == 'rhombic_prism':
            return np.array([(a, b, c) for c in (-z, z)
                             for a, b in ((x, 0), (0, y), (-x, 0), (0, -y))])
        return np.array([(a, b, c) for a in (-x, x) for b in (-y, y) for c in (-z, z)])


@dataclass
class MixedSceneConfig(SceneConfig):
    objects: list = field(default_factory=list)
    time_limit: float = 180.
    goal_half_width: float = .085
    task_version: int = 1  # Missing in archived metadata: retain original semantics.

    def __post_init__(self):
        if self.task_version not in (1, 2):
            raise ValueError('Unknown task version')
        self.objects = [ObjectSpec(**x) if isinstance(x, dict) else x for x in self.objects]
        if not 1 <= len(self.objects) <= 4:
            raise ValueError('Prototype supports 1–4 objects; default scenes contain four')
        if not np.isfinite(self.time_limit) or self.time_limit < DT or self.time_limit > 600:
            raise ValueError('time_limit must be between 0.1 and 600 simulated seconds')
        if not np.isclose(self.time_limit/DT, round(self.time_limit/DT)):
            raise ValueError('time_limit must be a multiple of the 0.1-second action interval')
        if not .05 <= self.goal_half_width <= .10:
            raise ValueError('goal_half_width must be 0.05–0.10 m')

    @classmethod
    def from_seed(cls, seed, time_limit=180.):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 142026]))
        shapes = rng.permutation(SHAPES)
        materials = list(rng.permutation(list(MATERIALS))) + [rng.choice(list(MATERIALS))]
        sizes = {'cube': (.055, .055, .055), 'cuboid': (.055, .055, .085),
                 'cylinder': (.058, .058, .065), 'rhombic_prism': (.074, .062, .050)}
        positions = [(-.17, -.15), (-.17, .15), (-.34, -.15), (-.34, .15)]
        specs = []
        for shape, material, xy in zip(shapes, materials, positions):
            scale = float(rng.uniform(.96, 1.04))
            specs.append(ObjectSpec(str(shape), tuple(np.array(sizes[shape])*scale), str(material),
                                    tuple(np.array(xy)+rng.uniform(-.006, .006, 2))))
        base = asdict(SceneConfig.from_seed(seed))
        base.update(cube_edge=.055, cube_yaw=0., cube_x=specs[0].xy[0], cube_y=specs[0].xy[1],
                    goal_x=.16, goal_y=0.)
        return cls(**base, objects=specs, time_limit=time_limit)


def world_bounds(spec, position, rotation):
    """Exact axis-aligned bounds of our solid, including tilted cylinders."""
    if spec.shape == 'cylinder':
        axis = rotation[:, 2]
        half = spec.size[2]/2*np.abs(axis) + spec.size[0]/2*np.sqrt(np.maximum(0, 1-axis**2))
        return position-half, position+half
    vertices = spec.vertices() @ rotation.T + position
    return vertices.min(axis=0), vertices.max(axis=0)


def stable_height(records, goal_xy, goal_half_width=.085, window=31):
    """Conservative terminal height of a released, table-rooted contact component.

    No cube alignment or upright assumption. A component must remain connected
    through persistent object contacts over the entire three-second window.
    Any transient hand contact or unstable member disqualifies that component.
    Only roots fully inside the build square may touch the table.
    """
    empty = dict(height_m=0., object_indices=[], object_count=0)
    if len(records) < window:
        return empty
    rows = list(records)[-window:]
    velocity = np.stack([r['object_velocity'] for r in rows])
    stable = ((np.linalg.norm(velocity[:, :, :3], axis=-1) < .1)
              & (np.linalg.norm(velocity[:, :, 3:], axis=-1) < .02)).all(axis=0)
    hand = np.stack([r['object_hand_contacts'] for r in rows]).any(axis=0)
    ground = np.stack([r['object_ground_contacts'] for r in rows]).any(axis=0)
    table = np.stack([r['object_table_contacts'] for r in rows]) > 0
    bounds = np.stack([r['object_bounds'] for r in rows])
    stable &= (np.ptp(bounds, axis=0) < .005).all(axis=(1, 2))
    in_zone = ((bounds[:, :, 0, :2] >= np.asarray(goal_xy)-goal_half_width)
               & (bounds[:, :, 1, :2] <= np.asarray(goal_xy)+goal_half_width)).all(axis=(0, 2))
    roots = table.all(axis=0) & in_zone
    contacts = np.stack([r['object_contacts'] for r in rows]) > 0
    persistent, ever = contacts.all(axis=0), contacts.any(axis=0)
    seen, best = set(), empty
    for root in range(len(stable)):
        if root in seen:
            continue
        component, todo = set(), [root]
        while todo:
            i = todo.pop()
            if i in component:
                continue
            component.add(i)
            todo.extend(np.flatnonzero(ever[i]).tolist())
        seen.update(component)
        ids = sorted(component)
        if (not roots[ids].any() or not stable[ids].all() or hand[ids].any() or ground[ids].any()
                or np.any(table.any(axis=0)[ids] & ~roots[ids])):
            continue
        reachable = set(np.flatnonzero(roots).tolist()) & component
        todo = list(reachable)
        while todo:
            for i in np.flatnonzero(persistent[todo.pop()]):
                if i in component and i not in reachable:
                    reachable.add(int(i))
                    todo.append(int(i))
        if reachable != component:
            continue
        height = float(bounds[:, ids, 1, 2].max(axis=1).min())
        if height > best['height_m']:
            best = dict(height_m=height, object_indices=ids, object_count=len(ids))
    return best


class MixedStackEnv(SharpaWholeHandEnv):
    """25-action XYZ/finger interface with STOP and an enforced simulated clock."""
    def __init__(self, seed=0, resolution=192, config=None, time_limit=180.):
        self.history = deque(maxlen=31)
        super().__init__(seed=seed, resolution=resolution,
                         config=config or MixedSceneConfig.from_seed(seed, time_limit))
        self.object_names = ['cube'] + [f'object_{i}' for i in range(1, len(self.config.objects))]
        self.object_ids = [self.model.body(n).id for n in self.object_names]
        self.object_geoms = [self.model.geom(n).id for n in self.object_names]
        self.object_qids = [self.model.joint('cube_free' if i == 0 else f'object_{i}_free').qposadr[0]
                            for i in range(len(self.object_names))]
        self.hand_geoms.difference_update(self.object_geoms)
        self.ground_geom = self.model.geom('ground').id
        poses = json.loads((ROOT/'assets/sharpa-size-grasps.json').read_text())
        self.open_fingers = np.array(poses['profiles'][1]['open'])
        self.reset(fingers=self.open_fingers)

    @staticmethod
    def build_xml(config):
        root = ET.fromstring(SharpaWholeHandEnv.build_xml(config))
        root.set('model', 'sharpa_mixed_stacking_v0')
        root.find(".//joint[@name='wrist_z']").set('range', '.03 .75')
        root.find(".//site[@name='goal']").set('size', f'{config.goal_half_width} {config.goal_half_width} .0005')
        for camera in root.findall('.//camera'):
            camera.set('fovy', '58')
        world = root.find('worldbody')
        world.remove(world.find("body[@name='cube']"))
        hand_geoms = []
        for i, geom in enumerate(world.findall('.//geom')):
            if geom.get('contype', '1') == '0' and geom.get('conaffinity', '1') == '0':
                continue
            if geom.get('name') in ('table', 'ground'):
                continue
            if not geom.get('name'):
                geom.set('name', f'mixed_hand_geom_{i}')
            hand_geoms.append(geom.get('name'))
        contact = root.find('contact')
        if contact is None:
            contact = ET.SubElement(root, 'contact')
        names = []
        for i, spec in enumerate(config.objects):
            name = 'cube' if i == 0 else f'object_{i}'
            names.append(name)
            material = MATERIALS[spec.material]
            body = ET.SubElement(world, 'body', name=name,
                                 pos=f'{spec.xy[0]} {spec.xy[1]} {spec.size[2]/2+.001}',
                                 quat=f'{math.cos(spec.yaw/2)} 0 0 {math.sin(spec.yaw/2)}')
            ET.SubElement(body, 'freejoint', name='cube_free' if i == 0 else f'{name}_free')
            attrs = dict(name=name, mass=str(spec.mass), rgba=' '.join(map(str, material['rgb']))+' 1',
                         friction=f"{material['friction']} .005 .0001", condim='4',
                         solref='.008 1', solimp='.95 .99 .001')
            if spec.shape == 'rhombic_prism':
                mesh = f'{name}_mesh'
                ET.SubElement(root.find('asset'), 'mesh', name=mesh,
                              vertex=' '.join(map(str, spec.vertices().ravel())))
                attrs.update(type='mesh', mesh=mesh)
            elif spec.shape == 'cylinder':
                attrs.update(type='cylinder', size=f'{spec.size[0]/2} {spec.size[2]/2}')
            else:
                attrs.update(type='box', size=' '.join(str(x/2) for x in spec.size))
            ET.SubElement(body, 'geom', **attrs)
        # Explicit symmetric geometric-mean mixing is a task convention, not a
        # claim about real materials. It avoids default max-friction mixing.
        for i, name in enumerate(names):
            partners = [(other, .9) for other in hand_geoms] + [('table', .7), ('ground', .7)]
            partners += [(names[j], MATERIALS[config.objects[j].material]['friction']) for j in range(i)]
            for other, friction in partners:
                mu = math.sqrt(MATERIALS[config.objects[i].material]['friction']*friction)
                ET.SubElement(contact, 'pair', geom1=name, geom2=other, condim='4',
                              friction=f'{mu} {mu} .005 .0001 .0001', solref='.008 1', solimp='.95 .99 .001')
        return ET.tostring(root, encoding='unicode')

    def reset(self, fingers=None):
        result = super().reset(fingers=fingers if fingers is not None else getattr(self, 'open_fingers', None))
        self.elapsed_steps = 0
        self.max_steps = round(self.config.time_limit/DT)
        self.status, self.stop_reason = 'running', None
        self.finalization_steps = 0
        self.history.clear()
        if hasattr(self, 'object_ids'):
            self.history.append(self.privileged())
        return result

    @property
    def time_remaining(self):
        return max(0., (self.max_steps-self.elapsed_steps)*DT)

    def observation(self, rgb=True):
        obs = dict(proprio=self.proprio(), time_remaining=np.float32(self.time_remaining),
                   goal_xy=np.array([self.config.goal_x, self.config.goal_y], np.float32))
        if rgb:
            obs['rgb'] = self.render()
        return obs

    def step(self, action):
        if self.status != 'running':
            raise RuntimeError('Episode stopped: call finalize_step(), or reset for a new episode')
        result = super().step(action)
        self.elapsed_steps += 1
        self.history.append(self.privileged())
        if self.elapsed_steps >= self.max_steps:
            self.stop('time_limit')
        return result

    def stop(self, reason='agent_stop'):
        if self.status != 'running':
            raise RuntimeError('STOP is only valid during an active episode')
        if reason not in ('agent_stop', 'time_limit'):
            raise ValueError('Invalid stop reason')
        self.status, self.stop_reason = 'finalizing', reason
        self.retreat_xy = self.data.qpos[self.qids[:2]].copy()
        self.lift_steps, self.withdraw_steps = 20, 40
        if self.config.task_version == 2:
            lift = np.r_[self.retreat_xy, .70, self.open_fingers]
            park = np.r_[-.35, -.25, .70, self.open_fingers]
            self.lift_steps = int(np.ceil(np.max(np.abs(lift-self.targets)/self.action_scale))) + 10
            move_steps = int(np.ceil(np.max(np.abs(park-lift)/self.action_scale))) + 10
            self.withdraw_steps = self.lift_steps + move_steps

    def finalize_step(self):
        """Release/withdrawal, then hands-off observation (3.1 s).

        V1 retains 40 withdrawal steps; V2 budgets enough bounded actions to
        reach both lift and park targets, plus a settling margin per movement.

        No policy actions are accepted after STOP/deadline. Returns the applied
        action and phase so the entire physical transition can be recorded.
        """
        if self.status != 'finalizing':
            raise RuntimeError('Not in finalization')
        k = self.finalization_steps
        if k < self.lift_steps:
            desired = np.r_[self.retreat_xy, .70, self.open_fingers]
        else:
            desired = np.r_[-.35, -.25, .70, self.open_fingers]
        action = np.clip((desired-self.targets)/self.action_scale, -1, 1).astype(np.float32)
        if k >= self.withdraw_steps:
            action[:] = 0
        super().step(action)
        self.history.append(self.privileged())
        self.finalization_steps += 1
        if self.finalization_steps == self.withdraw_steps + 31:
            self.status = 'done'
        return action, 'withdraw' if k < self.withdraw_steps else 'verify'

    def score(self, window=31):
        from .support_score import supported_height
        scorer = stable_height if self.config.task_version == 1 else supported_height
        return scorer(self.history, [self.config.goal_x, self.config.goal_y],
                      self.config.goal_half_width, window=window)

    def outcome(self):
        if self.status != 'done':
            raise RuntimeError('A terminal score requires completed hands-off verification')
        score = self.score()
        result = dict(final_stable_height_m=score['height_m'], scored_objects=score['object_indices'],
                    scored_object_count=score['object_count'], elapsed_seconds=round(self.elapsed_steps*DT, 6),
                    time_limit_seconds=self.config.time_limit, stop_reason=self.stop_reason,
                    verification_seconds=3., finalization_seconds=round(self.finalization_steps*DT, 6))
        if self.config.task_version == 2:
            result.update(task_version=2, scoring_profile='upward-support-v2',
                          park_reached=bool(np.max(np.abs(self.data.qpos[self.qids[:3]]
                                                         - [-.35, -.25, .70])) < .005))
        return result

    def privileged(self):
        n = len(self.object_ids)
        pose = np.stack([self.data.qpos[q:q+7].copy() for q in self.object_qids])
        velocity, bounds = np.zeros((n, 6)), np.zeros((n, 2, 3))
        for i, body in enumerate(self.object_ids):
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity[i], 0)
            bounds[i] = world_bounds(self.config.objects[i], self.data.xpos[body], self.data.xmat[body].reshape(3, 3))
        pairs, table, ground, hand = np.zeros((n, n), np.int16), np.zeros(n, np.int16), np.zeros(n, np.int16), np.zeros(n, np.int16)
        forces = np.zeros((n, 6))
        support, table_support = np.zeros((n, n), bool), np.zeros(n, bool)
        lookup = {g: i for i, g in enumerate(self.object_geoms)}
        for ci, contact in enumerate(self.data.contact):
            a, b = lookup.get(contact.geom1), lookup.get(contact.geom2)
            if self.config.task_version == 2:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, ci, wrench)
                # Contact normal points from geom1 to geom2. Require an upward
                # force component and positive compression, not lateral touch.
                nz = float(contact.frame[2])
                if wrench[0] > 1e-4 and abs(nz) > .5:
                    lower, upper = (a, b) if nz > 0 else (b, a)
                    lower_geom = contact.geom1 if nz > 0 else contact.geom2
                    if upper is not None:
                        if lower is not None and pose[upper, 2] > pose[lower, 2]:
                            support[lower, upper] = True
                        elif lower_geom == self.table_geom:
                            table_support[upper] = True
            if a is not None and b is not None:
                pairs[a, b] += 1
                pairs[b, a] += 1
            for geom, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                i = lookup.get(geom)
                if i is None:
                    continue
                if other == self.table_geom:
                    table[i] += 1
                elif other == self.ground_geom:
                    ground[i] += 1
                elif other in self.hand_geoms:
                    hand[i] += 1
                    name = self.model.body(self.model.geom_bodyid[other]).name
                    part = next((j for j, f in enumerate(self.contact_parts[:5]) if f in name), 5)
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, ci, wrench)
                    forces[i, part] += wrench[0]
        result = dict(object_pose=pose, object_velocity=velocity, object_bounds=bounds,
                    object_contacts=pairs, object_table_contacts=table, object_ground_contacts=ground,
                    object_hand_contacts=hand, object_hand_normal_force=forces)
        if self.config.task_version == 2:
            result.update(object_support=support, object_table_support=table_support)
        return result


class MixedStackCollector:
    """Privileged, geometry-based demonstration; no visual model or recovery.

    Conservative selects the tallest initial upright object and stops. Stack
    uses broad box bases first. These are illustrative policies, not oracles.
    """
    def __init__(self, env, mode='stack'):
        if mode not in ('conservative', 'stack'):
            raise ValueError('Choose conservative or stack')
        self.mode = mode
        specs = env.config.objects
        if mode == 'conservative':
            self.order = [max(range(len(specs)), key=lambda i: specs[i].size[2])]
        else:
            rank = {'cube': 0, 'cuboid': 1, 'cylinder': 2, 'rhombic_prism': 3}
            self.order = sorted(range(len(specs)), key=lambda i: rank[specs[i].shape])
        # Rounded shapes/heavier objects require gentler transport than cubes.
        self.stages = [(phase, 104 if phase == 'transport' else 48 if phase == 'approach'
                        else count) for phase, count in PLACEMENT_STAGES]
        self.cycle_steps = sum(n for _, n in self.stages)
        self.horizon = self.cycle_steps*len(self.order)
        self.calibration = json.loads((ROOT/'assets/sharpa-size-grasps.json').read_text())
        self.shape_profiles = json.loads((ROOT/'assets/sharpa-mixed-grasps.json').read_text())['profiles']
        self.active_cycle = -1
        self.top_height = 0.

    def action(self, env, t):
        if t >= self.horizon:
            return None, 'stop'
        cycle, local = divmod(t, self.cycle_steps)
        i = self.order[cycle]
        spec = env.config.objects[i]
        if cycle != self.active_cycle:
            self.active_cycle = cycle
            profiles = self.calibration['profiles']
            sizes = np.array([r['edge_mm'] for r in profiles])/1000
            grip_width = spec.size[0] if spec.shape != 'rhombic_prism' else .050
            profile = {key: np.array([np.interp(grip_width, sizes, column)
                                     for column in np.array([r[key] for r in profiles]).T])
                       for key in ('grasp_center_local', 'open', 'closed')}
            self.center, self.open, self.closed = (profile[k] for k in ('grasp_center_local', 'open', 'closed'))
            if spec.shape in self.shape_profiles:
                profile = self.shape_profiles[spec.shape]
                if profile['joint_names'] != env.joint_names[3:]:
                    raise ValueError('Shape grasp profile joint order mismatch')
                self.center, self.open, self.closed = (np.array(profile[k]) for k in ('grasp_center_local', 'open', 'closed'))
                # Whole-hand local X points down in world Z. This raises the
                # grasp point relative to the object's centre, including placement.
                self.center[0] += profile['grasp_height_offset']
            truth = env.privileged()
            self.start = truth['object_pose'][i, :3].copy()
            scored = env.score(window=21)
            self.top_height = scored['height_m']
            self.goal = np.array([env.config.goal_x, env.config.goal_y])
            if scored['object_indices']:
                top = max(scored['object_indices'], key=lambda j: truth['object_bounds'][j, 1, 2])
                self.goal = truth['object_pose'][top, :2].copy()
        cursor = 0
        for phase, count in self.stages:
            if local < cursor+count:
                break
            cursor += count
        height = spec.size[2]
        xy, z, fingers = self.start[:2].copy(), max(.20, self.top_height+height+.09), self.open.copy()
        if phase == 'clear':
            xy = env.data.qpos[env.qids[:2]] + (env.rotation@self.center)[:2]
        if phase in ('descend', 'close'):
            z = self.start[2]
        if phase in ('close', 'lift', 'transport', 'lower'):
            fingers = self.closed
        if phase in ('transport', 'lower', 'release', 'retreat', 'settle'):
            xy = self.goal
        if phase in ('lower', 'release'):
            z = self.top_height+height/2+.001
        desired = np.r_[np.r_[xy, z]-env.rotation@self.center, fingers]
        action = np.clip((desired-env.targets)/env.action_scale, -1, 1)
        if phase == 'lower':
            action[2] = np.clip(action[2], -.30, .30)
        if phase == 'transport':
            action[:2] = np.clip(action[:2], -.35, .35)
        return action.astype(np.float32), f'object_{i}/{phase}'


def make_mixed_controller(env, mode='stack'):
    """Keep the original demonstration modes reproducible alongside feedback."""
    if mode == 'feedback':
        from .mixed_feedback import FeedbackStackController
        return FeedbackStackController(env)
    return MixedStackCollector(env, mode)
