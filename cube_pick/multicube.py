"""Prototype sequential stacking of 2–5 free cubes, with measured tower labels."""
from dataclasses import dataclass, replace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .stacking import StackSceneConfig, SharpaStackEnv
from .whole_hand import SizeConditionedCollector


@dataclass
class MultiCubeConfig(StackSceneConfig):
    num_cubes: int = 3  # Includes the support already on the table.
    pickup_xy: list | None = None


class MultiCubeStackEnv(SharpaStackEnv):
    def __init__(self, seed=0, resolution=192, config=None, *, num_cubes=3, cube_edge=.050):
        if config is None:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 82818]))
            config = replace(MultiCubeConfig.from_seed(seed), num_cubes=num_cubes,
                             cube_edge=cube_edge, cube_yaw=0., goal_x=.16, goal_y=0.,
                             support_mass=float(rng.uniform(.12, .24)),
                             support_friction=float(rng.uniform(.5, .9)))
            # Widely spaced stock positions keep the open hand away from neighbors.
            positions = [[-.17, -.15], [-.17, .15], [-.34, -.15], [-.34, .15]]
            jitter = rng.uniform(-.008, .008, (4, 2))
            positions = (np.asarray(positions) + jitter).tolist()[:num_cubes-1]
            config = replace(config, pickup_xy=positions,
                             cube_x=positions[0][0], cube_y=positions[0][1]) if positions else config
        if not isinstance(config.num_cubes, int) or not 2 <= config.num_cubes <= 5:
            raise ValueError('Prototype supports 2–5 total cubes, including the base')
        if not np.isfinite(config.cube_edge) or not .045 <= config.cube_edge <= .070:
            raise ValueError('Calibrated pickup cube edges must be 45–70 mm')
        xy = np.asarray(config.pickup_xy, dtype=float)
        if xy.shape != (config.num_cubes-1, 2) or not np.isfinite(xy).all():
            raise ValueError('pickup_xy must contain one finite XY position per pickup cube')
        super().__init__(seed=seed, resolution=resolution, config=config)
        self.object_names = ['support_cube', 'cube'] + [f'cube_{i}' for i in range(2, config.num_cubes)]
        self.object_ids = [self.model.body(n).id for n in self.object_names]
        self.object_geoms = [self.model.geom(n).id for n in self.object_names]
        self.object_qids = [self.model.joint('support_free' if i == 0 else 'cube_free' if i == 1 else f'cube_{i}_free').qposadr[0]
                            for i in range(config.num_cubes)]
        self.hand_geoms.difference_update(self.object_geoms)
        self.object_edges = np.r_[config.support_edge, np.full(config.num_cubes-1, config.cube_edge)]

    @staticmethod
    def build_xml(config):
        root = ET.fromstring(SharpaStackEnv.build_xml(config))
        root.set('model', 'sharpa_multicube_prototype')
        # Extra clearance for the hand above a five-layer tower; only this task changes.
        root.find(".//joint[@name='wrist_z']").set('range', '.03 .65')
        for camera in root.findall('.//camera'):
            camera.set('fovy', '55')
        colors = ((.25, .65, .9), (.8, .3, .45), (.35, .8, .45))
        for i in range(2, config.num_cubes):
            x, y = config.pickup_xy[i-1]
            body = ET.SubElement(root.find('worldbody'), 'body', name=f'cube_{i}',
                                 pos=f'{x} {y} {config.cube_edge/2+.001}')
            ET.SubElement(body, 'freejoint', name=f'cube_{i}_free')
            ET.SubElement(body, 'geom', name=f'cube_{i}', type='box',
                          size=' '.join([str(config.cube_edge/2)]*3), mass=str(config.cube_mass),
                          friction=f'{config.cube_friction} .005 .0001', priority='1', condim='4',
                          solref='.008 1', solimp='.95 .99 .001',
                          rgba=' '.join(map(str, colors[i-2]))+' 1')
        return ET.tostring(root, encoding='unicode')

    def privileged(self):
        truth = super().privileged()
        n = self.config.num_cubes
        velocities = np.zeros((n, 6))
        for i, body in enumerate(self.object_ids):
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, velocities[i], 0)
        pairs = np.zeros((n, n), np.int16)
        table = np.zeros(n, np.int16)
        hand = np.zeros((n, 6))
        lookup = {g: i for i, g in enumerate(self.object_geoms)}
        for contact in self.data.contact:
            a, b = lookup.get(contact.geom1), lookup.get(contact.geom2)
            if a is not None and b is not None:
                pairs[a, b] += 1
                pairs[b, a] += 1
        for ci, contact in enumerate(self.data.contact):
            for obj_geom, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                i = lookup.get(obj_geom)
                if i is None:
                    continue
                if other == self.table_geom:
                    table[i] += 1
                elif other in self.hand_geoms:
                    name = self.model.body(self.model.geom_bodyid[other]).name
                    part = next((j for j, f in enumerate(self.contact_parts[:5]) if f in name), 5)
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, ci, wrench)
                    hand[i, part] += wrench[0]
        # Contact count, rather than force threshold, is used to verify release.
        hand_counts = np.zeros(n, np.int16)
        for contact in self.data.contact:
            for obj_geom, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                if obj_geom in lookup and other in self.hand_geoms:
                    hand_counts[lookup[obj_geom]] += 1
        truth.update(object_pose=np.stack([self.data.qpos[q:q+7].copy() for q in self.object_qids]),
                     object_velocity=velocities, object_contacts=pairs, object_table_contacts=table,
                     object_hand_contacts=hand_counts, object_hand_normal_force=hand)
        return truth


PLACEMENT_STAGES = [('clear', 16), ('approach', 32), ('descend', 32), ('close', 16),
                    ('lift', 32), ('transport', 44), ('lower', 40), ('release', 18),
                    ('retreat', 24), ('settle', 30)]


class MultiCubeCollector(SizeConditionedCollector):
    """Privileged, timed script: repeat pickup/place, without moving object poses."""
    def __init__(self, env, mode='nominal', seed=0):
        if mode not in ('nominal', 'offset', 'early_release'):
            raise ValueError('Modes: nominal, offset, early_release')
        super().__init__(env, mode, seed)
        self.cycle_stages = list(PLACEMENT_STAGES)
        self.cycle_steps = sum(n for _, n in self.cycle_stages)
        self.stages = [(f'cube_{i}/{phase}', steps) for i in range(1, env.config.num_cubes)
                       for phase, steps in self.cycle_stages]
        self.horizon = self.cycle_steps*(env.config.num_cubes-1)

    def action(self, env, t):
        if not 0 <= t < self.horizon:
            raise IndexError(t)
        cycle, local = divmod(t, self.cycle_steps)
        i = cycle+1
        cursor = 0
        for phase, count in self.cycle_stages:
            if local < cursor+count:
                phase_t = local-cursor
                break
            cursor += count
        c = env.config
        start = np.asarray(c.pickup_xy[cycle])
        goal = np.array([c.goal_x, c.goal_y])
        # Fail on the last placement so partially built towers are represented.
        perturbed = i == c.num_cubes-1
        if self.mode == 'offset' and perturbed:
            goal[0] += .045
        clearance = c.support_edge + i*c.cube_edge + .08
        xy, z, fingers = start.copy(), clearance, self.open.copy()
        if phase == 'clear':
            xy = np.asarray(c.pickup_xy[0]) if cycle == 0 else np.array([c.goal_x, c.goal_y])
        if phase in ('descend', 'close'):
            z = c.cube_edge/2+.001
        if phase in ('close', 'lift', 'transport', 'lower'):
            fingers = self.closed.copy()
        if phase in ('transport', 'lower', 'release', 'retreat', 'settle'):
            xy = goal
        if phase in ('lower', 'release'):
            z = c.support_edge + (i-1)*c.cube_edge + c.cube_edge/2 + .001
        if self.mode == 'early_release' and perturbed and phase == 'transport' and phase_t >= 6:
            fingers = self.open.copy()
        desired = np.r_[env.wrist_for_cube([*xy, z]), fingers]
        action = np.clip((desired-env.targets)/env.action_scale, -1, 1)
        # Limit vertical placement speed; actuator dynamics still determine motion.
        if phase == 'lower':
            action[2] = np.clip(action[2], -.30, .30)
        return action.astype(np.float32), f'cube_{i}/{phase}'


def stable_tower(truth, edges, goal_xy, window=21):
    """Longest identical contact chain stable at every sample of a time window.

    Cube zero is the designated base. A base by itself counts as one layer.
    A frame spike, airborne pile or hand-supported stack cannot earn layers.
    """
    if len(truth) < window:
        return {'layers': 0, 'height_m': 0., 'object_indices': []}
    records = truth[-window:]
    poses = np.stack([r['object_pose'] for r in records])
    velocity = np.stack([r['object_velocity'] for r in records])
    upright = 1-2*(poses[:, :, 4]**2+poses[:, :, 5]**2)
    valid = ((upright > np.cos(np.deg2rad(10)))
             & (np.linalg.norm(velocity[:, :, 3:], axis=-1) < .02)
             & (np.linalg.norm(velocity[:, :, :3], axis=-1) < .1)
             & (np.stack([r['object_hand_contacts'] for r in records]) == 0)).all(axis=0)
    base_ok = (valid[0] and all(r['object_table_contacts'][0] > 0 for r in records)
               and np.all(np.linalg.norm(poses[:, 0, :2]-goal_xy, axis=-1) < .02))
    if not base_ok:
        return {'layers': 0, 'height_m': 0., 'object_indices': []}
    contacts = np.stack([r['object_contacts'] for r in records]).all(axis=0)
    n = len(edges)
    above = np.zeros((n, n), bool)
    for lower in range(n):
        for upper in range(n):
            dz = poses[:, upper, 2]-poses[:, lower, 2]
            above[lower, upper] = (lower != upper and valid[upper] and contacts[lower, upper]
                and np.all(np.linalg.norm(poses[:, upper, :2]-poses[:, lower, :2], axis=-1) < .015)
                and np.all(np.abs(dz-(edges[upper]+edges[lower])/2) < .006))
    def longest(chain):
        choices = [longest(chain+[i]) for i in range(n) if i not in chain and above[chain[-1], i]]
        return max([chain, *choices], key=len)
    chain = longest([0])
    height = float(np.min(poses[:, chain[-1], 2]+edges[chain[-1]]/2))
    return {'layers': len(chain), 'height_m': height, 'object_indices': chain}


def summarize_multicube(env, truth, proprio=None):
    goal = np.array([env.config.goal_x, env.config.goal_y])
    final = stable_tower(truth, env.object_edges, goal)
    best = max((stable_tower(truth[:t+1], env.object_edges, goal) for t in range(20, len(truth))),
               key=lambda r: (r['layers'], r['height_m']), default={'layers': 0, 'height_m': 0., 'object_indices': []})
    lifted = [bool(any(r['object_pose'][i, 2] > env.object_edges[i]/2+.05 and
                      r['object_table_contacts'][i] == 0 for r in truth)) for i in range(1, len(env.object_edges))]
    cycle_steps = sum(steps for _, steps in PLACEMENT_STAGES)
    placements = []
    for i in range(1, env.config.num_cubes):
        end = i*cycle_steps
        if end >= len(truth):
            break
        tower = stable_tower(truth[:end+1], env.object_edges, goal)
        placements.append({'pickup_object': i, 'stable_layers': tower['layers'],
                           'placed_on_tower': bool(i in tower['object_indices'])})
    return {'success': bool(final['layers'] == env.config.num_cubes and all(lifted)),
            'stack_success': bool(final['layers'] == env.config.num_cubes and all(lifted)),
            'target_layers': env.config.num_cubes, 'final_stable_layers': final['layers'],
            'final_stable_height_m': final['height_m'], 'final_tower_objects': final['object_indices'],
            'max_stable_layers': best['layers'], 'max_stable_height_m': best['height_m'],
            'pickup_lifted': lifted, 'placements': placements, 'stability_window_seconds': 2.}
