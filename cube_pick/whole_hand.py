"""Experimental whole-hand grasp; per-finger physical contacts are measured."""
import xml.etree.ElementTree as ET
from dataclasses import replace
import json
import mujoco
import numpy as np
from .sharpa import SharpaCubeEnv, build_xml, FINGERS, ROOT
from .sharpa_policy import SharpaCollector, summarize_sharpa
from .env import SceneConfig


class SharpaWholeHandEnv(SharpaCubeEnv):
    default_cube_edge = .055
    rotation = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
    grasp_center = np.array([.035, 0., .115])
    contact_parts = (*FINGERS, 'palm')

    def __init__(self, seed=0, resolution=192, config=None, *, cube_edge=None):
        """cube_edge is in meters and changes collision geometry and inertia too.

        SizeConditionedCollector covers 45–70 mm. Other positive sizes are
        permitted for experiments, without a scripted-success guarantee.
        """
        if cube_edge is not None:
            edge = float(cube_edge)
            if not np.isfinite(edge) or edge <= 0:
                raise ValueError('cube_edge must be a finite positive length in meters')
            if config is None:
                config = replace(SceneConfig.from_seed(seed), cube_edge=edge, cube_yaw=0.)
            else:
                config = replace(config, cube_edge=edge)
        elif config is not None and (not np.isfinite(config.cube_edge) or config.cube_edge <= 0):
            raise ValueError('cube_edge must be a finite positive length in meters')
        super().__init__(seed=seed, resolution=resolution, config=config)

    @staticmethod
    def build_xml(config):
        root = ET.fromstring(build_xml(config))
        root.set('model', 'sharpa_whole_hand_cube_v0')
        root.find(".//body[@name='right_hand_C_MC']").set('quat', '.707106781 0 .707106781 0')
        root.find(".//site[@name='goal']").set('size', '.065 .065 .0005')
        return ET.tostring(root, encoding='unicode')

    def privileged(self):
        truth = super().privileged()
        counts = np.zeros(6, np.int16)
        forces = np.zeros(6)
        for i, contact in enumerate(self.data.contact):
            if self.cube_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.cube_geom else contact.geom1
            if other not in self.hand_geoms:
                continue
            name = self.model.body(self.model.geom_bodyid[other]).name
            part = next((j for j, f in enumerate(FINGERS) if f in name), 5)
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            counts[part] += 1
            forces[part] += wrench[0]
        truth.update(hand_part_contact_counts=counts, hand_part_normal_force=forces)
        return truth


class WholeHandCollector(SharpaCollector):
    pose_path = 'assets/sharpa-whole-hand-grasp.json'


class SizeConditionedCollector(WholeHandCollector):
    """Scripted collector with interpolated, size-specific finger/wrist targets.

    Uses known simulation size. This is not a visual or learned policy. The
    original WholeHandCollector remains frozen for replaying the v0.2 examples.
    """
    profiles_path = 'assets/sharpa-size-grasps.json'

    def __init__(self, env, mode, seed):
        calibration = json.loads((ROOT/self.profiles_path).read_text())
        assert calibration['joint_names'] == env.joint_names[3:]
        profiles = calibration['profiles']
        sizes = np.array([r['edge_mm'] for r in profiles])/1000
        edge = env.config.cube_edge
        if not sizes[0] <= edge <= sizes[-1]:
            raise ValueError(f'Size-conditioned example controller requires {sizes[0]*1000:g}–{sizes[-1]*1000:g} mm cubes')
        self.profile = {key: np.array([np.interp(edge, sizes, column)
                                      for column in np.asarray([r[key] for r in profiles]).T])
                        for key in ('grasp_center_local', 'open', 'contact', 'closed')}
        env.grasp_center = self.profile['grasp_center_local'].copy()
        super().__init__(env, mode, seed)
        for key in ('open', 'contact', 'closed'):
            setattr(self, key, self.profile[key].copy())
        env.reset(fingers=self.open)


def summarize_whole_hand(env, truth, proprio):
    summary = summarize_sharpa(env, truth, proprio)
    force = np.stack([x['hand_part_normal_force'] for x in truth])
    airborne = np.array([x['cube_pose'][2] > env.config.cube_edge/2 + .05
                         and x['contact_counts'][0] == 0 for x in truth])
    active = force[:, :5] > .05
    run = longest = 0
    for flag in airborne & active.all(axis=1):
        run = run + 1 if flag else 0
        longest = max(longest, run)
    summary.update(
        finger_names=list(FINGERS),
        airborne_seconds=float(airborne.sum() * .1),
        finger_contact_fraction=active[airborne].mean(axis=0).tolist() if airborne.any() else [0.] * 5,
        finger_mean_normal_force_n=force[airborne, :5].mean(axis=0).tolist() if airborne.any() else [0.] * 5,
        all_five_contact_longest_seconds=max(0, longest-1) * .1,
        whole_hand_success=bool(summary['success'] and longest >= 6),
    )
    return summary
