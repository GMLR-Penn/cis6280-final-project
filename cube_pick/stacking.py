"""Place one free rigid cube on another; both can slide, rotate or topple."""
from dataclasses import dataclass, replace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .env import SceneConfig
from .whole_hand import SharpaWholeHandEnv, SizeConditionedCollector, summarize_whole_hand


@dataclass
class StackSceneConfig(SceneConfig):
    support_edge: float = .060
    support_mass: float = .16
    support_friction: float = .7


class SharpaStackEnv(SharpaWholeHandEnv):
    def __init__(self, seed=0, resolution=192, config=None, *, cube_edge=None, support_edge=.060):
        if config is None:
            config = StackSceneConfig.from_seed(seed)
            rng = np.random.default_rng(np.random.SeedSequence([seed, 82818]))
            config = replace(config, cube_edge=.050 if cube_edge is None else cube_edge, cube_yaw=0., support_edge=support_edge,
                             support_mass=float(rng.uniform(.12,.24)),
                             support_friction=float(rng.uniform(.5,.9)))
        if not np.isfinite(config.support_edge) or config.support_edge<=0:
            raise ValueError('support_edge must be a finite positive length in meters')
        super().__init__(seed=seed,resolution=resolution,config=config)
        self.support_id = self.model.body('support_cube').id
        self.support_geom = self.model.geom('support_cube').id
        self.support_qid = self.model.joint('support_free').qposadr[0]
        self.hand_geoms.discard(self.support_geom)

    @staticmethod
    def build_xml(config):
        root = ET.fromstring(SharpaWholeHandEnv.build_xml(config))
        root.set('model','sharpa_two_cube_stack_v0')
        body = ET.SubElement(root.find('worldbody'),'body',name='support_cube',
                             pos=f'{config.goal_x} {config.goal_y} {config.support_edge/2+.001}')
        ET.SubElement(body,'freejoint',name='support_free')
        ET.SubElement(body,'geom',name='support_cube',type='box',
                      size=' '.join([str(config.support_edge/2)]*3),mass=str(config.support_mass),
                      friction=f'{config.support_friction} .005 .0001',priority='1',condim='4',
                      solref='.008 1',solimp='.95 .99 .001',rgba='.85 .64 .22 1')
        return ET.tostring(root,encoding='unicode')

    def privileged(self):
        truth = super().privileged()
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.support_id,vel,0)
        counts = np.zeros(2,np.int16); forces = np.zeros(2)
        stack_count = 0; stack_force = 0.
        for i,c in enumerate(self.data.contact):
            if self.support_geom not in (c.geom1,c.geom2): continue
            other = c.geom2 if c.geom1==self.support_geom else c.geom1
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model,self.data,i,wrench)
            if other==self.cube_geom:
                stack_count += 1; stack_force += wrench[0]
            else:
                j = 0 if other==self.table_geom else 1 if other in self.hand_geoms else None
                if j is not None: counts[j] += 1; forces[j] += wrench[0]
        truth.update(support_pose=self.data.qpos[self.support_qid:self.support_qid+7].copy(),
                     support_velocity=vel,support_contact_counts=counts,
                     support_normal_force=forces,stack_contact_count=np.array([stack_count],np.int16),
                     stack_normal_force=np.array([stack_force]))
        return truth


class StackingCollector(SizeConditionedCollector):
    """Size-aware five-finger pickup, lower onto the support, release and wait."""
    def __init__(self, env, mode, seed):
        super().__init__(env,mode,seed)
        self.hover_height = max(self.hover_height,env.config.support_edge+env.config.cube_edge+.07)
        self.stages = [(name,30 if name=='settle' else length) for name,length in self.stages]
        self.horizon = sum(n for _,n in self.stages)
        if mode=='offset': self.goal[0] += .045

    def action(self, env, t):
        action,phase = super().action(env,t)
        if phase in ('lower','release') and self.mode!='probe':
            height = env.config.support_edge+env.config.cube_edge/2+.001
            target_z = env.wrist_for_cube([*self.goal,height])[2]
            action[2] = np.clip((target_z-env.targets[2])/env.action_scale[2],-1,1)
        return action,phase


def summarize_stack(env, truth, proprio):
    # For the lift/contact diagnostic, resting on the support is not airborne.
    lift_truth = [{**r,'contact_counts':r['contact_counts']+np.array([r['stack_contact_count'][0],0])}
                  for r in truth]
    summary = summarize_whole_hand(env,lift_truth,proprio)
    top = np.stack([r['cube_pose'] for r in truth]); base = np.stack([r['support_pose'] for r in truth])
    top_vel = np.stack([r['cube_velocity'] for r in truth]); base_vel = np.stack([r['support_velocity'] for r in truth])
    # z-axis vertical cosine for MuJoCo wxyz quaternions.
    top_upright = 1-2*(top[:,4]**2+top[:,5]**2)
    base_upright = 1-2*(base[:,4]**2+base[:,5]**2)
    xy_error = np.linalg.norm(top[:,:2]-base[:,:2],axis=1)
    height_error = np.abs(top[:,2]-base[:,2]-(env.config.cube_edge+env.config.support_edge)/2)
    contact = np.array([r['stack_contact_count'][0]>0 and r['support_contact_counts'][0]>0 for r in truth])
    released = np.array([r['contact_counts'][1]==0 and r['support_contact_counts'][1]==0 for r in truth])
    stable = ((xy_error<.015) & (height_error<.006) & (top_upright>np.cos(np.deg2rad(10)))
              & (base_upright>np.cos(np.deg2rad(10)))
              & (np.linalg.norm(top_vel[:,3:],axis=1)<.02) & (np.linalg.norm(base_vel[:,3:],axis=1)<.02)
              & (np.linalg.norm(top_vel[:,:3],axis=1)<.1) & (np.linalg.norm(base_vel[:,:3],axis=1)<.1)
              & (np.linalg.norm(base[:,:2]-[env.config.goal_x,env.config.goal_y],axis=1)<.02)
              & contact & released)
    success = bool(summary['lifted'] and len(stable)>=21 and stable[-21:].all())
    summary.update(success=success,stack_success=success,
                   whole_hand_success=bool(success and summary['all_five_contact_longest_seconds']>=.5),
                   stability_window_seconds=2.,final_stack_xy_error_m=float(xy_error[-1]),
                   final_stack_height_error_m=float(height_error[-1]),
                   final_top_tilt_degrees=float(np.rad2deg(np.arccos(np.clip(top_upright[-1],-1,1)))),
                   final_support_tilt_degrees=float(np.rad2deg(np.arccos(np.clip(base_upright[-1],-1,1)))),
                   final_support_displacement_m=float(np.linalg.norm(base[-1,:2]-[env.config.goal_x,env.config.goal_y])))
    return summary
