"""Scripted collection with official Sharpa joints; no learned control claim."""
import json
import numpy as np

from .sharpa import ROOT
from .policy import STAGES


class SharpaCollector:
    pose_path = 'assets/sharpa-cube-grasp.json'

    def __init__(self, env, mode, seed):
        self.mode=mode
        self.rng=np.random.default_rng(np.random.SeedSequence([seed,31415]))
        self.stages=[(name,length if name=='settle' else int(length+self.rng.integers(-2,3))) for name,length in STAGES]
        self.horizon=sum(n for _,n in self.stages)
        self.hover_height=float(self.rng.uniform(.15,.19))
        poses=json.loads((ROOT/self.pose_path).read_text())
        assert poses['joint_names']==env.joint_names[3:]
        self.open=np.array(poses['open']);self.closed=np.array(poses['closed']);self.contact=np.array(poses['contact'])
        c=env.config
        self.start=np.array([c.cube_x,c.cube_y]);self.goal=np.array([c.goal_x,c.goal_y])
        self.grasp_xy=self.start.copy()
        if mode=='miss':self.grasp_xy[0]+=.10
        env.reset(fingers=self.open)

    def action(self,env,t):
        cursor=0
        for name,length in self.stages:
            if t<cursor+length:phase=name;phase_t=t-cursor;break
            cursor+=length
        else:raise IndexError(t)
        c=env.config
        xyz=np.r_[self.grasp_xy,self.hover_height]
        fingers=self.open.copy()
        if phase in ('descend','close'):xyz[2]=c.cube_edge/2+.001
        if phase in ('close','lift','transport','lower'):
            fingers=self.contact.copy() if self.mode=='weak_grip' else self.closed.copy()
        if phase in ('transport','lower','release','retreat','settle'):xyz[:2]=self.goal
        if phase in ('lower','release'):xyz[2]=c.cube_edge/2+.001
        if self.mode=='early_release' and phase=='transport' and phase_t>=6:fingers=self.open.copy()
        if self.mode=='probe' and phase in ('lift','transport','lower','release'):
            xyz=np.r_[self.grasp_xy,c.cube_edge/2+.001]
            alpha=(np.sin(t*.5)+1)/2
            fingers=(1-alpha)*self.open+alpha*self.closed
        if self.mode=='jitter' and phase in ('descend','close','lift','transport'):
            xyz+=self.rng.normal(0,.001,3)
        desired=np.r_[env.wrist_for_cube(xyz),fingers]
        return np.clip((desired-env.targets)/env.action_scale,-1,1).astype(np.float32),phase


def summarize_sharpa(env,truth,proprio):
    poses=np.stack([x['cube_pose'] for x in truth]);v=np.stack([x['cube_velocity'] for x in truth])
    contacts=np.stack([x['contact_counts'] for x in truth]);c=env.config
    lifted=bool(np.any((poses[:,2]>c.cube_edge/2+.05)&(contacts[:,0]==0)))
    last=slice(-11,None)
    in_goal=np.linalg.norm(poses[last,:2]-[c.goal_x,c.goal_y],axis=1)<.025
    at_height=np.abs(poses[last,2]-c.cube_edge/2)<.008
    stable=np.linalg.norm(v[last,3:],axis=1)<.03
    released=contacts[last,1]==0
    return {'success':bool(lifted and np.all(in_goal&at_height&stable&released)),
            'lifted':lifted,'final_goal_error_m':float(np.linalg.norm(poses[-1,:2]-[c.goal_x,c.goal_y])),
            'max_cube_height_m':float(poses[:,2].max()),'final_cube_height_m':float(poses[-1,2]),
            'final_speed_m_s':float(np.linalg.norm(v[-1,3:]))}
