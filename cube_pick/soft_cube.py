"""Experimental unpinned elastic cube and the official Sharpa five-finger hand.

MuJoCo 3.13 flex elasticity with full vertex DOFs, not a rigid collision proxy.
Material parameters are synthetic; they have not been fit to a physical object.
"""
from dataclasses import dataclass, replace
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .env import SceneConfig, DT
from .sharpa import ROOT, FINGERS, assets
from .whole_hand import SharpaWholeHandEnv, SizeConditionedCollector


@dataclass
class SoftSceneConfig(SceneConfig):
    young: float = 100000.
    poisson: float = .3
    elastic_damping: float = .003
    grid_count: int = 4
    physics_dt: float = .0005
    contact_radius: float = .0005


class SharpaSoftCubeEnv(SharpaWholeHandEnv):
    def __init__(self, seed=0, resolution=192, config=None, *, cube_edge=.055, young=100000.):
        if tuple(map(int,mujoco.__version__.split('.')[:2]))<(3,13):
            raise RuntimeError('Elastic cube requires MuJoCo >= 3.13 with the discrete integrator')
        if config is None:
            config = replace(SoftSceneConfig.from_seed(seed), cube_edge=cube_edge, cube_yaw=0., young=young)
        c = self.config = config
        if not np.isfinite([c.cube_edge,c.young,c.poisson,c.elastic_damping,c.physics_dt]).all():
            raise ValueError('Soft-body parameters must be finite')
        if c.cube_edge<=0 or c.young<=0 or not 0<=c.poisson<.5 or c.elastic_damping<0:
            raise ValueError('Invalid elastic material or cube dimensions')
        if c.grid_count<3 or not isinstance(c.grid_count,int) or not 0<c.physics_dt<=.005 or not np.isclose(DT/c.physics_dt,round(DT/c.physics_dt)):
            raise ValueError('Use grid_count >= 3 and a timestep <= 0.005 seconds that divides 0.1 seconds')
        if not 0<=c.contact_radius<c.cube_edge/2 or c.cube_yaw!=0:
            raise ValueError('This prototype requires zero initial yaw and a radius smaller than half the edge')
        self.xml = self.build_xml(c)
        self.model = mujoco.MjModel.from_xml_string(self.xml,assets())
        self.data = mujoco.MjData(self.model)
        self.resolution = resolution; self.renderer = None
        self.joint_names = [self.model.joint(j).name for j in self.model.actuator_trnid[:,0]]
        self.qids = self.model.jnt_qposadr[self.model.actuator_trnid[:,0]]
        self.vids = self.model.jnt_dofadr[self.model.actuator_trnid[:,0]]
        self.low,self.high = self.model.actuator_ctrlrange.T.copy()
        self.action_scale = np.r_[np.full(3,.015),np.full(22,.15)]
        self.table_geom = self.model.geom('table').id
        self.hand_geoms = {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g]!=0}
        self.flex_id = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_FLEX,'soft_cube')
        assert self.model.nflex==1 and self.model.neq==self.model.nmocap==0
        bodies = self.model.flex_vertbodyid
        assert np.all(bodies>0)  # No vertices pinned to the world.
        self.vertex_mass = self.model.body_mass[bodies].copy()
        np.testing.assert_allclose(self.vertex_mass.sum(),c.cube_mass,rtol=1e-10)
        self.vertex_dofs = self.model.body_dofadr[bodies,None]+np.arange(3)
        assert np.all(self.model.body_dofnum[bodies]==3)
        self.tetrahedra = self.model.flex_elem.reshape(-1,4).copy()
        mujoco.mj_forward(self.model,self.data)
        self.rest_vertices = self.data.flexvert_xpos.copy()
        rest_edges = self._edges(self.rest_vertices)
        self.rest_determinants = np.linalg.det(rest_edges)
        self.rest_inverse = np.linalg.inv(rest_edges)
        self.visual_options = mujoco.MjvOption(); self.visual_options.geomgroup[3] = 0
        # Display the same flex surface that participates in collisions.
        self.visual_options.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
        self.visual_options.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = False
        self.visual_options.flags[mujoco.mjtVisFlag.mjVIS_FLEXVERT] = False
        pose = json.loads((ROOT/'assets/grasp.json').read_text())['joints']
        self.reference = np.array([pose[n] for n in self.joint_names[3:]])
        self.reset()

    @staticmethod
    def build_xml(c):
        root = ET.fromstring(SharpaWholeHandEnv.build_xml(c))
        root.set('model','sharpa_elastic_cube_v0')
        option = root.find('option')
        option.set('integrator','discrete'); option.set('timestep',str(c.physics_dt))
        world = root.find('worldbody'); world.remove(world.find("body[@name='cube']"))
        spacing = (c.cube_edge-2*c.contact_radius)/(c.grid_count-1)
        flex = ET.SubElement(world,'flexcomp',name='soft_cube',type='grid',dim='3',dof='full',
                             count=' '.join([str(c.grid_count)]*3),spacing=' '.join([str(spacing)]*3),
                             pos=f'{c.cube_x} {c.cube_y} {c.cube_edge/2+.001}',
                             mass=str(c.cube_mass),radius=str(c.contact_radius),
                             rgba=' '.join(map(str,c.cube_rgb))+' 1')
        ET.SubElement(flex,'contact',condim='4',priority='1',selfcollide='none',internal='false',
                      friction=f'{c.cube_friction} .005 .0001',solref='.008 1',solimp='.95 .99 .001')
        ET.SubElement(flex,'elasticity',young=str(c.young),poisson=str(c.poisson),damping=str(c.elastic_damping))
        return ET.tostring(root,encoding='unicode')

    def _edges(self, vertices):
        points = vertices[self.tetrahedra]
        return np.transpose(points[:,1:]-points[:,:1],(0,2,1))

    def _advance(self, nstep):
        # Check element orientation every <= 5 ms, including between RGB frames.
        chunk = max(1,int(.005/self.config.physics_dt+1e-9))
        for offset in range(0,nstep,chunk):
            mujoco.mj_step(self.model,self.data,nstep=min(chunk,nstep-offset))
            mujoco.mj_forward(self.model,self.data)
            ratio = np.linalg.det(self._edges(self.data.flexvert_xpos))/self.rest_determinants
            self.min_volume_ratio_run = min(self.min_volume_ratio_run,float(ratio.min()))
            if not np.isfinite(self.data.qpos).all() or np.any(self.data.warning.number) or not np.isfinite(ratio).all() or ratio.min()<=0:
                raise RuntimeError('Invalid soft-body state: warning, nonfinite state or inverted tetrahedron')

    def reset(self, fingers=None):
        mujoco.mj_resetData(self.model,self.data)
        c = self.config
        self.targets = np.r_[self.wrist_for_cube([c.cube_x,c.cube_y,.17]),
                             self.reference if fingers is None else fingers]
        self.data.qpos[self.qids] = self.targets; self.data.ctrl[:] = self.targets
        self.min_volume_ratio_run = 1.
        self._advance(round(.3/c.physics_dt))
        self.start_time = float(self.data.time)
        return self.proprio()

    def step(self, action):
        a = np.asarray(action,dtype=np.float64)
        if a.shape!=(25,) or not np.isfinite(a).all() or np.any(np.abs(a)>1+1e-6):
            raise ValueError('Expected 25 finite normalized actions')
        self.targets = np.clip(self.targets+np.clip(a,-1,1)*self.action_scale,self.low,self.high)
        self.data.ctrl[:] = self.targets
        self._advance(round(DT/self.config.physics_dt))
        return self.proprio()

    def privileged(self):
        counts = np.zeros(2,np.int16); forces = np.zeros(2)
        parts = np.zeros(6,np.int16); part_forces = np.zeros(6)
        for i,c in enumerate(self.data.contact):
            if self.flex_id not in c.flex: continue
            other = c.geom[1] if c.flex[0]==self.flex_id else c.geom[0]
            j = 0 if other==self.table_geom else 1 if other in self.hand_geoms else None
            if j is None: continue
            wrench = np.zeros(6); mujoco.mj_contactForce(self.model,self.data,i,wrench)
            counts[j] += 1; forces[j] += wrench[0]
            if j==1:
                name = self.model.body(self.model.geom_bodyid[other]).name
                part = next((k for k,f in enumerate(FINGERS) if f in name),5)
                parts[part] += 1; part_forces[part] += wrench[0]
        vertices = self.data.flexvert_xpos.copy()
        edges = self._edges(vertices)
        singular_values = np.linalg.svd(edges@self.rest_inverse,compute_uv=False)
        return {'object_vertices':vertices,'object_vertex_velocity':self.data.qvel[self.vertex_dofs].copy(),
                'object_center':np.average(vertices,axis=0,weights=self.vertex_mass),'contact_counts':counts,'contact_normal_force':forces,
                'hand_part_contact_counts':parts,'hand_part_normal_force':part_forces,
                'principal_stretches':singular_values,
                'tetra_volume_ratio':np.linalg.det(edges)/self.rest_determinants,
                'min_volume_ratio_run':np.array([self.min_volume_ratio_run])}


class SoftCubeCollector(SizeConditionedCollector):
    def __init__(self, env, mode, seed, grip_scale=.4):
        super().__init__(env,mode,seed)
        if not 0<=grip_scale<=1: raise ValueError('grip_scale must be in [0,1]')
        self.grip_scale = grip_scale
        self.closed = self.contact+grip_scale*(self.closed-self.contact)
        self.stages = [(name,30 if name=='settle' else length) for name,length in self.stages]
        self.horizon = sum(n for _,n in self.stages)


def summarize_soft(env, truth, proprio):
    centers = np.stack([r['object_center'] for r in truth])
    velocities = np.stack([r['object_vertex_velocity'] for r in truth])
    contacts = np.stack([r['contact_counts'] for r in truth])
    forces = np.stack([r['hand_part_normal_force'] for r in truth])
    strain = np.stack([np.abs(r['principal_stretches']-1) for r in truth])
    airborne = (centers[:,2]>env.config.cube_edge/2+.05)&(contacts[:,0]==0)
    stable = ((np.linalg.norm(centers[:,:2]-[env.config.goal_x,env.config.goal_y],axis=1)<.025)
              & (contacts[:,0]>0)&(contacts[:,1]==0)
              & (np.linalg.norm(velocities,axis=2).max(axis=1)<.03))
    longest = run = 0
    for flag in airborne & (forces[:,:5]>.05).all(axis=1):
        run = run+1 if flag else 0; longest = max(longest,run)
    success = bool(airborne.any() and stable[-11:].all())
    return {'success':success,'lifted':bool(airborne.any()),'whole_hand_success':bool(success and longest>=6),
            'all_five_contact_longest_seconds':max(0,longest-1)*DT,
            'final_goal_error_m':float(np.linalg.norm(centers[-1,:2]-[env.config.goal_x,env.config.goal_y])),
            'max_center_height_m':float(centers[:,2].max()),
            'max_principal_strain':float(strain.max()),'final_max_principal_strain':float(strain[-1].max()),
            'min_tetra_volume_ratio':float(env.min_volume_ratio_run),
            'inverted_elements':0,'physics_warnings':int(env.data.warning.number.sum())}
