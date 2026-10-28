"""Official Sharpa Wave hand on a simulated XYZ wrist stage for cube data.

The hand's manufacturer joints, inertias, actuators and collision exclusions
are preserved. The XYZ stage, scene, object and scripted controller are local
additions. This does not model a complete robot arm or establish sim-to-real.
"""
import functools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .env import SceneConfig, DT, PHYSICS_DT, CAMERAS

ROOT = Path(__file__).resolve().parents[1]
HAND = ROOT / "assets/sharpa/wave_01/right_sharpa_wave"
GRASP_CENTER = np.array([.065, .040, .115])
ROT = np.array([[1.,0,0],[0,0,1],[0,-1,0]])
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


@functools.lru_cache(maxsize=1)
def assets():
    tree = ET.parse(HAND/"right_sharpa_wave.xml")
    return {m.attrib["file"]:(HAND/"meshes"/m.attrib["file"]).read_bytes()
            for m in tree.findall("./asset/mesh")}


def build_xml(c):
    root = ET.parse(HAND/"right_sharpa_wave.xml").getroot()
    root.set("model", "sharpa_cube_pick_v0")
    root.find("compiler").set("meshdir", "")
    ET.SubElement(root,"option",timestep=str(PHYSICS_DT),gravity="0 0 -9.81",integrator="implicitfast",iterations="80",cone="elliptic")
    visual=ET.SubElement(root,"visual")
    ET.SubElement(visual,"global",offwidth="640",offheight="480")
    ET.SubElement(visual,"headlight",ambient=".3 .3 .3",diffuse=".65 .65 .65")
    ET.SubElement(visual,"map",znear=".005")
    ET.SubElement(root.find("asset"),"texture",name="sky",type="skybox",builtin="flat",rgb1=".10 .13 .18",rgb2=".10 .13 .18",width="32",height="32")
    world=root.find("worldbody");hand=world.find("body");world.remove(hand)
    stage=ET.SubElement(world,"body",name="wrist_stage")
    ET.SubElement(stage,"inertial",mass=".1",pos="0 0 0",diaginertia=".001 .001 .001")
    limits=((-0.5,.5),(-.5,.5),(.03,.4))
    for name,axis,lim in zip(("wrist_x","wrist_y","wrist_z"),("1 0 0","0 1 0","0 0 1"),limits):
        ET.SubElement(stage,"joint",name=name,type="slide",axis=axis,range=" ".join(map(str,lim)),damping="10")
    hand.set("pos","0 0 0");hand.set("quat",".707106781 -.707106781 0 0")
    stage.append(hand)
    actuators=root.find("actuator")
    for i,name in enumerate(("wrist_x","wrist_y","wrist_z")):
        actuators.insert(i,ET.Element("position",name=name+"_ctrl",joint=name,kp="10000",kv="200",forcerange="-200 200",inheritrange="1"))
    ET.SubElement(world,"light",pos="0 -.3 1.1",dir="0 .2 -1",diffuse=" ".join([str(c.light_scale)]*3))
    ET.SubElement(world,"camera",name="front",pos=".48 -.65 .58",xyaxes=".804 .594 0 -.330 .447 .831",fovy="44")
    ET.SubElement(world,"camera",name="side",pos="-.50 -.55 .52",xyaxes=".740 -.673 0 .363 .399 .842",fovy="44")
    ET.SubElement(world,"geom",name="table",type="box",pos="0 0 -.025",size=".40 .30 .025",rgba=" ".join(map(str,c.table_rgb))+" 1")
    ET.SubElement(world,"geom",name="ground",type="plane",pos="0 0 -.2",size="1 1 .1",rgba=".12 .15 .20 1")
    ET.SubElement(world,"site",name="goal",type="box",pos=f"{c.goal_x} {c.goal_y} .0005",size=".047 .047 .0005",rgba=".2 .8 .5 .45")
    cube=ET.SubElement(world,"body",name="cube",pos=f"{c.cube_x} {c.cube_y} {c.cube_edge/2+.001}",quat=f"{np.cos(c.cube_yaw/2)} 0 0 {np.sin(c.cube_yaw/2)}")
    ET.SubElement(cube,"freejoint",name="cube_free")
    ET.SubElement(cube,"geom",name="cube",type="box",size=" ".join([str(c.cube_edge/2)]*3),mass=str(c.cube_mass),friction=f"{c.cube_friction} .005 .0001",priority="1",solref=".008 1",solimp=".95 .99 .001",condim="4",rgba=" ".join(map(str,c.cube_rgb))+" 1")
    return ET.tostring(root,encoding="unicode")


class SharpaCubeEnv:
    build_xml = staticmethod(build_xml)
    default_cube_edge = .035
    rotation = ROT
    grasp_center = GRASP_CENTER

    def __init__(self, seed=0, resolution=192, config=None):
        self.config=config or SceneConfig.from_seed(seed)
        if config is None:
            self.config.cube_edge=self.default_cube_edge
            self.config.cube_yaw=0.
        self.xml=self.build_xml(self.config)
        self.model=mujoco.MjModel.from_xml_string(self.xml,assets())
        self.data=mujoco.MjData(self.model)
        self.resolution=resolution;self.renderer=None
        self.joint_names=[self.model.joint(j).name for j in self.model.actuator_trnid[:,0]]
        self.qids=self.model.jnt_qposadr[self.model.actuator_trnid[:,0]]
        self.vids=self.model.jnt_dofadr[self.model.actuator_trnid[:,0]]
        self.low,self.high=self.model.actuator_ctrlrange.T.copy()
        self.action_scale=np.r_[np.full(3,.015),np.full(22,.15)]
        self.cube_qid=self.model.joint("cube_free").qposadr[0]
        self.cube_id=self.model.body("cube").id;self.cube_geom=self.model.geom("cube").id
        self.table_geom=self.model.geom("table").id
        self.hand_geoms={i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] not in (0,self.cube_id)}
        self.visual_options=mujoco.MjvOption();self.visual_options.geomgroup[3]=0
        pose=json.loads((ROOT/"assets/grasp.json").read_text())["joints"]
        self.reference=np.array([pose[n] for n in self.joint_names[3:]])
        self.reset()

    def wrist_for_cube(self, xyz):
        return np.asarray(xyz)-self.rotation@self.grasp_center

    def reset(self, fingers=None):
        mujoco.mj_resetData(self.model,self.data)
        c=self.config
        wrist=self.wrist_for_cube([c.cube_x,c.cube_y,.17])
        self.targets=np.r_[wrist,self.reference if fingers is None else fingers]
        self.data.qpos[self.qids]=self.targets
        self.data.ctrl[:]=self.targets
        mujoco.mj_forward(self.model,self.data)
        mujoco.mj_step(self.model,self.data,nstep=150)
        mujoco.mj_forward(self.model,self.data)
        self.start_time=float(self.data.time)
        return self.proprio()

    def proprio(self):
        return np.r_[self.data.qpos[self.qids],self.data.qvel[self.vids],self.targets].astype(np.float32)

    def step(self, action):
        a=np.asarray(action,dtype=np.float64)
        if a.shape!=(25,) or not np.isfinite(a).all() or np.any(np.abs(a)>1+1e-6):
            raise ValueError("Expected 25 finite normalized actions")
        self.targets=np.clip(self.targets+np.clip(a,-1,1)*self.action_scale,self.low,self.high)
        self.data.ctrl[:]=self.targets
        mujoco.mj_step(self.model,self.data,nstep=50)
        mujoco.mj_forward(self.model,self.data)
        if not np.isfinite(self.data.qpos).all() or np.any(self.data.warning.number):
            raise RuntimeError("Invalid physics state or warning")
        return self.proprio()

    def render(self):
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=self.resolution,width=self.resolution)
        frames=[]
        for name in CAMERAS:
            self.renderer.update_scene(self.data,camera=name,scene_option=self.visual_options)
            frames.append(self.renderer.render().copy())
        return np.stack(frames)

    def privileged(self):
        vel=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.cube_id,vel,0)
        counts=np.zeros(2,np.int16);force=np.zeros(2)
        for i,contact in enumerate(self.data.contact):
            if self.cube_geom not in (contact.geom1,contact.geom2):continue
            other=contact.geom2 if contact.geom1==self.cube_geom else contact.geom1
            j=0 if other==self.table_geom else 1 if other in self.hand_geoms else None
            if j is not None:
                counts[j]+=1;w=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,w);force[j]+=w[0]
        return {"cube_pose":self.data.qpos[self.cube_qid:self.cube_qid+7].copy(),"cube_velocity":vel,
                "contact_counts":counts,"contact_normal_force":force}

    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None
