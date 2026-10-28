"""Procedural Cartesian two-finger gripper with a freely moving rigid cube.

This is a simplified data-generation robot, not a model of a particular real
gripper/arm. Only actuator targets are commanded after reset. No object weld,
mocap, pose playback or external object force is used.
"""
from dataclasses import asdict, dataclass

import mujoco
import numpy as np


DT = 0.1
PHYSICS_DT = 0.002
ACTION_SCALE = np.array([0.018, 0.018, 0.018, 0.02])
TARGET_LOW = np.array([-0.32, -0.22, 0.032, 0.006])
TARGET_HIGH = np.array([0.32, 0.22, 0.30, 0.096])
CAMERAS = ("front", "side")


@dataclass
class SceneConfig:
    seed: int
    cube_edge: float
    cube_mass: float
    cube_friction: float
    cube_x: float
    cube_y: float
    cube_yaw: float
    goal_x: float
    goal_y: float
    cube_rgb: list
    table_rgb: list
    light_scale: float

    @classmethod
    def from_seed(cls, seed):
        # Appearance is sampled from a separate stream, without a material label.
        rng = np.random.default_rng(seed)
        visual = np.random.default_rng(np.random.SeedSequence([seed, 7919]))
        return cls(int(seed), float(rng.uniform(.036, .046)),
                   float(rng.uniform(.04, .12)), float(rng.uniform(.45, 1.0)),
                   float(rng.uniform(-.17, -.10)), float(rng.uniform(-.08, .08)),
                   float(rng.uniform(-.15, .15)), float(rng.uniform(.10, .17)),
                   float(rng.uniform(-.08, .08)), visual.uniform(.2, .85, 3).tolist(),
                   visual.uniform(.18, .32, 3).tolist(), float(visual.uniform(.75, 1.15)))


def scene_xml(c):
    rgb = " ".join(map(str, c.cube_rgb))
    table = " ".join(map(str, c.table_rgb))
    q = f"{np.cos(c.cube_yaw/2)} 0 0 {np.sin(c.cube_yaw/2)}"
    half = c.cube_edge / 2
    return f'''<mujoco model="cube_pick_rigid_v0">
 <compiler angle="radian"/>
 <option timestep="{PHYSICS_DT}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" iterations="80"/>
 <visual><global offwidth="640" offheight="480"/><headlight ambient=".3 .3 .3" diffuse=".6 .6 .6"/><map znear=".005"/></visual>
 <default><joint damping="2"/><geom friction=".8 .005 .0001" solref=".008 1" solimp=".95 .99 .001" condim="4"/></default>
 <asset><texture name="sky" type="skybox" builtin="flat" rgb1=".10 .13 .18" rgb2=".10 .13 .18" width="32" height="32"/></asset>
 <worldbody>
  <light pos="0 -.3 1.1" dir="0 .2 -1" diffuse="{c.light_scale} {c.light_scale} {c.light_scale}"/>
  <camera name="front" pos=".48 -.65 .58" xyaxes=".804 .594 0 -.330 .447 .831" fovy="44"/>
  <camera name="side" pos="-.50 -.55 .52" xyaxes=".740 -.673 0 .363 .399 .842" fovy="44"/>
  <geom name="table" type="box" pos="0 0 -.025" size=".40 .30 .025" rgba="{table} 1"/>
  <geom name="ground" type="plane" pos="0 0 -.20" size="1 1 .1" rgba=".12 .15 .20 1"/>
  <site name="goal" type="box" pos="{c.goal_x} {c.goal_y} .0005" size=".047 .047 .0005" rgba=".2 .8 .5 .45"/>
  <body name="gripper" pos="0 0 0" gravcomp="1">
   <joint name="x" type="slide" axis="1 0 0" range="-.32 .32"/>
   <joint name="y" type="slide" axis="0 1 0" range="-.22 .22"/>
   <joint name="z" type="slide" axis="0 0 1" range=".032 .30"/>
   <geom name="palm" type="box" pos="0 0 .075" size=".072 .03 .018" mass="1" rgba=".65 .70 .77 1"/>
   <geom name="stem" type="cylinder" pos="0 0 .13" size=".018 .038" mass=".2" rgba=".42 .47 .54 1"/>
   <body name="left" pos="-.012 0 0" gravcomp="1">
    <joint name="left_open" type="slide" axis="-1 0 0" range="0 .045" damping="1"/>
    <geom name="left_pad" type="box" size=".009 .025 .027" mass=".04" rgba=".12 .15 .19 1"/>
    <geom name="left_link" type="box" pos="0 0 .042" size=".009 .016 .015" mass=".03" rgba=".6 .65 .72 1"/>
   </body>
   <body name="right" pos=".012 0 0" gravcomp="1">
    <joint name="right_open" type="slide" axis="1 0 0" range="0 .045" damping="1"/>
    <geom name="right_pad" type="box" size=".009 .025 .027" mass=".04" rgba=".12 .15 .19 1"/>
    <geom name="right_link" type="box" pos="0 0 .042" size=".009 .016 .015" mass=".03" rgba=".6 .65 .72 1"/>
   </body>
  </body>
  <body name="cube" pos="{c.cube_x} {c.cube_y} {half+.002}" quat="{q}">
   <freejoint name="cube_free"/>
   <geom name="cube" type="box" size="{half} {half} {half}" mass="{c.cube_mass}" friction="{c.cube_friction} .005 .0001" priority="1" rgba="{rgb} 1"/>
  </body>
 </worldbody>
 <actuator>
  <position name="move_x" joint="x" kp="2500" kv="100" forcerange="-150 150"/>
  <position name="move_y" joint="y" kp="2500" kv="100" forcerange="-150 150"/>
  <position name="move_z" joint="z" kp="2500" kv="100" forcerange="-150 150"/>
  <position name="open_left" joint="left_open" kp="400" kv="5" forcerange="-8 8"/>
  <position name="open_right" joint="right_open" kp="400" kv="5" forcerange="-8 8"/>
 </actuator>
</mujoco>'''


class CubePickEnv:
    def __init__(self, seed=0, resolution=192, config=None):
        self.config = config or SceneConfig.from_seed(seed)
        self.xml = scene_xml(self.config)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.resolution = resolution
        self.renderer = None
        self.robot_qids = np.array([self.model.joint(n).qposadr[0] for n in ("x", "y", "z", "left_open", "right_open")])
        self.robot_vids = np.array([self.model.joint(n).dofadr[0] for n in ("x", "y", "z", "left_open", "right_open")])
        self.cube_qid = self.model.joint("cube_free").qposadr[0]
        self.cube_id = self.model.body("cube").id
        self.cube_geom = self.model.geom("cube").id
        self.contact_geoms = [self.model.geom(n).id for n in ("table", "left_pad", "right_pad")]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.targets = np.array([-.20, 0, .23, .096])
        self.data.qpos[self.robot_qids] = [*self.targets[:3], .045, .045]
        self._set_ctrl()
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=150)
        mujoco.mj_forward(self.model, self.data)
        self.start_time = float(self.data.time)
        return self.proprio()

    def _set_ctrl(self):
        self.data.ctrl[:] = [*self.targets[:3], *((self.targets[3]-.006)/2,)*2]

    def proprio(self):
        # Actual joint positions/velocities + the actuator target accumulator.
        return np.concatenate([self.data.qpos[self.robot_qids], self.data.qvel[self.robot_vids], self.targets]).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (4,) or not np.isfinite(action).all() or np.any(np.abs(action)>1+1e-6):
            raise ValueError("Expected four finite normalized actions in [-1, 1]")
        self.targets = np.clip(self.targets + np.clip(action, -1, 1)*ACTION_SCALE, TARGET_LOW, TARGET_HIGH)
        self._set_ctrl()
        mujoco.mj_step(self.model, self.data, nstep=50)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or np.any(self.data.warning.number):
            raise RuntimeError("Invalid physics state or MuJoCo warning")
        return self.proprio()

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.resolution, width=self.resolution)
        frames = []
        for camera in CAMERAS:
            self.renderer.update_scene(self.data, camera=camera)
            frames.append(self.renderer.render().copy())
        return np.stack(frames)

    def privileged(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.cube_id, velocity, 0)
        counts = np.zeros(3, np.int16)
        force = np.zeros(3)
        for i, contact in enumerate(self.data.contact):
            if self.cube_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.cube_geom else contact.geom1
            if other in self.contact_geoms:
                j = self.contact_geoms.index(other)
                counts[j] += 1
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, wrench)
                force[j] += wrench[0]
        return {"cube_pose": self.data.qpos[self.cube_qid:self.cube_qid+7].copy(),
                "cube_velocity": velocity, "contact_counts": counts, "contact_normal_force": force}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
