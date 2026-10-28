"""Privileged scripted DATA COLLECTOR, not a learned visual agent."""
import numpy as np
from .env import ACTION_SCALE

MODES = ("nominal", "jitter", "miss", "weak_grip", "early_release", "probe")
STAGES = (("approach", 12), ("descend", 12), ("close", 12), ("lift", 12),
          ("transport", 24), ("lower", 12), ("release", 12), ("retreat", 12), ("settle", 10))
HORIZON = sum(length for _, length in STAGES)


class ScriptedCollector:
    def __init__(self, env, mode, seed):
        self.mode = mode
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 31415]))
        self.stages = [(name, length if name == "settle" else int(length + self.rng.integers(-3, 4)))
                       for name, length in STAGES]
        self.horizon = sum(length for _, length in self.stages)
        self.hover_height = float(self.rng.uniform(.15, .22))
        c = env.config
        self.start = np.array([c.cube_x, c.cube_y])
        self.goal = np.array([c.goal_x, c.goal_y])
        self.grasp_xy = self.start + self.rng.uniform(-.003, .003, 2)
        if mode == "miss":
            self.grasp_xy[1] += self.rng.choice([-1, 1]) * .075
        self.closed = .010 if mode != "weak_grip" else c.cube_edge + self.rng.uniform(-.001, .002)

    def action(self, env, t):
        cursor = 0
        for name, length in self.stages:
            if t < cursor + length:
                phase = name
                phase_t = t-cursor
                break
            cursor += length
        else:
            raise IndexError(t)
        c = env.config
        target = np.array([*self.grasp_xy, self.hover_height, .096])
        if phase in ("descend", "close"):
            target[2] = .033
        if phase in ("close", "lift", "transport", "lower"):
            target[3] = self.closed
        if phase in ("transport", "lower", "release", "retreat", "settle"):
            target[:2] = self.goal
        if phase in ("lower", "release"):
            target[2] = .033
        if self.mode == "early_release" and phase == "transport" and phase_t >= 4:
            target[3] = .096
        if self.mode == "probe" and phase in ("lift", "transport", "lower", "release"):
            target[:2] = self.grasp_xy
            target[2] = .033
            target[3] = c.cube_edge + .014*np.sin(t*.6)
        if self.mode == "jitter" and phase in ("descend", "close", "lift", "transport"):
            target[:3] += self.rng.normal(0, .002, 3)
        action = np.clip((target-env.targets)/ACTION_SCALE, -1, 1).astype(np.float32)
        return action, phase


def summarize(env, truth, proprio):
    poses = np.stack([x["cube_pose"] for x in truth])
    velocities = np.stack([x["cube_velocity"] for x in truth])
    contacts = np.stack([x["contact_counts"] for x in truth])
    c = env.config
    lifted = bool(np.any((poses[:, 2] > c.cube_edge/2 + .05) & (contacts[:, 0] == 0)))
    last = slice(-11, None)  # Ten intervals / one full second, including endpoints.
    in_goal = np.linalg.norm(poses[last, :2]-[c.goal_x, c.goal_y], axis=1) < .025
    at_height = np.abs(poses[last, 2]-c.cube_edge/2) < .008
    stable = np.linalg.norm(velocities[last, 3:], axis=1) < .03
    released = (contacts[last, 1:].sum(axis=1) == 0) & (proprio[last, 3:5].sum(axis=1) + .006 > c.cube_edge + .01)
    return {"success": bool(lifted and np.all(in_goal & at_height & stable & released)),
            "lifted": lifted, "final_goal_error_m": float(np.linalg.norm(poses[-1, :2]-[c.goal_x, c.goal_y])),
            "max_cube_height_m": float(poses[:, 2].max()), "final_cube_height_m": float(poses[-1, 2]),
            "final_speed_m_s": float(np.linalg.norm(velocities[-1, 3:]))}
