"""Same-footprint boxes: two heights, three fixed synthetic materials."""
from dataclasses import asdict

import numpy as np

from .env import SceneConfig
from .mixed_stacking import MATERIALS, MixedSceneConfig, MixedStackEnv, ObjectSpec
from .mixed_feedback import FeedbackStackController


def box_scene(seed=0, time_limit=180.):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 152026]))
    heights = rng.permutation([.055, .055, .085, .085])
    materials = rng.permutation(list(MATERIALS) + [str(rng.choice(list(MATERIALS)))])
    positions = [(-.17, -.15), (-.17, .15), (-.34, -.15), (-.34, .15)]
    objects = [ObjectSpec('cube' if h == .055 else 'cuboid', (.055, .055, float(h)),
                          str(material), tuple(np.asarray(xy)+rng.uniform(-.006, .006, 2)))
               for h, material, xy in zip(heights, materials, positions)]
    base = asdict(SceneConfig.from_seed(seed))
    base.update(cube_edge=.055, cube_yaw=0., cube_x=objects[0].xy[0], cube_y=objects[0].xy[1],
                goal_x=.16, goal_y=0.)
    return MixedSceneConfig(**base, objects=objects, time_limit=time_limit, task_version=2)


class BoxStackEnv(MixedStackEnv):
    def __init__(self, seed=0, resolution=128, time_limit=180., config=None):
        super().__init__(seed=seed, resolution=resolution,
                         config=config or box_scene(seed, time_limit))


class BoxStackController(FeedbackStackController):
    """Reuse actuator-only pose/contact feedback with short boxes first."""
    def __init__(self, env):
        super().__init__(env)
        self.order = sorted(range(len(env.config.objects)), key=lambda i:
                            (env.config.objects[i].size[2], -env.config.objects[i].mass))
        self.profiles.order = self.order.copy()

    def select(self, env):
        super().select(env)
        if not self.done:
            # A small flexion margin keeps heavy, low-friction boxes gripped.
            for j, name in enumerate(env.joint_names[3:]):
                if name.endswith(('PIP', 'DIP', 'IP', 'MCP_FE')):
                    self.closed[j] += .2 if self.spec.material == 'purple' else .1
            self.closed = np.clip(self.closed, env.low[3:], env.high[3:])


def full_stack_success(env):
    """Dataset admission rule, stricter than the task's continuous height score.

    All four upright layers must be retained. This rejects a connected pile of
    fallen boxes that would still earn a legitimate, smaller task height score.
    """
    result = env.outcome()
    nominal_height = sum(s.size[2] for s in env.config.objects)
    return (result['scored_object_count'] == len(env.config.objects)
            and result['final_stable_height_m'] >= .95*nominal_height)
