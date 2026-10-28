"""Same-footprint scene contract and a heavy-material physical regression."""
from dataclasses import asdict
import unittest

import mujoco
import numpy as np

from cube_pick.box_stacking import BoxStackEnv, BoxStackController, box_scene, full_stack_success
from cube_pick.mixed_stacking import MATERIALS


class BoxStacking(unittest.TestCase):
    def test_same_footprint_two_heights_and_materials(self):
        for seed in range(12):
            a, b = box_scene(seed), box_scene(seed)
            self.assertEqual(asdict(a), asdict(b))
            self.assertEqual(sorted(o.size[2] for o in a.objects), [.055, .055, .085, .085])
            self.assertEqual(set(o.material for o in a.objects), set(MATERIALS))
            for o in a.objects:
                self.assertEqual(o.size[:2], (.055, .055))
                self.assertEqual(o.yaw, 0.)
                self.assertAlmostEqual(o.mass, np.prod(o.size)*MATERIALS[o.material]['density'])

    def test_heavy_tall_box_full_physics_and_no_state_writes(self):
        env = BoxStackEnv(seed=2)
        try:
            policy = BoxStackController(env)
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            before = np.empty(mujoco.mj_stateSize(env.model, spec))
            after = before.copy()
            while env.status == 'running':
                mujoco.mj_getState(env.model, env.data, before, spec)
                targets = env.targets.copy()
                action, _ = policy.action(env, env.elapsed_steps)
                mujoco.mj_getState(env.model, env.data, after, spec)
                np.testing.assert_array_equal(after, before)
                np.testing.assert_array_equal(env.targets, targets)
                if action is None:
                    env.stop()
                else:
                    self.assertEqual(action.shape, (25,))
                    self.assertTrue(np.isfinite(action).all())
                    self.assertLessEqual(np.abs(action).max(), 1.)
                    env.step(action)
            while env.status == 'finalizing':
                env.finalize_step()
            self.assertTrue(full_stack_success(env))
            self.assertGreater(env.outcome()['final_stable_height_m'], .275)
        finally:
            env.close()

    def test_deadline_has_no_success_override(self):
        env = BoxStackEnv(seed=2, time_limit=.2)
        try:
            policy = BoxStackController(env)
            while env.status == 'running':
                action, _ = policy.action(env, env.elapsed_steps)
                env.step(action)
            while env.status == 'finalizing':
                env.finalize_step()
            self.assertFalse(full_stack_success(env))
            self.assertEqual(env.outcome()['stop_reason'], 'time_limit')
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
