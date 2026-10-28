"""End-to-end actuator-only feedback and enforced-deadline regressions."""
import unittest

import mujoco
import numpy as np

from cube_pick.mixed_stacking import MixedStackEnv, make_mixed_controller


class MixedFeedback(unittest.TestCase):
    def rollout(self, env):
        policy = make_mixed_controller(env, 'feedback')
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(env.model, state_spec))
        after = before.copy()
        while env.status == 'running':
            mujoco.mj_getState(env.model, env.data, before, state_spec)
            targets = env.targets.copy()
            elapsed = env.elapsed_steps
            action, _ = policy.action(env, elapsed)
            mujoco.mj_getState(env.model, env.data, after, state_spec)
            np.testing.assert_array_equal(after, before, err_msg='Policy changed live physics')
            np.testing.assert_array_equal(env.targets, targets)
            self.assertEqual(env.elapsed_steps, elapsed)
            if action is None:
                env.stop()
            else:
                self.assertEqual(action.shape, (25,))
                self.assertTrue(np.isfinite(action).all())
                self.assertLessEqual(float(np.abs(action).max()), 1.)
                env.step(action)
        while env.status == 'finalizing':
            env.finalize_step()
        return policy

    def test_four_shapes_remain_stable_after_hand_withdrawal(self):
        env = MixedStackEnv(seed=0)
        try:
            self.rollout(env)
            result = env.outcome()
            self.assertEqual(result['scored_object_count'], 4)
            self.assertGreater(result['final_stable_height_m'], .25)
            self.assertLess(result['elapsed_seconds'], env.config.time_limit)
            self.assertEqual(result['stop_reason'], 'agent_stop')
        finally:
            env.close()

    def test_deadline_interrupts_controller(self):
        env = MixedStackEnv(seed=0, time_limit=2.)
        try:
            self.rollout(env)
            result = env.outcome()
            self.assertEqual(result['stop_reason'], 'time_limit')
            self.assertEqual(result['elapsed_seconds'], 2.)
            self.assertEqual(result['scored_object_count'], 0)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
