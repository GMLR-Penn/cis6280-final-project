"""Variable geometry, physically paired interventions and scene-disjoint splits."""
from dataclasses import asdict
import unittest

import mujoco
import numpy as np

from cube_pick.box_stacking import box_scene
from cube_pick.mixed_stacking import MATERIALS
from cube_pick.varied_boxes import VariedBoxEnv, VariedBoxController, varied_box_scene, MODES
from scripts.collect_varied_box_dataset import finalize_splits


class VariedBoxes(unittest.TestCase):
    def test_geometry_and_mass_change_without_changing_v1(self):
        for seed in range(12):
            scene = varied_box_scene(seed)
            self.assertEqual(asdict(scene), asdict(varied_box_scene(seed)))
            self.assertEqual(set(o.size[0] for o in scene.objects), {.05, .055, .06})
            self.assertEqual(sorted(o.size[2] for o in scene.objects), [.055, .055, .085, .085])
            for o in scene.objects:
                self.assertEqual(o.size[0], o.size[1])
                self.assertAlmostEqual(o.mass, np.prod(o.size)*MATERIALS[o.material]['density'])
            self.assertTrue(all(o.size[:2] == (.055, .055) for o in box_scene(seed).objects))

    def test_scene_pairs_never_cross_splits(self):
        for count in (1, 2, 3, 6, 20):
            m = dict(episodes=[dict(scene_seed=s, mode=mode, episode_id=f'{s}-{mode}')
                               for s in range(count) for mode in MODES])
            finalize_splits(m)
            ids = [i for split in m['splits'].values() for i in split]
            self.assertEqual(len(ids), count*len(MODES))
            self.assertEqual(len(set(ids)), len(ids))
            for seed in range(count):
                groups = [k for k, values in m['splits'].items() if f'{seed}-nominal' in values]
                self.assertEqual(len(groups), 1)
                self.assertTrue(all(f'{seed}-{mode}' in m['splits'][groups[0]] for mode in MODES))

    def test_paired_initial_state_and_distinct_order(self):
        states, orders = [], []
        for mode in MODES:
            env = VariedBoxEnv(seed=1)
            try:
                controller = VariedBoxController(env, mode, seed=1)
                state = np.empty(mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_INTEGRATION))
                mujoco.mj_getState(env.model, env.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
                states.append(state)
                orders.append(controller.order)
            finally:
                env.close()
        for state in states[1:]:
            np.testing.assert_array_equal(state, states[0])
        self.assertNotEqual(orders[0], orders[1])

    def test_intervention_changes_actions_without_direct_state_mutation(self):
        results = []
        for mode in ('nominal', 'weak_grip'):
            env = VariedBoxEnv(seed=1)
            controller = VariedBoxController(env, mode, seed=1)
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            before = np.empty(mujoco.mj_stateSize(env.model, spec))
            after = before.copy()
            actions = []
            try:
                while env.status == 'running' and env.elapsed_steps < 600:
                    mujoco.mj_getState(env.model, env.data, before, spec)
                    targets = env.targets.copy()
                    action, _ = controller.action(env, env.elapsed_steps)
                    mujoco.mj_getState(env.model, env.data, after, spec)
                    np.testing.assert_array_equal(before, after)
                    np.testing.assert_array_equal(targets, env.targets)
                    if action is None:
                        break
                    self.assertTrue(np.isfinite(action).all())
                    self.assertLessEqual(np.abs(action).max(), 1)
                    actions.append(action.copy())
                    env.step(action)
                results.append((np.array(actions), controller.intervention_step))
            finally:
                env.close()
        nominal, changed = results[0][0], results[1][0]
        step = results[1][1]
        self.assertIsNotNone(step)
        self.assertGreater(step, 100)
        np.testing.assert_array_equal(nominal[:step], changed[:step])
        n = min(len(nominal), len(changed))
        self.assertTrue(np.any(nominal[step:n] != changed[step:n]))


if __name__ == '__main__':
    unittest.main()
