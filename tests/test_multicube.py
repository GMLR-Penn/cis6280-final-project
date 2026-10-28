"""Outcome regressions: physically unsupported or transient towers must not score."""
import copy
import unittest

import numpy as np

from cube_pick.multicube import MultiCubeStackEnv, stable_tower


def tower_records():
    pose = np.array([[0, 0, .03, 1, 0, 0, 0], [0, 0, .085, 1, 0, 0, 0],
                     [0, 0, .135, 1, 0, 0, 0]], dtype=float)
    sample = dict(object_pose=pose, object_velocity=np.zeros((3, 6)),
                  object_hand_contacts=np.zeros(3, int), object_table_contacts=np.array([1, 0, 0]),
                  object_contacts=np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    return [copy.deepcopy(sample) for _ in range(21)]


class TowerLabels(unittest.TestCase):
    def score(self, rows):
        return stable_tower(rows, np.array([.06, .05, .05]), np.zeros(2))

    def test_stable_three_layers(self):
        result = self.score(tower_records())
        self.assertEqual(result['layers'], 3)
        self.assertAlmostEqual(result['height_m'], .16)

    def test_requires_full_two_seconds(self):
        self.assertEqual(self.score(tower_records()[:20])['layers'], 0)

    def test_hand_supported_top_does_not_count(self):
        rows = tower_records()
        rows[10]['object_hand_contacts'][2] = 1
        self.assertEqual(self.score(rows)['layers'], 2)

    def test_contact_and_alignment_both_required(self):
        rows = tower_records()
        rows[0]['object_contacts'][1, 2] = 0
        self.assertEqual(self.score(rows)['layers'], 2)
        rows = tower_records()
        rows[-1]['object_pose'][2, 0] = .04
        self.assertEqual(self.score(rows)['layers'], 2)

    def test_falling_top_does_not_count(self):
        rows = tower_records()
        rows[-1]['object_velocity'][2, 5] = -.2
        self.assertEqual(self.score(rows)['layers'], 2)

    def test_tilted_top_does_not_count(self):
        rows = tower_records()
        rows[-1]['object_pose'][2, 3:] = [np.cos(.2), np.sin(.2), 0, 0]
        self.assertEqual(self.score(rows)['layers'], 2)

    def test_displaced_or_airborne_base_does_not_count(self):
        rows = tower_records()
        rows[-1]['object_table_contacts'][0] = 0
        self.assertEqual(self.score(rows)['layers'], 0)
        rows = tower_records()
        for row in rows:
            row['object_pose'][:, 0] += .05
        self.assertEqual(self.score(rows)['layers'], 0)

    def test_layers_can_use_different_pickup_order(self):
        rows = tower_records()
        for row in rows:
            row['object_pose'][[1, 2]] = row['object_pose'][[2, 1]]
            row['object_contacts'] = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]])
        self.assertEqual(self.score(rows)['object_indices'], [0, 2, 1])

    def test_free_body_dimensions_and_no_hand_misclassification(self):
        for n in (2, 3, 5):
            env = MultiCubeStackEnv(num_cubes=n)
            try:
                self.assertEqual(env.model.nq, 25+7*n)
                self.assertEqual(env.model.neq, 0)
                self.assertEqual(env.model.nmocap, 0)
                self.assertFalse(set(env.object_geoms) & env.hand_geoms)
                self.assertEqual(env.privileged()['object_pose'].shape, (n, 7))
                self.assertEqual(env.proprio().shape, (75,))
            finally:
                env.close()


if __name__ == '__main__':
    unittest.main()
