"""Unit tests for scene splitting, action noise, support structure, and stacking events."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))

from assignment.training.data.collect import parse_condition, slug
from assignment.training.data.signals import ActionNoise, NoiseSpec, events, scene_split, structure, task_rewards, tower

GOAL, HALF = (.16, 0.), .085


def scene(n, k=2):
    """Synthetic privileged arrays: objects resting on the table outside the goal zone."""
    truth = dict(object_hand_contacts=np.zeros((n, k), np.int16), object_ground_contacts=np.zeros((n, k), np.int16),
                 object_table_contacts=np.ones((n, k), np.int16), object_table_support=np.ones((n, k), bool),
                 object_contacts=np.zeros((n, k, k), np.int16), object_support=np.zeros((n, k, k), bool),
                 object_hand_normal_force=np.zeros((n, k, 6)), object_pose=np.zeros((n, k, 7)),
                 object_bounds=np.zeros((n, k, 2, 3)))
    for i in range(k):
        place(truth, slice(None), i, (-.2, .1*i), 0.)
    return truth


def place(truth, t, i, xy, bottom, edge=.05):
    truth['object_bounds'][t, i, 0] = (xy[0]-edge/2, xy[1]-edge/2, bottom)
    truth['object_bounds'][t, i, 1] = (xy[0]+edge/2, xy[1]+edge/2, bottom+edge)
    truth['object_pose'][t, i, :3] = (xy[0], xy[1], bottom+edge/2)


def hold(truth, t, i, bottom):
    truth['object_hand_contacts'][t, i] = 6
    truth['object_hand_normal_force'][t, i] = 1.
    truth['object_table_contacts'][t, i] = 0
    truth['object_table_support'][t, i] = False
    place(truth, t, i, truth['object_pose'][t, i, :2], bottom)


class StackingSignals(unittest.TestCase):
    def test_scene_split_is_fixed_and_proportional(self):
        splits = [scene_split(s) for s in range(10000)]
        self.assertEqual(splits, [scene_split(s) for s in range(10000)])
        fractions = {k: splits.count(k)/len(splits) for k in ('train', 'val', 'test')}
        self.assertTrue(abs(fractions['train']-.8) < .02 and abs(fractions['val']-.1) < .02)

    def test_noise_parse_and_statistics(self):
        self.assertEqual(NoiseSpec.parse('nominal'), NoiseSpec())
        spec = NoiseSpec.parse('ou:0.02-0.08:0.9,f=0,p=0.1')
        self.assertTrue(spec.kind == 'ou' and spec.sigma == (.02, .08) and spec.rho == .9)
        self.assertTrue(spec.finger_scale == 0 and spec.p_random == .1)
        zero = np.zeros(25, np.float32)
        self.assertTrue(np.array_equal(ActionNoise(NoiseSpec(), 0)(zero)[0], zero))
        iid = ActionNoise(NoiseSpec('iid', .05), 1)
        x = np.stack([iid(zero)[0] for _ in range(4000)])
        self.assertTrue(abs(x.std()-.05) < .003 and abs(np.corrcoef(x[:-1, 0], x[1:, 0])[0, 1]) < .1)
        ou = ActionNoise(NoiseSpec('ou', .05, .9), 2)
        y = np.stack([ou(zero)[0] for _ in range(20000)])
        self.assertTrue(abs(y.std()-.05) < .01 and abs(np.corrcoef(y[:-1, 0], y[1:, 0])[0, 1]-.9) < .03)
        ranged = ActionNoise(NoiseSpec('iid', (.02, .08)), 3)
        self.assertTrue(.02 <= ranged.sigma <= .08 and ranged.resolved()['sampled_sigma'] == ranged.sigma)

    def test_noise_log_uniform_sigma_and_zero_fraction(self):
        spec = NoiseSpec.parse('iid:0.005-0.2,log,z=0.1')
        self.assertTrue(spec.sigma == (.005, .2) and spec.log_sigma and spec.p_zero == .1)
        sigmas = np.array([ActionNoise(spec, seed).sigma for seed in range(4000)])
        self.assertTrue(abs(np.mean(sigmas == 0)-.1) < .02)
        logs = np.log(sigmas[sigmas > 0])
        self.assertTrue(logs.min() >= np.log(.005) and logs.max() <= np.log(.2))
        self.assertTrue(abs(np.mean(logs < np.log(.005)+np.log(40)/2)-.5) < .03)
        zero = np.zeros(25, np.float32)
        off = next(ActionNoise(spec, s) for s in range(100) if ActionNoise(spec, s).sigma == 0)
        self.assertTrue(np.array_equal(off(zero)[0], zero))
        with self.assertRaises(ValueError):
            NoiseSpec('iid', .1, log_sigma=True)

    def test_structure_requires_goal_zone_support_and_free_hands(self):
        truth = scene(1)
        place(truth, 0, 0, GOAL, 0.)
        place(truth, 0, 1, GOAL, .05)
        truth['object_table_support'][0, 1] = False
        truth['object_support'][0, 0, 1] = True
        row = {k: v[0] for k, v in truth.items()}
        members, height = structure(row, GOAL, HALF)
        self.assertTrue(members.tolist() == [True, True] and np.isclose(height, .1))
        row['object_hand_contacts'] = np.array([0, 3])
        members, height = structure(row, GOAL, HALF)
        self.assertTrue(members.tolist() == [True, False] and np.isclose(height, .05))

    def test_tower_counts_only_one_connected_structure(self):
        truth = scene(1, 4)
        for i, dx in enumerate((-.06, -.02, .02, .06)):
            place(truth, 0, i, (GOAL[0]+dx, GOAL[1]), 0., edge=.03)
        row = {k: v[0] for k, v in truth.items()}
        self.assertEqual(int(structure(row, GOAL, HALF)[0].sum()), 4)
        self.assertEqual(int(tower(row, GOAL, HALF)[0].sum()), 1)
        place(truth, 0, 1, (GOAL[0]-.06, GOAL[1]), .03, edge=.03)
        truth['object_table_support'][0, 1] = False
        truth['object_support'][0, 0, 1] = True
        row = {k: v[0] for k, v in truth.items()}
        members, height = tower(row, GOAL, HALF)
        self.assertTrue(members.tolist() == [True, True, False, False] and np.isclose(height, .06))

    def test_events_placement_drop_and_collapse(self):
        n = 120
        truth = scene(n)
        for t in range(10, 30):
            hold(truth, t, 0, .0 if t < 12 else .1)
        truth['object_table_contacts'][10:12, 0] = 1
        place(truth, slice(30, n), 0, GOAL, 0.)
        for t in range(40, 55):
            hold(truth, t, 1, .0 if t < 42 else .1)
        truth['object_table_contacts'][40:42, 1] = 1
        place(truth, slice(90, n), 0, (.3, .3), 0.)
        found = [(e['event'], e['object']) for e in events(truth, GOAL, HALF)]
        self.assertEqual(found, [('grasp_attempt', 0), ('lift', 0), ('placement', 0), ('grasp_attempt', 1),
                                 ('lift', 1), ('drop', 1), ('collapse', 0)])
        reward, phi = task_rewards(truth, GOAL, HALF, final_height=0.)
        self.assertTrue(np.isclose(phi.max(), .05) and np.isclose(reward.sum(), 0.))

    def test_collector_condition_parsing(self):
        name, policy, noise = parse_condition('iid:0.05@offset+recover+sweep')
        self.assertEqual(policy, dict(kind='scripted', mode='offset', recovery=True, variation=None, sweep=True))
        self.assertTrue(noise['kind'] == 'iid' and noise['sigma'] == .05)
        self.assertEqual(slug('ou:0.02-0p08:0.9,f=0+recover'.replace('0p08', '0.08')), 'ou-0p02-0p08-0p9_f0-recover')


if __name__ == '__main__':
    unittest.main()
