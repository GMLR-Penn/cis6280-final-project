"""Assignment entry points: module imports, self-contained index limits, and sampler reseeding."""
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from assignment.training import _cli

try:
    import torch
except ImportError:
    torch = None

PROBE = '''
def work(x):
    return x + 1


def main():
    import pickle
    return pickle.loads(pickle.dumps(work))(1)
'''


class ScriptLoading(unittest.TestCase):
    def test_script_functions_pickle_for_worker_processes(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'scripts').mkdir()
            (Path(root)/'scripts'/'assignment_probe.py').write_text(PROBE)
            with mock.patch.object(_cli, 'ROOT', Path(root)):
                self.assertEqual(_cli.run('assignment_probe.py'), 2)

    def test_stage_entrypoints_exist_and_import(self):
        stages = [
            'assignment.training.data.collect',
            'assignment.training.data.cache',
            'assignment.training.data.encode',
            'assignment.training.data.collect_onpolicy',
            'assignment.training.training.train_jepa',
            'assignment.training.training.train_policy',
            'assignment.training.training.finetune_onpolicy',
            'assignment.training.planning.evaluate',
            'assignment.training.planning.goals',
        ]
        for name in stages:
            mod = importlib.import_module(name)
            self.assertTrue(callable(getattr(mod, 'main', None)), name)


@unittest.skipIf(torch is None, 'torch is not installed')
class AgentAndPolicyModules(unittest.TestCase):
    def test_modules_import(self):
        for name in ['assignment.training.training.jepa', 'assignment.training.training.policy',
                     'assignment.training.planning.agent', 'cube_pick.stack_latent_policy',
                     'cube_pick.stack_jepa']:
            importlib.import_module(name)

    def test_episode_limits_work_without_raw_manifest(self):
        from assignment.training.training.policy import episode_limits
        index_with_steps = {
            'roots': ['/nonexistent/training/machine/path'],
            'episodes': [
                {'episode_id': 'ep0', 'length': 50, 'control_steps': 200},
                {'episode_id': 'ep1', 'length': 30, 'control_steps': 200},
            ],
        }
        np.testing.assert_array_equal(episode_limits(index_with_steps), np.array([40, 29]))
        index_fallback = {
            'roots': ['/nonexistent/training/machine/path'],
            'episodes': [{'episode_id': 'ep0', 'length': 50}],
        }
        with self.assertWarns(UserWarning):
            np.testing.assert_array_equal(episode_limits(index_fallback), np.array([49]))

    def test_reset_seed_makes_samples_independent_of_history(self):
        from assignment.training.planning.agent import GoalPlanningAgent, LatentPolicyAgent
        for cls in (LatentPolicyAgent, GoalPlanningAgent):
            agent = cls.__new__(cls)
            agent.generator = torch.Generator().manual_seed(0)
            draws = []
            for burn in (0, 5):
                torch.randn(burn, generator=agent.generator)
                LatentPolicyAgent.reset(agent, seed=100038)
                draws.append(torch.randn(4, generator=agent.generator))
            self.assertTrue(torch.equal(*draws))

    def _goal_agent(self):
        from assignment.training.planning.agent import GoalPlanningAgent
        agent = GoalPlanningAgent.__new__(GoalPlanningAgent)
        agent.generator = torch.Generator().manual_seed(0)
        agent.goals = None
        agent._encode = lambda frames: torch.as_tensor(frames, dtype=torch.float32).flatten(1)
        return agent

    def test_goal_reset_reseeds_and_sets_switch_gaps(self):
        agent = self._goal_agent()
        goals, init = np.array([[[1.]], [[3.]]]), np.array([[0.]])
        draws = []
        for burn in (0, 5):
            torch.randn(burn, generator=agent.generator)
            agent.reset(goals, init, seed=100038)
            draws.append(torch.randn(4, generator=agent.generator))
        self.assertTrue(torch.equal(*draws))
        self.assertEqual(agent.gap.tolist(), [1., 4.])

    def test_goal_reset_validates_before_changing_state(self):
        agent = self._goal_agent()
        agent.reset(np.array([[[1.]], [[3.]]]), np.array([[0.]]))
        with self.assertRaises(ValueError):
            agent.reset(np.zeros((3, 1, 1)))
        self.assertEqual(len(agent.goals), 2)
        self.assertEqual(len(agent.gap), 2)
        agent.reset()
        with self.assertRaises(RuntimeError):
            agent.step(np.zeros((1, 1)))

    def test_finetune_reload_picks_up_shards_that_become_readable(self):
        from assignment.training.training.finetune_onpolicy import KEYS, Train
        shard = {k: np.zeros((1, 1, 1), np.float16) for k in KEYS}
        with tempfile.TemporaryDirectory() as root:
            np.savez(Path(root)/'train-r000-00000.npz', **shard)
            Path(root, 'train-r001-00000.npz').write_bytes(b'partial')
            train = Train(root, 'cpu')
            self.assertEqual(train.n, 1)
            np.savez(Path(root)/'train-r001-00000.npz', **shard)
            self.assertTrue(train.reload())
            self.assertEqual(train.n, 2)
            self.assertFalse(train.reload())


if __name__ == '__main__':
    unittest.main()
