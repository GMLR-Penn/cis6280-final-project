"""Regression coverage for support scoring, bounded withdrawal and error recovery."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from cube_pick.box_stacking import BoxStackEnv
from cube_pick.box_data import record_attempt
from cube_pick.varied_box_data import record_attempt as varied_attempt
from cube_pick.recording import scene_config
from cube_pick.sharpa import SharpaCubeEnv
from cube_pick.support_score import supported_height


def tower_rows():
    row = dict(object_velocity=np.zeros((4, 6)), object_hand_contacts=np.zeros(4, int),
        object_ground_contacts=np.zeros(4, int), object_table_contacts=np.array([1, 0, 0, 1]),
        object_table_support=np.array([1, 0, 0, 1]),
        object_support=np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        object_contacts=np.zeros((4, 4), int),
        object_bounds=np.array([[[-.025, -.025, 0], [.025, .025, .055]],
          [[-.025, -.025, .055], [.025, .025, .11]],
          [[-.025, -.025, .11], [.025, .025, .195]],
          [[.07, -.025, 0], [.125, .025, .055]]]))
    return [copy.deepcopy(row) for _ in range(31)]


class SupportScore(unittest.TestCase):
    def test_incidental_outside_contact_preserves_tower(self):
        rows = tower_rows()
        rows[0]['object_contacts'][0, 3] = rows[0]['object_contacts'][3, 0] = 1
        self.assertEqual(supported_height(rows, [0, 0])['object_indices'], [0, 1, 2])
        self.assertAlmostEqual(supported_height(rows, [0, 0])['height_m'], .195)

    def test_falling_top_preserves_base_but_invalid_support_excludes_upper(self):
        rows = tower_rows()
        rows[0]['object_velocity'][2, 5] = -.5
        self.assertEqual(supported_height(rows, [0, 0])['object_indices'], [0, 1])
        rows = tower_rows()
        rows[0]['object_hand_contacts'][1] = 1
        self.assertEqual(supported_height(rows, [0, 0])['object_indices'], [0])

    def test_lateral_contact_and_temporary_support_cannot_support_upper(self):
        rows = tower_rows()
        for r in rows:
            r['object_support'][0, 1] = 0
            r['object_contacts'][0, 1] = r['object_contacts'][1, 0] = 1
        self.assertEqual(supported_height(rows, [0, 0])['object_indices'], [0])
        rows = tower_rows()
        rows[-1]['object_support'][0, 1] = 0
        self.assertEqual(supported_height(rows, [0, 0])['object_indices'], [0])

    def test_extra_invalid_lower_support_blocks_upper(self):
        rows = tower_rows()
        rows[0]['object_support'][3, 1] = 1
        self.assertEqual(supported_height(rows, [0, 0])['object_indices'], [0])

    def test_isolated_roots_are_not_a_full_stack(self):
        rows = tower_rows()
        for r in rows:
            r['object_bounds'][3] = [[.03, -.025, 0], [.08, .025, .055]]
        self.assertEqual(supported_height(rows, [0, 0])['object_count'], 3)
        self.assertEqual(supported_height(rows[:-1], [0, 0])['height_m'], 0)


class RecordingRecovery(unittest.TestCase):
    def test_error_preserves_valid_prefix_and_is_not_retried(self):
        original = SharpaCubeEnv.step
        def fail(env, action):
            if getattr(env, 'elapsed_steps', -1) == 9:
                # Model a failed step that already modified the simulator.
                env.data.qpos[0] += .001
                raise RuntimeError('injected failure after partial mutation')
            return original(env, action)
        for collector, extra in ((record_attempt, ()), (varied_attempt, ('nominal',))):
            with self.subTest(collector=collector.__module__), tempfile.TemporaryDirectory() as temp:
                with patch.object(SharpaCubeEnv, 'step', fail):
                    record = collector(temp, 1, *extra, resolution=64, time_limit=2.)
                self.assertEqual(record['execution_status'], 'error')
                self.assertIsNone(record['success'])
                self.assertIsNone(record['outcome'])
                self.assertEqual(record['transitions'], 9)
                with np.load(Path(temp)/record['observations']) as obs, np.load(Path(temp)/record['privileged']) as truth:
                    self.assertEqual(obs['actions'].shape, (9, 25))
                    self.assertEqual(obs['proprio'].shape, (10, 75))
                    self.assertEqual(len(truth['integration_state']), 10)
                    self.assertFalse(obs['terminated'].any() or obs['truncated'].any())
                    meta = json.loads((Path(temp)/record['metadata']).read_text())
                    env = BoxStackEnv(config=scene_config(meta))
                    try:
                        state = np.empty(truth['integration_state'].shape[1])
                        for t in range(10):
                            mujoco.mj_getState(env.model, env.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
                            np.testing.assert_array_equal(state, truth['integration_state'][t])
                            if t < 9:
                                env.step(obs['actions'][t])
                    finally:
                        env.close()
                error = json.loads((Path(temp)/record['error']).read_text())
                self.assertEqual(error['stage'], 'step')
                self.assertEqual(len(error['failed_action']), 25)
                with patch.object(SharpaCubeEnv, 'step', side_effect=AssertionError('must not retry')):
                    self.assertEqual(collector(temp, 1, *extra, resolution=64, time_limit=2.), record)
                following = collector(temp, 2, *extra, resolution=64, time_limit=.1)
                self.assertEqual(following['execution_status'], 'ok')
                self.assertFalse(following['success'])  # physical failure remains a real label

    def test_initialization_error_is_audited(self):
        with tempfile.TemporaryDirectory() as temp, patch('cube_pick.box_data.BoxStackEnv', side_effect=RuntimeError('init')):
            record = record_attempt(temp, 0)
            self.assertEqual(record['transitions'], 0)
            self.assertIsNone(record['success'])
            error = json.loads((Path(temp)/record['error']).read_text())
            self.assertEqual(error['stage'], 'initialize')
            self.assertEqual(error['valid_observations'], 0)

    def test_metadata_roundtrip_and_low_wrist_withdrawal(self):
        with tempfile.TemporaryDirectory() as temp:
            record = record_attempt(temp, 2, resolution=64, time_limit=.1)
            self.assertEqual(record['execution_status'], 'ok')
            meta = json.loads((Path(temp)/record['metadata']).read_text())
            self.assertNotIn('cube_mass', meta['scene'])
            self.assertIn('cube_mass', meta['legacy_unused_scene_fields'])
            env = BoxStackEnv(config=scene_config(meta))
            try:
                import hashlib
                self.assertEqual(hashlib.sha256(env.xml.encode()).hexdigest(), meta['mjcf_sha256'])
                self.assertEqual(meta['physics']['objects'][0]['mass_kg'], env.model.body_mass[env.object_ids[0]])
                # Start near the bottom of the wrist range to exercise a lift
                # requiring more than the legacy twenty actions.
                env.data.qpos[env.qids[2]] = .04
                env.targets[2] = .04
                mujoco.mj_forward(env.model, env.data)
                env.stop()
                self.assertGreater(env.lift_steps, 20)
                for _ in range(200):
                    if env.status == 'done':
                        break
                    env.finalize_step()
                self.assertEqual(env.status, 'done')
                self.assertTrue(env.outcome()['park_reached'])
                self.assertGreater(env.finalization_steps, 71)
                self.assertLess(env.finalization_steps, 200)
            finally:
                env.close()


class CollectorCLI(unittest.TestCase):
    def test_all_errors_are_bounded_and_resume_does_not_repeat(self):
        import os
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'sitecustomize.py').write_text(
                "from cube_pick.sharpa import SharpaCubeEnv\n"
                "def fail(self, action):\n    raise RuntimeError('CLI injected error')\n"
                "SharpaCubeEnv.step = fail\n")
            process_env = dict(os.environ, PYTHONPATH=str(root))
            variants = [
                ('box', ['--successes', '1', '--max-attempts', '2', '--stage', 'collect'], 1),
                ('varied_box', ['--scenes', '1', '--modes', 'nominal', 'weak_grip'], 0),
            ]
            for kind, options, exit_code in variants:
                out = root/kind
                command = [sys.executable, f'scripts/collect_{kind}_dataset.py', '--out', str(out),
                           '--workers', '1', '--render-workers', '1', '--resolution', '64',
                           '--time-limit', '.1', *options]
                saved = None
                for resume in (False, True):
                    result = subprocess.run(command+(['--resume'] if resume else []),
                        env=process_env, capture_output=True, text=True, timeout=45)
                    self.assertEqual(result.returncode, exit_code, result.stderr)
                    manifest = json.loads((out/'manifest.json').read_text())
                    self.assertEqual(len(manifest['attempts']), 2)
                    self.assertEqual(manifest['episodes'], [])
                    self.assertTrue(all(r['execution_status'] == 'error' for r in manifest['attempts']))
                    if saved is not None:
                        self.assertEqual(saved, manifest['attempts'])
                    saved = manifest['attempts']
                if kind == 'varied_box':
                    self.assertEqual(manifest['status'], 'complete_with_errors')
                    validation = json.loads((out/'validation.json').read_text())
                    self.assertFalse(validation['all_passed'])
                    self.assertEqual(len(validation['execution_errors']), 2)
