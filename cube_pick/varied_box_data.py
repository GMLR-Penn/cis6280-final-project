"""Record, render and replay all variable-size paired rollouts, including failures.

Separate from the frozen success-only v0.1 schema and collector.
"""
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from .box_stacking import full_stack_success
from .varied_boxes import VariedBoxEnv, VariedBoxController
from .env import DT, PHYSICS_DT, CAMERAS
from .recording import record_rollout, scene_config
from .sharpa import ROOT

SCHEMA = 'varied-box-interventions-v0.2'
SOURCES = ('cube_pick/recording.py', 'cube_pick/support_score.py', 'cube_pick/box_stacking.py', 'cube_pick/box_data.py', 'cube_pick/mixed_feedback.py',
           'cube_pick/mixed_stacking.py', 'cube_pick/multicube.py', 'cube_pick/whole_hand.py',
           'cube_pick/sharpa.py', 'cube_pick/env.py', 'assets/sharpa-size-grasps.json',
           'assets/sharpa-mixed-grasps.json', 'cube_pick/varied_boxes.py',
           'cube_pick/varied_box_data.py', 'scripts/collect_varied_box_dataset.py')
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {f: digest(ROOT/f) for f in SOURCES}


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def score_labels(env):
    score = env.score()
    members = np.zeros(len(env.config.objects), bool)
    members[score['object_indices']] = True
    return np.float32(score['height_m']), members


def record_attempt(root, seed, mode, resolution=128, time_limit=180.):
    return record_rollout(root, seed, mode, resolution, time_limit,
        env_factory=VariedBoxEnv, policy_factory=lambda env: VariedBoxController(env, mode, seed),
        schema=SCHEMA, source_hashes=source_hashes)


def render_episode(root, record):
    if record.get('execution_status') == 'error':
        raise ValueError('Execution errors are audit records, not renderable episodes')
    root = Path(root)
    folder = (root/record['metadata']).parent
    current = json.loads((folder/'record.json').read_text())
    if current['rendered']:
        for key in ('observations', 'privileged', 'metadata', 'video'):
            if digest(root/current[key]) != current['sha256'][key]:
                raise ValueError(f'Corrupt rendered file: {current[key]}')
        return current
    meta = json.loads((root/record['metadata']).read_text())
    env = VariedBoxEnv(config=scene_config(meta), resolution=meta['resolution'])
    video_temp = folder/'preview.tmp.mp4'
    video_path = folder/'preview.mp4'
    writer = imageio.get_writer(video_temp, fps=round(1/DT), codec='libx264', quality=7,
                               ffmpeg_params=['-movflags', '+faststart'])
    try:
        with np.load(root/record['privileged']) as truth:
            states = truth['integration_state']
            frames = np.empty((len(states), 2, meta['resolution'], meta['resolution'], 3), np.uint8)
            for t, state in enumerate(states):
                mujoco.mj_setState(env.model, env.data, state, STATE_SPEC)
                mujoco.mj_forward(env.model, env.data)
                frames[t] = env.render()
                writer.append_data(np.concatenate(frames[t], axis=1))
        writer.close()
        video_temp.replace(video_path)
        with np.load(root/record['observations']) as obs:
            values = {k: obs[k] for k in obs.files if k != 'rgb'}
        temp = folder/'observations.tmp.npz'
        np.savez_compressed(temp, rgb=frames, **values)
        temp.replace(root/record['observations'])
        result = dict(record, rendered=True, video=str(video_path.relative_to(root)))
        result['sha256'] = dict(record['sha256'], observations=digest(root/record['observations']),
                                video=digest(video_path))
        write_json(folder/'record.json', result)
        return result
    finally:
        writer.close()
        env.close()


def validate_episode(root, record):
    root = Path(root)
    for key in ('observations', 'privileged', 'metadata', 'video'):
        if digest(root/record[key]) != record['sha256'][key]:
            raise ValueError(f'Checksum mismatch: {record[key]}')
    meta = json.loads((root/record['metadata']).read_text())
    env = VariedBoxEnv(config=scene_config(meta), resolution=meta['resolution'])
    try:
        assert hashlib.sha256(env.xml.encode()).hexdigest() == meta['mjcf_sha256']
        with np.load(root/record['observations']) as obs, np.load(root/record['privileged']) as truth:
            obs = {k: obs[k] for k in obs.files}
            truth = {k: truth[k] for k in truth.files}
            n = len(obs['actions'])
            assert obs['rgb'].shape == (n+1, 2, meta['resolution'], meta['resolution'], 3)
            assert obs['proprio'].shape == (n+1, 75)
            assert obs['actions'].shape == (n, 25)
            assert np.isfinite(obs['actions']).all() and np.max(np.abs(obs['actions'])) <= 1
            assert np.array_equal(obs['agent_action_mask'], np.arange(n) < meta['control_steps'])
            stopped = meta['outcome']['stop_reason'] == 'agent_stop'
            assert np.flatnonzero(obs['stop_requested']).tolist() == ([meta['control_steps']] if stopped else [])
            assert np.flatnonzero(obs['terminated']).tolist() == ([n-1] if stopped else [])
            assert np.flatnonzero(obs['truncated']).tolist() == ([] if stopped else [n-1])
            frames = obs['rgb']
            sampled = {0, n//2, n}
            state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
            for t in range(n+1):
                mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
                np.testing.assert_array_equal(state, truth['integration_state'][t])
                np.testing.assert_array_equal(env.proprio(), obs['proprio'][t])
                np.testing.assert_equal(np.float32(env.time_remaining), obs['time_remaining'][t])
                np.testing.assert_allclose(env.data.time-env.start_time, obs['timestamps'][t], atol=1e-10, rtol=0)
                for key, value in env.privileged().items():
                    np.testing.assert_array_equal(value, truth[key][t])
                height, members = score_labels(env)
                np.testing.assert_equal(height, truth['stable_height_m'][t])
                np.testing.assert_array_equal(members, truth['stable_component_mask'][t])
                if t in sampled:
                    np.testing.assert_array_equal(env.render(), frames[t])
                if t == n:
                    break
                before = env.targets.copy()
                if obs['agent_action_mask'][t]:
                    env.step(obs['actions'][t])
                else:
                    if env.status == 'running':
                        env.stop()
                    action, phase = env.finalize_step()
                    np.testing.assert_array_equal(action, obs['actions'][t])
                    assert phase == str(truth['phases'][t])
                np.testing.assert_array_equal(env.targets-before, obs['applied_target_delta'][t])
            assert bool(full_stack_success(env)) == meta['success'] == record['success']
            assert env.outcome() == meta['outcome'] == record['outcome']
            reader = imageio.get_reader(root/record['video'])
            try:
                assert reader.count_frames() == n+1
                assert reader.get_meta_data()['fps'] == round(1/DT)
                assert reader.get_data(n).shape == (meta['resolution'], 2*meta['resolution'], 3)
            finally:
                reader.close()
        return dict(episode_id=record['episode_id'], transitions=n, replay='exact',
                    sampled_rgb='exact', video_frames=n+1, success=bool(record['success']))
    finally:
        env.close()


def load_episode(root, record, include_privileged=False):
    """Default student loader excludes simulation truth and success/failure labels."""
    if record.get('execution_status') == 'error':
        raise ValueError('Execution errors are excluded from the student dataset')
    root = Path(root)
    with np.load(root/record['observations']) as f:
        data = {k: f[k] for k in f.files}
    if include_privileged:
        with np.load(root/record['privileged']) as f:
            data['privileged'] = {k: f[k] for k in f.files}
    return data
