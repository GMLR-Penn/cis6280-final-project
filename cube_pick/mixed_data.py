"""Recording and action replay for the separate mixed-stacking prototype."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .mixed_stacking import MATERIALS, MixedSceneConfig, MixedStackEnv, make_mixed_controller
from .sharpa import ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_mixed_episode(out, seed=0, mode='stack', resolution=128, time_limit=180.):
    out = Path(out)
    name = f'seed-{seed:04d}-{mode}'
    paths = {key: out/folder/f'{name}.{suffix}' for key, folder, suffix in
             [('observations', 'observations', 'npz'), ('privileged', 'privileged', 'npz'),
              ('metadata', 'privileged', 'json')]}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    env = MixedStackEnv(seed=seed, resolution=resolution, time_limit=time_limit)
    policy = make_mixed_controller(env, mode)
    frames, proprio, truth, states, times, remaining = [], [], [], [], [], []
    actions, phases, agent_mask = [], [], []
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION

    def record():
        frames.append(env.render())
        proprio.append(env.proprio())
        truth.append(env.privileged())
        state = np.empty(mujoco.mj_stateSize(env.model, state_spec))
        mujoco.mj_getState(env.model, env.data, state, state_spec)
        states.append(state)
        times.append(float(env.data.time-env.start_time))
        remaining.append(env.time_remaining)

    try:
        record()
        while env.status != 'done':
            agent = env.status == 'running'
            if agent:
                action, phase = policy.action(env, env.elapsed_steps)
                if action is None:
                    env.stop()
                    continue
                env.step(action)
            else:
                action, phase = env.finalize_step()
            actions.append(action)
            phases.append(phase)
            agent_mask.append(agent)
            record()
        np.savez_compressed(paths['observations'], rgb=np.stack(frames), proprio=np.stack(proprio),
                            actions=np.stack(actions), agent_action_mask=np.asarray(agent_mask),
                            timestamps=np.asarray(times), time_remaining=np.asarray(remaining, np.float32),
                            goal_xy=np.array([env.config.goal_x, env.config.goal_y], np.float32))
        np.savez_compressed(paths['privileged'], **{key: np.stack([r[key] for r in truth]) for key in truth[0]},
                            integration_state=np.stack(states), phases=np.asarray(phases))
        meta = dict(schema='mixed-stacking-v0.1', scene=asdict(env.config), mode=mode,
                    controller=f'{type(policy).__name__}: privileged scripted, no learned model', order=policy.order,
                    controller_events=getattr(policy, 'events', []),
                    material_catalog=MATERIALS, resolution=resolution, outcome=env.outcome(),
                    control_steps=env.elapsed_steps, mjcf_sha256=hashlib.sha256(env.xml.encode()).hexdigest(),
                    mujoco=mujoco.__version__, integration_state_spec=int(state_spec),
                    source_sha256={f: digest(ROOT/f) for f in ['cube_pick/mixed_stacking.py',
                        'cube_pick/mixed_data.py', 'cube_pick/mixed_feedback.py',
                        'cube_pick/sharpa.py', 'cube_pick/whole_hand.py',
                        'cube_pick/multicube.py', 'assets/sharpa-size-grasps.json',
                        'assets/sharpa-mixed-grasps.json']})
        paths['metadata'].write_text(json.dumps(meta, indent=2)+'\n')
        return dict(episode_id=name, scene_seed=seed, mode=mode, transitions=len(actions),
                    control_steps=env.elapsed_steps, outcome=env.outcome(),
                    **{key: str(path.relative_to(out)) for key, path in paths.items()},
                    sha256={key: digest(path) for key, path in paths.items()})
    finally:
        env.close()


def replay_mixed_episode(root, record):
    root = Path(root)
    for key in ('observations', 'privileged', 'metadata'):
        if digest(root/record[key]) != record['sha256'][key]:
            raise ValueError(f'Checksum mismatch: {record[key]}')
    meta = json.loads((root/record['metadata']).read_text())
    env = MixedStackEnv(config=MixedSceneConfig(**meta['scene']), resolution=meta['resolution'])
    try:
        assert hashlib.sha256(env.xml.encode()).hexdigest() == meta['mjcf_sha256'], 'Scene XML changed'
        with np.load(root/record['observations']) as obs, np.load(root/record['privileged']) as truth:
            n = len(obs['actions'])
            assert len(obs['proprio']) == n+1
            sampled = {0, n//2, n}
            for t in range(n+1):
                np.testing.assert_array_equal(env.proprio(), obs['proprio'][t])
                np.testing.assert_equal(np.float32(env.time_remaining), obs['time_remaining'][t])
                np.testing.assert_allclose(env.data.time-env.start_time, obs['timestamps'][t], atol=1e-10, rtol=0)
                for key, value in env.privileged().items():
                    np.testing.assert_array_equal(value, truth[key][t])
                state = np.empty(mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_INTEGRATION))
                mujoco.mj_getState(env.model, env.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
                np.testing.assert_array_equal(state, truth['integration_state'][t])
                if t in sampled:
                    np.testing.assert_array_equal(env.render(), obs['rgb'][t])
                if t == n:
                    break
                if obs['agent_action_mask'][t]:
                    env.step(obs['actions'][t])
                else:
                    if env.status == 'running':
                        env.stop()
                    action, phase = env.finalize_step()
                    np.testing.assert_array_equal(action, obs['actions'][t])
                    assert phase == str(truth['phases'][t])
            assert env.outcome() == meta['outcome'] == record['outcome'], 'Outcome mismatch'
        return dict(episode_id=record['episode_id'], transitions=n, replay='exact', sampled_rgb='exact', outcome=env.outcome())
    finally:
        env.close()
