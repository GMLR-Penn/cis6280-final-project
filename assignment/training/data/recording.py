"""Common trajectory format for every stacking data source (scripted controller, BC teacher).

One episode = T executed actions aligned with T+1 observations. Executed
actions (policy output + action noise, clipped) are what the simulator saw;
policy actions and the noise are stored separately. Physical failures are
ordinary episodes; simulator/controller exceptions are audit-only records.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import zlib

import imageio.v2 as imageio
import mujoco
import numpy as np

from cube_pick.box_stacking import full_stack_success
from cube_pick.env import DT, PHYSICS_DT, CAMERAS
from cube_pick.mixed_stacking import MATERIALS
from cube_pick.recording import physics_metadata, scene_config
from cube_pick.sharpa import ROOT
from assignment.training.data.signals import (LABELER_VERSION, ActionNoise, NoiseSpec, SHAPING, STAGES, episode_summary, events,
                            hand_point, scene_split, shaping_terms, stages, task_rewards)
from cube_pick.varied_boxes import VariedBoxEnv

SCHEMA = 'stacking-trajectories-v0.3'
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
SOURCES = ('cube_pick/recording.py', 'cube_pick/support_score.py', 'cube_pick/box_stacking.py',
           'cube_pick/mixed_feedback.py', 'cube_pick/mixed_stacking.py', 'cube_pick/multicube.py',
           'cube_pick/whole_hand.py', 'cube_pick/sharpa.py', 'cube_pick/env.py',
           'assets/sharpa-size-grasps.json', 'assets/sharpa-mixed-grasps.json', 'cube_pick/varied_boxes.py',
           'assignment/training/data/signals.py', 'assignment/training/data/scripted.py', 'assignment/training/data/recording.py',
           'assignment/training/data/teacher.py', 'assignment/training/data/task.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {f: digest(ROOT/f) for f in SOURCES}


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def derived_seed(*parts):
    return zlib.crc32('/'.join(map(str, parts)).encode())


def make_policy(env, spec, scene_seed, policy_seed):
    """spec['kind'] selects the source; every policy exposes action(env, t) -> (action|None, phase)."""
    if spec['kind'] == 'scripted':
        from assignment.training.data.scripted import ScriptedStackPolicy
        return ScriptedStackPolicy(env, spec.get('mode', 'nominal'), scene_seed, recovery=spec.get('recovery', False),
                                   variation=spec.get('variation'), policy_seed=policy_seed,
                                   sweep=spec.get('sweep', False))
    if spec['kind'] == 'bc':
        from assignment.training.data.teacher import BCPolicy
        return BCPolicy(env, spec, policy_seed)
    raise ValueError(f'Unknown policy kind: {spec["kind"]}')


def render_states(env, states):
    frames = np.empty((len(states), len(CAMERAS), env.resolution, env.resolution, 3), np.uint8)
    for t, state in enumerate(states):
        mujoco.mj_setState(env.model, env.data, state, STATE_SPEC)
        mujoco.mj_forward(env.model, env.data)
        frames[t] = env.render()
    return frames


def rgb_steps(n, every):
    """Observation indices that get RGB: every `every`-th step plus the final observation."""
    return np.unique(np.r_[np.arange(0, n+1, every), n]).astype(np.int32)


def write_video(path, frames, every=1):
    temp = path.with_name(path.stem+'.tmp.mp4')
    writer = imageio.get_writer(temp, fps=max(1, round(1/DT/every)), codec='libx264', quality=7,
                                ffmpeg_params=['-movflags', '+faststart'])
    try:
        for frame in frames:
            writer.append_data(np.concatenate(frame, axis=1))
    finally:
        writer.close()
    temp.replace(path)


def record_episode(root, job, render=True):
    """job: episode_id, source, condition, scene_seed, policy (dict), noise (NoiseSpec fields),
    resolution, time_limit, and optional provenance (e.g. checkpoint, training step) and
    rgb_every (render every k-th observation; states stay full-rate for exact re-rendering)."""
    root = Path(root)
    folder = root/'episodes'/job['episode_id']
    if (folder/'record.json').exists():
        record = json.loads((folder/'record.json').read_text())
        for key, checksum in record['sha256'].items():
            if digest(root/record[key]) != checksum:
                raise ValueError(f'Corrupt existing episode: {job["episode_id"]}/{key}')
        return record
    folder.mkdir(parents=True, exist_ok=True)
    timing = dict(start=time.time())
    seed = int(job['scene_seed'])
    noise = ActionNoise(NoiseSpec(**job.get('noise', {})), derived_seed(job['episode_id'], 'noise'))
    rows, states, proprio, times, remaining, heights, members, hands = [], [], [], [], [], [], [], []
    actions, policy_actions, replaced, phases, agent_mask, deltas = [], [], [], [], [], []
    env = policy = error = None
    stage, action = 'initialize', None

    def snapshot():
        row = env.privileged()
        state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
        mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
        prop = env.proprio()
        score = env.score()
        mask = np.zeros(len(env.config.objects), bool)
        mask[score['object_indices']] = True
        if not (np.isfinite(state).all() and np.isfinite(prop).all()):
            raise RuntimeError('Non-finite recorded state')
        rows.append(row); states.append(state); proprio.append(prop); hands.append(hand_point(env))
        times.append(float(env.data.time-env.start_time)); remaining.append(env.time_remaining)
        heights.append(np.float32(score['height_m'])); members.append(mask)

    try:
        try:
            start = job.get('start')
            if start:
                # Curriculum start: continue from a recorded state of another episode of this scene.
                from assignment.training.data.task import restore_env
                source_meta = json.loads((Path(start['root'])/start['metadata']).read_text())
                env = VariedBoxEnv(config=scene_config(source_meta), resolution=job['resolution'])
                with np.load(Path(start['root'])/start['privileged']) as f:
                    restore_env(env, f['integration_state'], int(start['step']))
            else:
                env = VariedBoxEnv(seed=seed, resolution=job['resolution'], time_limit=job['time_limit'])
            policy = make_policy(env, job['policy'], seed, derived_seed(job['episode_id'], 'policy'))
            snapshot()
            while env.status != 'done':
                agent = env.status == 'running'
                before = env.targets.copy()
                stage, action = 'policy' if agent else 'finalize', None
                if agent:
                    planned, phase = policy.action(env, env.elapsed_steps)
                    if planned is None:
                        env.stop(getattr(policy, 'stop_reason', None) or 'agent_stop')
                        continue
                    action, random = noise(planned)
                    stage = 'step'
                    env.step(action)
                else:
                    action, phase = env.finalize_step()
                    planned, random = action, False
                stage = 'snapshot'
                snapshot()
                actions.append(np.asarray(action, np.float32).copy())
                policy_actions.append(np.asarray(planned, np.float32).copy())
                replaced.append(random); phases.append(phase); agent_mask.append(agent)
                deltas.append(env.targets-before)
            stage = 'outcome'
            outcome = env.outcome()
            success = bool(full_stack_success(env))
        except Exception as exc:
            error = dict(type=type(exc).__name__, message=str(exc), stage=stage,
                         valid_transitions=len(actions), valid_observations=len(states),
                         failed_action=None if action is None else np.asarray(action).tolist(),
                         traceback=traceback.format_exc())
            outcome, success = None, None
        timing['simulate_s'] = time.time()-timing['start']
        n = len(actions)
        if len(states) > n+1:
            del rows[n+1:], states[n+1:], proprio[n+1:], hands[n+1:], times[n+1:], remaining[n+1:], \
                heights[n+1:], members[n+1:]
        k = len(env.config.objects) if env else 0

        def stack(values, shape, dtype=float):
            return np.stack(values) if values else np.empty((0, *shape), dtype=dtype)
        truth = {key: np.stack([r[key] for r in rows]) for key in rows[0]} if rows else {}
        goal = (env.config.goal_x, env.config.goal_y) if env else (0., 0.)
        half = env.config.goal_half_width if env else 0.
        executed = stack(actions, (25,), np.float32)
        reward = phi = shaping = labels = None
        summary = found = None
        if truth:
            final = None if error else outcome['final_stable_height_m']
            reward, phi = task_rewards(truth, goal, half, final)
            shaping = shaping_terms(truth, np.stack(hands), executed, goal, half)
            labels = stages(truth)
            found = events(truth, goal, half)
            if not error:
                summary = episode_summary(truth, goal, half, final, outcome['scored_object_count'], reward)
        stop_requested = np.zeros(len(states), bool)
        terminated, truncated = np.zeros(n, bool), np.zeros(n, bool)
        if error is None:
            if env.stop_reason == 'agent_stop':
                stop_requested[env.elapsed_steps] = True
            if n:
                terminated[-1] = env.stop_reason == 'agent_stop'
                truncated[-1] = env.stop_reason == 'time_limit'
        timing['labels_s'] = time.time()-timing['start']-timing['simulate_s']
        frames = None
        every = int(job.get('rgb_every', 1))
        if render and error is None:
            t0 = time.time()
            steps = rgb_steps(n, every)
            frames = render_states(env, [states[t] for t in steps])
            write_video(folder/'preview.mp4', frames, every)
            timing['render_s'] = time.time()-t0
        t0 = time.time()
        paths = dict(observations=folder/'observations.npz', privileged=folder/'privileged.npz',
                     metadata=folder/'metadata.json')
        np.savez_compressed(paths['observations'],
            **({'rgb': frames, 'rgb_steps': steps} if frames is not None else {}),
            proprio=stack(proprio, (75,)), time_remaining=np.asarray(remaining, np.float32),
            goal_xy=np.array(goal, np.float32), actions=executed,
            policy_actions=stack(policy_actions, (25,), np.float32),
            action_noise=executed-stack(policy_actions, (25,), np.float32),
            random_action=np.asarray(replaced, bool), applied_target_delta=stack(deltas, (25,)),
            agent_action_mask=np.asarray(agent_mask, bool), timestamps=np.asarray(times),
            reward=np.zeros(0, np.float32) if reward is None else reward,
            stop_requested=stop_requested, terminated=terminated, truncated=truncated)
        np.savez_compressed(paths['privileged'], **truth,
            integration_state=stack(states, (mujoco.mj_stateSize(env.model, STATE_SPEC) if env else 0,)),
            hand_point=stack(hands, (3,)), phases=np.asarray(phases, str),
            stable_height_m=np.asarray(heights, np.float32), stable_component_mask=stack(members, (k,), bool),
            instant_height_m=np.zeros(0, np.float32) if phi is None else phi,
            stage=np.zeros(0, np.int8) if labels is None else labels,
            shaping=np.zeros((0, len(SHAPING)), np.float32) if shaping is None else shaping)
        outcome_class = ('execution_error' if error else 'full_stack' if success else
                         'partial_stack' if outcome['final_stable_height_m'] > 0 else 'no_stable_height')
        meta = dict(schema=SCHEMA, episode_id=job['episode_id'], source=job['source'], condition=job['condition'],
            scene_seed=seed, split=scene_split(seed), policy=job['policy'], noise=noise.resolved(),
            noise_seed=derived_seed(job['episode_id'], 'noise'),
            policy_seed=derived_seed(job['episode_id'], 'policy'), start=job.get('start') or dict(kind='init'),
            provenance=job.get('provenance', {}),
            execution_status='error' if error else 'ok', outcome_class=outcome_class, outcome=outcome,
            success=success, summary=summary, events=found, resolution=job['resolution'], rgb_every=every,
            render_backend=os.environ.get('MUJOCO_GL'),
            time_limit=job['time_limit'], source_sha256=source_hashes(), mujoco=mujoco.__version__,
            control_steps=int(sum(agent_mask)), camera_names=list(CAMERAS), action_dt=DT, physics_dt=PHYSICS_DT,
            integration_state_spec=int(STATE_SPEC), stages=list(STAGES), shaping_terms=list(SHAPING),
            labeler_version=LABELER_VERSION,
            reward='instant supported-height progress; terminal step adds verified final height minus last potential')
        if env is not None:
            scene = asdict(env.config)
            legacy = {key: scene.pop(key) for key in ('cube_mass', 'cube_friction', 'cube_rgb')}
            meta.update(scene=scene, legacy_unused_scene_fields=legacy, physics=physics_metadata(env),
                material_catalog=MATERIALS, action_scale=env.action_scale.tolist(), joint_names=env.joint_names,
                actuator_limits=np.stack([env.low, env.high], axis=1).tolist(),
                mjcf_sha256=hashlib.sha256(env.xml.encode()).hexdigest())
        if policy is not None:
            meta.update(order=getattr(policy, 'order', None), controller_events=getattr(policy, 'events', None),
                        policy_parameters=getattr(policy, 'parameters', None),
                        intervention_step=getattr(policy, 'intervention_step', None))
        if error:
            paths['error'] = folder/'error.json'
            write_json(paths['error'], error)
        write_json(paths['metadata'], meta)
        if frames is not None:
            paths['video'] = folder/'preview.mp4'
        timing['write_s'] = time.time()-t0
        timing['total_s'] = time.time()-timing.pop('start')
        result = dict(episode_id=job['episode_id'], source=job['source'], condition=job['condition'],
            mode=job['condition'], scene_seed=seed, split=meta['split'], success=success,
            execution_status=meta['execution_status'], outcome_class=outcome_class, outcome=outcome,
            transitions=n, control_steps=meta['control_steps'], rendered=frames is not None,
            task_return=None if reward is None else float(reward.sum()), summary=summary,
            provenance=meta['provenance'], timing=timing,
            bytes={key: p.stat().st_size for key, p in paths.items()},
            **{key: str(p.relative_to(root)) for key, p in paths.items()},
            sha256={key: digest(p) for key, p in paths.items()})
        write_json(folder/'record.json', result)
        return result
    finally:
        if env is not None:
            env.close()


def validate_episode(root, record, rgb_samples=3):
    """Exact replay of executed actions plus recomputation of every derived label."""
    root = Path(root)
    for key, checksum in record['sha256'].items():
        if digest(root/record[key]) != checksum:
            raise ValueError(f'Checksum mismatch: {record[key]}')
    meta = json.loads((root/record['metadata']).read_text())
    env = VariedBoxEnv(config=scene_config(meta), resolution=meta['resolution'])
    try:
        assert hashlib.sha256(env.xml.encode()).hexdigest() == meta['mjcf_sha256']
        start = meta.get('start', {})
        if start.get('kind', 'init') != 'init':
            from assignment.training.data.task import restore_env
            with np.load(Path(start['root'])/start['privileged']) as f:
                restore_env(env, f['integration_state'], int(start['step']))
        with np.load(root/record['observations']) as f, np.load(root/record['privileged']) as g:
            obs = {k: f[k] for k in f.files}
            truth = {k: g[k] for k in g.files}
        n = len(obs['actions'])
        assert obs['proprio'].shape == (n+1, 75) and obs['actions'].shape == (n, 25)
        assert obs['reward'].shape == (n,) and truth['stage'].shape == (n+1,)
        assert np.max(np.abs(obs['actions'])) <= 1
        np.testing.assert_array_equal(obs['action_noise'], obs['actions']-obs['policy_actions'])
        assert np.array_equal(obs['agent_action_mask'], np.arange(n) < meta['control_steps'])
        assert not obs['action_noise'][~obs['agent_action_mask']].any()
        stopped = meta['outcome']['stop_reason'] == 'agent_stop'
        assert np.flatnonzero(obs['terminated']).tolist() == ([n-1] if stopped else [])
        assert np.flatnonzero(obs['truncated']).tolist() == ([] if stopped else [n-1])
        sampled = {}
        if 'rgb' in obs:
            steps = obs['rgb_steps'] if 'rgb_steps' in obs else np.arange(n+1)
            assert np.array_equal(steps, rgb_steps(n, meta.get('rgb_every', 1)))
            assert obs['rgb'].shape == (len(steps), len(CAMERAS), meta['resolution'], meta['resolution'], 3)
            picks = np.unique(np.linspace(0, len(steps)-1, rgb_samples).round().astype(int))
            sampled = {int(steps[i]): int(i) for i in picks}
        state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
        for t in range(n+1):
            mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
            np.testing.assert_array_equal(state, truth['integration_state'][t])
            np.testing.assert_array_equal(env.proprio(), obs['proprio'][t])
            np.testing.assert_array_equal(hand_point(env), truth['hand_point'][t])
            for key, value in env.privileged().items():
                np.testing.assert_array_equal(value, truth[key][t])
            if t in sampled:
                np.testing.assert_array_equal(env.render(), obs['rgb'][sampled[t]])
            if t == n:
                break
            before = env.targets.copy()
            if obs['agent_action_mask'][t]:
                env.step(obs['actions'][t])
            else:
                if env.status == 'running':
                    env.stop(meta['outcome']['stop_reason'])
                action, phase = env.finalize_step()
                np.testing.assert_array_equal(action, obs['actions'][t])
            np.testing.assert_array_equal(env.targets-before, obs['applied_target_delta'][t])
        assert env.outcome() == meta['outcome'] == record['outcome']
        assert bool(full_stack_success(env)) == meta['success']
        goal, half = tuple(obs['goal_xy'].tolist()), meta['scene']['goal_half_width']
        reward, phi = task_rewards(truth, goal, half, meta['outcome']['final_stable_height_m'])
        np.testing.assert_array_equal(reward, obs['reward'])
        np.testing.assert_array_equal(phi, truth['instant_height_m'])
        np.testing.assert_array_equal(stages(truth), truth['stage'])
        assert events(truth, goal, half) == meta['events']
        np.testing.assert_allclose(reward.sum(), meta['outcome']['final_stable_height_m']-phi[0], atol=1e-6)
        if 'video' in record:
            reader = imageio.get_reader(root/record['video'])
            try:
                assert reader.count_frames() == len(obs['rgb'])
            finally:
                reader.close()
        return dict(episode_id=record['episode_id'], transitions=n, replay='exact',
                    sampled_rgb=len(sampled), labels='exact')
    finally:
        env.close()


def relabel_episode(root, record):
    """Recompute events/summary with the current labeler; physics data are untouched."""
    root = Path(root)
    meta = json.loads((root/record['metadata']).read_text())
    if meta.get('labeler_version') == LABELER_VERSION:
        return record
    with np.load(root/record['privileged']) as f:
        truth = {k: f[k] for k in f.files if k.startswith('object_')}
        stage_labels = f['stage']
    with np.load(root/record['observations']) as f:
        reward, goal = f['reward'], tuple(f['goal_xy'].tolist())
    half = meta['scene']['goal_half_width']
    assert np.array_equal(stages(truth), stage_labels)
    meta['events'] = events(truth, goal, half)
    meta['summary'] = episode_summary(truth, goal, half, meta['outcome']['final_stable_height_m'],
                                      meta['outcome']['scored_object_count'], reward)
    meta['labeler_version'] = LABELER_VERSION
    write_json(root/record['metadata'], meta)
    record = dict(record, summary=meta['summary'], sha256=dict(record['sha256'], metadata=digest(root/record['metadata'])),
                  bytes=dict(record['bytes'], metadata=(root/record['metadata']).stat().st_size))
    write_json((root/record['metadata']).parent/'record.json', record)
    return record


def load_manifest(root):
    return json.loads((Path(root)/'manifest.json').read_text())


def load_episode(root, record, include_privileged=False):
    """Student loader: observations, executed actions, rewards and termination flags."""
    if record.get('execution_status') == 'error':
        raise ValueError('Execution errors are audit records, not training data')
    root = Path(root)
    with np.load(root/record['observations']) as f:
        data = {k: f[k] for k in f.files}
    if include_privileged:
        with np.load(root/record['privileged']) as f:
            data['privileged'] = {k: f[k] for k in f.files}
    return data


def select_episodes(roots, split=None, weights=None, sources=None, conditions=None):
    """Episodes from one or more dataset roots, filtered and tagged with their root and mix weight."""
    roots = [roots] if isinstance(roots, (str, Path)) else list(roots)
    weights = weights or [1.]*len(roots)
    chosen = []
    for root, weight in zip(roots, weights):
        for r in load_manifest(root)['episodes']:
            if ((split is None or r['split'] == split) and (sources is None or r['source'] in sources)
                    and (conditions is None or r['condition'] in conditions)):
                chosen.append(dict(r, root=str(root), mix_weight=weight))
    return chosen
