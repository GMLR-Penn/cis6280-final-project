"""Native variable-footprint box/intervention demo; use mjpython on macOS for the viewer."""
import argparse
import json
import time

from cube_pick.box_stacking import full_stack_success
from cube_pick.varied_boxes import VariedBoxEnv, VariedBoxController, MODES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=2000)
    p.add_argument('--mode', choices=MODES, default='nominal')
    p.add_argument('--time-limit', type=float, default=180.)
    p.add_argument('--speed', type=float, default=1.)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    if not 0 < args.speed <= 100:
        p.error('--speed must be in (0, 100]')
    env = VariedBoxEnv(seed=args.seed, time_limit=args.time_limit)
    policy = VariedBoxController(env, args.mode, args.seed)

    def advance():
        if env.status == 'running':
            action, _ = policy.action(env, env.elapsed_steps)
            if action is None:
                env.stop()
            else:
                env.step(action)
        else:
            env.finalize_step()

    try:
        if args.headless:
            while env.status != 'done':
                advance()
        else:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                viewer.cam.lookat[:] = [-.04, 0, .16]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.15, 130, -30
                viewer.opt.geomgroup[3] = 0
                while viewer.is_running() and env.status != 'done':
                    start = time.perf_counter()
                    with viewer.lock():
                        advance()
                    viewer.sync()
                    time.sleep(max(0, .1/args.speed-(time.perf_counter()-start)))
                if env.status != 'done':
                    return
                print(json.dumps(dict(**env.outcome(), full_stack_success=full_stack_success(env))), flush=True)
                while viewer.is_running() and not args.once:
                    viewer.sync()
                    time.sleep(.05)
                return
        print(json.dumps(dict(**env.outcome(), full_stack_success=full_stack_success(env))), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
