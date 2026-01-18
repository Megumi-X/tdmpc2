import gymnasium as gym

from envs.wrappers.timeout import Timeout
from envs.envs_baselines.fly_walk_all_free import FlyWalkAllFreeEnv
from envs.envs_baselines.fly_walk_pit_all_free import FlyWalkPitAllFreeEnv
from envs.envs_baselines.roll_plane_all_free import RollPlaneAllFreeEnv
from envs.envs_baselines.car_leg_all_free import CarLegAllFreeEnv
from envs.envs_baselines.car_leg_narrow_all_free import CarLegNarrowAllFreeEnv
from envs.envs_baselines.wheel_leg_all_free import WheelLegAllFreeEnv


BASELINE_TASKS = {
    'fly-walk-all-free': FlyWalkAllFreeEnv,
    'fly-walk-pit-all-free': FlyWalkPitAllFreeEnv,
    'roll-plane-all-free': RollPlaneAllFreeEnv,
    'car-leg-all-free': CarLegAllFreeEnv,
    'car-leg-narrow-all-free': CarLegNarrowAllFreeEnv,
    'wheel-leg-all-free': WheelLegAllFreeEnv,
}
XMLS = {
    'fly-walk-all-free': 'fly_plane_real_local',
    'fly-walk-pit-all-free': 'fly_plane_pit_local',
    'roll-plane-all-free': 'roll_plane',
    'car-leg-all-free': 'car_leg_neo_local',
    'car-leg-narrow-all-free': 'car_leg_narrow_local',
    'wheel-leg-all-free': 'wheel_leg',
}


class GymnasiumToTDWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
        else:
            obs = out
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info = dict(info)
        info.setdefault('success', 0.0)
        info['terminated'] = terminated
        return obs, reward, done, info


def make_env(cfg):
    if cfg.task not in BASELINE_TASKS:
        raise ValueError('Unknown task:', cfg.task)
    assert cfg.obs == 'state', 'This task only supports state observations.'
    env_name = cfg.get('env_name', XMLS[cfg.task])
    env = BASELINE_TASKS[cfg.task](env_name=env_name, settings=None)
    env = GymnasiumToTDWrapper(env)
    env = Timeout(env, max_episode_steps=env.max_episode_steps)
    return env
