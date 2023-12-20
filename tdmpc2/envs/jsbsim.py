from collections import deque, defaultdict
from typing import Any, NamedTuple
import numpy as np
from envs.exceptions import UnknownTaskError
import gymnasium as gym
import jsbsim_gym


class ExtendedTimeStep(NamedTuple):
	step_type: Any
	reward: Any
	discount: Any
	observation: Any
	action: Any

	def first(self):
		return self.step_type == StepType.FIRST

	def mid(self):
		return self.step_type == StepType.MID

	def last(self):
		return self.step_type == StepType.LAST



class TimeStepToGymWrapper:
	def __init__(self, env, domain, task):
		obs_shp = []
		for v in env.observation_spec().values():
			try:
				shp = np.prod(v.shape)
			except:
				shp = 1
			obs_shp.append(shp)
		obs_shp = (int(np.sum(obs_shp)),)
		act_shp = env.action_spec().shape
		self.observation_space = gym.spaces.Box(
			low=np.full(
				obs_shp,
				-np.inf,
				dtype=np.float32),
			high=np.full(
				obs_shp,
				np.inf,
				dtype=np.float32),
			dtype=np.float32,
		)
		self.action_space = gym.spaces.Box(
			low=np.full(act_shp, env.action_spec().minimum),
			high=np.full(act_shp, env.action_spec().maximum),
			dtype=env.action_spec().dtype)
		self.env = env
		self.domain = domain
		self.task = task
		self.max_episode_steps = 500
		self.t = 0
	
	@property
	def unwrapped(self):
		return self.env

	@property
	def reward_range(self):
		return None

	@property
	def metadata(self):
		return None
	
	def _obs_to_array(self, obs):
		return np.concatenate([v.flatten() for v in obs.values()])

	def reset(self):
		self.t = 0
		return self._obs_to_array(self.env.reset().observation)
	
	def step(self, action):
		self.t += 1
		time_step = self.env.step(action)
		return self._obs_to_array(time_step.observation), time_step.reward, time_step.last() or self.t == self.max_episode_steps, defaultdict(float)

	def render(self, mode='rgb_array', width=384, height=384, camera_id=0):
		camera_id = dict(quadruped=2).get(self.domain, camera_id)
		return self.env.physics.render(height, width, camera_id)

class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)



def make_env(cfg):
    """
    Make JSBSim environment.
    """
    domain, task = cfg.task.replace('-', '_').split('_', 1)
    env = gym.make(task, config={"root": cfg.root})
    env.max_episode_steps = 6001


    # env = ActionDTypeWrapper(env, np.float32)
    env = NormalizeActions(env)
    # env = ActionRepeatWrapper(env, 2)
    # env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
    # env = ExtendedTimeStepWrapper(env)
    # env = TimeStepToGymWrapper(env, domain, task)
    return env

