import gym
import numpy as np
from gym import spaces


class DictToFlatWrapper(gym.ObservationWrapper):
    """Flatten dict observations by concatenating values in insertion order.

    Unlike gym.wrappers.FlattenObservation which sorts keys alphabetically,
    this wrapper preserves the original dict key order. This is critical
    when training data was generated with np.concatenate(obs.values()).
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict)
        # Preserve insertion order of the OrderedDict
        self._keys = list(env.observation_space.spaces.keys())
        sizes = []
        for key in self._keys:
            space = env.observation_space.spaces[key]
            sizes.append(int(np.prod(space.shape)))
        total = sum(sizes)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total,), dtype=np.float32
        )

    def observation(self, observation):
        return np.concatenate(
            [np.atleast_1d(observation[k]).flatten() for k in self._keys]
        )
