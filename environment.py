import numpy as np
from chainerrl.env import Env


class Space(object):

    def __init__(self, space_dim, low=None, high=None):
        self.space = np.zeros((space_dim, 1), dtype=np.float32)
        if low is not None:
            self.low = low
            self.high = high
        self.shape = self.space.shape

    def sample(self):
        return np.random.uniform(self.low, self.high)

    def set_seed(self, seed):
        np.random.seed(seed)


class Easy2D(Env):

    def __init__(self, test=False,
                 obs_dim=1, action_dim=1,
                 action_low=-10, action_high=10,
                 timestep_limit=1000):
        super().__init__()
        self.action_space = \
            Space(space_dim=action_dim, low=action_low, high=action_high)
        self.observation_space = Space(space_dim=obs_dim)
        self.test = test
        self.timestep_limit = timestep_limit

        self.pre_obs = None
        self.pre_action = None

    def reset(self):
        # self.pre_obs = -np.finfo(np.float32).max/1e20
        x = self.action_space.sample()
        y = -1*(np.cos(3*x) + x**2 - x)
        self.pre_obs = [x, y]
        # self.pre_obs = [y]
        return np.asarray([self.pre_obs], dtype=np.float32)

    def reword(self, x):
        r = -1*(np.cos(3*x) + x**2 - x)
        # reword = 100 if r > self.pre_obs or r > -5 else -100
        reword = r
        self.pre_obs = np.asarray([x, r], dtype=np.float32)
        return np.asarray([[reword]], dtype=np.float32)

    def seed(self, seed):
        self.action_space.set_seed(seed)

    def step(self, action):
        reword = self.reword(action)
        done = False if np.isnan(reword) or np.isinf(reword) else True
        return self.pre_obs, reword, done, None

    def close(self):
        pass
