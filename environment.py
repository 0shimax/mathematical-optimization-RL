import numpy as np
from chainerrl.env import Env


class Space(object):

    def __init__(self, space_dim, low=None, high=None):
        self.space = np.zeros((space_dim, 1), dtype=np.float32)
        self.space_dim = space_dim
        if low is not None:
            self.low = low
            self.high = high
        self.shape = self.space.shape

    def sample(self):
        return np.random.uniform(self.low, self.high, self.space_dim)

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
        x = self.action_space.sample()
        y = -1*(np.cos(3*x) + x**2 - x)
        self.pre_obs = [x, y]
        return np.asarray([self.pre_obs], dtype=np.float32)
        # self.pre_obs = np.asarray([[x, y]], dtype=np.float32)
        # return self.pre_obs

    def reword(self, x):
        r = -1*(np.cos(3*x) + x**2 - x)
        reword = r
        self.pre_obs = np.asarray([x, r], dtype=np.float32)
        # self.pre_obs[:, 0] = x
        # self.pre_obs[:,1] = r
        return np.asarray([[reword]], dtype=np.float32)

    def seed(self, seed):
        self.action_space.set_seed(seed)

    def step(self, action):
        self.pre_obs[0] += action
        print("self.pre_obs0", self.pre_obs)
        reword = self.reword(self.pre_obs[0])
        print("self.pre_obs1", self.pre_obs)
        # done = False if np.isnan(reword) or np.isinf(reword) else True
        done = True
        return self.pre_obs, reword, done, None

    def close(self):
        pass


class Griewank(Env):

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
        x = self.action_space.sample()
        cos_x = x/np.sqrt(np.arange(len(x))+1)
        r = np.sum(np.power(x, 2))/4000 - np.prod(np.cos(cos_x))
        y = -r
        self.pre_obs = [*x, y]
        return np.asarray([self.pre_obs], dtype=np.float32)

    def reword(self, x):
        cos_x = x/np.sqrt(np.arange(len(x))+1)
        r = np.sum(np.power(x, 2))/4000 - np.prod(np.cos(cos_x))
        r = -r
        reword = r
        self.pre_obs = np.asarray([*x, r], dtype=np.float32)
        return np.asarray([[reword]], dtype=np.float32)

    def seed(self, seed):
        self.action_space.set_seed(seed)

    def step(self, action):
        print("action:", action.shape)
        print("self.pre_obs0", len(self.pre_obs))
        self.pre_obs[:-1] += action
        print("self.pre_obs1", len(self.pre_obs))
        reword = self.reword(self.pre_obs[:-1])
        print("self.pre_obs2", len(self.pre_obs))
        done = True
        return self.pre_obs, reword, done, None

    def close(self):
        pass
