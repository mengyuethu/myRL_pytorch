from atari_wrappers_old import *


def get_env(name_of_game):
    env = gym.make(name_of_game)
    if env.unwrapped.__class__.__name__ == 'AtariEnv':
        env = wrap_deepmind(env)
    return env


class Game(object):

    def __init__(self, name_of_game):
        super(Game, self).__init__()

        self.env = get_env(name_of_game)
        if self.env.action_space.__class__.__name__ == 'Discrete':
            self.num_action = self.env.action_space.n
            self.action_space = range(self.num_action)
        elif self.env.action_space.__class__.__name__ == 'Box':
            self.dim_action = self.env.action_space.shape
            self.action_interval = np.array([self.env.action_space.low, self.env.action_space.high])
        self.dim_obs = self.env.observation_space.shape
        self.obs_interval = np.array([self.env.observation_space.low, self.env.observation_space.high])
