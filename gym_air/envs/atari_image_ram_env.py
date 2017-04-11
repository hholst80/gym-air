import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

try:
    import atari_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you can install Atari dependencies by running 'pip install "
        "gym[atari]'.)".format(e))

import logging
logger = logging.getLogger(__name__)


class AIREnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game='pong', frameskip=(2, 5), use_tuple=True,
                 repeat_action_probability=0.):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        utils.EzPickle.__init__(self, game)

        self.game_path = atari_py.get_game_path(game)
        if not os.path.exists(self.game_path):
            raise IOError('You asked for game %s but path %s does not '
                          'exist' % (game, self.game_path))
        self.frameskip = frameskip
        self.ale = atari_py.ALEInterface()
        self.viewer = None

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        assert isinstance(repeat_action_probability, (float, int)), \
            "Invalid repeat_action_probability: {!r}" \
            .format(repeat_action_probability)
        self.ale.setFloat('repeat_action_probability'.encode('utf-8'),
                          repeat_action_probability)

        self._seed()

        (screen_width, screen_height) = self.ale.getScreenDims()
        self._buffer = np.empty((screen_height, screen_width, 4),
                                dtype=np.uint8)

        self._action_set = self.ale.getMinimalActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))

        (screen_width, screen_height) = self.ale.getScreenDims()
        obs_image = spaces.Box(low=0, high=255, shape=(screen_height,
                                                       screen_width, 3))
        obs_ram = spaces.Box(low=0, high=255, shape=(128,))

        if use_tuple:
            self.observation_space = spaces.Tuple((obs_image, obs_ram))
        else:
            self.observation_space = obs_image

        self._use_tuple = use_tuple

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b'random_seed', seed2)
        self.ale.loadROM(self.game_path)
        return [seed1, seed2]

    def _step(self, a):
        reward = 0.0
        action = self._action_set[a]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0],
                                               self.frameskip[1])
        for _ in range(num_steps):
            reward += self.ale.act(action)

        obs = self._get_obs()

        done = self.ale.game_over()

        info = ({'ale.lives': self.ale.lives(), 'ram': self._get_ram()} if not
                self._use_tuple else {'ale.lives': self.ale.lives()})

        return obs, reward, done, info

    def _get_image(self):
        return self.ale.getScreenRGB2()

    def _get_ram(self):
        ram_size = self.ale.getRAMSize()
        ram = np.zeros((ram_size), dtype=np.uint8)
        self.ale.getRAM(ram)
        return ram

    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        if self._use_tuple:
            return (self._get_image(), self._get_ram())
        else:
            return self._get_image()

    # return: (states, observations)
    def _reset(self):
        self.ale.reset_game()
        return self._get_obs()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}
