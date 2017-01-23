import random
from gym import Env, spaces
from .util import *
from .music_env import MusicEnv

target_compositions = []

class MusicGenEnv(MusicEnv):
    """
    A music environment for music melody generation.
    """
    def _step(self, action):
        # Force play the first couple of notes as inspiration
        if self.beat < 4:
            action = self.target_composition[self.beat]

        state, reward, done, info = super()._step(action)
        return state, reward, done, {}

    def _reset(self):
        # Composition is a list of notes composed
        # TODO: Avoid globals
        # A global list of compositions used as example for training
        global target_compositions
        self.target_composition = random.choice(target_compositions)
        self.composition = [self.target_composition[0]]
        self.beat = 0
        return self._current_state()
