import random
from gym import Env, spaces
from .util import *
from .music_env import MusicEnv

class MusicGenEnv(MusicEnv):
    """
    A music environment for music melody generation.
    """

    def __init__(self, target_composition):
        super().__init__()
        self.target_composition = target_composition

    def _step(self, action):
        # Force play the first couple of notes as inspiration
        if self.beat < NOTES_PER_BAR:
            action = self.target_composition[self.beat]

        return super()._step(action)

    def _reset(self):
        # Composition is a list of notes composed
        self.composition = [self.target_composition[0]]
        self.beat = 0
        return self._current_state()
