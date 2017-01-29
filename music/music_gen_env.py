import random
from gym import Env, spaces
from .util import *
from .music_env import MusicEnv

class MusicGenEnv(MusicEnv):
    """
    A music environment for music melody generation.
    """

    def __init__(self, inspiration):
        """
        Args
            inspiration: One bar of inspirational melody to prime the
                         composition
        """
        super().__init__()
        self.inspiration = inspiration
        self.num_notes += NOTES_PER_BAR

    def _step(self, action):
        # Force play the first couple of notes as inspiration
        if self.beat < NOTES_PER_BAR:
            action = self.inspiration[self.beat]

        return super()._step(action)

    def _reset(self):
        # Composition is a list of notes composed
        self.composition = [self.inspiration[0]]
        self.beat = 0
        return self._current_state()
