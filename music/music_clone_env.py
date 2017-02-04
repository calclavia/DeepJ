import numpy as np
import random
from .music_env import MusicEnv
from .util import NOTES_PER_BAR, is_sublist

class MusicCloneEnv(MusicEnv):
    """
    Music environment that attempts to clone existing melodies.
    """

    def __init__(self, targets):
        """
        Args:
            targets: A global list of compositions used as example for training
                     This is a list of actions that should have been taken.
                     Integers of preprocessed melodies
        """
        super().__init__()
        self.targets = targets

    def _step(self, action):
        state, reward, done, info = super()._step(action)

        # Reward by pattern matching with target compositions.
        # A match is defined as 3 consecutive notes.
        if self.beat >= 3:
            last_notes = self.composition[:-3]

            for target in self.targets:
                if is_sublist(target, last_notes):
                    reward += 1
                    break

        return state, reward, done, info
