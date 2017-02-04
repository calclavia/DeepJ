import numpy as np
import random
from .music_env import MusicEnv
from .util import NOTES_PER_BAR, similarity


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
        # TODO: Avoid local optima?
        for t in self.targets:
            similar = similarity(t, self.composition[-8:])

            if similar >= 3:
                reward += similar

        reward /= len(self.targets)

        return state, reward, done, info
