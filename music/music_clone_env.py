import numpy as np
import random
from .music_env import MusicEnv
from .util import NOTES_PER_BAR, similarity, is_sublist


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
        if self.beat >= 3:  # and self.beat % 3 == 0:
            last_notes = self.composition[-8:]
            """
            for t in self.targets:
                 if is_sublist(t, last_notes):
                     reward += 1
                     break
            """
            for t in self.targets:
                similar = similarity(t, last_notes)

                if similar >= 3:
                    reward += 1 #max(reward, similar - 2)
                    break

        return state, reward, done, info
