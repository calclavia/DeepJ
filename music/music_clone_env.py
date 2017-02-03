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
            last_notes = self.compositions[:-3]

            for target in self.targets:
                if is_sublist(target, last_notes):
                    reward += 1
                    break

        return state, reward, done, info

    def _reset(self):
        self.target_composition = random.choice(target_compositions)
        self.num_notes = len(self.target_composition)
        # Every correct prediction gets 1 reward
        self.reward_amount = 1. / (self.num_notes - 1)
        self.beat = 0
        return self._current_state()

    def _current_state(self):
        return (self.target_composition[self.beat], self.beat % NOTES_PER_BAR)
