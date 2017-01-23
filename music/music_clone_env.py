import numpy as np
import random
from .music_env import MusicEnv
from .util import NOTES_PER_BAR

target_compositions = []

class MusicCloneEnv(MusicEnv):
    """
    Music environment that attempts to clone an existing melody.
    """
    def _step(self, action):
        reward = 0
        # The target we want our action to match.
        target = self.target_composition[self.beat + 1]

        # Award for action matching example composition
        if action == target:
            reward += self.reward_amount

        state = self._current_state()
        done = self.beat == len(self.target_composition) - 2
        self.beat += 1
        return state, reward, done, {}

    def _reset(self):
        # TODO: Avoid globals
        # A global list of compositions used as example for training
        global target_compositions
        self.target_composition = random.choice(target_compositions)
        self.num_notes = len(self.target_composition)
        # Every correct prediction gets 1 reward
        self.reward_amount = 1. / (self.num_notes - 1)
        self.beat = 0
        return self._current_state()

    def _current_state(self):
        return (self.target_composition[self.beat], self.beat % NOTES_PER_BAR)
