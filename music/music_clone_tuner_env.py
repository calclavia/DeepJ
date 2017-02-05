import numpy as np
import random
from .music_env import MusicEnv
from .music_theory_env import MusicTheoryEnv
from .music_clone_env import MusicCloneEnv
from .util import NUM_CLASSES, NOTES_PER_BAR


class MusicCloneTunerEnv(MusicEnv):
    """
    Music environment that combines tuning rewards and theory rewards
    https://arxiv.org/pdf/1611.02796v4.pdf
    """

    def __init__(self, targets, theory_scalar=1, preprocess=lambda x: x):
        super().__init__()
        self.theory = MusicTheoryEnv()
        self.clone = MusicCloneEnv(targets)
        self.theory_scalar = theory_scalar

    def _step(self, action):
        # Compute music theory rewards
        super()._step(action)
        state, clone_reward, done, info = self.clone._step(action)
        state, theory_reward, done, info = self.theory._step(action)

        reward = clone_reward + theory_reward * self.theory_scalar
        return state, reward, done, info

    def _reset(self):
        self.clone.reset()
        self.theory.reset()
        return super()._reset()
