import numpy as np
import random
from .music_theory_env import MusicTheoryEnv
from .util import NUM_CLASSES, NOTES_PER_BAR

from rl import Memory

class MusicTunerEnv(MusicTheoryEnv):
    """
    Music environment that combines tuning rewards and theory rewards
    https://arxiv.org/pdf/1611.02796v4.pdf
    """
    def __init__(self, g_rnn, model, theory_scalar=1, time_steps=8, preprocess=lambda x: x):
        super().__init__()
        # Separate the graph!
        self.g_rnn = g_rnn
        self.model = model
        self.preprocess = preprocess
        self.theory_scalar = theory_scalar
        self.time_steps = time_steps
        self.memory = Memory(self.time_steps)

    def _reset(self):
        state = super()._reset()
        self.memory.reset(self.preprocess(self, state))
        return state

    def _step(self, action):
        # Ask the Melody RNN to make a prediction
        with self.g_rnn.as_default():
            s = [np.array([s_i]) for s_i in self.memory.to_states()]
            predictions = self.model.predict(s)[0]
            norm_constant = np.log(np.sum(np.exp(predictions)))
            # np.log(np.clip(predictions[action], 1e-20, 1))
            prob = predictions[action] - norm_constant

        # Compute music theory rewards
        state, reward, done, info = super()._step(action)

        # Total reward = log(P(a | s)) + r_TM * c
        reward = prob + reward * self.theory_scalar

        self.memory.remember(self.preprocess(self, state))
        return state, reward, done, info
