import numpy as np
import random
from .music_theory_env import MusicTheoryEnv
from .util import NUM_CLASSES, NOTES_PER_BAR

class MusicTunerEnv(MusicTheoryEnv):
    """
    Music environment that combines tuning rewards and theory rewards
    https://arxiv.org/pdf/1611.02796v4.pdf
    """

    def __init__(self,
                 g_rnn,
                 model,
                 memory,
                 theory_scalar=1,
                 preprocess=lambda x: x):
        super().__init__()
        # Separate the graph!
        self.g_rnn = g_rnn
        self.model = model
        self.preprocess = preprocess
        self.theory_scalar = theory_scalar
        self.memory = memory

    def _reset(self):
        state = super()._reset()
        self.memory.reset(self.preprocess(self, state))
        return state

    def _step(self, action):
        # Ask the Melody RNN to make a prediction
        #with self.g_rnn.as_default():
        s = [np.array([s_i]) for s_i in self.memory.to_states()]
        predictions = self.model.predict(s)[0]
        norm_constant = np.log(np.sum(np.exp(predictions)))
        # np.log(np.clip(predictions[action], 1e-20, 1))
        prob = predictions[action] - norm_constant

        # Compute music theory rewards
        state, reward, done, info = super()._step(action)

        # Total reward = log(P(a | s)) + r_TM
        reward = prob + reward * self.theory_scalar

        self.memory.remember(self.preprocess(self, state))
        return state, reward, done, info
