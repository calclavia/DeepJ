import numpy as np
import random
from .music_theory_env import MusicTheoryEnv
from .util import NOTES_PER_BAR

# Keras supervised model
supervised_model = None

class MusicTunerEnv(MusicTheoryEnv):
    """
    Music environment that combines tuning rewards and theory rewards
    https://arxiv.org/pdf/1611.02796v4.pdf
    """
    def _step(self, action):
        # TODO: Avoid globals!
        global supervised_model

        # Ask the Melody RNN to make a prediction
        s = [[i] for i in self._current_state()]
        predictions = supervised_model.predict(s)[0]
        prob = predictions[action]

        # Compute music theory rewards
        state, reward, done, info = super()_.step(action)

        # Total reward = log(P(a | s)) + r_TM / c
        reward = np.log(prob) + reward

        return state, reward, done, info
