import random
from gym import Env, spaces
from .util import *


class MusicEnv(Env):
    """
    An abstract music environment for music melody composition.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.observation_space = spaces.Tuple((
            spaces.Discrete(NUM_CLASSES),
            spaces.Discrete(NOTES_PER_BAR)
        ))
        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(2) for _ in range(NUM_CLASSES))
        )

        # Total number of notes
        self.num_notes = 32
        self.key = C_MAJOR_KEY

    def _step(self, action):
        """
        Args:
            action: An integer that represents the note chosen
        """
        self.composition.append(action)
        self.beat += 1
        return self._current_state(), 0, self.beat == self.num_notes, {}

    def _reset(self):
        # Composition is a list of notes composed
        self.composition = [random.choice(self.key)]
        self.beat = 0
        return self._current_state()

    def _current_state(self):
        return (self.composition[-1], self.beat % NOTES_PER_BAR)

    def _render(self, mode='human', close=False):
        pass
