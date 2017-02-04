import unittest
from music.util import *

class Test(unittest.TestCase):
    def test_similarity(self):
        self.assertEqual(similarity([], []), 0)
        self.assertEqual(similarity([], [1]), 0)
        self.assertEqual(similarity([1], [1]), 1)
        self.assertEqual(similarity([1, 2], [1]), 1)
        self.assertEqual(similarity([1, 2, 3], [2, 3]), 2)
        self.assertEqual(similarity([1, 2, 3], [1, 2, 3]), 3)
        self.assertEqual(similarity([1, 2, 2, 3], [1, 2, 3]), 2)
        self.assertEqual(similarity([1, 2, 3, 2, 3], [1, 2, 3]), 3)
        self.assertEqual(similarity([2, 4, 78, 2, 1], [2]), 1)
        self.assertEqual(similarity([2, 4, 78, 2, 1], [0]), 0)

if __name__ == '__main__':
    unittest.main()
