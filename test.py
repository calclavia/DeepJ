import unittest
from music.util import *

class Test(unittest.TestCase):
    def test_sublist(self):
        self.assertEqual(is_sublist([], []), None)
        self.assertEqual(is_sublist([], [1]), None)
        self.assertEqual(is_sublist([1], [1]), 0)
        self.assertEqual(is_sublist([1, 2], [1]), 0)
        self.assertEqual(is_sublist([1, 2, 3], [2, 3]), 1)
        self.assertEqual(is_sublist([1, 2, 3], [1, 2, 3]), 0)
        self.assertEqual(is_sublist([1, 2, 2, 3], [1, 2, 3]), None)

if __name__ == '__main__':
    unittest.main()
