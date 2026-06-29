"""Portability tests that do not require tensorflow or python-midi."""
import unittest

from style_registry import flatten_styles, list_styles_text


class TestStyleRegistry(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(flatten_styles()), 23)

    def test_list_contains_bach(self):
        self.assertIn('bach', list_styles_text())


if __name__ == '__main__':
    unittest.main()