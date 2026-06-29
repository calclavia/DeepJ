#!/usr/bin/env python3
"""Print DeepJ composer style indices (no tensorflow/midi required)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from style_registry import list_styles_text

if __name__ == '__main__':
    print(list_styles_text())