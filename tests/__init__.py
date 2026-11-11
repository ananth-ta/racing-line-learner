"""Unit tests. Run from the repo root with: python -m unittest discover -s tests -t ."""

import os
import sys

# Make ``src/`` importable without installing the package.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
