#!/usr/bin/env python3
"""Compatibility wrapper for assignment.training.training.train_jepa."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assignment.training.training.train_jepa import *  # noqa: F401,F403,E402

if __name__ == '__main__':
    main()
