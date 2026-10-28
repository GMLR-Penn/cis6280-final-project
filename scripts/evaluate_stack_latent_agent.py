#!/usr/bin/env python3
"""Compatibility wrapper for assignment.training.planning.evaluate."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assignment.training.planning.evaluate import *  # noqa: F401,F403,E402

if __name__ == '__main__':
    main()
