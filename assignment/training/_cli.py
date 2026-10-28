"""Run an existing script entry point as a training-track stage."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(script):
    path = ROOT / 'scripts' / script
    for entry in (str(ROOT), str(path.parent)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module  # worker processes unpickle functions by this module name
    spec.loader.exec_module(module)
    return module.main()
