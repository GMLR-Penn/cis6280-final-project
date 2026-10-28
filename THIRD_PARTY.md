# Third-Party Assets and Licenses

The Sharpa Wave five-finger hand model and meshes are downloaded from
[sharpa-robotics/sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml)
at commit `0d19cac602f46456b819e4b6a2c09a74982c9a3e`.
Its Apache-2.0 license and notices are preserved by `scripts/fetch_assets.py`
under `assets/sharpa/LICENSE.txt` and `assets/sharpa/NOTICE.txt`.
The downloaded mesh and XML files are fetched on setup by `scripts/fetch_assets.py`
and are not committed directly to Git.

The JEPA world-model architecture in `assignment/training/training/jepa.py` adapts components
from the reference implementation of LeWM ([lucas-maes/le-wm](https://github.com/lucas-maes/le-wm),
MIT License, Copyright (c) 2026 Lucas Maes). MuJoCo, Gymnasium, PyTorch, and other
dependencies retain their respective licenses.
