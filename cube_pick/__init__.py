"""Visual cube pick-and-place data prototype; independent of the Sharpa task."""
__all__ = ["CubePickEnv"]


def __getattr__(name):
    if name == "CubePickEnv":
        from .env import CubePickEnv
        return CubePickEnv
    raise AttributeError(name)
