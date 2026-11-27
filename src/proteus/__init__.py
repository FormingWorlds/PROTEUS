from __future__ import annotations

from .utils.julia_env import set_julia_threads

# Before the first import of juliacall, which starts Julia with these settings
set_julia_threads()

from .proteus import Proteus  # noqa: E402

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    # Fallback for source-only runs where setuptools-scm has not yet
    # generated _version.py (e.g. inspecting the tree without an editable
    # install). The published wheel and `pip install -e .` both write
    # _version.py, so this branch only fires on bare-clone introspection.
    __version__ = '0.0.0.dev0'
    __version_tuple__ = (0, 0, 0, 'dev0')

__all__ = ['Proteus', '__version__', '__version_tuple__']
