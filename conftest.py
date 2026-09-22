"""
Repository-level pytest bootstrap.

Hoist editable-install finders ahead of ``PathFinder`` on ``sys.meta_path``
so that ``import aragog`` resolves to the installed ``fwl-aragog`` package
rather than the gitignored ``aragog/`` dev checkout that sits at the repo
root. Without this, ``PathFinder`` finds ``aragog/`` as a namespace package
during pytest collection (the cwd entry in ``sys.path`` matches the directory
name exactly), shadowing the real package and breaking every test that does
``from aragog import aragog_file_logger``.

The same shadowing does not affect ``Zalmoxis``, ``CALLIOPE``, ``JANUS``,
``MORS`` because those dev checkouts use mixed-case directory names that do
not match the lowercase Python import names.
"""

from __future__ import annotations

import sys


def _hoist_editable_finders() -> None:
    from importlib.machinery import PathFinder

    try:
        path_finder_idx = next(i for i, f in enumerate(sys.meta_path) if f is PathFinder)
    except StopIteration:
        return

    to_hoist = [
        f
        for i, f in enumerate(sys.meta_path)
        if i > path_finder_idx and '__editable__' in getattr(f, '__module__', '')
    ]
    if not to_hoist:
        return

    remaining = [f for f in sys.meta_path if f not in to_hoist]
    insert_at = remaining.index(PathFinder)
    remaining[insert_at:insert_at] = to_hoist
    sys.meta_path = remaining


_hoist_editable_finders()
