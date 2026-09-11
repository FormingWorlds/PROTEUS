# Julia helper functions shared between julia based modules.
from __future__ import annotations

import os

import juliacall
import numpy as np
from juliacall import Main as jl

from proteus.utils.logs import GetCurrentLogfileIndex, GetLogfilePath


def to_julia_dict(obj):
    """Recursively convert Python dict/list to native Julia Dict/Vector."""
    if isinstance(obj, dict):
        jd = jl.Dict()
        for k, v in obj.items():
            jd[k] = to_julia_dict(v)
        return jd
    elif isinstance(obj, list):
        return [to_julia_dict(v) for v in obj]
    else:
        return obj


def make_julia_converters(jl_module_name: str):
    """Build array/scalar-to-Julia converters bound to one Julia
    submodule's own precision type.

    Parameters
    ----------
        jl_module_name : str
            Name of the imported Julia submodule as it appears in the
            `jl` namespace, e.g. 'Obliqua' or 'LovePy'.

    Returns
    -------
        jlarr, jlsca_float, jlsca_prec : callables
            `jlarr(arr)` converts a numpy array to a Julia `Array{<jl_module>.prec, 1}`.
            `jlsca_float(sca)` converts a Python scalar to `<jl_module>.Float64`.
            `jlsca_prec(sca)` converts a Python scalar to `<jl_module>.prec`.
    """

    def _jl_module():
        return getattr(jl, jl_module_name)

    def jlarr(arr: np.ndarray):
        # Make copy of array, and convert to Julia type
        cop = np.array(arr, copy=True, dtype=float).flatten()
        return juliacall.convert(jl.Array[_jl_module().prec, 1], cop)

    def jlsca_float(sca: float):
        # Make a copy of a scalar, and convert to Julia type
        return juliacall.convert(_jl_module().Float64, sca)

    def jlsca_prec(sca: float):
        # Make a copy of a scalar, and convert to Julia type
        return juliacall.convert(_jl_module().prec, sca)

    return jlarr, jlsca_float, jlsca_prec


def make_log_syncer(module_logfile_name: str):
    """Build a ``sync_log_files(outdir) -> list[str]`` bound to one Julia
    submodule's own recent-run logfile name (e.g. ``'obliqua_recent.log'``,
    ``'agni_recent.log'``).

    Each Julia-backed submodule (Obliqua, AGNI, ...) writes its own
    solver-run log to a fixed filename in ``outdir``; the returned function
    moves that content into PROTEUS's own logfile and clears the
    submodule's copy, so callers can scan the just-synced lines for the
    submodule's own failure-mode markers (e.g. AGNI's
    ``_extract_agni_failure_reason``).
    """

    def sync_log_files(outdir: str) -> list[str]:
        """Move the submodule's logfile content into the PROTEUS logfile
        and clear it.

        Returns the list of lines that were copied, so that callers can
        scan them for failure-mode markers. Returns an empty list if the
        submodule's logfile cannot be read.
        """
        # Logfile paths
        module_logpath = os.path.join(outdir, module_logfile_name)
        logpath = GetLogfilePath(outdir, GetCurrentLogfileIndex(outdir))

        # Copy logfile content
        try:
            with open(module_logpath) as infile:
                inlines = infile.readlines()
        except OSError:
            return []

        with open(logpath, 'a') as outfile:
            for i, line in enumerate(inlines):
                # First line of the submodule's logfile has NULL chars at
                # the start, for some reason
                if i == 0 and '[' in line:
                    line = '[' + line.split('[', 1)[1]
                # copy the line
                outfile.write(line)

        # Remove logfile content
        with open(module_logpath, 'w') as hdl:
            hdl.write('')

        return inlines

    return sync_log_files
