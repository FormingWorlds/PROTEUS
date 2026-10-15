# Generic atmospheric chemistry wrapper
from __future__ import annotations

import logging
import os
import shutil
from typing import TYPE_CHECKING

import pandas as pd

from proteus.atmos_chem.common import (
    chem_csv_name,
    helpfile_row_at,
    read_result,
    resolve_chemistry_times,
)
from proteus.utils.helper import UpdateStatusfile

log = logging.getLogger('fwl.' + __name__)

if TYPE_CHECKING:
    from proteus.config import Config


def run_chemistry(dirs: dict, config: Config, hf_all: pd.DataFrame) -> pd.DataFrame | None:
    """
    Post-process atmosphere snapshots with the atmospheric chemistry model.

    Runs at the snapshot times set by `atmos_chem.times`, in ascending order,
    after the simulation. Each run writes ``offchem/<module>_<time>.csv``;
    the latest successful one is also copied to ``offchem/<module>.csv``.
    The ``offchem`` folder is cleared once at the start.

    Parameters
    ----------
    dirs : dict
        Dictionary of directories.
    config : Config
        Configuration object.
    hf_all : pd.DataFrame
        Helpfile of the run. Each run uses the row nearest its snapshot time.

    Returns
    ----------
        result : pd.DataFrame | None
            Result of the latest successful run, or None if none succeeded.
    """

    log.info('Running atmospheric chemistry...')

    # The config converter maps the TOML string 'none' to None
    module = config.atmos_chem.module
    if not module:
        log.warning('Cannot run atmospheric chemistry, no module specified')
        return None

    # Resolve the runner function (lazy imports keep heavy backends optional)
    if module == 'vulcan':
        from proteus.atmos_chem.vulcan import run_vulcan as _run
    elif module == 'dummy':
        from proteus.atmos_chem.dummy import run_dummy_chem as _run
    else:
        if dirs.get('output'):
            UpdateStatusfile(dirs, 20)
        raise ValueError(f"Invalid atmos_chem module: '{module}'")

    times = resolve_chemistry_times(dirs['output'], hf_all, config.atmos_chem.times)
    if not times:
        log.warning('No snapshots to run atmospheric chemistry on')
        return None
    log.info(f'    Using {module} module at {len(times)} snapshot(s)')

    shutil.rmtree(dirs['output/offchem'], ignore_errors=True)
    os.makedirs(dirs['output/offchem'])

    # Rows report a column absent from an old helpfile, rather than a KeyError
    from proteus.utils.coupler import HelpfileRow, helpfile_path

    latest = None
    for i, time in enumerate(times):
        log.info('    Chemistry snapshot %d of %d, t=%.3e yr' % (i + 1, len(times), time))
        # Snapshot file names hold the time to 1e-3 yr, so the row takes that time
        snapshot = helpfile_row_at(hf_all, time)
        snapshot['Time'] = time
        hf_row = HelpfileRow(snapshot, helpfile_path(dirs['output']))
        if _run(dirs, config, hf_row, first=(i == 0)):
            latest = time
        else:
            log.warning('%s chemistry failed at t=%.3e yr; skipping it' % (module, time))

    # Surface the failure to the caller rather than a None DataFrame downstream
    if latest is None:
        log.warning(f'{module} chemistry produced no output')
        return None

    # The fixed name holds the latest result, read by observe and the plots
    offchem = dirs['output/offchem']
    shutil.copyfile(
        os.path.join(offchem, chem_csv_name(module, latest)),
        os.path.join(offchem, f'{module}.csv'),
    )
    return read_result(dirs['output'], module)
