# Package the most important results of a PROTEUS parameter grid into one archive
from __future__ import annotations

import logging
import os
from glob import glob
from pathlib import Path
from shutil import copyfile, rmtree
from zipfile import ZIP_DEFLATED, ZipFile

log = logging.getLogger('fwl.' + __name__)


def pack(grid: str, plots: bool = True, zip: bool = True, rmdir_pack: bool = True):
    """Pack most-important data for all cases into a single folder; optionally zip it."""
    if (not os.path.exists(grid)) or (not os.path.isdir(grid)):
        raise FileNotFoundError("Invalid path '%s'" % grid)

    grid = os.path.abspath(grid)
    log.info('Grid dir: %s', grid)

    pack = os.path.join(grid, 'pack')
    log.info('Pack dir: %s', pack)
    rmtree(pack, ignore_errors=True)
    os.mkdir(pack)

    # find case_* subdirectories
    case_dirs = list(glob(grid + '/case_*'))
    if not case_dirs:
        raise FileNotFoundError('Cannot find any subfolders containing grid cases!')

    # copy top-level files in grid output folder
    log.info('Copy top-level files...')
    for tf in ['manager.log', 'ref_config.toml', 'copy.grid.toml']:
        try:
            copyfile(os.path.join(grid, tf), os.path.join(pack, tf))
        except FileNotFoundError:
            log.warning("Top-level file '%s' not found; skipping", tf)

    # copy per-case data
    log.info('Copy results for each grid point...')
    log.info('Found %d subfolders', len(case_dirs))
    for case in case_dirs:
        log.info('   %s', os.path.basename(case))
        dest = os.path.join(pack, os.path.basename(case))
        os.mkdir(dest)

        # lower level files
        llfs = ['runtime_helpfile.csv', 'init_coupler.toml', 'status']
        for lf in llfs:
            try:
                copyfile(os.path.join(case, lf), os.path.join(dest, lf))
            except FileNotFoundError:
                pass

        # proteus segment logs (any number of restarts, not just the first 100)
        for lf in glob('proteus_*.log', root_dir=case):
            copyfile(os.path.join(case, lf), os.path.join(dest, lf))

        # plots directory
        if plots:
            for pf in glob('plot_*', root_dir=os.path.join(case, 'plots')):
                copyfile(os.path.join(case, 'plots', pf), os.path.join(dest, pf))

    # create zip at grid/pack.zip containing "pack/..."
    if zip:
        zip_path = os.path.join(grid, 'pack.zip')
        log.info('Make zip: %s', zip_path)
        if os.path.isfile(zip_path):
            os.remove(zip_path)
        with ZipFile(zip_path, 'w', compression=ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(pack):
                root_path = Path(root)
                for name in files:
                    file_path = root_path / name
                    # keep "pack" as top-level folder inside the zip
                    arcname = Path('pack') / file_path.relative_to(pack)
                    zf.write(file_path, arcname)

        # give size of file
        log.info('Archive size is %.1f MB', os.path.getsize(zip_path) / 1e6)

        # remove `pack` folder now that we have zipped it
        if rmdir_pack:
            rmtree(pack)

    return True
