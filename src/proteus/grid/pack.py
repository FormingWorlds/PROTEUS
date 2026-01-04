# Check the status of a PROTEUS parameter grid's cases
from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from shutil import copyfile, rmtree
from zipfile import ZIP_DEFLATED, ZipFile


def pack(grid: str, plots: bool = True, zip: bool = True, rmdir_pack: bool = True):
    """Pack most-important data for all cases into a single folder; optionally zip it."""
    if (not os.path.exists(grid)) or (not os.path.isdir(grid)):
        raise FileNotFoundError("Invalid path '%s'" % grid)

    grid = os.path.abspath(grid)
    print(f'Grid dir: {grid}')

    pack = os.path.join(grid, 'pack')
    print(f'Pack dir: {pack}')
    rmtree(pack, ignore_errors=True)
    os.mkdir(pack)

    # find case_* subdirectories
    case_dirs = list(glob(grid + '/case_*'))
    if not case_dirs:
        raise FileNotFoundError('Cannot find any subfolders containing grid cases!')

    # copy top-level files in grid output folder
    print('Copy top-level files...')
    for tf in ['manager.log', 'ref_config.toml', 'copy.grid.toml']:
        copyfile(os.path.join(grid, tf), os.path.join(pack, tf))

    # copy per-case data
    print('Copy results for each grid point...')
    print(f'Found {len(case_dirs)} subfolders')
    for case in case_dirs:
        print('   ' + os.path.basename(case))
        dest = os.path.join(pack, os.path.basename(case))
        os.mkdir(dest)

        # lower level files
        llfs = ['runtime_helpfile.csv', 'init_coupler.toml', 'status']
        llfs.extend([f'proteus_{i:02d}.log' for i in range(100)])
        for lf in llfs:
            try:
                copyfile(os.path.join(case, lf), os.path.join(dest, lf))
            except FileNotFoundError:
                pass

        # plots directory
        if plots:
            for pf in glob('plot_*', root_dir=os.path.join(case, 'plots')):
                copyfile(os.path.join(case, 'plots', pf), os.path.join(dest, pf))

    # create zip at grid/pack.zip containing "pack/..."
    if zip:
        zip_path = os.path.join(grid, 'pack.zip')
        print(f'Make zip: {zip_path}')
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
        print(f'Archive size is {os.path.getsize(zip_path) / 1e6:.1f} MB')

        # remove `pack` folder now that we have zipped it
        if rmdir_pack:
            rmtree(pack)

    return True
