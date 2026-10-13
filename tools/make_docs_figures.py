#!/usr/bin/env python3
"""Redraw the figures of the tutorial pages as light and dark pairs.

Usage:
    python tools/make_docs_figures.py --runs <folder of run folders> [--only <id or page>]
        [--run <name>=<folder>] [--out <root>] [--chili-repo <folder>] [--manifest <file>]
    python tools/make_docs_figures.py --check

The manifest `tools/docs_figures.toml` has one entry per figure: the page, the asset
path stem, the run folders and the maker. For each entry the tool activates the
proteus-mpl style of a variant, calls the maker with a temporary output path and writes
`<asset>.avif` (light) and `<asset>_dark.avif` (dark). It never writes into a run folder.
For every saved figure it prints a text geometry table; a text that the saved file cuts
off is an error.

`--check` needs no run folder: it compares the manifest with the images that the pages
under `docs/Tutorials` embed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import re
import sys
import tempfile
import tomllib
from pathlib import Path

import _docs_pages
import _figure_geometry

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / 'tools' / 'docs_figures.toml'
SUFFIX = _docs_pages.SUFFIX
# Manifest keys with their TOML types; the keys in OPTIONAL can be absent.
TOP_TYPES = dict(proteus_mpl=str, avif_quality=int, pages=str, exempt_pages=list, figure=list)
ENTRY_TYPES = dict(id=str, page=str, asset=str, runs=list, maker=str, args=dict, variants=list)
OPTIONAL = {'exempt_pages', 'args', 'variants'}
TYPE_NAMES = {str: 'a string', int: 'a whole number', list: 'a list', dict: 'a table'}


class FigureError(Exception):
    """A manifest, a maker or a run folder does not meet the contract of the tool."""


def _table_fault(table, types: dict) -> str | None:
    """Name a missing key, an unknown key or a value of the wrong type in a manifest table."""
    if not isinstance(table, dict):
        return 'is not a table'
    missing = [key for key in types if key not in table and key not in OPTIONAL]
    unknown = sorted(set(table) - set(types))
    wrong = [key for key in types if key in table and type(table[key]) is not types[key]]
    if missing:
        return f'has no `{missing[0]}`'
    if unknown:
        return f'unknown key `{unknown[0]}`'
    if wrong:
        return f'`{wrong[0]}` must be {TYPE_NAMES[types[wrong[0]]]}'
    return None


def _plain(path) -> bool:
    """True for a relative path (a string) without an empty, `.` or `..` part."""
    return isinstance(path, str) and all(p not in ('', '.', '..') for p in path.split('/'))


def _entry_fault(fig) -> str | None:
    """Say what is wrong with one `[[figure]]` table of the manifest; None when it is valid."""
    fault = _table_fault(fig, ENTRY_TYPES)
    if fault:
        return fault
    if '/' in fig['id'] or not _plain(fig['id']) or not _plain(fig['page']):
        return '`id` must be a name and `page` a path under the pages folder'
    kind = fig['maker'].rpartition(':')[0]
    if not kind:
        return f'maker `{fig["maker"]}` is not `<kind>:<name>`'
    if kind != 'proteus' and not (kind.startswith('tools/') and _plain(kind)):
        return f'maker script `{kind}` is not under tools/'
    if not (fig['asset'].startswith('docs/assets/') and _plain(fig['asset'])):
        return f'asset `{fig["asset"]}` is not a file stem under docs/assets'
    if not fig['runs'] or not all(_plain(name) for name in fig['runs']):
        return '`runs` must list at least 1 folder name'
    variants = fig.get('variants', list(SUFFIX))
    known = tuple(SUFFIX)
    if not variants or any(v not in known or variants.count(v) > 1 for v in variants):
        return f'variants must be light and dark, got {variants}'
    return None


def load_manifest(path: Path) -> dict:
    """Read and validate the figure manifest.

    Parameters
    ----------
    path : Path
        TOML file with the keys `proteus_mpl`, `avif_quality`, `pages`, `exempt_pages`
        and one `[[figure]]` table per figure.

    Returns
    -------
    dict
        The manifest; each figure has `args` and `variants` filled in.

    Raises
    ------
    FigureError
        If the file cannot be read, a key is missing or unknown or has a value of the
        wrong type, an id or a file of an asset repeats, or an entry has a maker without
        the `kind:name` form, a maker script outside `tools/`, an asset outside
        `docs/assets`, no run folder, or a variant that is not `light` or `dark`.
    """
    try:
        with open(path, 'rb') as f:
            manifest = tomllib.load(f)
    except (OSError, ValueError) as exc:
        raise FigureError(f'{path.name}: {exc}') from exc
    fault = _table_fault(manifest, TOP_TYPES)
    if fault:
        raise FigureError(f'{path.name}: {fault}')
    if not 1 <= manifest['avif_quality'] <= 100:
        raise FigureError(f'{path.name}: `avif_quality` must be from 1 to 100')
    if not re.fullmatch(r'\d+(\.\d+)*', manifest['proteus_mpl']):
        raise FigureError(f'{path.name}: `proteus_mpl` must be a version such as 1.3.0')
    manifest.setdefault('exempt_pages', [])
    if not all(isinstance(name, str) for name in manifest['exempt_pages']):
        raise FigureError(f'{path.name}: `exempt_pages` must list page names')
    ids, files = [], []
    for fig in manifest['figure']:
        fault = _entry_fault(fig)
        if fault:
            name = fig.get('id', '?') if isinstance(fig, dict) else '?'
            raise FigureError(f'{path.name}: figure {name}: {fault}')
        fig.setdefault('args', {})
        fig.setdefault('variants', list(SUFFIX))
        ids.append(fig['id'])
        files += [f'{fig["asset"]}{SUFFIX[v]}.avif' for v in fig['variants']]
    for kind, values in (('id', ids), ('file', files)):
        twice = sorted({v for v in values if values.count(v) > 1})
        if twice:
            raise FigureError(f'{path.name}: {kind} {twice[0]} is used twice')
    return manifest


def style_version(minimum: str) -> str:
    """Return the installed version of proteus-mpl.

    Raises
    ------
    FigureError
        If the package is absent or older than `minimum`.
    """
    from packaging.version import Version

    try:
        found = importlib.metadata.version('proteus-mpl')
    except importlib.metadata.PackageNotFoundError:
        raise FigureError(
            f'proteus-mpl is not installed; install proteus-mpl>={minimum} '
            '(it is in the docs and develop extras of PROTEUS)'
        ) from None
    if Version(found) < Version(minimum):
        raise FigureError(f'proteus-mpl {found} is older than {minimum}, the manifest value')
    return found


def _run_inputs(run_dir: Path):
    """Read the resolved config and the helpfile of a run folder."""
    import pandas as pd

    from proteus.config import read_config_object

    config = read_config_object(str(run_dir / 'init_coupler.toml'))
    config.params.out.plot_fmt = 'png'
    return config, pd.read_csv(run_dir / 'runtime_helpfile.csv', sep=r'\s+')


def _proteus_global(run_dir: Path, out_dir: Path, logt: bool = True, tmin: float = 1e3) -> str:
    from proteus.plot.cpl_global import plot_global

    config, hf_all = _run_inputs(run_dir)
    plot_global(hf_all=hf_all, output_dir=str(out_dir), config=config, logt=logt, tmin=tmin)
    return 'plot_global_log.png' if logt else 'plot_global_lin.png'


def _proteus_orbit(run_dir: Path, out_dir: Path) -> str:
    from proteus.plot.cpl_orbit import plot_orbit

    config, hf_all = _run_inputs(run_dir)
    has_sat = config.orbit.satellite.include_satellite
    plot_orbit(hf_all, str(out_dir), has_sat, plot_format='png')
    return 'plot_orbit.png'


def _proteus_fluxes_global(run_dir: Path, out_dir: Path) -> str:
    from proteus.plot.cpl_fluxes_global import plot_fluxes_global

    config, hf_all = _run_inputs(run_dir)
    plot_fluxes_global(hf_all, str(out_dir), config)
    return 'plot_fluxes_global.png'


def _proteus_lovenumber(run_dir: Path, out_dir: Path) -> str:
    """Love number spectrum from every tides snapshot of the run."""
    from proteus.orbit.wrapper import read_tides_data
    from proteus.plot.cpl_orbit import plot_lovenumber

    config, _ = _run_inputs(run_dir)
    module = config.orbit.module
    snapshots = (run_dir / 'data').glob(f'*_{module}.nc')
    times = sorted(int(f.name.split('_')[0]) for f in snapshots)
    data = read_tides_data(str(run_dir), module, times)
    plot_lovenumber(output_dir=str(out_dir), times=times, data=data, plot_format='png')
    return 'plot_lovenumber.png'


# `proteus plot` functions: name -> adapter(run folder, output folder, **args) -> file name
PROTEUS_MAKERS = {
    'global': _proteus_global,
    'orbit': _proteus_orbit,
    'fluxes_global': _proteus_fluxes_global,
    'lovenumber': _proteus_lovenumber,
}


def _script_maker(script: str, name: str):
    """Load the function `name` of a tools script given by its path in the repository."""
    path = REPO_ROOT / script
    if not path.is_file():
        raise FigureError(f'maker script {script} does not exist')
    spec = importlib.util.spec_from_file_location(f'docs_figure_maker_{path.stem}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, name, None)):
        raise FigureError(f'{script} has no function {name}')
    return getattr(module, name)


def _fill(args: dict, fig_id: str, places: dict) -> dict:
    """Replace `{runs}` and `{chili_repo}` in the string arguments of an entry."""
    filled = {}
    for key, value in args.items():
        if isinstance(value, str):
            for place, path in places.items():
                if place in value:
                    if path is None:
                        option = '--' + place.strip('{}').replace('_', '-')
                        raise FigureError(f'{fig_id}: argument `{key}` needs {option}')
                    value = value.replace(place, str(path))
        filled[key] = value
    return filled


def _draw(fig: dict, run_dirs: list[Path], work: Path, places: dict) -> Path:
    """Call the maker of an entry under the active style; return the PNG it wrote."""
    kind, name = fig['maker'].rsplit(':', 1)
    args = _fill(fig['args'], fig['id'], places)
    if kind == 'proteus':
        if name not in PROTEUS_MAKERS:
            raise FigureError(f'{fig["id"]}: no `proteus plot` maker named {name}')
        (work / 'plots').mkdir(parents=True)
        png = work / 'plots' / PROTEUS_MAKERS[name](run_dirs[0], work, **args)
    else:
        work.mkdir(parents=True)
        png = work / f'{fig["id"]}.png'
        _script_maker(kind, name)(run_dirs, png, **args)
    if not png.is_file():
        raise FigureError(f'{fig["id"]}: the maker wrote no file {png.name}')
    return png


def _draw_variant(fig: dict, variant: str, run_dirs: list[Path], work: Path, places: dict):
    """Draw one variant under its proteus-mpl style; return the PNG and the geometry tables."""
    import matplotlib
    import matplotlib.pyplot as plt
    import proteus_mpl

    with matplotlib.rc_context():
        proteus_mpl.use(variant)
        try:
            with _figure_geometry.watch_savefig() as tables:
                png = _draw(fig, run_dirs, work, places)
        finally:
            plt.close('all')
    if not tables:
        raise FigureError(
            f'{fig["id"]}: the maker saved no figure with `Figure.savefig`, '
            'so the text geometry cannot be measured'
        )
    return png, tables


def _write_entry(fig: dict, drawn: list, out_root: Path, quality: int) -> None:
    """Encode every drawn variant next to its target, then put the AVIF files into place.

    A failure of an encode leaves the files in `out_root` as they were.
    """
    from PIL import Image

    staged = []
    try:
        for variant, png in drawn:
            target = out_root / f'{fig["asset"]}{SUFFIX[variant]}.avif'
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + '.part')
            with Image.open(png) as image:
                staged.append(
                    (f'{variant}: {image.size[0]} x {image.size[1]} px', part, target)
                )
                image.convert('RGB').save(part, 'AVIF', quality=quality)
        for label, part, target in staged:
            os.replace(part, target)
            print(f'{fig["id"]} {label} -> {target}')
    finally:
        for _, part, _ in staged:
            part.unlink(missing_ok=True)


def _tree_state(run_dirs: list[Path]) -> dict:
    """Modification time and size of every file and folder under the run folders.

    A symbolic link inside a run folder is recorded as the link, not as its target.
    """
    state = {}
    for run_dir in run_dirs:
        for path in [run_dir, *run_dir.rglob('*')]:
            try:
                stat = path.lstat()
            except FileNotFoundError:
                raise FigureError(
                    f'a run folder changed during the draw: {path} is gone'
                ) from None
            state[path] = (stat.st_mtime_ns, stat.st_size)
    return state


def _report(fig_id: str, variant: str, tables: list) -> int:
    """Print the geometry table of each saved figure; return the number of cut-off texts."""
    errors = 0
    for fname, table in tables:
        print(f'  geometry {fig_id} {variant} ({Path(fname).name}): {table["texts"]} texts')
        for key in ('cut_off', 'outside_figure', 'tick_overlaps', 'tick_crowding', 'covered'):
            rows = table[key]
            flag = 'ERROR ' if key == 'cut_off' and rows else ''
            print(f'    {flag}{key}: {len(rows)}' + ''.join(f'\n      {row}' for row in rows))
        errors += len(table['cut_off'])
    return errors


def make_figures(
    manifest: dict,
    runs: Path,
    out_root: Path | None = None,
    only: str | None = None,
    chili_repo: Path | None = None,
    run_map: dict[str, Path] | None = None,
) -> int:
    """Draw the figures of the manifest in every variant and write the AVIF files.

    Parameters
    ----------
    manifest : dict
        Manifest from `load_manifest`.
    runs : Path
        Folder that holds the run folders named by the entries.
    out_root : Path, optional
        Folder that stands for the repository root; the files go to `<out_root>/<asset>`.
        The repository itself when None.
    only : str, optional
        Id or page of the entries to draw; all when None.
    chili_repo : Path, optional
        Value of the `{chili_repo}` placeholder.
    run_map : dict, optional
        Folder of a run by its manifest name, for a run folder that has another name or
        lies outside `runs`.

    Returns
    -------
    int
        Number of texts that a saved file would cut off, over all figures. The files of
        an entry with such a text are not written.

    Raises
    ------
    FigureError
        If proteus-mpl or AVIF support is missing, no entry matches `only`, a run
        folder is absent, a maker fails its contract, or a run folder changed.
    """
    import matplotlib

    matplotlib.use('Agg')
    import PIL
    from PIL import features

    version = style_version(manifest['proteus_mpl'])
    print(
        f'proteus-mpl {version}, matplotlib {matplotlib.__version__}, Pillow {PIL.__version__}'
    )
    if not features.check('avif'):
        raise FigureError('this Pillow build cannot write AVIF')
    out_root = out_root or REPO_ROOT
    figures = [f for f in manifest['figure'] if only in (None, f['id'], f['page'])]
    if not figures:
        raise FigureError(f'no figure with id or page `{only}` in the manifest')
    places = {'{runs}': runs, '{chili_repo}': chili_repo}
    errors = 0
    for fig in figures:
        run_dirs = [(run_map or {}).get(name, runs / name) for name in fig['runs']]
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                raise FigureError(f'{fig["id"]}: run folder {run_dir} does not exist')
        before = _tree_state(run_dirs)
        with tempfile.TemporaryDirectory() as tmp:
            drawn, cut_off = [], 0
            for variant in fig['variants']:
                png, tables = _draw_variant(fig, variant, run_dirs, Path(tmp) / variant, places)
                if _tree_state(run_dirs) != before:
                    raise FigureError(
                        f'{fig["id"]}: a run folder changed during the draw; '
                        'run folders are read only'
                    )
                cut_off += _report(fig['id'], variant, tables)
                drawn.append((variant, png))
            if not cut_off:
                _write_entry(fig, drawn, out_root, manifest['avif_quality'])
        if cut_off:
            print(f'{fig["id"]}: NOT WRITTEN, the saved file would cut off {cut_off} text(s)')
        errors += cut_off
    return errors


def main(argv: list[str] | None = None) -> int:
    """Command line entry point; returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--runs', type=Path, help='Folder that holds the run folders')
    parser.add_argument(
        '--run',
        action='append',
        default=[],
        metavar='NAME=FOLDER',
        help='Folder of the run with this manifest name (repeatable)',
    )
    parser.add_argument('--only', help='Draw the entries with this id or page only')
    parser.add_argument('--chili-repo', type=Path, help='Checkout of the CHILI repository')
    parser.add_argument('--out', type=Path, help='Write under this root, not the repository')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--check', action='store_true', help='Compare pages and manifest')
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.check:
            problems = _docs_pages.check_pages(manifest, REPO_ROOT)
            for name in manifest['exempt_pages']:
                print(f'not checked (exempt): {manifest["pages"]}/{name}')
            print(
                '\n'.join(problems) or f'{len(manifest["figure"])} figures agree with the pages'
            )
            return 1 if problems else 0
        if args.runs is None:
            parser.error('--runs is required unless --check is given')
        names = {name for fig in manifest['figure'] for name in fig['runs']}
        run_map = {
            name: Path(folder) for name, _, folder in (r.partition('=') for r in args.run)
        }
        unknown = sorted(
            name for name, folder in run_map.items() if name not in names or not folder.name
        )
        if unknown:
            raise FigureError(
                f'--run {unknown[0]}: give NAME=FOLDER with a run name of the manifest'
            )
        errors = make_figures(
            manifest, args.runs, args.out, args.only, args.chili_repo, run_map
        )
    except FigureError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    print(f'{errors} text(s) cut off in the saved files')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
