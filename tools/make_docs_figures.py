#!/usr/bin/env python3
"""Redraw the figures of the tutorial pages as light and dark pairs.

Usage:
    python tools/make_docs_figures.py --runs <folder of run folders> [--only <id or page>]
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
import re
import sys
import tempfile
import tomllib
from pathlib import Path

import _figure_geometry

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / 'tools' / 'docs_figures.toml'
SUFFIX = {'light': '', 'dark': '_dark'}
EMBED = re.compile(r'!\[[^\]]*\]\(([^)\s]+)\)')
PLACEHOLDERS = ('{runs}', '{chili_repo}')


class FigureError(Exception):
    """A manifest, a maker or a run folder does not meet the contract of the tool."""


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
        If a key is missing, an id repeats, a maker has no `kind:name` form, a variant
        is not `light` or `dark`, or an asset lies outside `docs/assets`.
    """
    with open(path, 'rb') as f:
        manifest = tomllib.load(f)
    for key in ('proteus_mpl', 'avif_quality', 'pages', 'figure'):
        if key not in manifest:
            raise FigureError(f'{path.name}: key `{key}` is missing')
    manifest.setdefault('exempt_pages', [])
    seen = set()
    for fig in manifest['figure']:
        for key in ('id', 'page', 'asset', 'runs', 'maker'):
            if key not in fig:
                raise FigureError(f'{path.name}: figure {fig.get("id", "?")} has no `{key}`')
        if fig['id'] in seen:
            raise FigureError(f'{path.name}: id {fig["id"]} is used twice')
        seen.add(fig['id'])
        if ':' not in fig['maker']:
            raise FigureError(f'{fig["id"]}: maker `{fig["maker"]}` is not `<kind>:<name>`')
        if not fig['asset'].startswith('docs/assets/') or '..' in fig['asset']:
            raise FigureError(f'{fig["id"]}: asset `{fig["asset"]}` is not under docs/assets')
        fig.setdefault('args', {})
        fig.setdefault('variants', list(SUFFIX))
        unknown = set(fig['variants']) - set(SUFFIX)
        if unknown or not fig['variants']:
            raise FigureError(
                f'{fig["id"]}: variants must be light and dark, got {fig["variants"]}'
            )
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
    files = (run_dir / 'data').glob(f'*_{module}.nc')
    times = sorted(int(f.name.split('_')[0]) for f in files)
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
            for place in PLACEHOLDERS:
                if place in value and places[place] is None:
                    option = '--' + place.strip('{}').replace('_', '-')
                    raise FigureError(f'{fig_id}: argument `{key}` needs {option}')
                if place in value:
                    value = value.replace(place, str(places[place]))
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


def _tree_state(run_dirs: list[Path]) -> dict:
    """Modification time and size of every file and folder under the run folders."""
    state = {}
    for run_dir in run_dirs:
        for path in [run_dir, *run_dir.rglob('*')]:
            stat = path.stat()
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

    Returns
    -------
    int
        Number of texts that a saved file cuts off, over all figures.

    Raises
    ------
    FigureError
        If proteus-mpl or AVIF support is missing, no entry matches `only`, a run
        folder is absent, a maker fails its contract, or a run folder changed.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import PIL
    from PIL import Image, features

    version = style_version(manifest['proteus_mpl'])
    print(
        f'proteus-mpl {version}, matplotlib {matplotlib.__version__}, Pillow {PIL.__version__}'
    )
    if not features.check('avif'):
        raise FigureError('this Pillow build cannot write AVIF')
    import proteus_mpl

    out_root = out_root or REPO_ROOT
    figures = [f for f in manifest['figure'] if only in (None, f['id'], f['page'])]
    if not figures:
        raise FigureError(f'no figure with id or page `{only}` in the manifest')
    places = {'{runs}': runs, '{chili_repo}': chili_repo}
    errors = 0
    for fig in figures:
        run_dirs = [runs / name for name in fig['runs']]
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                raise FigureError(f'{fig["id"]}: run folder {run_dir} does not exist')
        before = _tree_state(run_dirs)
        for variant in fig['variants']:
            target = out_root / f'{fig["asset"]}{SUFFIX[variant]}.avif'
            with tempfile.TemporaryDirectory() as tmp, matplotlib.rc_context():
                proteus_mpl.use(variant)
                try:
                    with _figure_geometry.watch_savefig() as tables:
                        png = _draw(fig, run_dirs, Path(tmp) / variant, places)
                finally:
                    plt.close('all')
                if _tree_state(run_dirs) != before:
                    raise FigureError(
                        f'{fig["id"]}: the maker changed a run folder; run folders are read only'
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(png) as image:
                    image.convert('RGB').save(target, 'AVIF', quality=manifest['avif_quality'])
                    print(
                        f'{fig["id"]} {variant}: {image.size[0]} x {image.size[1]} px -> {target}'
                    )
            errors += _report(fig['id'], variant, tables)
    return errors


def _repo_target(page: Path, target: str, repo: Path) -> str:
    """Path of an embedded image from the repository root, with its `#fragment` kept.

    A web address or a path that leaves the repository is returned as written.
    """
    path, _, fragment = target.partition('#')
    resolved = (page.parent / path).resolve()
    if '://' in path or not resolved.is_relative_to(repo.resolve()):
        return target
    return resolved.relative_to(repo.resolve()).as_posix() + (
        f'#{fragment}' if fragment else ''
    )


def check_pages(manifest: dict, repo: Path) -> list[str]:
    """Compare the manifest with the images that the tutorial pages embed.

    Parameters
    ----------
    manifest : dict
        Manifest from `load_manifest`.
    repo : Path
        Repository root.

    Returns
    -------
    list[str]
        One line per problem: an exempt page that does not exist, an embedded image
        without an entry, an entry without its files, or an entry that its page does
        not embed as a light and dark pair on adjacent embeds. Empty when all agree.
    """
    pages_dir = repo / manifest['pages']
    problems = [
        f'exempt page {name} does not exist'
        for name in manifest['exempt_pages']
        if not (pages_dir / name).is_file()
    ]
    wanted = {}
    for fig in manifest['figure']:
        pair = [f'{fig["asset"]}{SUFFIX[v]}.avif#only-{v}' for v in fig['variants']]
        wanted.setdefault(fig['page'], []).append((fig['id'], pair))
        for target in pair:
            if not (repo / target.split('#')[0]).is_file():
                problems.append(f'{fig["id"]}: file {target.split("#")[0]} is missing')
        if fig['page'] in manifest['exempt_pages'] or not (pages_dir / fig['page']).is_file():
            problems.append(f'{fig["id"]}: page {fig["page"]} is exempt or does not exist')
    for page in sorted(pages_dir.glob('*.md')):
        if page.name in manifest['exempt_pages']:
            continue
        embeds = [
            _repo_target(page, target, repo) for target in EMBED.findall(page.read_text())
        ]
        entries = wanted.get(page.name, [])
        known = {target for _, pair in entries for target in pair}
        problems += [
            f'{page.name}: image {e} has no manifest entry' for e in embeds if e not in known
        ]
        for fig_id, pair in entries:
            at = [embeds.index(t) if t in embeds else None for t in pair]
            if None in at or at != list(range(at[0], at[0] + len(at))):
                problems.append(
                    f'{fig_id}: {page.name} does not embed {" and ".join(pair)} as a pair'
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    """Command line entry point; returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--runs', type=Path, help='Folder that holds the run folders')
    parser.add_argument('--only', help='Draw the entries with this id or page only')
    parser.add_argument('--chili-repo', type=Path, help='Checkout of the CHILI repository')
    parser.add_argument('--out', type=Path, help='Write under this root, not the repository')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--check', action='store_true', help='Compare pages and manifest')
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.check:
            problems = check_pages(manifest, REPO_ROOT)
            for name in manifest['exempt_pages']:
                print(f'not checked (exempt): {manifest["pages"]}/{name}')
            print(
                '\n'.join(problems) or f'{len(manifest["figure"])} figures agree with the pages'
            )
            return 1 if problems else 0
        if args.runs is None:
            parser.error('--runs is required unless --check is given')
        errors = make_figures(manifest, args.runs, args.out, args.only, args.chili_repo)
    except FigureError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    print(f'{errors} text(s) cut off in the saved files')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
