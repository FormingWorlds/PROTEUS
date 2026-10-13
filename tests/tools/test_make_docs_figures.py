"""Unit tests for tools/make_docs_figures.py, the tool that draws the tutorial figures.

The makers are small scripts written under `tmp_path`, so the tests draw real figures
with the real proteus-mpl style and never read a PROTEUS run.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import inspect
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import proteus_mpl  # noqa: E402
import pytest  # noqa: E402
import tomlkit  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from PIL import Image, features  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    """Load a tools/ script by path and register it, so that sibling imports resolve."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / 'tools' / f'{name}.py')
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


_load('_figure_geometry')
_load('_docs_pages')
tool = _load('make_docs_figures')

MAKER = """
import matplotlib
import matplotlib.pyplot as plt


def draw(run_dirs, out_path, record=None, note='', stray=False):
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1], [0, 1])
    if stray:
        fig.text(1.5, 0.5, 'stray').set_in_layout(False)
    if record:
        with open(record, 'a') as f:
            f.write(f"{matplotlib.rcParams['text.color']} {note} {len(run_dirs)}\\n")
    fig.savefig(out_path)
    return fig


def add_file_to_run(run_dirs, out_path):
    (run_dirs[0] / 'plots' / 'new.png').write_text('x')
    draw(run_dirs, out_path)


def rewrite_file_of_run(run_dirs, out_path):
    (run_dirs[0] / 'runtime_helpfile.csv').write_text('Time\\n2.0\\n')
    draw(run_dirs, out_path)


def write_nothing(run_dirs, out_path):
    pass


def fail_in_dark(run_dirs, out_path):
    if sum(matplotlib.colors.to_rgb(matplotlib.rcParams['figure.facecolor'])) < 1.0:
        raise RuntimeError('no dark figure')
    draw(run_dirs, out_path)


def save_through_the_canvas(run_dirs, out_path):
    fig, ax = plt.subplots(figsize=(2, 1.5))
    fig.canvas.print_figure(out_path)


NOT_A_FUNCTION = 1
"""
PAIR = (
    '![Alt](../assets/tutorials/fig_a.avif#only-light){ width="100%" }\n'
    '![Alt](../assets/tutorials/fig_a_dark.avif#only-dark){ width="100%" }\n'
)


def _entry(**changes) -> dict:
    """A valid manifest entry for the maker script of the `site` fixture."""
    entry = {
        'id': 'fig_a',
        'page': 'page_a.md',
        'asset': 'docs/assets/tutorials/fig_a',
        'runs': ['run_a'],
        'maker': 'tools/fake_maker.py:draw',
    }
    entry.update(changes)
    return {k: v for k, v in entry.items() if v is not None}


def _manifest(path: Path, figures=None, **top) -> dict:
    """Write a manifest file and read it back through the tool."""
    data = {'proteus_mpl': '1.3.0', 'avif_quality': 80, 'pages': 'docs/Tutorials'}
    data.update(top)
    data['figure'] = [_entry()] if figures is None else figures
    path.write_text(tomlkit.dumps({k: v for k, v in data.items() if v is not None}))
    return tool.load_manifest(path)


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A repository root with a maker script, 1 page with a pair, and 1 run folder."""
    repo = tmp_path / 'repo'
    (repo / 'tools').mkdir(parents=True)
    (repo / 'tools' / 'fake_maker.py').write_text(MAKER)
    (repo / 'docs' / 'Tutorials').mkdir(parents=True)
    (repo / 'docs' / 'Tutorials' / 'page_a.md').write_text(f'# Page A\n\n{PAIR}')
    assets = repo / 'docs' / 'assets' / 'tutorials'
    assets.mkdir(parents=True)
    for name in ('fig_a.avif', 'fig_a_dark.avif'):
        (assets / name).write_bytes(b'placeholder')
    run = tmp_path / 'runs' / 'run_a'
    (run / 'plots').mkdir(parents=True)
    (run / 'runtime_helpfile.csv').write_text('Time\n1.0\n')
    (run / 'dangling_link').symlink_to(run / 'gone')
    monkeypatch.setattr(tool, 'REPO_ROOT', repo)
    return SimpleNamespace(repo=repo, runs=tmp_path / 'runs', run=run, toml=tmp_path / 'm.toml')


def _files(root: Path) -> list[str]:
    paths = [p for p in root.rglob('*') if p.is_file() or p.is_symlink()]
    return sorted(p.relative_to(root).as_posix() for p in paths)


# --- manifest ---------------------------------------------------------------------


def test_manifest_fills_the_defaults_of_an_entry(tmp_path):
    """An entry without `args` and `variants` gets no argument and both variants."""
    manifest = _manifest(tmp_path / 'm.toml')
    assert manifest['exempt_pages'] == []
    assert manifest['figure'][0]['args'] == {}
    assert manifest['figure'][0]['variants'] == ['light', 'dark']


@pytest.mark.parametrize(
    'top, figures, message',
    [
        ({'proteus_mpl': None}, None, 'm.toml: has no `proteus_mpl`'),
        ({'pages': None}, None, 'm.toml: has no `pages`'),
        ({'page': 'docs/Tutorials'}, None, 'unknown key `page`'),
        ({'pages': 3}, None, '`pages` must be a string'),
        ({'proteus_mpl': 1.3}, None, '`proteus_mpl` must be a string'),
        ({'exempt_pages': 'old.md'}, None, '`exempt_pages` must be a list'),
        ({'exempt_pages': [3]}, None, '`exempt_pages` must list page names'),
        ({'avif_quality': 0}, None, '`avif_quality` must be from 1 to 100'),
        ({'avif_quality': 101}, None, '`avif_quality` must be from 1 to 100'),
        ({'avif_quality': '80'}, None, '`avif_quality` must be a whole number'),
        ({'avif_quality': True}, None, '`avif_quality` must be a whole number'),
        ({'avif_quality': 80.0}, None, '`avif_quality` must be a whole number'),
        ({}, 3, '`figure` must be a list'),
        ({}, [3], 'figure ?: is not a table'),
        ({}, [_entry(runs=None)], 'figure fig_a: has no `runs`'),
        ({}, [_entry(id=None)], 'figure ?: has no `id`'),
        ({}, [_entry(id=['fig_a'])], '`id` must be a string'),
        (
            {},
            [_entry(id='a/b')],
            '`id` must be a name and `page` a path under the pages folder',
        ),
        ({}, [_entry(id='..')], '`id` must be a name and `page` a path under the pages folder'),
        ({}, [_entry(page='./page_a.md')], '`id` must be a name and `page` a path under'),
        ({}, [_entry(page='../page_a.md')], '`id` must be a name and `page` a path under'),
        ({'proteus_mpl': 'abc'}, None, '`proteus_mpl` must be a version such as 1.3.0'),
        ({'proteus_mpl': '1.3.'}, None, '`proteus_mpl` must be a version such as 1.3.0'),
        ({}, [_entry(page=3)], '`page` must be a string'),
        ({}, [_entry(asset=3)], '`asset` must be a string'),
        ({}, [_entry(maker=3)], '`maker` must be a string'),
        ({}, [_entry(), _entry()], 'id fig_a is used twice'),
        (
            {},
            [_entry(), _entry(id='fig_b')],
            'file docs/assets/tutorials/fig_a.avif is used twice',
        ),
        (
            {},
            [_entry(), _entry(id='fig_b', asset='docs/assets/tutorials/fig_a_dark')],
            'file docs/assets/tutorials/fig_a_dark.avif is used twice',
        ),
        ({}, [_entry(variant=['dark'])], 'figure fig_a: unknown key `variant`'),
        ({}, [_entry(arg={'logt': False})], 'figure fig_a: unknown key `arg`'),
        ({}, [_entry(maker='draw')], 'maker `draw` is not `<kind>:<name>`'),
        ({}, [_entry(maker='scripts/fake_maker.py:draw')], 'is not under tools/'),
        ({}, [_entry(maker='/tmp/tools/fake_maker.py:draw')], 'is not under tools/'),
        ({}, [_entry(maker='tools/../../fake_maker.py:draw')], 'is not under tools/'),
        ({}, [_entry(asset='docs/other/fig_a')], 'is not a file stem under docs/assets'),
        ({}, [_entry(asset='docs/assets/../../fig_a')], 'is not a file stem under docs/assets'),
        ({}, [_entry(asset='docs/assets/')], 'is not a file stem under docs/assets'),
        ({}, [_entry(asset='docs/assets//fig_a')], 'is not a file stem under docs/assets'),
        ({}, [_entry(runs=[])], '`runs` must list at least 1 folder name'),
        ({}, [_entry(runs='run_a')], '`runs` must be a list'),
        ({}, [_entry(runs=[3])], '`runs` must list at least 1 folder name'),
        ({}, [_entry(runs=['run_a', '../other'])], '`runs` must list at least 1 folder name'),
        ({}, [_entry(runs=['/abs/run_a'])], '`runs` must list at least 1 folder name'),
        ({}, [_entry(runs=['.'])], '`runs` must list at least 1 folder name'),
        ({}, [_entry(args=['logt'])], '`args` must be a table'),
        ({}, [_entry(variants='dark')], '`variants` must be a list'),
        ({}, [_entry(variants=['light', 'white'])], 'variants must be light and dark'),
        ({}, [_entry(variants=['light', 'light'])], 'variants must be light and dark'),
        ({}, [_entry(variants=[['light']])], 'variants must be light and dark'),
        ({}, [_entry(variants=[])], 'variants must be light and dark'),
    ],
)
def test_manifest_with_a_fault_is_refused_by_name(tmp_path, top, figures, message):
    """Each fault of a manifest raises `FigureError` with the entry and the key in it."""
    with pytest.raises(tool.FigureError) as err:
        _manifest(tmp_path / 'm.toml', figures, **top)
    assert message in str(err.value)
    assert str(err.value).startswith('m.toml: ')


def test_manifest_that_cannot_be_read_is_refused(tmp_path):
    """A missing file and a file with a TOML fault stop the tool with the file name."""
    with pytest.raises(tool.FigureError, match='absent.toml: .*No such file'):
        tool.load_manifest(tmp_path / 'absent.toml')
    (tmp_path / 'broken.toml').write_text('proteus_mpl = "1.3.0\n[[figure]\n')
    with pytest.raises(tool.FigureError, match='broken.toml: '):
        tool.load_manifest(tmp_path / 'broken.toml')
    (tmp_path / 'latin.toml').write_bytes('pages = "caf\xe9"\n'.encode('latin-1'))
    with pytest.raises(tool.FigureError, match='latin.toml: '):
        tool.load_manifest(tmp_path / 'latin.toml')
    assert tool.main(['--check', '--manifest', str(tmp_path / 'broken.toml')]) == 1


def test_manifest_of_the_repository_agrees_with_pyproject_and_the_pages():
    """The style floor is the same in the manifest and in both extras, and `--check` passes."""
    manifest = tool.load_manifest(tool.MANIFEST)
    with open(_REPO_ROOT / 'pyproject.toml', 'rb') as f:
        extras = tomllib.load(f)['project']['optional-dependencies']
    for extra in ('docs', 'develop'):
        pins = [d for d in extras[extra] if d.startswith('proteus-mpl')]
        assert pins == [f'proteus-mpl>={manifest["proteus_mpl"]}']
    assert tool.main(['--check']) == 0
    # Every figure is a pair, and every maker that an entry names can be loaded
    for fig in manifest['figure']:
        assert fig['variants'] == ['light', 'dark']
        kind, name = fig['maker'].rsplit(':', 1)
        assert (
            name in tool.PROTEUS_MAKERS if kind == 'proteus' else tool._script_maker(kind, name)
        )
    # The environment of the tests has a proteus-mpl that this manifest accepts
    installed = importlib.metadata.version('proteus-mpl')
    assert tool.style_version(manifest['proteus_mpl']) == installed


def test_committed_pairs_are_images_of_equal_size_on_the_surface_of_their_variant():
    """Each file of the manifest opens as an image; a pair has 1 size and the 2 surface colours."""
    surface = {}
    for variant in ('light', 'dark'):
        with matplotlib.rc_context():
            proteus_mpl.use(variant)
            surface[variant] = [
                round(255 * c) for c in to_rgb(matplotlib.rcParams['figure.facecolor'])
            ]
    assert sum(surface['light']) > 600 > 60 > sum(surface['dark'])
    for fig in tool.load_manifest(tool.MANIFEST)['figure']:
        sizes = []
        for variant in fig['variants']:
            with Image.open(_REPO_ROOT / f'{fig["asset"]}{tool.SUFFIX[variant]}.avif') as image:
                sizes.append(image.size)
                corner = image.convert('RGB').getpixel((2, 2))
            assert corner == pytest.approx(surface[variant], abs=6), (fig['id'], variant)
        assert sizes[0] == sizes[-1] and min(sizes[0]) > 500, fig['id']


# --- style version ----------------------------------------------------------------


@pytest.mark.parametrize(
    'found, minimum, above', [('1.3.0', '1.3.0', '1.3.1'), ('1.10.0', '1.9.0', '1.11.0')]
)
def test_style_version_accepts_the_floor_and_compares_numbers(
    monkeypatch, found, minimum, above
):
    """The floor itself passes, and 1.10 is newer than 1.9 (not older, as text order says)."""
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: found)
    assert tool.style_version(minimum) == found
    with pytest.raises(tool.FigureError, match=f'{found} is older than {above}'):
        tool.style_version(above)


def test_style_version_refuses_an_older_and_an_absent_package(monkeypatch):
    """An older package and a missing package stop the tool, each with its own message."""
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '1.2.9')
    with pytest.raises(tool.FigureError, match='proteus-mpl 1.2.9 is older than 1.3.0'):
        tool.style_version('1.3.0')

    def absent(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, 'version', absent)
    with pytest.raises(tool.FigureError, match='proteus-mpl is not installed'):
        tool.style_version('1.3.0')


# --- drawing ----------------------------------------------------------------------


def test_both_variants_are_written_under_the_active_style(site, capsys):
    """The maker sees the text colour of each variant, and each file has its surface colour."""
    record = site.repo / 'record.txt'
    entry = _entry(args={'record': str(record), 'note': '{runs}'})
    before, text_before = _files(site.runs), matplotlib.rcParams['text.color']
    assert tool.make_figures(_manifest(site.toml, [entry]), site.runs) == 0

    lines = [line.split() for line in record.read_text().splitlines()]
    light_text, dark_text = lines[0][0], lines[1][0]
    assert [line[1:] for line in lines] == [[str(site.runs), '1']] * 2
    assert to_rgb(light_text) != to_rgb(dark_text)
    assert to_rgb(dark_text) != to_rgb('black') and sum(to_rgb(dark_text)) > 2.0
    assert sum(to_rgb(light_text)) < 1.0

    assets = site.repo / 'docs' / 'assets' / 'tutorials'
    surface = {}
    for name, variant in (('fig_a.avif', 'light'), ('fig_a_dark.avif', 'dark')):
        with matplotlib.rc_context():
            proteus_mpl.use(variant)
            surface[name] = matplotlib.rcParams['figure.facecolor']
    assert to_rgb(surface['fig_a.avif']) != to_rgb(surface['fig_a_dark.avif'])
    sizes = []
    for name, colour in surface.items():
        with Image.open(assets / name) as image:
            sizes.append(image.size)
            corner = image.convert('RGB').getpixel((2, 2))
        expected = [round(255 * c) for c in to_rgb(colour)]
        assert corner == pytest.approx(expected, abs=6)
    assert sizes[0] == sizes[1] and sizes[0][0] > 400
    assert _files(site.runs) == before
    assert matplotlib.rcParams['text.color'] == text_before
    assert plt.get_fignums() == []
    out = capsys.readouterr().out
    assert f'proteus-mpl {importlib.metadata.version("proteus-mpl")}, matplotlib' in out
    assert out.count('cut_off: 0') == 2 and 'ERROR' not in out


def test_only_selects_by_id_and_by_page(site):
    """`only` draws the entries of one id or one page; a name that matches none is an error."""
    second = _entry(id='fig_b', page='page_b.md', asset='docs/assets/tutorials/fig_b')
    manifest = _manifest(site.toml, [_entry(), second])
    out = site.repo / 'out'
    tool.make_figures(manifest, site.runs, out, only='fig_b')
    assert _files(out) == [
        'docs/assets/tutorials/fig_b.avif',
        'docs/assets/tutorials/fig_b_dark.avif',
    ]
    tool.make_figures(manifest, site.runs, out, only='page_a.md')
    assert len(_files(out)) == 4
    with pytest.raises(tool.FigureError, match='no figure with id or page `page_c.md`'):
        tool.make_figures(manifest, site.runs, out, only='page_c.md')


def test_single_variant_entry_writes_one_file(site):
    """An entry with the dark variant only writes the dark file only."""
    out = site.repo / 'out'
    manifest = _manifest(site.toml, [_entry(variants=['dark'])])
    assert tool.make_figures(manifest, site.runs, out) == 0
    assert _files(out) == ['docs/assets/tutorials/fig_a_dark.avif']
    with Image.open(out / 'docs/assets/tutorials/fig_a_dark.avif') as image:
        assert sum(image.convert('RGB').getpixel((2, 2))) < 60


@pytest.mark.parametrize('maker', ['add_file_to_run', 'rewrite_file_of_run'])
def test_maker_that_changes_a_run_folder_stops_the_tool(site, maker):
    """A new file and a rewritten file in a run folder both stop the tool before a write."""
    out = site.repo / 'out'
    manifest = _manifest(site.toml, [_entry(maker=f'tools/fake_maker.py:{maker}')])
    with pytest.raises(tool.FigureError, match='fig_a: a run folder changed during the draw'):
        tool.make_figures(manifest, site.runs, out)
    assert not out.exists()
    assert plt.get_fignums() == []


def test_file_that_goes_away_during_the_walk_is_a_named_fault(site, monkeypatch):
    """A path that is gone when the tool reads its state is a `FigureError`, not a traceback."""
    monkeypatch.setattr(
        Path, 'rglob', lambda self, pattern: iter([self / 'removed_by_another_process'])
    )
    with pytest.raises(tool.FigureError) as err:
        tool.make_figures(_manifest(site.toml), site.runs, site.repo / 'out')
    assert 'a run folder changed during the draw' in str(err.value)
    assert 'removed_by_another_process is gone' in str(err.value)
    assert not (site.repo / 'out').exists()


def test_quality_of_the_manifest_reaches_the_encoder(site):
    """A low `avif_quality` gives smaller files than a high one, for both variants."""
    sizes = {}
    for quality in (5, 95):
        out = site.repo / f'out_{quality}'
        tool.make_figures(_manifest(site.toml, avif_quality=quality), site.runs, out)
        sizes[quality] = [(out / name).stat().st_size for name in _files(out)]
    assert len(sizes[5]) == len(sizes[95]) == 2
    assert all(low < 0.8 * high for low, high in zip(sizes[5], sizes[95]))


def test_run_option_maps_a_manifest_name_to_a_folder_under_the_watch(site, tmp_path, capsys):
    """`--run name=folder` draws from a folder of another name; that folder is read only too."""
    elsewhere = tmp_path / 'elsewhere' / 'run_20260101_000000_abcd'
    elsewhere.parent.mkdir()
    site.run.rename(elsewhere)
    record, out = site.repo / 'record.txt', site.repo / 'out'
    _manifest(site.toml, [_entry(runs=['quick_start'], args={'record': str(record)})])
    base = ['--manifest', str(site.toml), '--runs', str(site.runs), '--out', str(out)]
    assert tool.main(base) == 1
    assert 'quick_start does not exist' in capsys.readouterr().err
    assert tool.main([*base, '--run', f'quick_start={elsewhere}']) == 0
    assert len(_files(out)) == 2 and len(record.read_text().splitlines()) == 2
    # A name that is not in the manifest and an entry without a folder are refused
    for option in (f'quick={elsewhere}', 'quick_start', 'quick_start='):
        assert tool.main([*base, '--run', option]) == 1
        assert 'give NAME=FOLDER with a run name of the manifest' in capsys.readouterr().err
    _manifest(
        site.toml, [_entry(runs=['quick_start'], maker='tools/fake_maker.py:add_file_to_run')]
    )
    (elsewhere / 'plots').mkdir(exist_ok=True)
    assert tool.main([*base, '--run', f'quick_start={elsewhere}']) == 1
    assert 'a run folder changed during the draw' in capsys.readouterr().err


def test_main_passes_the_output_root_and_the_chili_folder(site, tmp_path, capsys):
    """`--out` and `--chili-repo` reach the maker: the files go under the root, the path into the argument."""
    record, out, chili = site.repo / 'record.txt', tmp_path / 'root', tmp_path / 'chili-repo'
    _manifest(site.toml, [_entry(args={'record': str(record), 'note': '{chili_repo}/data'})])
    options = ['--manifest', str(site.toml), '--runs', str(site.runs)]
    assert tool.main([*options, '--out', str(out)]) == 1
    assert 'argument `note` needs --chili-repo' in capsys.readouterr().err
    assert tool.main([*options, '--out', str(out), '--chili-repo', str(chili)]) == 0
    assert [line.split()[1] for line in record.read_text().splitlines()] == [
        f'{chili}/data'
    ] * 2
    assert _files(out) == [
        'docs/assets/tutorials/fig_a.avif',
        'docs/assets/tutorials/fig_a_dark.avif',
    ]
    assets = site.repo / 'docs' / 'assets' / 'tutorials'
    assert (assets / 'fig_a.avif').read_bytes() == b'placeholder'


@pytest.mark.parametrize(
    'changes, message',
    [
        ({'runs': ['run_b']}, 'run_b does not exist'),
        ({'maker': 'tools/absent.py:draw'}, 'maker script tools/absent.py does not exist'),
        ({'maker': 'tools/fake_maker.py:absent'}, 'tools/fake_maker.py has no function absent'),
        ({'maker': 'tools/fake_maker.py:NOT_A_FUNCTION'}, 'has no function NOT_A_FUNCTION'),
        ({'maker': 'tools/fake_maker.py:write_nothing'}, 'the maker wrote no file fig_a.png'),
        (
            {'maker': 'tools/fake_maker.py:save_through_the_canvas'},
            'the maker saved no figure with `Figure.savefig`',
        ),
        ({'maker': 'proteus:absent'}, 'no `proteus plot` maker named absent'),
        ({'args': {'note': '{chili_repo}/data'}}, 'argument `note` needs --chili-repo'),
    ],
)
def test_entry_that_breaks_the_contract_writes_nothing(site, changes, message):
    """A missing run, script, function, output file or option stops the tool with its name."""
    out = site.repo / 'out'
    with pytest.raises(tool.FigureError) as err:
        tool.make_figures(_manifest(site.toml, [_entry(**changes)]), site.runs, out)
    assert message in str(err.value)
    assert not out.exists()


def test_placeholders_are_filled_from_the_options(site):
    """`{chili_repo}` and `{runs}` are replaced inside a string; other values pass unchanged."""
    places = {'{runs}': Path('/r'), '{chili_repo}': Path('/c')}
    args = {'a': '{chili_repo}/x', 'b': '{runs}', 'c': 3, 'd': 'plain', 'e': True}
    assert tool._fill(args, 'fig_a', places) == {
        'a': '/c/x',
        'b': '/r',
        'c': 3,
        'd': 'plain',
        'e': True,
    }
    with pytest.raises(tool.FigureError, match='fig_a: argument `b` needs --runs'):
        tool._fill(args, 'fig_a', {'{runs}': None, '{chili_repo}': Path('/c')})


def test_text_that_the_file_cuts_off_is_counted_and_fails_the_command(site, capsys):
    """A text outside the saved area is 1 error per variant, and `main` returns 1."""
    _manifest(site.toml, [_entry(args={'stray': True})])
    out = site.repo / 'out'
    code = tool.main(
        ['--manifest', str(site.toml), '--runs', str(site.runs), '--out', str(out)]
    )
    assert code == 1
    printed = capsys.readouterr().out
    assert printed.count('ERROR cut_off: 1') == 2
    assert "(-1, 'text', 'stray')" in printed
    assert 'fig_a: NOT WRITTEN, the saved file would cut off 2 text(s)' in printed
    assert '2 text(s) cut off in the saved files' in printed
    assert not out.exists()


def test_failure_of_the_second_encode_leaves_the_committed_pair(site, monkeypatch):
    """When the dark file cannot be encoded, the light file in place stays the old one."""
    assets = site.repo / 'docs' / 'assets' / 'tutorials'
    real_save = Image.Image.save

    def save(self, target, *args, **kwargs):
        if '_dark.avif' in Path(target).name:
            raise OSError('no space left')
        return real_save(self, target, *args, **kwargs)

    monkeypatch.setattr(Image.Image, 'save', save)
    with pytest.raises(OSError, match='no space left'):
        tool.make_figures(_manifest(site.toml), site.runs)
    assert (assets / 'fig_a.avif').read_bytes() == b'placeholder'
    assert (assets / 'fig_a_dark.avif').read_bytes() == b'placeholder'
    assert sorted(p.name for p in assets.iterdir()) == ['fig_a.avif', 'fig_a_dark.avif']
    monkeypatch.setattr(Image.Image, 'save', real_save)
    assert tool.make_figures(_manifest(site.toml), site.runs) == 0
    assert all(len((assets / n).read_bytes()) > 100 for n in ('fig_a.avif', 'fig_a_dark.avif'))


def test_failure_in_the_second_variant_leaves_the_first_unwritten(site):
    """A maker that fails for the dark variant leaves no light file of that entry behind."""
    out = site.repo / 'out'
    first = _entry(id='fig_0', asset='docs/assets/tutorials/fig_0')
    failing = _entry(maker='tools/fake_maker.py:fail_in_dark')
    with pytest.raises(tool.FigureError, match='fig_a: RuntimeError: no dark figure'):
        tool.make_figures(_manifest(site.toml, [first, failing]), site.runs, out)
    assert _files(out) == [
        'docs/assets/tutorials/fig_0.avif',
        'docs/assets/tutorials/fig_0_dark.avif',
    ]
    assert plt.get_fignums() == []


def test_pillow_without_avif_stops_the_tool(site, monkeypatch):
    """Without AVIF support nothing is drawn."""
    monkeypatch.setattr(features, 'check', lambda name: False)
    with pytest.raises(tool.FigureError, match='cannot write AVIF'):
        tool.make_figures(_manifest(site.toml), site.runs, site.repo / 'out')
    assert not (site.repo / 'out').exists()


# --- makers for the `proteus plot` functions --------------------------------------


@pytest.fixture
def plot_calls(monkeypatch):
    """Replace the 4 plot functions and the run reader by recorders."""
    import proteus.orbit.wrapper
    import proteus.plot.cpl_fluxes_global
    import proteus.plot.cpl_global
    import proteus.plot.cpl_orbit

    calls = []
    config = SimpleNamespace(
        orbit=SimpleNamespace(
            module='obliqua', satellite=SimpleNamespace(include_satellite=True)
        )
    )
    monkeypatch.setattr(
        tool, '_run_inputs', lambda run_dir: (config, f'helpfile of {run_dir.name}')
    )

    def record(module, function, name):
        """Replace a plot function; a call that its real signature refuses raises TypeError."""
        signature = inspect.signature(getattr(module, function))

        def recorder(*args, **kwargs):
            signature.bind(*args, **kwargs)
            calls.append((name, args, kwargs))

        monkeypatch.setattr(module, function, recorder)

    record(proteus.plot.cpl_global, 'plot_global', 'global')
    record(proteus.plot.cpl_orbit, 'plot_orbit', 'orbit')
    record(proteus.plot.cpl_orbit, 'plot_lovenumber', 'lovenumber')
    record(proteus.plot.cpl_fluxes_global, 'plot_fluxes_global', 'fluxes')
    assert list(inspect.signature(proteus.orbit.wrapper.read_tides_data).parameters) == [
        'output_dir',
        'model',
        'times',
    ]
    monkeypatch.setattr(
        proteus.orbit.wrapper,
        'read_tides_data',
        lambda output_dir, model, times: [f'{model} {t}' for t in times],
    )
    return SimpleNamespace(calls=calls, config=config, record=record)


def test_proteus_makers_pass_the_run_data_and_the_output_folder(
    site, plot_calls, tmp_path, monkeypatch
):
    """Each maker calls its plot function with the data of the run and names the file it writes."""
    out, hf = tmp_path / 'work', 'helpfile of run_a'
    makers = tool.PROTEUS_MAKERS
    assert makers['global'](site.run, out, logt=False, tmin=10.0) == 'plot_global_lin.png'
    assert makers['global'](site.run, out) == 'plot_global_log.png'
    assert makers['orbit'](site.run, out) == 'plot_orbit.png'
    assert makers['fluxes_global'](site.run, out) == 'plot_fluxes_global.png'
    expected = {'hf_all': hf, 'output_dir': str(out), 'config': plot_calls.config}
    assert plot_calls.calls == [
        ('global', (), {**expected, 'logt': False, 'tmin': 10.0}),
        ('global', (), {**expected, 'logt': True, 'tmin': 1e3}),
        ('orbit', (hf, str(out), True), {'plot_format': 'png'}),
        ('fluxes', (hf, str(out), plot_calls.config), {}),
    ]
    # A plot function that loses a parameter refuses the call of its adapter
    import proteus.plot.cpl_orbit as cpl_orbit

    monkeypatch.setattr(cpl_orbit, 'plot_orbit', lambda hf_all, output_dir, has_sat: None)
    plot_calls.record(cpl_orbit, 'plot_orbit', 'orbit')
    with pytest.raises(TypeError, match="unexpected keyword argument 'plot_format'"):
        makers['orbit'](site.run, out)
    assert len(plot_calls.calls) == 4


def test_lovenumber_maker_reads_every_snapshot_in_time_order(site, plot_calls, tmp_path):
    """The snapshot times come from the file names of the tides module, sorted as numbers."""
    data = site.run / 'data'
    data.mkdir()
    for name in ('100_obliqua.nc', '3_obliqua.nc', '20_obliqua.nc', '7_other.nc', '5_int.nc'):
        (data / name).write_text('')
    out = tmp_path / 'work'
    assert tool.PROTEUS_MAKERS['lovenumber'](site.run, out) == 'plot_lovenumber.png'
    assert plot_calls.calls == [
        (
            'lovenumber',
            (),
            {
                'output_dir': str(out),
                'times': [3, 20, 100],
                'data': ['obliqua 3', 'obliqua 20', 'obliqua 100'],
                'plot_format': 'png',
            },
        )
    ]


def test_proteus_maker_draws_into_a_folder_outside_the_run(site, monkeypatch):
    """The tool gives a `proteus plot` maker a temporary folder and converts the file it names."""
    seen = []

    def fake_global(run_dir, out_dir, logt=True):
        seen.append((run_dir, out_dir, logt, matplotlib.rcParams['figure.facecolor']))
        fig = plt.figure(figsize=(1, 1))
        fig.savefig(out_dir / 'plots' / 'plot_x.png')
        return 'plot_x.png'

    monkeypatch.setitem(tool.PROTEUS_MAKERS, 'global', fake_global)
    entry = _entry(maker='proteus:global', args={'logt': False})
    out = site.repo / 'out'
    assert tool.make_figures(_manifest(site.toml, [entry]), site.runs, out) == 0
    assert [(s[0], s[2]) for s in seen] == [(site.run, False)] * 2
    assert all(site.runs not in s[1].parents and not s[1].exists() for s in seen)
    assert to_rgb(seen[0][3]) != to_rgb(seen[1][3])
    assert len(_files(out)) == 2


def test_run_inputs_reads_the_resolved_config_and_the_helpfile(tmp_path):
    """The config of the run folder is read with PNG as plot format, and the helpfile by column."""
    run = tmp_path / 'run'
    run.mkdir()
    resolved = tomlkit.parse((_REPO_ROOT / 'input' / 'dummy.toml').read_text())
    resolved['params']['out']['plot_fmt'] = 'pdf'
    (run / 'init_coupler.toml').write_text(tomlkit.dumps(resolved))
    (run / 'runtime_helpfile.csv').write_text('Time\tT_surf\n0.0\t3000.0\n10.5\t2900.0\n')
    from proteus.config import read_config_object

    assert read_config_object(str(run / 'init_coupler.toml')).params.out.plot_fmt == 'pdf'
    config, hf_all = tool._run_inputs(run)
    assert config.params.out.plot_fmt == 'png'
    assert list(hf_all.columns) == ['Time', 'T_surf']
    assert hf_all['Time'].tolist() == pytest.approx([0.0, 10.5])


# --- command line -----------------------------------------------------------------


def test_main_check_names_the_exempt_pages_and_fails_on_a_problem(site, capsys):
    """`--check` prints each exempt page, returns 0 on agreement and 1 with the problems."""
    tutorials = site.repo / 'docs' / 'Tutorials'
    (tutorials / 'old.md').write_text('![Old](../assets/tutorials/old.png)\n')
    _manifest(site.toml, exempt_pages=['old.md'])
    assert tool.main(['--check', '--manifest', str(site.toml)]) == 0
    printed = capsys.readouterr().out
    assert 'not checked (exempt): docs/Tutorials/old.md' in printed
    assert '1 figures agree with the pages' in printed

    _manifest(site.toml, exempt_pages=['old.md', 'gone.md'])
    assert tool.main(['--check', '--manifest', str(site.toml)]) == 1
    assert 'exempt page gone.md does not exist' in capsys.readouterr().out
    _manifest(site.toml)
    assert tool.main(['--check', '--manifest', str(site.toml)]) == 1
    assert 'old.md: image docs/assets/tutorials/old.png has no manifest entry' in (
        capsys.readouterr().out
    )


def test_main_requires_the_run_folder_unless_it_checks(site, capsys):
    """Without `--runs` and without `--check` the command exits with a usage error."""
    _manifest(site.toml)
    with pytest.raises(SystemExit) as err:
        tool.main(['--manifest', str(site.toml)])
    assert err.value.code == 2
    assert '--runs is required unless --check is given' in capsys.readouterr().err


def test_main_draws_into_the_repository_and_reports_a_fault(site, capsys):
    """The command writes the pair into the repository root; a fault gives exit code 1."""
    _manifest(site.toml)
    assert tool.main(['--manifest', str(site.toml), '--runs', str(site.runs)]) == 0
    assets = site.repo / 'docs' / 'assets' / 'tutorials'
    assert all((assets / n).stat().st_size > 100 for n in ('fig_a.avif', 'fig_a_dark.avif'))
    assert '0 text(s) cut off in the saved files' in capsys.readouterr().out

    code = tool.main(['--manifest', str(site.toml), '--runs', str(site.runs), '--only', 'x'])
    assert code == 1
    assert 'ERROR: no figure with id or page `x`' in capsys.readouterr().err


def test_main_reports_an_error_of_a_maker_as_its_own_error_line(site, capsys):
    """An exception of a maker ends the command with exit code 1 and one `ERROR:` line that
    names the figure, with no traceback, and leaves no file of that figure."""
    _manifest(site.toml, [_entry(maker='tools/fake_maker.py:fail_in_dark')])
    assert tool.main(['--manifest', str(site.toml), '--runs', str(site.runs)]) == 1
    printed = capsys.readouterr()
    assert printed.err.strip() == 'ERROR: fig_a: RuntimeError: no dark figure'
    assert 'Traceback' not in printed.out + printed.err
    assets = site.repo / 'docs' / 'assets' / 'tutorials'
    assert (assets / 'fig_a.avif').read_bytes() == b'placeholder'
