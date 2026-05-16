"""Unit tests for ``proteus.grid.pack``.

Covers the ``pack`` entrypoint: invalid-path error, no-case-dir error,
file-copy with and without plots, and zip-with-cleanup logic. Uses real
filesystem operations because pack is a thin wrapper around shutil +
zipfile primitives that would amount to mocking the function under test.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from zipfile import ZipFile

import pytest

import proteus.grid.pack as pack_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid_with_cases(tmp_path, n_cases: int = 2, with_plots: bool = True):
    """Build a minimal grid layout:

        grid/
            manager.log
            ref_config.toml
            copy.grid.toml
            case_000000/
                runtime_helpfile.csv
                init_coupler.toml
                status
                proteus_00.log
                plots/
                    plot_global.png
                    plot_orbit.pdf
            case_000001/
                ...
    """
    grid = tmp_path / 'grid'
    grid.mkdir()
    # Top-level files
    (grid / 'manager.log').write_text('top-level log', encoding='utf-8')
    (grid / 'ref_config.toml').write_text('ref-config', encoding='utf-8')
    (grid / 'copy.grid.toml').write_text('grid-copy', encoding='utf-8')

    for i in range(n_cases):
        case = grid / f'case_{i:06d}'
        case.mkdir()
        (case / 'runtime_helpfile.csv').write_text(f'helpfile-{i}', encoding='utf-8')
        (case / 'init_coupler.toml').write_text(f'init-{i}', encoding='utf-8')
        (case / 'status').write_text('10\nDone\n', encoding='utf-8')
        (case / 'proteus_00.log').write_text(f'log-{i}', encoding='utf-8')
        if with_plots:
            plots = case / 'plots'
            plots.mkdir()
            (plots / 'plot_global.png').write_bytes(b'png-bytes')
            (plots / 'plot_orbit.pdf').write_bytes(b'pdf-bytes')
            # File that does NOT match the plot_* prefix; must be skipped.
            (plots / 'auxiliary.txt').write_text('aux', encoding='utf-8')

    return grid


# ---------------------------------------------------------------------------
# pack: invalid input
# ---------------------------------------------------------------------------


def test_pack_raises_when_grid_directory_missing(tmp_path):
    """Non-existent grid path raises FileNotFoundError. A regression that
    proceeded would create a pack/ folder at the parent.
    """
    with pytest.raises(FileNotFoundError, match='Invalid path'):
        pack_mod.pack(str(tmp_path / 'missing'))


def test_pack_raises_when_grid_has_no_cases(tmp_path):
    """An empty grid directory (no case_* subfolders) raises
    FileNotFoundError. Discrimination: the pack/ folder is created by
    pack before the check, but the error is still raised loudly.
    """
    grid = tmp_path / 'grid'
    grid.mkdir()
    (grid / 'manager.log').write_text('log', encoding='utf-8')
    (grid / 'ref_config.toml').write_text('ref', encoding='utf-8')
    (grid / 'copy.grid.toml').write_text('copy', encoding='utf-8')

    with pytest.raises(FileNotFoundError, match='subfolders containing grid cases'):
        pack_mod.pack(str(grid))


# ---------------------------------------------------------------------------
# pack: file-copy semantics
# ---------------------------------------------------------------------------


def test_pack_without_zip_creates_pack_directory_with_copied_files(tmp_path):
    """pack(zip=False) leaves the pack/ folder intact with the copied
    top-level files and per-case data. Discrimination: the zip step is
    skipped (no pack.zip), and the pack/ folder is not removed.
    """
    grid = _make_grid_with_cases(tmp_path, n_cases=2)
    result = pack_mod.pack(str(grid), zip=False)

    pack_dir = grid / 'pack'
    assert result is True
    # Discrimination: pack folder exists with the 3 top-level files
    assert (pack_dir / 'manager.log').exists()
    assert (pack_dir / 'ref_config.toml').exists()
    assert (pack_dir / 'copy.grid.toml').exists()
    # Per-case data present in both case subfolders
    for i in range(2):
        case_dest = pack_dir / f'case_{i:06d}'
        assert case_dest.exists()
        assert (case_dest / 'runtime_helpfile.csv').exists()
        assert (case_dest / 'status').exists()
        # Plot file copied flat into case_dest (not under plots/)
        assert (case_dest / 'plot_global.png').exists()
        assert (case_dest / 'plot_orbit.pdf').exists()
        # The auxiliary.txt file must NOT have been copied (only plot_* names)
        assert not (case_dest / 'auxiliary.txt').exists()
    # No zip created when zip=False
    assert not (grid / 'pack.zip').exists()


def test_pack_with_plots_false_skips_plot_copies(tmp_path):
    """plots=False prevents the inner plot_* copy loop. Discrimination:
    the case folder in pack/ has runtime_helpfile.csv but no plot files.
    """
    grid = _make_grid_with_cases(tmp_path, n_cases=1)
    pack_mod.pack(str(grid), plots=False, zip=False)

    case_dest = grid / 'pack' / 'case_000000'
    assert (case_dest / 'runtime_helpfile.csv').exists()
    # Discrimination: no plot_* files in the destination
    assert not (case_dest / 'plot_global.png').exists()
    assert not (case_dest / 'plot_orbit.pdf').exists()


# ---------------------------------------------------------------------------
# pack: zip + cleanup
# ---------------------------------------------------------------------------


def test_pack_with_zip_creates_archive_and_removes_pack_dir(tmp_path):
    """zip=True (default) creates pack.zip and (when rmdir_pack=True
    default) removes the pack/ folder. Discrimination: the zip contains
    pack/<files> entries (the 'pack' top-level folder is preserved
    inside the archive).
    """
    grid = _make_grid_with_cases(tmp_path, n_cases=1)
    pack_mod.pack(str(grid), zip=True, rmdir_pack=True)

    zip_path = grid / 'pack.zip'
    assert zip_path.exists()
    # Pack folder removed
    assert not (grid / 'pack').exists()
    # Discrimination: zip contains pack/<file> entries, NOT raw <file> entries
    with ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
    assert any(n.startswith('pack/') for n in names)
    assert any(n.endswith('manager.log') for n in names)
    assert any(n.endswith('runtime_helpfile.csv') for n in names)


def test_pack_with_zip_keeps_pack_dir_when_rmdir_pack_false(tmp_path):
    """rmdir_pack=False keeps the pack/ folder alongside the new zip.
    Discrimination: both pack.zip and pack/ coexist.
    """
    grid = _make_grid_with_cases(tmp_path, n_cases=1)
    pack_mod.pack(str(grid), zip=True, rmdir_pack=False)

    assert (grid / 'pack.zip').exists()
    assert (grid / 'pack').exists()
    assert (grid / 'pack' / 'manager.log').exists()


def test_pack_replaces_existing_zip_on_rerun(tmp_path):
    """A pre-existing pack.zip is removed and recreated. Discrimination:
    the new zip's mtime is later than the old one's, OR the contents
    reflect the latest grid state.
    """
    grid = _make_grid_with_cases(tmp_path, n_cases=1)
    # Pre-seed an old pack.zip with bogus content
    old_zip = grid / 'pack.zip'
    old_zip.write_bytes(b'old-junk')
    old_size = old_zip.stat().st_size

    pack_mod.pack(str(grid), zip=True, rmdir_pack=True)

    new_size = old_zip.stat().st_size
    # Discrimination: the new zip is a real archive (much bigger than 8
    # bytes of junk) and is openable as a zip
    assert new_size > old_size + 100
    with ZipFile(old_zip, 'r') as zf:
        assert len(zf.namelist()) > 0
