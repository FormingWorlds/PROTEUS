"""Unit tests for ``proteus.utils.archive``.

Covers ``archive_exists``, ``create``, ``append``, ``extract``,
``update``, and ``remove_old``. Uses real tarfile + tmp_path filesystem
operations because the functions are thin wrappers around tarfile and
os.* primitives; mocking those would amount to mocking the function
under test.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import os
import tarfile
from collections import Counter

import pytest

import proteus.utils.archive as archive_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _tar_member_counts(tar_path) -> Counter:
    """Return a Counter of member names inside a tar (duplicates counted)."""
    with tarfile.open(tar_path, 'r') as tar:
        return Counter(tar.getnames())


# ---------------------------------------------------------------------------
# archive_exists
# ---------------------------------------------------------------------------


def test_archive_exists_returns_false_when_no_tar(tmp_path, caplog):
    """A directory with no `<basename>.tar` returns False and logs a
    warning unless ignore_warnings is True. Discrimination against a
    regression that returned None or raised: the return must be exactly
    False.
    """
    with caplog.at_level('WARNING'):
        result = archive_mod.archive_exists(str(tmp_path), ignore_warnings=False)
    assert result is False
    # Discrimination: a warning must have been logged about the missing tar
    assert any('does not exist' in record.message for record in caplog.records)


def test_archive_exists_returns_true_when_tar_present(tmp_path, caplog):
    """When `<basename>.tar` exists inside the directory, the function
    returns True. The basename is taken from os.path.split(dir)[-1].
    Discrimination: no warning should be emitted on the True path
    (mirroring the False-path test). A regression that always logged
    the missing-tar warning would fail this.
    """
    tar_path = tmp_path / f'{tmp_path.name}.tar'
    tar_path.write_bytes(b'fake-tar-content')

    with caplog.at_level('WARNING'):
        result = archive_mod.archive_exists(str(tmp_path))
    assert result is True
    assert not any('does not exist' in record.message for record in caplog.records)


def test_archive_exists_suppresses_warning_with_flag(tmp_path, caplog):
    """ignore_warnings=True must suppress the missing-tar warning even
    when the tar is absent. Discrimination: the caplog must be empty of
    warnings even though the function returns False.
    """
    with caplog.at_level('WARNING'):
        result = archive_mod.archive_exists(str(tmp_path), ignore_warnings=True)
    assert result is False
    assert not any('does not exist' in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_archives_files_and_removes_them(tmp_path):
    """Create packs a directory's files into <dir>/<basename>.tar and,
    by default, removes the originals. The tar must exist after create
    returns, and the original files must be gone.
    """
    # Populate directory with two files
    (tmp_path / 'a.txt').write_text('hello', encoding='utf-8')
    (tmp_path / 'b.txt').write_text('world', encoding='utf-8')

    tar_path = archive_mod.create(str(tmp_path), remove_files=True)

    expected_tar = str(tmp_path / f'{tmp_path.name}.tar')
    assert tar_path == expected_tar
    assert os.path.exists(expected_tar)
    # Discrimination: the originals must be removed when remove_files=True
    assert not (tmp_path / 'a.txt').exists()
    assert not (tmp_path / 'b.txt').exists()


def test_create_preserves_files_when_remove_files_false(tmp_path):
    """remove_files=False keeps the original files alongside the new tar.
    A regression that always removed files would leave only the tar.
    """
    (tmp_path / 'a.txt').write_text('hello', encoding='utf-8')

    archive_mod.create(str(tmp_path), remove_files=False)

    # Discrimination: both the tar AND the original must coexist
    assert (tmp_path / f'{tmp_path.name}.tar').exists()
    assert (tmp_path / 'a.txt').exists()


def test_create_refuses_if_archive_already_exists(tmp_path, caplog):
    """When the tar already exists, create logs an error and returns
    None rather than overwriting. Discrimination: a regression that
    overwrote the tar would silently lose the original content.
    """
    tar_path = tmp_path / f'{tmp_path.name}.tar'
    tar_path.write_bytes(b'original')

    with caplog.at_level('ERROR'):
        result = archive_mod.create(str(tmp_path))

    assert result is None
    assert any('already exists' in record.message for record in caplog.records)
    # Discrimination: the original tar content must be unchanged
    assert tar_path.read_bytes() == b'original'


def test_create_returns_none_when_directory_missing(tmp_path, caplog):
    """If the input dir doesn't exist, create logs an error and returns
    None. Discrimination: no tar file should be created at the parent.
    """
    missing = tmp_path / 'missing'
    with caplog.at_level('ERROR'):
        result = archive_mod.create(str(missing))
    assert result is None
    assert any('does not exist' in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# append + update
# ---------------------------------------------------------------------------


def test_append_adds_new_files_to_existing_archive(tmp_path):
    """append() adds files to an existing tar. After create() + a new
    file + append(), the tar must contain both the original and the
    new file. Discrimination via reading back the tar contents.
    """
    (tmp_path / 'a.txt').write_text('first', encoding='utf-8')
    archive_mod.create(str(tmp_path), remove_files=True)
    # Add a second file and append it
    (tmp_path / 'b.txt').write_text('second', encoding='utf-8')
    archive_mod.append(str(tmp_path), remove_files=False)

    import tarfile

    tar_path = tmp_path / f'{tmp_path.name}.tar'
    with tarfile.open(tar_path, 'r') as tar:
        names = sorted(tar.getnames())
    # Discrimination: both 'a.txt' and 'b.txt' must be in the archive
    assert 'a.txt' in names
    assert 'b.txt' in names


def test_append_returns_none_when_archive_missing(tmp_path, caplog):
    """When the tar doesn't exist, append() logs an error and returns
    None. A regression that silently created an archive would mask the
    user's misuse of the API.
    """
    (tmp_path / 'a.txt').write_text('content', encoding='utf-8')
    with caplog.at_level('ERROR'):
        result = archive_mod.append(str(tmp_path))
    assert result is None
    # Discrimination: no tar must have been created
    assert not (tmp_path / f'{tmp_path.name}.tar').exists()


def test_update_creates_when_missing_and_appends_when_present(tmp_path):
    """update() is the high-level entrypoint: it creates a fresh archive
    if none exists, otherwise appends. After two update() calls on a
    directory with one file each time, the archive must contain both.
    """
    # First call: directory empty except for a.txt -> creates
    (tmp_path / 'a.txt').write_text('first', encoding='utf-8')
    archive_mod.update(str(tmp_path), remove_files=True)
    assert (tmp_path / f'{tmp_path.name}.tar').exists()

    # Second call: new file in directory -> appends
    (tmp_path / 'b.txt').write_text('second', encoding='utf-8')
    archive_mod.update(str(tmp_path), remove_files=True)

    import tarfile

    with tarfile.open(tmp_path / f'{tmp_path.name}.tar', 'r') as tar:
        names = sorted(tar.getnames())
    # Discrimination: both files must be in the archive (one via create,
    # one via append).
    assert 'a.txt' in names
    assert 'b.txt' in names


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def test_extract_restores_files_and_optionally_removes_tar(tmp_path):
    """extract() unpacks the tar's contents into the directory. When
    remove_tar=True, the tar is deleted afterwards. Discrimination:
    file content must match the pre-archive state.
    """
    # Build a tar from a.txt with known content
    (tmp_path / 'a.txt').write_text('hello', encoding='utf-8')
    archive_mod.create(str(tmp_path), remove_files=True)
    assert not (tmp_path / 'a.txt').exists()  # confirms create() removed it

    archive_mod.extract(str(tmp_path), remove_tar=True)

    # Discrimination: a.txt restored with original content; tar removed
    assert (tmp_path / 'a.txt').read_text(encoding='utf-8') == 'hello'
    assert not (tmp_path / f'{tmp_path.name}.tar').exists()


def test_extract_keeps_tar_when_remove_tar_false(tmp_path):
    """remove_tar=False (default) leaves the tar in place after extraction."""
    (tmp_path / 'a.txt').write_text('content', encoding='utf-8')
    archive_mod.create(str(tmp_path), remove_files=True)
    archive_mod.extract(str(tmp_path), remove_tar=False)

    assert (tmp_path / 'a.txt').exists()
    assert (tmp_path / f'{tmp_path.name}.tar').exists()


def test_extract_returns_none_when_directory_missing(tmp_path, caplog):
    """If the directory doesn't exist, extract logs an error and returns
    None. Discrimination: an error message must have been recorded, and
    no tar must have been created at the parent (i.e. extract didn't
    silently fall through and write something unexpected).
    """
    missing = tmp_path / 'absent'
    with caplog.at_level('ERROR'):
        result = archive_mod.extract(str(missing))
    assert result is None
    assert any('does not exist' in record.message for record in caplog.records)
    assert not (tmp_path / f'{missing.name}.tar').exists()


# ---------------------------------------------------------------------------
# remove_old
# ---------------------------------------------------------------------------


def test_remove_old_keeps_archive_and_recent_snapshots(tmp_path):
    """remove_old prunes timestamped snapshots whose age (parsed from the
    filename prefix) is below the `before` cutoff and keeps everything
    else: the .tar archive, recent snapshots, and fixed-name runtime
    files that the interior modules re-read between structure re-solves.

    Discrimination: with three snapshots at ages 100, 1000, 10000 and a
    cutoff of 500, only the age-100 snapshot may disappear. The runtime
    hand-off files (zalmoxis_output.dat, spider_mesh.dat) and the
    spider_eos/ table directory must survive: deleting them kills the
    next Aragog solver.reset(), which re-reads zalmoxis_output.dat.
    """
    # Pre-existing tar (must survive)
    (tmp_path / f'{tmp_path.name}.tar').write_bytes(b'archive')
    # Snapshots: <age>_int.nc and <age>_atm.nc
    (tmp_path / '100_int.nc').write_text('a', encoding='utf-8')
    (tmp_path / '1000_int.nc').write_text('b', encoding='utf-8')
    (tmp_path / '10000_atm.nc').write_text('c', encoding='utf-8')
    # Fixed-name runtime files re-read by the interior modules
    (tmp_path / 'zalmoxis_output.dat').write_text('mesh', encoding='utf-8')
    (tmp_path / 'spider_mesh.dat').write_text('mesh', encoding='utf-8')
    # EOS table directory (directories are kept wholesale)
    (tmp_path / 'spider_eos').mkdir()
    (tmp_path / 'spider_eos' / 'table.dat').write_text('eos', encoding='utf-8')

    archive_mod.remove_old(str(tmp_path), before=500)

    # tar archive remains
    assert (tmp_path / f'{tmp_path.name}.tar').exists()
    # Snapshots: age 100 removed (< 500), 1000 + 10000 kept (>= 500)
    assert not (tmp_path / '100_int.nc').exists()
    assert (tmp_path / '1000_int.nc').exists()
    assert (tmp_path / '10000_atm.nc').exists()
    # Runtime hand-off files survive the prune
    assert (tmp_path / 'zalmoxis_output.dat').exists()
    assert (tmp_path / 'spider_mesh.dat').exists()
    # The table directory survives with its contents intact
    assert (tmp_path / 'spider_eos' / 'table.dat').read_text(encoding='utf-8') == 'eos'


def test_remove_old_keeps_non_timestamped_nc_and_json_names(tmp_path):
    """A .nc/.json name without a leading integer time token is not a
    snapshot and must be kept rather than crashing the int() parse.

    Edge case for the prefix parser: 'notes.json' and 'mesh_summary.nc'
    have no integer prefix, while '0_int.nc' is the boundary-age
    snapshot (age 0 < any positive cutoff, so it is pruned). A
    regression that re-applied delete-by-default or raised ValueError on
    the parse would fail this test.
    """
    (tmp_path / 'notes.json').write_text('a', encoding='utf-8')
    (tmp_path / 'mesh_summary.nc').write_text('b', encoding='utf-8')
    # Boundary snapshot: age 0 parses cleanly and is below the cutoff
    (tmp_path / '0_int.nc').write_text('c', encoding='utf-8')

    archive_mod.remove_old(str(tmp_path), before=500)

    # Non-timestamped names kept; no exception raised on the parse
    assert (tmp_path / 'notes.json').exists()
    assert (tmp_path / 'mesh_summary.nc').exists()
    # Discrimination: the parseable age-0 snapshot IS pruned, so the
    # kept files above survive because of the name check, not because
    # the prune was a no-op.
    assert not (tmp_path / '0_int.nc').exists()


def test_remove_old_keeps_json_snapshots_alongside_nc(tmp_path):
    """The age-based gate applies to .json snapshots in addition to .nc.
    A regression that only handled .nc would leave old .json files behind.
    """
    (tmp_path / '100_int.json').write_text('a', encoding='utf-8')
    (tmp_path / '5000_int.json').write_text('b', encoding='utf-8')

    archive_mod.remove_old(str(tmp_path), before=500)

    # Discrimination: 100 (< 500) removed; 5000 (>= 500) kept
    assert not (tmp_path / '100_int.json').exists()
    assert (tmp_path / '5000_int.json').exists()


# ---------------------------------------------------------------------------
# _snapshot_time (shared timestamped-snapshot predicate)
# ---------------------------------------------------------------------------


def test_snapshot_time_recognizes_only_integer_prefixed_nc_json():
    """`_snapshot_time` returns the integer simulated time only for names
    ending in .nc/.json whose leading token parses as an integer, and None
    for everything else. This predicate is the single source of truth
    shared by the archive filter and remove_old, so its boundary
    behaviour is pinned here.

    Edge cases: the age-0 boundary snapshot parses to 0 (falsy but not
    None, so a naive truthiness check would misclassify it); the
    fixed-name runtime files, the EOS table directory name, and the
    integer-prefixed stellar flux file (.sflux, wrong extension) all
    return None; a .nc/.json name with a non-integer prefix returns None
    via the ValueError branch rather than raising.
    """
    st = archive_mod._snapshot_time
    # Recognised snapshots -> integer time
    assert st('1000_int.nc') == 1000
    assert st('1000_atm.nc') == 1000
    assert st('5000.json') == 5000
    # Boundary: age 0 must be 0, not None (distinct from "not a snapshot")
    assert st('0_int.nc') == 0
    # Fixed-name runtime files and runtime dirs -> None
    assert st('zalmoxis_output.dat') is None
    assert st('zalmoxis_output.dat.prev') is None
    assert st('spider_mesh.dat') is None
    assert st('spider_eos') is None
    # Integer prefix but wrong extension (stellar flux) -> None
    assert st('1.sflux') is None
    assert st('-1.sflux') is None
    # Right extension but non-integer prefix -> None via the ValueError guard
    assert st('notes.json') is None
    assert st('mesh_summary.nc') is None


# ---------------------------------------------------------------------------
# snapshots_only: rolling in-loop archive must not re-append runtime files
# ---------------------------------------------------------------------------


def test_update_snapshots_only_excludes_fixed_name_files_and_dirs(tmp_path):
    """snapshots_only=True archives only timestamped snapshots; the
    fixed-name runtime files, the stellar-flux inputs, and the EOS table
    directory are neither added to the tar nor removed from disk.

    This is the rolling in-loop configuration (remove_files=False): the
    interior modules re-read the fixed-name files between structure
    re-solves, so they must survive on disk while the snapshot is
    archived. Discrimination: the snapshot is in the tar, the runtime
    entries are not, and every runtime entry is still loose afterwards.
    """
    # Timestamped snapshots (should be archived)
    (tmp_path / '1000_int.nc').write_text('snap', encoding='utf-8')
    (tmp_path / '1000_atm.nc').write_text('snap', encoding='utf-8')
    # Fixed-name runtime files + flux inputs (should NOT be archived)
    (tmp_path / 'zalmoxis_output.dat').write_text('mesh', encoding='utf-8')
    (tmp_path / 'spider_mesh.dat').write_text('mesh', encoding='utf-8')
    (tmp_path / '1.sflux').write_text('flux', encoding='utf-8')
    # EOS table directory (the worst-case duplicated payload)
    (tmp_path / 'spider_eos').mkdir()
    (tmp_path / 'spider_eos' / 'table.dat').write_text('eos', encoding='utf-8')

    archive_mod.update(str(tmp_path), remove_files=False, snapshots_only=True)

    counts = _tar_member_counts(tmp_path / f'{tmp_path.name}.tar')
    # Snapshots archived
    assert counts['1000_int.nc'] == 1
    assert counts['1000_atm.nc'] == 1
    # Runtime entries NOT archived (discrimination against the unfiltered path)
    assert counts['zalmoxis_output.dat'] == 0
    assert counts['spider_mesh.dat'] == 0
    assert counts['1.sflux'] == 0
    assert counts['spider_eos'] == 0
    assert counts['spider_eos/table.dat'] == 0
    # remove_files=False plus the snapshots_only skip: runtime entries stay loose
    assert (tmp_path / 'zalmoxis_output.dat').exists()
    assert (tmp_path / 'spider_mesh.dat').exists()
    assert (tmp_path / '1.sflux').exists()
    assert (tmp_path / 'spider_eos' / 'table.dat').exists()


def test_update_snapshots_only_bounds_tar_growth_across_cycles(tmp_path):
    """Across repeated rolling-archive cycles, snapshots_only=True keeps the
    fixed-name EOS directory out of the tar entirely, so the archive does
    not accumulate one extra copy of it per cycle.

    Discrimination guard: an otherwise-identical directory archived with
    snapshots_only=False (the unfiltered behaviour) gains one EOS-dir copy
    every cycle, so after N cycles the EOS member count equals N there but
    stays 0 under the fix. This proves the assertion fails for the
    unfixed code path rather than passing trivially.
    """

    def build_dir(root):
        d = tmp_path / root
        d.mkdir()
        (d / 'spider_eos').mkdir()
        (d / 'spider_eos' / 'table.dat').write_text('eos' * 100, encoding='utf-8')
        (d / 'zalmoxis_output.dat').write_text('mesh', encoding='utf-8')
        return d

    fixed = build_dir('fixed')
    unfixed = build_dir('unfixed')
    n_cycles = 4
    times = [100, 1000, 10000, 100000]

    for i in range(n_cycles):
        t = times[i]
        for d, snaps_only in ((fixed, True), (unfixed, False)):
            (d / f'{t}_int.nc').write_text('snap', encoding='utf-8')
            archive_mod.update(str(d), remove_files=False, snapshots_only=snaps_only)
            # Prune snapshots older than ~1% below the current time
            archive_mod.remove_old(str(d), before=t * 0.99)

    fixed_counts = _tar_member_counts(fixed / 'fixed.tar')
    unfixed_counts = _tar_member_counts(unfixed / 'unfixed.tar')

    # Under the fix the EOS dir and runtime file never enter the tar
    assert fixed_counts['spider_eos'] == 0
    assert fixed_counts['spider_eos/table.dat'] == 0
    assert fixed_counts['zalmoxis_output.dat'] == 0
    # The unfiltered path duplicates the EOS dir once per cycle: proves the
    # guard would fail on the pre-fix code and is not trivially satisfied
    assert unfixed_counts['spider_eos'] == n_cycles
    assert unfixed_counts['zalmoxis_output.dat'] == n_cycles
    # The fix still archives the snapshots themselves (archive is not empty)
    assert sum(v for k, v in fixed_counts.items() if k.endswith('_int.nc')) >= 1
    # The EOS directory is still on disk for the next interior solve
    assert (fixed / 'spider_eos' / 'table.dat').exists()


def test_create_snapshots_only_first_cycle_skips_runtime_files(tmp_path):
    """On the first rolling-archive cycle the tar does not yet exist, so
    update() routes through create(); snapshots_only must apply there too.

    Otherwise the very first archive would bake one copy of the EOS
    directory and mesh files into the tar. Discrimination: the snapshot is
    present, the runtime entries are absent, and (remove_files=False) the
    runtime files remain on disk.
    """
    (tmp_path / '500_int.nc').write_text('snap', encoding='utf-8')
    (tmp_path / 'spider_mesh.dat').write_text('mesh', encoding='utf-8')
    (tmp_path / 'spider_eos').mkdir()
    (tmp_path / 'spider_eos' / 'table.dat').write_text('eos', encoding='utf-8')

    # Route through create(): no tar exists yet
    assert not (tmp_path / f'{tmp_path.name}.tar').exists()
    archive_mod.update(str(tmp_path), remove_files=False, snapshots_only=True)

    counts = _tar_member_counts(tmp_path / f'{tmp_path.name}.tar')
    assert counts['500_int.nc'] == 1
    assert counts['spider_mesh.dat'] == 0
    assert counts['spider_eos'] == 0
    # Runtime files kept loose for the next solve
    assert (tmp_path / 'spider_mesh.dat').exists()
    assert (tmp_path / 'spider_eos' / 'table.dat').exists()
