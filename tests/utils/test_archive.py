"""Unit tests for ``proteus.utils.archive``.

Covers ``archive_exists``, ``create``, ``append``, ``extract``,
``update``, and ``archive_old``. Uses real tarfile + tmp_path filesystem
operations because the functions are thin wrappers around tarfile and
os.* primitives; mocking those would amount to mocking the function
under test.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import hashlib
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

    tar_path = archive_mod.create(str(tmp_path))

    expected_tar = str(tmp_path / f'{tmp_path.name}.tar')
    assert tar_path == expected_tar
    assert os.path.exists(expected_tar)
    # Discrimination: the originals must be removed
    assert not (tmp_path / 'a.txt').exists()
    assert not (tmp_path / 'b.txt').exists()


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
    archive_mod.create(str(tmp_path))
    # Add a second file and append it
    (tmp_path / 'b.txt').write_text('second', encoding='utf-8')
    archive_mod.append(str(tmp_path))

    import tarfile

    tar_path = tmp_path / f'{tmp_path.name}.tar'
    with tarfile.open(tar_path, 'r') as tar:
        names = sorted(tar.getnames())
    # Discrimination: both 'a.txt' and 'b.txt' must be in the archive
    assert 'a.txt' in names
    assert 'b.txt' in names
    # append moves the file: it is no longer loose
    assert not (tmp_path / 'b.txt').exists()


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
    archive_mod.update(str(tmp_path))
    assert (tmp_path / f'{tmp_path.name}.tar').exists()

    # Second call: new file in directory -> appends
    (tmp_path / 'b.txt').write_text('second', encoding='utf-8')
    archive_mod.update(str(tmp_path))

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
    archive_mod.create(str(tmp_path))
    assert not (tmp_path / 'a.txt').exists()  # confirms create() removed it

    archive_mod.extract(str(tmp_path), remove_tar=True)

    # Discrimination: a.txt restored with original content; tar removed
    assert (tmp_path / 'a.txt').read_text(encoding='utf-8') == 'hello'
    assert not (tmp_path / f'{tmp_path.name}.tar').exists()


def test_extract_keeps_tar_when_remove_tar_false(tmp_path):
    """remove_tar=False (default) leaves the tar in place after extraction."""
    (tmp_path / 'a.txt').write_text('content', encoding='utf-8')
    archive_mod.create(str(tmp_path))
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
# archive_old: rolling in-loop archive
# ---------------------------------------------------------------------------


def _tar_contents(tar_path) -> dict:
    """Return member name -> bytes (the last member wins, as on extraction)."""
    with tarfile.open(tar_path, 'r') as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}


def test_archive_old_moves_old_snapshots_and_keeps_runtime_files(tmp_path):
    """archive_old moves the snapshots older than the cutoff into the tar and
    leaves everything else loose and out of the tar: recent snapshots, the
    fixed-name runtime files the interior modules re-read between structure
    re-solves, the stellar-flux inputs and the EOS table directory.

    Discrimination: with snapshots at 100, 1000 and 10000 yr and a cutoff of
    500 yr, only the 100 yr snapshot may leave the disk, and it must arrive
    in the tar with its content. Deleting zalmoxis_output.dat would kill the
    next Aragog reset(), and archiving the EOS directory every cycle would
    store the largest entry many times.
    """
    (tmp_path / '100_int.nc').write_text('a', encoding='utf-8')
    (tmp_path / '1000_int.nc').write_text('b', encoding='utf-8')
    (tmp_path / '10000_atm.nc').write_text('c', encoding='utf-8')
    (tmp_path / 'zalmoxis_output.dat').write_text('mesh', encoding='utf-8')
    (tmp_path / 'spider_mesh.dat').write_text('mesh', encoding='utf-8')
    (tmp_path / '1.sflux').write_text('flux', encoding='utf-8')
    (tmp_path / 'spider_eos').mkdir()
    (tmp_path / 'spider_eos' / 'table.dat').write_text('eos', encoding='utf-8')

    # No tar yet: the first cycle creates it
    archive_mod.archive_old(str(tmp_path), before=500)

    assert _tar_member_counts(tmp_path / f'{tmp_path.name}.tar') == Counter({'100_int.nc': 1})
    assert _tar_contents(tmp_path / f'{tmp_path.name}.tar') == {'100_int.nc': b'a'}
    assert not (tmp_path / '100_int.nc').exists()
    assert (tmp_path / '1000_int.nc').exists()
    assert (tmp_path / '10000_atm.nc').exists()
    assert (tmp_path / 'zalmoxis_output.dat').exists()
    assert (tmp_path / 'spider_mesh.dat').exists()
    assert (tmp_path / '1.sflux').exists()
    assert (tmp_path / 'spider_eos' / 'table.dat').read_text(encoding='utf-8') == 'eos'


def test_archive_old_moves_the_zalmoxis_copy_with_its_snapshot(tmp_path):
    """A <time>_zalmoxis.dat copy goes into the tar with the _int.nc of the same time."""
    for name in ('100_int.nc', '100_zalmoxis.dat', '1000_int.nc', '1000_zalmoxis.dat'):
        (tmp_path / name).write_text(name, encoding='utf-8')
    (tmp_path / 'zalmoxis_output.dat').write_text('mesh', encoding='utf-8')

    archive_mod.archive_old(str(tmp_path), before=500)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        '1000_int.nc',
        '1000_zalmoxis.dat',
        f'{tmp_path.name}.tar',
        'zalmoxis_output.dat',
    ]
    assert _tar_contents(tmp_path / f'{tmp_path.name}.tar') == {
        '100_int.nc': b'100_int.nc',
        '100_zalmoxis.dat': b'100_zalmoxis.dat',
    }


def test_archive_old_keeps_non_timestamped_nc_and_json_names(tmp_path):
    """A .nc/.json name without a leading time token is not a snapshot and
    stays loose rather than crashing the time parse.

    Edge case for the prefix parser: 'notes.json' and 'mesh_summary.nc'
    have no time prefix, while '0_int.nc' is the boundary-age snapshot
    (age 0 < any positive cutoff, so it is archived). The archived 0 yr
    snapshot shows the kept files survive because of the name check, not
    because the call was a no-op.
    """
    (tmp_path / 'notes.json').write_text('a', encoding='utf-8')
    (tmp_path / 'mesh_summary.nc').write_text('b', encoding='utf-8')
    (tmp_path / '0_int.nc').write_text('c', encoding='utf-8')

    archive_mod.archive_old(str(tmp_path), before=500)

    assert (tmp_path / 'notes.json').exists()
    assert (tmp_path / 'mesh_summary.nc').exists()
    assert not (tmp_path / '0_int.nc').exists()
    assert _tar_contents(tmp_path / f'{tmp_path.name}.tar') == {'0_int.nc': b'c'}


def test_archive_old_handles_json_snapshots_alongside_nc(tmp_path):
    """The age gate applies to the SPIDER .json snapshots as well as .nc.
    A regression that only handled .nc would leave old .json files behind.
    """
    (tmp_path / '100_int.json').write_text('a', encoding='utf-8')
    (tmp_path / '5000_int.json').write_text('b', encoding='utf-8')

    archive_mod.archive_old(str(tmp_path), before=500)

    assert not (tmp_path / '100_int.json').exists()
    assert (tmp_path / '5000_int.json').exists()
    assert _tar_member_counts(tmp_path / f'{tmp_path.name}.tar') == Counter({'100_int.json': 1})


def test_archive_old_without_old_snapshots_creates_no_tar(tmp_path):
    """With nothing older than the cutoff, archive_old writes no tar and
    removes nothing: an empty tar would make later code take the append path
    and an `archive_exists` check report data that is not there.
    """
    (tmp_path / '1000_int.nc').write_text('b', encoding='utf-8')
    (tmp_path / 'zalmoxis_output.dat').write_text('mesh', encoding='utf-8')

    archive_mod.archive_old(str(tmp_path), before=1000)

    # Boundary: a snapshot exactly at the cutoff is not older than it
    assert not (tmp_path / f'{tmp_path.name}.tar').exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['1000_int.nc', 'zalmoxis_output.dat']


def test_archive_old_then_extract_restores_every_snapshot(tmp_path):
    """Resume extracts the tar of a stopped run: the archived snapshots and
    the loose ones together give back every snapshot with its content.

    Edge case: SPIDER names its JSON on the whole year, so two rows less than
    a year apart write the same name. When both versions reach the tar, the
    later one must come back on extraction, as `extractall` keeps the last
    member of a name.
    """
    (tmp_path / '50.json').write_text('t=50.2', encoding='utf-8')
    archive_mod.archive_old(str(tmp_path), before=50.5)
    (tmp_path / '50.json').write_text('t=50.7', encoding='utf-8')
    (tmp_path / '100_int.nc').write_text('t=100', encoding='utf-8')
    archive_mod.archive_old(str(tmp_path), before=99)
    (tmp_path / '200_int.nc').write_text('t=200', encoding='utf-8')
    archive_mod.archive_old(str(tmp_path), before=198)
    (tmp_path / '300_int.nc').write_text('t=300', encoding='utf-8')

    assert _tar_member_counts(tmp_path / f'{tmp_path.name}.tar') == Counter(
        {'50.json': 2, '100_int.nc': 1}
    )
    archive_mod.extract(str(tmp_path), remove_tar=True)

    assert {p.name: p.read_text(encoding='utf-8') for p in tmp_path.iterdir()} == {
        '50.json': 't=50.7',
        '100_int.nc': 't=100',
        '200_int.nc': 't=200',
        '300_int.nc': 't=300',
    }


# ---------------------------------------------------------------------------
# _snapshot_time (shared timestamped-snapshot predicate)
# ---------------------------------------------------------------------------


def test_snapshot_time_recognizes_only_integer_prefixed_nc_json():
    """`_snapshot_time` returns the integer simulated time only for names
    ending in .nc/.json whose leading token parses as an integer, and None
    for everything else. This predicate is the single source of truth
    used by archive_old, so its boundary
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
    assert st('884p700_int.nc') == pytest.approx(884.7)
    assert st('884p700_atm.nc') == pytest.approx(884.7)
    # Dot-form filenames (legacy) parse to the same value
    assert st('884.700_int.nc') == pytest.approx(884.7)
    assert st('884.700_atm.nc') == pytest.approx(884.7)
    # Boundary: age 0 must be 0, not None (distinct from "not a snapshot")
    assert st('0_int.nc') == 0
    # The Zalmoxis structure copy saved with each snapshot
    assert st('884p700_zalmoxis.dat') == pytest.approx(884.7)
    assert st('mesh_zalmoxis.dat') is None
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
# End-to-end archive lifecycle
# ---------------------------------------------------------------------------

# Simulated times [yr]; each is more than 1/0.99 times the previous, so every
# cycle moves the previous snapshot into the tar and keeps the current one loose.
_CYCLE_TIMES = [100, 1000, 10000, 100000]


def _tar_content_stats(tar_path) -> dict:
    """Inspect a tar by member *content*, not just by name.

    Returns a dict with:
      ``gross``            total bytes over every file member (duplicates
                           counted every time they appear);
      ``distinct``         total bytes counting each unique content once
                           (keyed by SHA-256 of the member bytes);
      ``counts``           Counter of member name -> number of members.

    ``inflation = gross / distinct``; 1.0 means no byte-identical duplication.
    """
    gross = 0
    size_by_hash: dict[str, int] = {}
    counts: Counter = Counter()
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            counts[member.name] += 1
            if not member.isfile():
                continue
            data = tar.extractfile(member).read()
            digest = hashlib.sha256(data).hexdigest()
            gross += len(data)
            size_by_hash.setdefault(digest, len(data))
    return {
        'gross': gross,
        'distinct': sum(size_by_hash.values()),
        'counts': counts,
    }


def _build_data_dir(root):
    """Create a mini output/data/ mirroring the production layout.

    A sizeable, unchanging EOS lookup table, the fixed-name runtime hand-off
    files the interior modules re-read between structure re-solves, and no
    snapshots yet; :func:`_replay_lifecycle` writes those per cycle.
    """
    root.mkdir()
    (root / 'spider_eos').mkdir()
    # Unchanging reference data: the same bytes for the whole run.
    (root / 'spider_eos' / 'table.dat').write_bytes(b'eos-table' * 512)
    (root / 'spider_mesh.dat').write_bytes(b'mesh' * 64)
    (root / 'zalmoxis_output.dat').write_bytes(b'zalmoxis-v0' * 64)
    return root


def _replay_lifecycle(data_dir):
    """Replay the proteus.py archive lifecycle over :data:`_CYCLE_TIMES`.

    Each cycle writes the interior+atmosphere snapshots for the current time
    (content depends only on the time, so a given snapshot's bytes never
    change once written -- re-appending it is byte-identical waste), mutates
    the fixed-name ``zalmoxis_output.dat`` to a new content (as a dynamic run
    does) and runs the rolling archive. After the loop the end-of-run full
    archive packs whatever is still loose.
    """
    for i, t in enumerate(_CYCLE_TIMES):
        (data_dir / f'{t}_int.nc').write_bytes((b'int-%d' % t) * 64)
        (data_dir / f'{t}_atm.nc').write_bytes((b'atm-%d' % t) * 64)
        # Fixed-name file whose content legitimately evolves each cycle.
        (data_dir / 'zalmoxis_output.dat').write_bytes((b'zalmoxis-v%d' % i) * 64)
        archive_mod.archive_old(str(data_dir), before=t * 0.99)
    # End-of-run full archive: packs the still-loose snapshots plus every
    # fixed-name file and the EOS directory in one pass.
    archive_mod.update(str(data_dir))


def test_lifecycle_final_pass_stores_reference_data_once(tmp_path):
    """Over the full lifecycle, including the end-of-run archive, the static EOS
    table and the fixed-name runtime files enter data.tar exactly once.

    Discrimination: a rolling archive that appends every loose entry stores the
    EOS table once per cycle plus once at the end, failing the member counts.
    """
    data_dir = _build_data_dir(tmp_path / 'run')
    _replay_lifecycle(data_dir)
    stats = _tar_content_stats(data_dir / 'run.tar')

    assert stats['counts']['spider_eos/table.dat'] == 1
    assert stats['counts']['spider_mesh.dat'] == 1
    assert stats['counts']['zalmoxis_output.dat'] == 1
    # The one copy holds the last version of the evolving file
    last = (b'zalmoxis-v%d' % (len(_CYCLE_TIMES) - 1)) * 64
    assert _tar_contents(data_dir / 'run.tar')['zalmoxis_output.dat'] == last


def test_lifecycle_snapshot_reappend_no_inflation(tmp_path):
    """Replaying the proteus.py lifecycle leaves every timestamped snapshot
    in data.tar exactly once and an overall content-inflation factor of 1.0.

    Discrimination: a rolling archive that appends the loose snapshots each
    cycle stores every snapshot that stays loose for a cycle twice.
    """
    data_dir = _build_data_dir(tmp_path / 'run')
    _replay_lifecycle(data_dir)
    stats = _tar_content_stats(data_dir / 'run.tar')

    # Primary contract: no snapshot is stored more than once.
    snapshot_names = [
        name
        for name in stats['counts']
        if archive_mod._snapshot_time(os.path.basename(name)) is not None
    ]
    # Discrimination: the fixture really does produce snapshots to check, so a
    # regression that archived nothing could not make this pass vacuously.
    assert len(snapshot_names) == 2 * len(_CYCLE_TIMES)  # int + atm per cycle
    for name in snapshot_names:
        assert stats['counts'][name] == 1

    # Overall: bytes stored should equal bytes of distinct content (no waste).
    inflation = stats['gross'] / stats['distinct']
    assert inflation == pytest.approx(1.0, abs=0.01)
