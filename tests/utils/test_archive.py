"""Unit tests for ``proteus.utils.archive``.

Covers ``archive_exists``, ``create``, ``append``, ``extract``,
``extract_latest``, ``update``, and ``archive_old``. Uses real tarfile + tmp_path filesystem
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
from unittest.mock import patch

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
# extract_latest
# ---------------------------------------------------------------------------

# Whole-year names sort as text in another order ('100' < '9'), and the sub-year
# name 1000p500 sits between two whole years, so a text sort picks other times.
_RUN_TIMES = ['0p000', '9', '10', '100', '1000', '1000p500', '2000']
_LATEST_3 = ['1000', '1000p500', '2000']


def _snapshot_names(stem):
    return [f'{stem}_int.nc', f'{stem}_atm.nc', f'{stem}_zalmoxis.dat']


def _archived_run(data_dir):
    """Archive a run directory as the end-of-run pass leaves it; return the
    content of every file by its name in the tar."""
    data_dir.mkdir()
    files = {'spider_eos/table.dat': b'eos' * 64, 'zalmoxis_output.dat': b'mesh' * 8}
    for stem in _RUN_TIMES:
        files.update({name: f'{name} content'.encode() for name in _snapshot_names(stem)})
    (data_dir / 'spider_eos').mkdir()
    for name, content in files.items():
        (data_dir / name).write_bytes(content)
        os.utime(data_dir / name, (1.0e9, 1.0e9))  # archived long before the resume
    archive_mod.create(str(data_dir))
    return files


def _loose_files(data_dir):
    return {
        p.relative_to(data_dir).as_posix(): p.read_bytes()
        for p in data_dir.rglob('*')
        if p.is_file() and p.name != f'{data_dir.name}.tar'
    }


def test_extract_latest_unpacks_runtime_files_and_latest_snapshots(tmp_path):
    """A resume unpacks the runtime files and the snapshots of the latest
    snapshot times by value, and leaves the tar as it was.

    Discrimination: the times are chosen so that a sort of the names as text
    selects other snapshots.
    """
    data_dir = tmp_path / 'data'
    files = _archived_run(data_dir)
    tar_bytes = (data_dir / 'data.tar').read_bytes()

    archive_mod.extract_latest(str(data_dir), count=3)

    expected = ['spider_eos/table.dat', 'zalmoxis_output.dat']
    expected += [n for stem in _LATEST_3 for n in _snapshot_names(stem)]
    assert _loose_files(data_dir) == {n: files[n] for n in expected}
    assert (data_dir / 'data.tar').read_bytes() == tar_bytes


def test_final_archive_after_resume_stores_changed_files_only(tmp_path):
    """After a resume, the end-of-run archive adds the files that changed and
    skips the unpacked ones that did not, and a full unpack gives back the
    latest content of every file.

    Discrimination: a skip by name alone loses the new zalmoxis_output.dat,
    which keeps its size, and no skip stores the EOS table twice.
    """
    data_dir = tmp_path / 'data'
    files = _archived_run(data_dir)

    archive_mod.extract_latest(str(data_dir), count=3)
    (data_dir / 'zalmoxis_output.dat').write_bytes(b'MESH' * 8)
    (data_dir / '3000_int.nc').write_bytes(b'new step')
    archive_mod.update(str(data_dir))

    counts = _tar_member_counts(data_dir / 'data.tar')
    assert counts['spider_eos/table.dat'] == 1
    assert counts['2000_int.nc'] == 1
    assert counts['zalmoxis_output.dat'] == 2
    assert counts['3000_int.nc'] == 1
    assert _loose_files(data_dir) == {}
    archive_mod.extract(str(data_dir), remove_tar=True)
    latest = {'zalmoxis_output.dat': b'MESH' * 8, '3000_int.nc': b'new step'}
    assert _loose_files(data_dir) == {**files, **latest}


def test_second_resume_compares_with_the_last_archived_copy(tmp_path):
    """On a second resume, an unpacked file that did not change is skipped
    against its last archived copy, and a file that changed size is added
    even when its modification time is the archived one.

    Discrimination: a comparison with the first archived copy stores
    zalmoxis_output.dat a third time, and a comparison by time alone loses
    the new EOS table.
    """
    data_dir = tmp_path / 'data'
    _archived_run(data_dir)
    archive_mod.extract_latest(str(data_dir), count=3)
    (data_dir / 'zalmoxis_output.dat').write_bytes(b'MESH' * 8)
    archive_mod.update(str(data_dir))

    archive_mod.extract_latest(str(data_dir), count=3)
    table = data_dir / 'spider_eos' / 'table.dat'
    table.write_bytes(b'eos-v2')
    os.utime(table, (1.0e9, 1.0e9))  # the archived time, with another size
    archive_mod.update(str(data_dir))

    counts = _tar_member_counts(data_dir / 'data.tar')
    assert counts['zalmoxis_output.dat'] == 2
    assert counts['spider_eos/table.dat'] == 2
    archive_mod.extract(str(data_dir), remove_tar=True)
    assert (data_dir / 'zalmoxis_output.dat').read_bytes() == b'MESH' * 8
    assert table.read_bytes() == b'eos-v2'


def test_full_extract_keeps_a_newer_loose_file(tmp_path):
    """A full unpack at the end of a resumed run, as offline chemistry does,
    keeps a runtime file that changed after the resume instead of the older
    copy of the earlier run."""
    data_dir = tmp_path / 'data'
    files = _archived_run(data_dir)
    archive_mod.extract_latest(str(data_dir), count=3)
    (data_dir / 'zalmoxis_output.dat').write_bytes(b'MESH' * 8)

    archive_mod.extract(str(data_dir), remove_tar=True)

    assert (data_dir / 'zalmoxis_output.dat').read_bytes() == b'MESH' * 8
    assert _loose_files(data_dir) == {**files, 'zalmoxis_output.dat': b'MESH' * 8}


def test_extract_latest_keeps_loose_copies_and_takes_last_archived_copy(tmp_path):
    """A loose file is newer than its archived copy, so it stays as it is,
    and of two archived copies of one name the last one is unpacked."""
    data_dir = tmp_path / 'data'
    _archived_run(data_dir)
    (data_dir / '2000_atm.nc').write_bytes(b'second copy')
    archive_mod.append(str(data_dir))
    (data_dir / '2000_int.nc').write_bytes(b'loose')

    archive_mod.extract_latest(str(data_dir), count=3)

    assert (data_dir / '2000_int.nc').read_bytes() == b'loose'
    assert (data_dir / '2000_atm.nc').read_bytes() == b'second copy'
    assert _tar_member_counts(data_dir / 'data.tar')['2000_atm.nc'] == 2


@pytest.mark.parametrize('count', [7, 50], ids=['all_times', 'more_than_all_times'])
def test_extract_latest_unpacks_everything_when_count_covers_all_times(tmp_path, count):
    """When the latest times cover every snapshot, all files are unpacked."""
    data_dir = tmp_path / 'data'
    files = _archived_run(data_dir)

    archive_mod.extract_latest(str(data_dir), count=count)

    assert _loose_files(data_dir) == files
    assert (data_dir / 'data.tar').exists()


class _Killed(BaseException):
    """The job is stopped from outside, as by Slurm at its time limit."""


def test_a_stopped_unpack_leaves_no_cut_file_for_later_unpacks(tmp_path, monkeypatch):
    """A resume whose unpack is stopped while it writes a file leaves no cut copy
    under that file's name. The next resume unpacks the full file, and a full unpack
    that removes the tar afterwards (as postprocessing does) keeps every file whole.

    Discrimination: an unpack in place leaves the cut copy, every later unpack keeps
    it as a newer loose file, and removing the tar then loses the only full copy.
    """
    data_dir = tmp_path / 'data'
    files = _archived_run(data_dir)
    copy = tarfile.copyfileobj
    elsewhere = tmp_path / 'elsewhere'  # the job's working folder, which stays untouched
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    started = []

    def _stop_in_the_third_file(src, dst, length=None, exception=OSError, bufsize=None):
        started.append(dst.name)
        if len(started) == 3:
            dst.write(src.read(length // 3))
            raise _Killed
        return copy(src, dst, length, exception, bufsize)

    # a hard stop runs no cleanup, so the hidden folder stays
    with (
        patch.object(tarfile, 'copyfileobj', _stop_in_the_third_file),
        patch.object(archive_mod.shutil, 'rmtree'),
        pytest.raises(_Killed),
    ):
        archive_mod.extract_latest(str(data_dir), count=3)

    cut = next(n for n in files if started[2].endswith('/' + n))
    loose = {n: c for n, c in _loose_files(data_dir).items() if not n.startswith('.unpacking/')}
    assert cut not in loose
    assert loose == {n: files[n] for n in loose}
    assert (data_dir / '.unpacking').is_dir() and os.listdir(elsewhere) == []

    archive_mod.extract_latest(str(data_dir), count=3)
    assert _loose_files(data_dir)[cut] == files[cut]
    archive_mod.extract(str(data_dir), remove_tar=True)

    assert _loose_files(data_dir) == files
    assert sorted(os.listdir(data_dir)) == sorted({n.split('/')[0] for n in files})
    assert os.listdir(elsewhere) == []


def test_an_unpack_with_nothing_missing_keeps_loose_files_and_removes_an_old_unpack(tmp_path):
    """A second resume whose files are all loose already, as after a resume that
    stopped before it archived again, unpacks nothing, keeps every loose file and
    removes the hidden folder that a hard stop of an earlier unpack left."""
    data_dir = tmp_path / 'data'
    _archived_run(data_dir)
    archive_mod.extract(str(data_dir))
    loose = _loose_files(data_dir)
    (data_dir / '.unpacking').mkdir()
    (data_dir / '.unpacking' / 'cut.nc').write_bytes(b'cut')

    archive_mod.extract_latest(str(data_dir), count=3)
    archive_mod.extract(str(data_dir))

    assert _loose_files(data_dir) == loose
    assert not (data_dir / '.unpacking').exists()


def test_extract_refuses_a_member_outside_the_data_folder(tmp_path):
    """An archive member whose name points above the data folder is refused, and
    nothing is written outside the data folder."""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    source = tmp_path / 'payload.dat'
    source.write_bytes(b'payload')
    with tarfile.open(data_dir / 'data.tar', 'w') as tar_file:
        tar_file.add(source, arcname='../outside.dat')
    source.unlink()
    (data_dir / '.unpacking').mkdir()  # so that only the filter can refuse the member

    with pytest.raises(tarfile.OutsideDestinationError):
        archive_mod.extract(str(data_dir))

    assert sorted(p.name for p in tmp_path.iterdir()) == ['data']
    assert sorted(os.listdir(data_dir)) == ['data.tar']


def test_extract_keeps_an_absolute_folder_member_inside_the_data_folder(tmp_path):
    """A folder member with an absolute name, which only a tar that PROTEUS did not
    write holds, is made under the data folder, as the data filter strips the
    leading slash. Joining the raw name would make it at the absolute path."""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    folder = tarfile.TarInfo(str(tmp_path / 'outside' / 'ESC'))
    folder.type = tarfile.DIRTYPE
    with tarfile.open(data_dir / 'data.tar', 'w') as tar_file:
        tar_file.addfile(folder)

    archive_mod.extract(str(data_dir))

    assert sorted(p.name for p in tmp_path.iterdir()) == ['data']
    assert (data_dir / str(tmp_path / 'outside' / 'ESC').lstrip('/')).is_dir()
    assert not (data_dir / '.unpacking').exists()


def test_extract_unpacks_an_archive_with_full_paths_and_no_folder_members(tmp_path):
    """An archive written before 25.05.20 stores each file under its full path and
    holds no folder members; every file is unpacked under that path, whole."""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    files = {'home/u/run/data/1000_int.nc': b'interior at 1000 yr', 'g.nc': b'atmosphere'}
    source = tmp_path / 'source.dat'
    with tarfile.open(data_dir / 'data.tar', 'w') as tar_file:
        for name, content in files.items():
            source.write_bytes(content)
            tar_file.add(source, arcname=name)
    source.unlink()
    with tarfile.open(data_dir / 'data.tar') as tar_file:
        assert not any(m.isdir() for m in tar_file.getmembers())

    archive_mod.extract(str(data_dir), remove_tar=True)

    assert _loose_files(data_dir) == files
    assert sorted(os.listdir(data_dir)) == ['g.nc', 'home']


def test_extract_unpacks_a_folder_member_that_follows_its_files(tmp_path):
    """A folder member stored after a file inside it, as an append can order them,
    finds the folder made already for that file and unpacks the file whole."""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    source = tmp_path / 'source.dat'
    source.write_bytes(b'escape state')
    folder = tarfile.TarInfo('ESC')
    folder.type = tarfile.DIRTYPE
    with tarfile.open(data_dir / 'data.tar', 'w') as tar_file:
        tar_file.add(source, arcname='ESC/1000_esc.json')
        tar_file.addfile(folder)
    source.unlink()

    archive_mod.extract(str(data_dir), remove_tar=True)

    assert _loose_files(data_dir) == {'ESC/1000_esc.json': b'escape state'}
    assert sorted(os.listdir(data_dir)) == ['ESC']


def test_extract_latest_without_tar_changes_nothing(tmp_path, caplog):
    """A run directory without a tar keeps its files and logs a warning."""
    (tmp_path / '100_int.nc').write_bytes(b'snap')
    with caplog.at_level('WARNING'):
        archive_mod.extract_latest(str(tmp_path), count=1)
    assert _loose_files(tmp_path) == {'100_int.nc': b'snap'}
    assert any('does not exist' in record.message for record in caplog.records)


def test_rolling_archive_after_resume_stores_each_snapshot_once(tmp_path):
    """After a resume, the rolling archive moves the unpacked snapshots that fall
    below its cutoff back into the tar without a second copy, and adds the new
    snapshots once.

    Discrimination: without the skip in _move_into_tar, each unpacked snapshot
    below the cutoff enters the tar a second time.
    """
    data_dir = tmp_path / 'data'
    _archived_run(data_dir)
    archive_mod.extract_latest(str(data_dir), count=3)
    for name in _snapshot_names('3000'):
        (data_dir / name).write_bytes(b'new step')

    archive_mod.archive_old(str(data_dir), before=2500.0)

    counts = _tar_member_counts(data_dir / 'data.tar')
    snapshots = [n for n in counts if archive_mod._snapshot_time(n) is not None]
    assert len(snapshots) == 3 * len(_RUN_TIMES)
    assert max(counts[n] for n in snapshots) == 1
    loose = _loose_files(data_dir)
    assert sorted(n for n in loose if archive_mod._snapshot_time(n) is not None) == sorted(
        _snapshot_names('3000')
    )
    assert loose['zalmoxis_output.dat'] == b'mesh' * 8


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


def test_snapshot_time_recognizes_only_time_prefixed_snapshot_names():
    """`_snapshot_time` returns the simulated time [yr] for names ending in
    .nc, .json or _zalmoxis.dat whose leading token parses as a time (whole
    year, `p` or dot form), and None for everything else. archive_old relies
    on it, so its boundary behaviour is pinned here.

    Edge cases: the age-0 boundary snapshot parses to 0 (falsy but not
    None, so a naive truthiness check would misclassify it); the
    fixed-name runtime files, the EOS table directory name, and the
    integer-prefixed stellar flux file (.sflux, wrong extension) all
    return None; a .nc/.json name whose prefix is not a time returns None
    via the ValueError branch rather than raising.
    """
    st = archive_mod._snapshot_time
    # Recognised snapshots -> simulated time [yr]
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
    (content depends only on the time, so a snapshot's bytes never change
    once written and a second copy is waste), mutates
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
    counts = _tar_member_counts(data_dir / 'run.tar')

    assert counts['spider_eos/table.dat'] == 1
    assert counts['spider_mesh.dat'] == 1
    assert counts['zalmoxis_output.dat'] == 1
    # The one copy holds the last version of the evolving file
    last = (b'zalmoxis-v%d' % (len(_CYCLE_TIMES) - 1)) * 64
    assert _tar_contents(data_dir / 'run.tar')['zalmoxis_output.dat'] == last


def test_lifecycle_snapshot_reappend_no_inflation(tmp_path):
    """Replaying the proteus.py lifecycle leaves every timestamped snapshot,
    and every other entry, in data.tar exactly once.

    Discrimination: a rolling archive that appends the loose snapshots each
    cycle stores every snapshot that stays loose for a cycle twice.
    """
    data_dir = _build_data_dir(tmp_path / 'run')
    _replay_lifecycle(data_dir)
    counts = _tar_member_counts(data_dir / 'run.tar')

    snapshot_names = [
        n for n in counts if archive_mod._snapshot_time(os.path.basename(n)) is not None
    ]
    # The fixture writes an int and an atm snapshot per cycle, so an archive
    # that stored nothing cannot pass.
    assert len(snapshot_names) == 2 * len(_CYCLE_TIMES)
    assert set(counts.values()) == {1}
