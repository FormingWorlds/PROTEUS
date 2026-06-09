"""Branch coverage for ``proteus.utils.helper``.

Exercises the side branches in ``safe_rm`` and ``CleanDir``, the
recursive attribute helpers, the gas-VMR to elemental-mass converter
and its zero-mass guard, ``eval_gas_mmw`` for both an element-only
input and a molecular formula, and the full ``CommentFromStatus``
table including the unhandled-status fallback.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# safe_rm branches
# ---------------------------------------------------------------------------


def test_safe_rm_warns_and_returns_on_empty_path(tmp_path, caplog, monkeypatch):
    """``safe_rm('')`` must emit a warning and return without raising;
    this guards against accidental rm-rf-from-root if a caller forgets
    to supply a path. Discrimination: warning log captured AND the
    process working directory is untouched (a regression that fell
    through to a real ``shutil.rmtree('')`` could attempt to remove cwd).
    """
    from proteus.utils.helper import safe_rm

    sentinel = tmp_path / 'cwd'
    sentinel.mkdir()
    (sentinel / 'keep.txt').write_text('keep')
    monkeypatch.chdir(sentinel)

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.helper'):
        safe_rm('')

    assert any('empty path' in rec.message for rec in caplog.records)
    assert (sentinel / 'keep.txt').exists()


def test_safe_rm_refuses_to_delete_directory_containing_git_subfolder(tmp_path, caplog):
    """If a directory contains a ``.git`` subfolder, ``safe_rm`` must
    refuse to delete it and emit a warning. Discrimination: the
    directory must still exist after the call.
    """
    from proteus.utils.helper import safe_rm

    protected = tmp_path / 'repo'
    protected.mkdir()
    (protected / '.git').mkdir()
    (protected / 'data.txt').write_text('keep me')

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.helper'):
        safe_rm(str(protected))

    assert protected.exists()
    assert (protected / '.git').exists()
    assert (protected / 'data.txt').exists()
    assert any('Git repository' in rec.message for rec in caplog.records)


def test_safe_rm_removes_regular_directory(tmp_path):
    """Without ``.git`` inside, ``safe_rm`` removes a directory tree.
    Discrimination: a regression that only removed top-level files but
    left the directory would fail the second assertion; and a sibling
    directory must remain untouched (no scope creep into the parent).
    """
    from proteus.utils.helper import safe_rm

    target = tmp_path / 'scratch'
    target.mkdir()
    (target / 'file.txt').write_text('data')
    sibling = tmp_path / 'sibling'
    sibling.mkdir()

    safe_rm(str(target))

    assert not target.exists()
    assert sibling.exists()


def test_safe_rm_removes_regular_file(tmp_path):
    """``safe_rm`` removes a regular file. Discrimination: the parent
    directory must remain (a regression that removed the parent would
    pass a naive ``not target.exists()`` check).
    """
    from proteus.utils.helper import safe_rm

    target = tmp_path / 'file.txt'
    target.write_text('data')

    safe_rm(str(target))

    assert not target.exists()
    assert tmp_path.exists()


# ---------------------------------------------------------------------------
# CleanDir keep_stdlog branch
# ---------------------------------------------------------------------------


def test_clean_dir_with_keep_stdlog_preserves_log_files_but_removes_others(tmp_path):
    """``CleanDir(..., keep_stdlog=True)`` removes regular files and
    subdirectories but keeps any file with ``.log`` in the path.
    Discrimination: the kept log file's contents must match.
    """
    from proteus.utils.helper import CleanDir

    target = tmp_path / 'workspace'
    target.mkdir()
    (target / 'proteus_00.log').write_text('LOG_CONTENT')
    (target / 'helpfile.csv').write_text('csv,data')
    (target / 'sub').mkdir()
    (target / 'sub' / 'inner.txt').write_text('inner')

    CleanDir(str(target), keep_stdlog=True)

    assert (target / 'proteus_00.log').exists()
    assert (target / 'proteus_00.log').read_text() == 'LOG_CONTENT'
    assert not (target / 'helpfile.csv').exists()
    assert not (target / 'sub').exists()


def test_clean_dir_creates_directory_if_missing(tmp_path):
    """``CleanDir`` on a non-existent path creates an empty directory."""
    from proteus.utils.helper import CleanDir

    target = tmp_path / 'fresh'
    assert not target.exists()

    CleanDir(str(target), keep_stdlog=False)

    assert target.is_dir()
    assert list(target.iterdir()) == []


# ---------------------------------------------------------------------------
# recursive_getattr / recursive_setattr
# ---------------------------------------------------------------------------


def test_recursive_getattr_follows_dot_notation():
    """Dotted attr access traverses nested namespaces. Discrimination:
    pinned to a non-trivial value (42) so a no-op implementation would
    return the namespace, not the leaf int.
    """
    from proteus.utils.helper import recursive_getattr

    obj = SimpleNamespace(level1=SimpleNamespace(level2=SimpleNamespace(value=42)))

    assert recursive_getattr(obj, 'level1.level2.value') == 42
    assert recursive_getattr(obj, 'level1') is obj.level1


def test_recursive_setattr_updates_nested_attribute():
    """``recursive_setattr`` walks the chain and writes only the leaf."""
    from proteus.utils.helper import recursive_getattr, recursive_setattr

    obj = SimpleNamespace(a=SimpleNamespace(b=SimpleNamespace(c=1)))

    recursive_setattr(obj, 'a.b.c', 99)

    assert recursive_getattr(obj, 'a.b.c') == 99
    # Discrimination: leaf write must not destroy the intermediate
    # namespace; ``a.b`` must still be a SimpleNamespace.
    assert isinstance(obj.a.b, SimpleNamespace)


def test_recursive_setattr_single_segment_writes_directly():
    """A single-segment attribute is set without recursion.
    Discrimination: sibling attributes are not perturbed and the type of
    the written value is preserved (a regression that always cast to
    string would fail).
    """
    from proteus.utils.helper import recursive_setattr

    obj = SimpleNamespace(x=0, y=99)
    recursive_setattr(obj, 'x', 7)
    assert obj.x == 7
    assert obj.y == 99
    assert isinstance(obj.x, int)


# ---------------------------------------------------------------------------
# CommentFromStatus full code table including the fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'status,fragment',
    [
        (0, 'Started'),
        (1, 'Running'),
        (10, 'solidified'),
        (12, 'maximum iterations'),
        (13, 'target time'),
        (14, 'net flux'),
        (15, 'volatiles escaped'),
        (16, 'disintegrated'),
        (20, 'generic'),
        (21, 'Interior'),
        (22, 'Atmosphere'),
        (23, 'Stellar'),
        (24, 'Kinetics'),
        (25, 'died'),
        (26, 'Tides'),
        (27, 'Outgassing'),
        (28, 'Escape'),
    ],
)
def test_comment_from_status_maps_each_documented_code(status, fragment):
    """Every documented status code returns a description containing a
    distinctive fragment. Discrimination: distinct fragments per code
    catch accidental fall-through (a regression that mapped 21 onto the
    "Atmosphere" string would fail the 21 row); the result must also be
    non-empty (a regression that returned the empty string would pass a
    weak ``in`` check against an empty fragment).
    """
    from proteus.utils.helper import CommentFromStatus

    desc = CommentFromStatus(status)
    assert fragment.lower() in desc.lower()
    assert len(desc) >= len(fragment)


def test_comment_from_status_unhandled_code_emits_warning(caplog):
    """A code not in the table returns 'UNHANDLED STATUS (N)' and logs
    a warning so an operator can spot the regression. Discrimination:
    the unhandled fallback contains the offending number.
    """
    from proteus.utils.helper import CommentFromStatus

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.helper'):
        desc = CommentFromStatus(999)

    assert '999' in desc
    assert 'UNHANDLED' in desc
    assert any('Unhandled' in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# UpdateStatusfile
# ---------------------------------------------------------------------------


def test_update_statusfile_writes_code_then_description(tmp_path):
    """``UpdateStatusfile`` writes the integer status on line 1 and the
    description on line 2, into ``<output>/status``.
    """
    from proteus.utils.helper import UpdateStatusfile

    UpdateStatusfile({'output': str(tmp_path)}, 10)

    contents = (tmp_path / 'status').read_text().splitlines()
    assert contents[0] == '10'
    assert 'solidified' in contents[1].lower()


def test_update_statusfile_creates_missing_output_dir(tmp_path):
    """If the output directory does not yet exist, ``UpdateStatusfile``
    creates it.
    """
    from proteus.utils.helper import UpdateStatusfile

    output_dir = tmp_path / 'will_be_created'
    assert not output_dir.exists()

    UpdateStatusfile({'output': str(output_dir)}, 13)

    assert output_dir.is_dir()
    assert (output_dir / 'status').exists()


# ---------------------------------------------------------------------------
# gas_vmr_to_emr
# ---------------------------------------------------------------------------


def test_gas_vmr_to_emr_pure_h2o_gives_two_to_one_h_to_o_by_mass_ratio():
    """Pure H2O has mass 18.015 g/mol with H mass fraction = 2 * 1.008
    / 18.015 ~ 0.1119 and O mass fraction = 15.999 / 18.015 ~ 0.8881.
    Discrimination: a regression that swapped H and O atomic masses
    would invert the ratio.
    """
    from proteus.utils.helper import gas_vmr_to_emr

    emr = gas_vmr_to_emr({'H2O': 1.0})

    # H mass fraction is the smaller one.
    assert emr['H'] == pytest.approx(2 * 1.008 / 18.015, rel=1e-2)
    assert emr['O'] == pytest.approx(15.999 / 18.015, rel=1e-2)
    # Order-discrimination: emr['O'] must be the larger of the two.
    assert emr['O'] > emr['H']
    # Closure invariant: emr fractions sum to unity (only H and O are
    # present, others must be filtered out below the 1e-20 threshold).
    assert sum(emr.values()) == pytest.approx(1.0, abs=1e-9)


def test_gas_vmr_to_emr_zero_vmr_returns_empty_with_warning(caplog):
    """When every gas has zero VMR, ``M_ele`` is zero and the function
    returns ``{}`` after logging a warning. Discrimination guard: an
    empty dict is the only physically-meaningful return for a degenerate
    input; a non-empty fallback would mask the bug.
    """
    from proteus.utils.helper import gas_vmr_to_emr

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.helper'):
        result = gas_vmr_to_emr({'H2O': 0.0, 'CO2': 0.0})

    assert result == {}
    assert any('zero or invalid' in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# eval_gas_mmw
# ---------------------------------------------------------------------------


def test_eval_gas_mmw_returns_element_mmw_when_input_is_pure_element():
    """A pure element symbol (e.g. 'H') returns the atomic mass directly
    via the early-return branch.
    """
    from proteus.utils.constants import element_mmw
    from proteus.utils.helper import eval_gas_mmw

    assert eval_gas_mmw('H') == pytest.approx(element_mmw['H'])
    assert eval_gas_mmw('O') == pytest.approx(element_mmw['O'])


def test_eval_gas_mmw_sums_atomic_masses_for_water():
    """For 'H2O', the molecular mass is 2*H + 1*O. Discrimination: any
    formula that treated 'H2' as element 'H2' instead of 2*'H' would
    raise a KeyError and the test would fail loudly. Also discriminates
    a regression that omitted the oxygen contribution: ``2*H + O`` is
    well above ``2*H`` alone (the wrong-formula value would land near
    2 g/mol, not ~18).
    """
    from proteus.utils.constants import element_mmw
    from proteus.utils.helper import eval_gas_mmw

    expected = 2 * element_mmw['H'] + element_mmw['O']
    actual = eval_gas_mmw('H2O')
    assert actual == pytest.approx(expected)
    assert abs(actual - 2 * element_mmw['H']) > element_mmw['O'] * 0.5


# ---------------------------------------------------------------------------
# get_proteus_dir
# ---------------------------------------------------------------------------


def test_get_proteus_dir_returns_directory_containing_pyproject_toml():
    """The returned path must contain ``pyproject.toml`` at its root and
    that pyproject must declare PROTEUS (a regression that walked too
    far up the tree could land on an unrelated parent pyproject).
    """
    from proteus.utils.helper import get_proteus_dir

    root = get_proteus_dir()
    pyproject = os.path.join(root, 'pyproject.toml')
    assert os.path.isfile(pyproject)
    with open(pyproject, encoding='utf-8') as fh:
        assert 'fwl-proteus' in fh.read()


# ---------------------------------------------------------------------------
# create_tmp_folder uses TMPDIR when valid
# ---------------------------------------------------------------------------


def test_create_tmp_folder_respects_tmpdir_when_present(tmp_path, monkeypatch):
    """If ``TMPDIR`` points at a real directory, the created folder is
    placed under it. Discrimination: the returned path must start with
    the supplied TMPDIR, not the global ``/tmp``.
    """
    from proteus.utils.helper import create_tmp_folder, safe_rm

    monkeypatch.setenv('TMPDIR', str(tmp_path))
    created = create_tmp_folder()
    try:
        assert created.startswith(str(tmp_path))
        assert os.path.isdir(created)
    finally:
        safe_rm(created)


def test_create_tmp_folder_falls_back_to_slash_tmp_when_tmpdir_invalid(monkeypatch):
    """When ``TMPDIR`` is empty or points at a missing path, the folder
    is created under ``/tmp``.
    """
    from proteus.utils.helper import create_tmp_folder, safe_rm

    monkeypatch.setenv('TMPDIR', '/nonexistent/path/that/should/not/exist')
    created = create_tmp_folder()
    try:
        assert created.startswith('/tmp/')
        assert os.path.isdir(created)
    finally:
        safe_rm(created)
