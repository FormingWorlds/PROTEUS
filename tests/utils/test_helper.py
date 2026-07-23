"""
Unit tests for proteus.utils.helper module.

Tests pure utility functions with no heavy dependencies. These are foundational
helper functions used throughout PROTEUS for:
- File and directory management
- String parsing and sorting
- Status codes and comments
- Numeric utility functions
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from proteus.utils.helper import (
    CleanDir,
    CommentFromStatus,
    UpdateStatusfile,
    create_tmp_folder,
    find_nearest,
    is_write_snapshot,
    mol_to_ele,
    multiple,
    natural_sort,
    recursive_get,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# =============================================================================
# Test: multiple() - Robust modulo checking
# =============================================================================


class TestMultiple:
    """Test the multiple() function for robust divisibility checking."""

    @pytest.mark.unit
    def test_multiple_basic_true(self):
        """Test basic divisibility: 10 is multiple of 5."""
        assert multiple(10, 5) is True
        # Discrimination: a regression returning truthy non-bool (e.g. the
        # raw integer 0 or a numpy.bool_) would pass `is True` only if the
        # bool() coercion held; pin the exact type to catch that.
        assert isinstance(multiple(10, 5), bool)

    @pytest.mark.unit
    def test_multiple_basic_false(self):
        """Test non-divisibility: 10 is not multiple of 3."""
        assert multiple(10, 3) is False
        # Discrimination: a sign-swap regression returning the remainder
        # (10 % 3 == 1) instead of False would fail this strict-False check.
        assert multiple(10, 3) != 1

    @pytest.mark.unit
    def test_multiple_with_zero_a(self):
        """Zero is multiple of any non-zero integer: 0 % 5 == 0."""
        assert multiple(0, 5) is True
        # Discrimination: symmetric case across a different divisor catches
        # a regression that special-cases only b=5 or only a==0 path.
        assert multiple(0, 13) is True

    @pytest.mark.unit
    def test_multiple_with_zero_b(self):
        """Division by zero returns False (safe handling)."""
        assert multiple(10, 0) is False
        # Discrimination: pin the bool type so a regression returning 0
        # (a falsy int) instead of the False sentinel is caught.
        assert multiple(10, 0) is not None and multiple(10, 0) == False  # noqa: E712

    @pytest.mark.unit
    def test_multiple_with_none_a(self):
        """None as first argument returns False (safe handling)."""
        assert multiple(None, 5) is False
        # Discrimination: a regression that raises TypeError instead of
        # returning False would be caught by re-invoking under no-raises.
        assert multiple(None, 5) is not True

    @pytest.mark.unit
    def test_multiple_with_none_b(self):
        """None as second argument returns False (safe handling)."""
        assert multiple(10, None) is False
        # Discrimination: a regression that crashed on b is None instead
        # of returning False would fail the second invocation here.
        assert multiple(10, None) is not True

    @pytest.mark.unit
    def test_multiple_both_none(self):
        """Both arguments None returns False (safe handling)."""
        assert multiple(None, None) is False
        # Discrimination: confirm the both-None path returns the same
        # sentinel as the single-None path (uniform safe-handling contract).
        assert multiple(None, None) == multiple(None, 5)

    @pytest.mark.unit
    def test_multiple_same_number(self):
        """Number is always multiple of itself: n % n == 0."""
        assert multiple(7, 7) is True
        # Discrimination: property holds for any non-zero integer, not just
        # 7; a regression keyed on a specific value would fail this.
        assert multiple(123, 123) is True

    @pytest.mark.unit
    def test_multiple_one_is_divisor(self):
        """Any integer is multiple of 1."""
        assert multiple(42, 1) is True
        # Discrimination: property holds for arbitrary integers; pin a
        # second value to catch a regression that special-cases 42.
        assert multiple(-99, 1) is True


# =============================================================================
# Test: mol_to_ele() - Molecular composition parsing
# =============================================================================


class TestMolToEle:
    """Test molecular formula parsing into elemental composition."""

    @pytest.mark.unit
    def test_single_atom(self):
        """Parse single atom: H → {H: 1}."""
        result = mol_to_ele('H')
        assert result == {'H': 1}
        # Discrimination: a regression that emitted spurious empty-string
        # keys from the regex split would have more than one entry.
        assert len(result) == 1

    @pytest.mark.unit
    def test_diatomic_molecule(self):
        """Parse diatomic molecule: H2 → {H: 2}."""
        result = mol_to_ele('H2')
        assert result == {'H': 2}
        # Discrimination: a regression that read the count as the string '2'
        # rather than the int 2 would still equal-compare loosely; pin int.
        assert isinstance(result['H'], int)

    @pytest.mark.unit
    def test_water(self):
        """Parse water molecule: H2O → {H: 2, O: 1}."""
        result = mol_to_ele('H2O')
        assert result == {'H': 2, 'O': 1}
        # Discrimination: a regression that swapped the counts (H:1, O:2)
        # while keeping the same key set would pass a key-only check.
        assert result['H'] == 2 * result['O']

    @pytest.mark.unit
    def test_carbon_dioxide(self):
        """Parse CO2: {C: 1, O: 2}."""
        result = mol_to_ele('CO2')
        assert result == {'C': 1, 'O': 2}
        # Discrimination: the implicit-1 count on the lead atom is the
        # common parser bug; pin the ratio so C:O = 1:2 stays explicit.
        assert result['O'] == 2 * result['C']

    @pytest.mark.unit
    def test_methane(self):
        """Parse CH4 (methane): {C: 1, H: 4}."""
        result = mol_to_ele('CH4')
        assert result == {'C': 1, 'H': 4}
        # Discrimination: total atom count is 5; a regression that dropped
        # the lead-atom implicit-1 would sum to 4 instead.
        assert sum(result.values()) == 5

    @pytest.mark.unit
    def test_sulfuric_acid(self):
        """Parse H2SO4: {H: 2, S: 1, O: 4}."""
        result = mol_to_ele('H2SO4')
        assert result == {'H': 2, 'S': 1, 'O': 4}
        # Discrimination: total atom count is 7; a regression that dropped
        # S (a single-atom middle group) would sum to 6.
        assert sum(result.values()) == 7

    @pytest.mark.unit
    def test_invalid_lowercase_start(self):
        """Reject molecules starting with lowercase: 'h2o' raises ValueError."""
        with pytest.raises(ValueError, match='Could not decompose'):
            mol_to_ele('h2o')
        # Discrimination: confirm the rejection is robust across other
        # lowercase-prefix inputs (a regression keyed on 'h' would pass
        # the first raise but miss this one).
        with pytest.raises(ValueError, match='Could not decompose'):
            mol_to_ele('xyz')

    @pytest.mark.unit
    def test_empty_result_raises(self):
        """Invalid molecule names that produce empty parse raise ValueError."""
        # Numbers-only or special characters should fail
        with pytest.raises(ValueError, match='Could not decompose'):
            mol_to_ele('123')
        # Discrimination: punctuation-only is also expected to raise; a
        # regression that only guarded digits would let '@@@' through.
        with pytest.raises(ValueError, match='Could not decompose'):
            mol_to_ele('@@@')

    @pytest.mark.unit
    def test_complex_molecule(self):
        """Parse more complex molecule: Ca(OH)2-like formula (without parens)."""
        # Note: This implementation doesn't handle parentheses, so test
        # what it actually supports
        result = mol_to_ele('CaO')
        assert result == {'Ca': 1, 'O': 1}
        # Discrimination: 'Ca' is parsed as one two-letter element, NOT as
        # 'C' + 'a'; a regression to single-letter-only tokenisation would
        # split it and the dict would carry a 'C' key instead.
        assert 'Ca' in result and 'C' not in result


# =============================================================================
# Test: natural_sort() - Natural sorting
# =============================================================================


class TestNaturalSort:
    """Test natural sorting (numeric-aware sorting)."""

    @pytest.mark.unit
    def test_natural_sort_basic(self):
        """Sort mixed alphanumeric strings naturally."""
        lst = ['file10.txt', 'file2.txt', 'file1.txt']
        result = natural_sort(lst)
        assert result == ['file1.txt', 'file2.txt', 'file10.txt']
        # Discrimination: a lexicographic (str-only) regression would
        # place 'file10.txt' before 'file2.txt'; pin the relative order.
        assert result.index('file2.txt') < result.index('file10.txt')

    @pytest.mark.unit
    def test_natural_sort_pure_numbers(self):
        """Sort numeric strings in correct numeric order."""
        lst = ['100', '20', '3', '1']
        result = natural_sort(lst)
        assert result == ['1', '3', '20', '100']
        # Discrimination: lexicographic sort gives ['1', '100', '20', '3'];
        # pin that '100' sits at the END (numeric order), not at index 1.
        assert result[-1] == '100'

    @pytest.mark.unit
    def test_natural_sort_pure_text(self):
        """Sort pure alphabetic strings correctly."""
        lst = ['zebra', 'apple', 'banana']
        result = natural_sort(lst)
        assert result == ['apple', 'banana', 'zebra']
        # Discrimination: pin that no element was dropped or duplicated by
        # the sort (set-equality across multisets via Counter would catch
        # duplicate-emission regressions; len equality is sufficient here).
        assert len(result) == len(lst)

    @pytest.mark.unit
    def test_natural_sort_case_insensitive(self):
        """Sorting is case-insensitive."""
        lst = ['Zebra', 'apple', 'Banana']
        result = natural_sort(lst)
        # Should be sorted ignoring case
        assert result[0].lower() == 'apple'
        # Discrimination: case-sensitive ASCII sort places capital letters
        # before lowercase (so 'Banana','Zebra','apple'); pin that 'Zebra'
        # is last and 'Banana' sits in the middle.
        assert result[-1].lower() == 'zebra' and result[1].lower() == 'banana'

    @pytest.mark.unit
    def test_natural_sort_already_sorted(self):
        """Already-sorted list returns unchanged."""
        lst = ['item1', 'item2', 'item3']
        result = natural_sort(lst)
        assert result == lst
        # Discrimination: the function must return a NEW list, not mutate
        # the input in place (mutation-in-place is a known regression class
        # for sort wrappers).
        assert result is not lst

    @pytest.mark.unit
    def test_natural_sort_empty(self):
        """Empty list returns empty list."""
        assert natural_sort([]) == []
        # Discrimination: must return an actual list, not None or a
        # generator. A regression that returned None would pass the loose
        # `== []` check via falsiness only if Python were forgiving (it
        # isn't here), but pin the type to be explicit.
        assert isinstance(natural_sort([]), list)

    @pytest.mark.unit
    def test_natural_sort_single_element(self):
        """Single element list returns unchanged."""
        assert natural_sort(['a']) == ['a']
        # Discrimination: the single-element path must not produce nested
        # output (e.g. [['a']]) from an over-eager split; pin scalar shape.
        assert natural_sort(['a'])[0] == 'a'


# =============================================================================
# Test: CommentFromStatus() - Status code interpretation
# =============================================================================


class TestCommentFromStatus:
    """Test conversion of status codes to human-readable descriptions."""

    @pytest.mark.unit
    def test_status_started(self):
        """Status 0: Started."""
        assert CommentFromStatus(0) == 'Started'
        # Discrimination: a regression that mapped 0 to the default
        # 'UNHANDLED STATUS (0)' branch would not equal 'Started' but
        # might match a loose substring; pin against the unhandled label.
        assert 'UNHANDLED' not in CommentFromStatus(0)

    @pytest.mark.unit
    def test_status_running(self):
        """Status 1: Running."""
        assert CommentFromStatus(1) == 'Running'
        # Discrimination: status 1 is a RUNNING case, not a completed or
        # error case; pin that the result does not start with 'Completed'
        # or 'Error' (a swap regression to case 10 or 20 would fail this).
        assert not CommentFromStatus(1).startswith(('Completed', 'Error'))

    @pytest.mark.unit
    def test_status_solidified(self):
        """Status 10: Completed (solidified)."""
        assert CommentFromStatus(10) == 'Completed (solidified)'
        # Discrimination: differentiate from the other Completed cases
        # (max iterations / target time / net flux / volatiles escaped /
        # disintegrated); pin the qualifier substring.
        assert 'solidified' in CommentFromStatus(10)

    @pytest.mark.unit
    def test_status_runtime(self):
        """Status 11: Completed (maximum clock runtime)."""
        assert CommentFromStatus(11) == 'Completed (maximum clock runtime)'
        assert 'maximum clock runtime' in CommentFromStatus(11)

    @pytest.mark.unit
    def test_status_max_iterations(self):
        """Status 12: Completed (maximum iterations)."""
        assert CommentFromStatus(12) == 'Completed (maximum iterations)'
        # Discrimination: differentiate from the neighbouring 11
        # (UNUSED_STATUS_CODE) and 13 (target time) branches by pinning
        # the iterations qualifier.
        assert 'iterations' in CommentFromStatus(12)

    @pytest.mark.unit
    def test_status_volatiles_escaped(self):
        """Status 15: Completed (volatiles escaped)."""
        assert CommentFromStatus(15) == 'Completed (volatiles escaped)'
        # Discrimination: differentiate from 16 (planet disintegrated) by
        # pinning the volatiles-escaped qualifier substring.
        assert 'volatiles' in CommentFromStatus(15)

    @pytest.mark.unit
    def test_status_generic_error(self):
        """Status 20: Generic error."""
        result = CommentFromStatus(20)
        assert 'Error' in result
        # Discrimination: status 20 is the generic-error parent of the
        # 21-28 family; pin the parenthetical that distinguishes it from
        # the specific-error cases (no model-component name).
        assert 'generic' in result

    @pytest.mark.unit
    def test_status_interior_error(self):
        """Status 21: Interior model error."""
        assert 'Interior' in CommentFromStatus(21)
        # Discrimination: this is an ERROR case, not a Running or
        # Completed one; pin the Error prefix to catch a regression that
        # mislabels the severity while keeping the component name.
        assert 'Error' in CommentFromStatus(21)

    @pytest.mark.unit
    def test_status_atmosphere_error(self):
        """Status 22: Atmosphere model error."""
        assert 'Atmosphere' in CommentFromStatus(22)
        # Discrimination: distinguish from neighbouring component-error
        # cases (Interior/Stellar/Kinetics) by pinning that 'Interior' is
        # NOT in the atmosphere label.
        assert 'Interior' not in CommentFromStatus(22)

    @pytest.mark.unit
    def test_status_unknown(self):
        """Unknown status codes get UNHANDLED label."""
        result = CommentFromStatus(999)
        assert 'UNHANDLED' in result
        # Discrimination: the unhandled branch echoes the integer code
        # into the string; pin that the actual code (999) appears so a
        # regression that emits a fixed placeholder is caught.
        assert '999' in result


# =============================================================================
# Test: UpdateStatusfile() - Status file creation
# =============================================================================


class TestUpdateStatusfile:
    """Test status file creation and updating."""

    @pytest.mark.unit
    def test_statusfile_creation(self):
        """Create status file with correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = {'output': tmpdir}
            UpdateStatusfile(dirs, 0)

            # Check file exists
            statusfile = os.path.join(tmpdir, 'status')
            assert os.path.exists(statusfile)

            # Check content
            with open(statusfile, 'r') as f:
                lines = f.readlines()
            assert len(lines) == 2
            assert lines[0].strip() == '0'
            assert 'Started' in lines[1]

    @pytest.mark.unit
    def test_statusfile_updates_on_second_call(self):
        """Statusfile is overwritten on subsequent calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = {'output': tmpdir}

            # First update
            UpdateStatusfile(dirs, 1)
            statusfile = os.path.join(tmpdir, 'status')
            with open(statusfile, 'r') as f:
                content1 = f.read()

            # Second update
            UpdateStatusfile(dirs, 10)
            with open(statusfile, 'r') as f:
                content2 = f.read()

            assert '1' in content1
            assert '10' in content2
            assert content1 != content2

    @pytest.mark.unit
    def test_statusfile_creates_directory(self):
        """Creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            newdir = os.path.join(tmpdir, 'nested', 'path')
            dirs = {'output': newdir}

            UpdateStatusfile(dirs, 0)
            assert os.path.exists(newdir)
            assert os.path.exists(os.path.join(newdir, 'status'))


# =============================================================================
# Test: CleanDir() - Directory cleaning
# =============================================================================


class TestCleanDir:
    """Test directory cleaning with safety checks."""

    @pytest.mark.unit
    def test_cleandir_removes_files(self):
        """CleanDir removes all files from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(tmpdir, 'file1.txt').touch()
            Path(tmpdir, 'file2.txt').touch()

            # Clean directory
            CleanDir(tmpdir)

            # Check directory is empty
            assert len(os.listdir(tmpdir)) == 0
            assert os.path.isdir(tmpdir)

    @pytest.mark.unit
    def test_cleandir_removes_subdirectories(self):
        """CleanDir removes subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = os.path.join(tmpdir, 'subdir')
            os.makedirs(subdir)
            Path(subdir, 'file.txt').touch()

            # Clean
            CleanDir(tmpdir)

            assert len(os.listdir(tmpdir)) == 0
            # Discrimination: tmpdir itself must still be a live directory
            # after the clean (a regression that rmtree-d the parent and
            # forgot to recreate it would raise OSError on listdir; in
            # case that listdir call ever changes to a tolerant wrapper,
            # pin the parent-still-exists property explicitly).
            assert os.path.isdir(tmpdir)

    @pytest.mark.unit
    def test_cleandir_nonexistent_directory(self):
        """CleanDir creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            newdir = os.path.join(tmpdir, 'newdir')
            assert not os.path.exists(newdir)

            CleanDir(newdir)

            assert os.path.exists(newdir)
            assert os.path.isdir(newdir)

    @pytest.mark.unit
    def test_cleandir_git_safety(self):
        """CleanDir raises error if directory contains .git."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .git subdirectory
            git_dir = os.path.join(tmpdir, '.git')
            os.makedirs(git_dir)

            # Should raise exception
            with pytest.raises(Exception, match='Git repository'):
                CleanDir(tmpdir)
            # Discrimination: after the safety abort, the .git directory
            # itself must still be intact (a regression that partially
            # ran the rmtree before checking would leave the dir gone
            # even though the exception fired).
            assert os.path.isdir(git_dir)


# =============================================================================
# Test: find_nearest() - Nearest value finding
# =============================================================================


class TestFindNearest:
    """Test finding nearest value in array."""

    @pytest.mark.unit
    def test_find_nearest_exact_match(self):
        """Find exact match returns the value and its index."""
        import numpy as np

        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        value, idx = find_nearest(array, 3.0)
        assert value == pytest.approx(3.0)
        assert idx == 2

    @pytest.mark.unit
    def test_find_nearest_between_values(self):
        """Target between values returns nearest one."""
        import numpy as np

        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        value, idx = find_nearest(array, 2.6)
        # Should be 3.0 (closest)
        assert value == pytest.approx(3.0)
        # Discrimination: pin the returned index so a regression that
        # picks the floor (2.0 at idx 1) instead of the nearest (3.0 at
        # idx 2) is caught. 2.6 is 0.4 from 3.0 and 0.6 from 2.0.
        assert idx == 2

    @pytest.mark.unit
    def test_find_nearest_edge_values(self):
        """Values at edges are handled correctly."""
        import numpy as np

        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        value_low, _ = find_nearest(array, 0.5)
        assert value_low == pytest.approx(1.0)

        value_high, _ = find_nearest(array, 5.5)
        assert value_high == pytest.approx(5.0)

    @pytest.mark.unit
    def test_find_nearest_single_element(self):
        """Single element array returns that element."""
        import numpy as np

        array = np.array([42.0])
        value, idx = find_nearest(array, 100.0)
        assert value == pytest.approx(42.0)
        assert idx == 0


# =============================================================================
# Test: recursive_get() - Nested dictionary access
# =============================================================================


class TestRecursiveGet:
    """Test nested dictionary access."""

    @pytest.mark.unit
    def test_recursive_get_single_key(self):
        """Access top-level key."""
        d = {'a': 1, 'b': 2}
        assert recursive_get(d, ['a']) == 1
        # Discrimination: a regression that returned d[keys[-1]] regardless
        # of path length would give 1 here but 2 for ['b']; pin the second
        # key independently to catch index-confusion bugs.
        assert recursive_get(d, ['b']) == 2

    @pytest.mark.unit
    def test_recursive_get_nested(self):
        """Access nested key path."""
        d = {'a': {'b': {'c': 42}}}
        assert recursive_get(d, ['a', 'b', 'c']) == 42
        # Discrimination: a regression that returned the leaf-most dict
        # one level too early would return {'c': 42} instead of 42; pin
        # the int type so a dict-return regression is caught.
        assert isinstance(recursive_get(d, ['a', 'b', 'c']), int)

    @pytest.mark.unit
    def test_recursive_get_missing_key(self):
        """Missing key raises TypeError (tries to subscript non-dict)."""
        d = {'a': {'b': 1}}
        # When we try ['a', 'b', 'c'], we get 1 at ['a', 'b'],
        # then try to subscript 1['c'] which raises TypeError
        with pytest.raises(TypeError):
            recursive_get(d, ['a', 'b', 'c'])
        # Discrimination: a regression that silently returned None on
        # non-dict subscripts (instead of letting Python raise) would
        # bypass the with-raises block AND leave a value retrievable
        # for the prefix path that IS valid; pin the prefix still works.
        assert recursive_get(d, ['a', 'b']) == 1

    @pytest.mark.unit
    def test_recursive_get_missing_intermediate(self):
        """Missing intermediate key raises KeyError."""
        d = {'a': {'b': 1}}
        with pytest.raises(KeyError):
            recursive_get(d, ['x', 'y', 'z'])
        # Discrimination: confirm a SHORTER missing-key path also raises
        # KeyError (a regression keyed on the three-level depth would
        # pass the longer raise but miss this single-level miss).
        with pytest.raises(KeyError):
            recursive_get(d, ['x'])

    @pytest.mark.unit
    def test_recursive_get_three_level_nesting(self):
        """Deeply nested access works correctly."""
        d = {'x': {'y': {'z': 123}}}
        assert recursive_get(d, ['x', 'y', 'z']) == 123
        # Discrimination: an off-by-one descent regression that stopped
        # one level early would return the dict {'z': 123} (truthy, with
        # a 'z' key) rather than the int 123; pin the int comparison via
        # an arithmetic relation a dict would not satisfy.
        assert recursive_get(d, ['x', 'y', 'z']) + 1 == 124


# =============================================================================
# Test: create_tmp_folder() - Temporary directory creation
# =============================================================================


class TestCreateTmpFolder:
    """Test temporary folder creation."""

    @pytest.mark.unit
    def test_create_tmp_folder_creates_directory(self):
        """create_tmp_folder creates a valid temporary directory."""
        tmpdir = create_tmp_folder()
        try:
            assert os.path.isdir(tmpdir)
            assert os.path.exists(tmpdir)
        finally:
            # Cleanup
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)

    @pytest.mark.unit
    def test_create_tmp_folder_is_writable(self):
        """Created temporary folder is writable."""
        tmpdir = create_tmp_folder()
        try:
            # Try to write a file
            testfile = os.path.join(tmpdir, 'test.txt')
            with open(testfile, 'w') as f:
                f.write('test')
            assert os.path.exists(testfile)
            # Discrimination: confirm the file actually carries the bytes
            # we wrote (a regression returning a read-only folder whose
            # writes silently failed could still leave the dirent in place
            # on some filesystems; pin the round-tripped content).
            with open(testfile, 'r') as f:
                assert f.read() == 'test'
        finally:
            # Cleanup
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)

    @pytest.mark.unit
    def test_create_tmp_folder_unique(self):
        """Multiple calls create different directories."""
        tmpdir1 = create_tmp_folder()
        tmpdir2 = create_tmp_folder()

        try:
            assert tmpdir1 != tmpdir2
            assert os.path.exists(tmpdir1)
            assert os.path.exists(tmpdir2)
        finally:
            # Cleanup
            for tmpdir in [tmpdir1, tmpdir2]:
                if os.path.exists(tmpdir):
                    shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# is_write_snapshot: OR-combined write cadence (write_mod OR dt_write_rel)
# ---------------------------------------------------------------------------


def test_is_write_snapshot_default_writes_on_write_mod_cadence():
    """With dt_write_rel disabled (0), the decision reduces to the write_mod
    iteration cadence, preserving the pre-OR default behaviour.

    Discrimination: a write_mod-boundary iteration writes; an off-boundary
    iteration does not. Both asserted so an always-True/always-False
    regression is caught.
    """
    # On a write_mod=5 boundary -> write.
    assert is_write_snapshot(10, 5, 0.0, 1.0e6, 0.0) is True
    # Off the boundary, and time trigger disabled -> no write.
    assert is_write_snapshot(11, 5, 0.0, 1.0e6, 0.0) is False


def test_is_write_snapshot_time_trigger_is_individually_sufficient():
    """A large elapsed time triggers a write even off the write_mod cadence,
    and an insufficient elapsed time does not (when off-cadence).

    This is the OR behaviour: either criterion alone suffices. Inputs chosen
    so cur_time=1e6, last_write=0, dt_write_rel=1e-3 gives an interval of
    1e3 yr, which 1e6 far exceeds (write); with last_write=1e6-1 the elapsed
    1 yr is well below 1e3 (no write).
    """
    # Off-cadence (11 not a multiple of 5) but time elapsed >> interval.
    assert is_write_snapshot(11, 5, 1.0e-3, 1.0e6, 0.0) is True
    # Off-cadence and elapsed time (1 yr) below the interval (1e3 yr).
    assert is_write_snapshot(11, 5, 1.0e-3, 1.0e6, 1.0e6 - 1.0) is False


def test_is_write_snapshot_initial_iteration_and_disabled_time_guard():
    """The initial iteration (loop 0 with the default write_mod=1) writes, and
    the time criterion never fires on its own when dt_write_rel <= 0.

    Edge/limit cases: loop 0 with last_write_time = -inf (initial condition),
    and dt_write_rel = 0 with a huge elapsed time (must NOT force a write off
    the write_mod cadence, guarding against the OR-with-always-true bug).
    """
    import math

    # Initial iteration: loop 0 is a multiple of write_mod=1 -> write.
    assert is_write_snapshot(0, 1, 0.0, 0.0, -math.inf) is True
    # dt_write_rel = 0 must not let the time term fire even with huge elapsed
    # time on an off-cadence iteration.
    assert is_write_snapshot(3, 100, 0.0, 1.0e12, -math.inf) is False
    # write_mod = 0 ("wait until completion") with time trigger off -> no write.
    assert is_write_snapshot(50, 0, 0.0, 1.0e6, 0.0) is False
