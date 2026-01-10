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
    mol_to_ele,
    multiple,
    natural_sort,
    recursive_get,
)

# =============================================================================
# Test: multiple() - Robust modulo checking
# =============================================================================


class TestMultiple:
    """Test the multiple() function for robust divisibility checking."""

    @pytest.mark.unit
    def test_multiple_basic_true(self):
        """Test basic divisibility: 10 is multiple of 5."""
        assert multiple(10, 5) is True

    @pytest.mark.unit
    def test_multiple_basic_false(self):
        """Test non-divisibility: 10 is not multiple of 3."""
        assert multiple(10, 3) is False

    @pytest.mark.unit
    def test_multiple_with_zero_a(self):
        """Zero is multiple of any non-zero integer: 0 % 5 == 0."""
        assert multiple(0, 5) is True

    @pytest.mark.unit
    def test_multiple_with_zero_b(self):
        """Division by zero returns False (safe handling)."""
        assert multiple(10, 0) is False

    @pytest.mark.unit
    def test_multiple_with_none_a(self):
        """None as first argument returns False (safe handling)."""
        assert multiple(None, 5) is False

    @pytest.mark.unit
    def test_multiple_with_none_b(self):
        """None as second argument returns False (safe handling)."""
        assert multiple(10, None) is False

    @pytest.mark.unit
    def test_multiple_both_none(self):
        """Both arguments None returns False (safe handling)."""
        assert multiple(None, None) is False

    @pytest.mark.unit
    def test_multiple_same_number(self):
        """Number is always multiple of itself: n % n == 0."""
        assert multiple(7, 7) is True

    @pytest.mark.unit
    def test_multiple_one_is_divisor(self):
        """Any integer is multiple of 1."""
        assert multiple(42, 1) is True


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

    @pytest.mark.unit
    def test_diatomic_molecule(self):
        """Parse diatomic molecule: H2 → {H: 2}."""
        result = mol_to_ele('H2')
        assert result == {'H': 2}

    @pytest.mark.unit
    def test_water(self):
        """Parse water molecule: H2O → {H: 2, O: 1}."""
        result = mol_to_ele('H2O')
        assert result == {'H': 2, 'O': 1}

    @pytest.mark.unit
    def test_carbon_dioxide(self):
        """Parse CO2: {C: 1, O: 2}."""
        result = mol_to_ele('CO2')
        assert result == {'C': 1, 'O': 2}

    @pytest.mark.unit
    def test_methane(self):
        """Parse CH4 (methane): {C: 1, H: 4}."""
        result = mol_to_ele('CH4')
        assert result == {'C': 1, 'H': 4}

    @pytest.mark.unit
    def test_sulfuric_acid(self):
        """Parse H2SO4: {H: 2, S: 1, O: 4}."""
        result = mol_to_ele('H2SO4')
        assert result == {'H': 2, 'S': 1, 'O': 4}

    @pytest.mark.unit
    def test_invalid_lowercase_start(self):
        """Reject molecules starting with lowercase: 'h2o' raises ValueError."""
        with pytest.raises(ValueError, match='Could not decompose'):
            mol_to_ele('h2o')

    @pytest.mark.unit
    def test_empty_result_raises(self):
        """Invalid molecule names that produce empty parse raise ValueError."""
        # Numbers-only or special characters should fail
        with pytest.raises(ValueError, match='Could not decompose'):
            mol_to_ele('123')

    @pytest.mark.unit
    def test_complex_molecule(self):
        """Parse more complex molecule: Ca(OH)2-like formula (without parens)."""
        # Note: This implementation doesn't handle parentheses, so test
        # what it actually supports
        result = mol_to_ele('CaO')
        assert result == {'Ca': 1, 'O': 1}


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

    @pytest.mark.unit
    def test_natural_sort_pure_numbers(self):
        """Sort numeric strings in correct numeric order."""
        lst = ['100', '20', '3', '1']
        result = natural_sort(lst)
        assert result == ['1', '3', '20', '100']

    @pytest.mark.unit
    def test_natural_sort_pure_text(self):
        """Sort pure alphabetic strings correctly."""
        lst = ['zebra', 'apple', 'banana']
        result = natural_sort(lst)
        assert result == ['apple', 'banana', 'zebra']

    @pytest.mark.unit
    def test_natural_sort_case_insensitive(self):
        """Sorting is case-insensitive."""
        lst = ['Zebra', 'apple', 'Banana']
        result = natural_sort(lst)
        # Should be sorted ignoring case
        assert result[0].lower() == 'apple'

    @pytest.mark.unit
    def test_natural_sort_already_sorted(self):
        """Already-sorted list returns unchanged."""
        lst = ['item1', 'item2', 'item3']
        result = natural_sort(lst)
        assert result == lst

    @pytest.mark.unit
    def test_natural_sort_empty(self):
        """Empty list returns empty list."""
        assert natural_sort([]) == []

    @pytest.mark.unit
    def test_natural_sort_single_element(self):
        """Single element list returns unchanged."""
        assert natural_sort(['a']) == ['a']


# =============================================================================
# Test: CommentFromStatus() - Status code interpretation
# =============================================================================


class TestCommentFromStatus:
    """Test conversion of status codes to human-readable descriptions."""

    @pytest.mark.unit
    def test_status_started(self):
        """Status 0: Started."""
        assert CommentFromStatus(0) == 'Started'

    @pytest.mark.unit
    def test_status_running(self):
        """Status 1: Running."""
        assert CommentFromStatus(1) == 'Running'

    @pytest.mark.unit
    def test_status_solidified(self):
        """Status 10: Completed (solidified)."""
        assert CommentFromStatus(10) == 'Completed (solidified)'

    @pytest.mark.unit
    def test_status_max_iterations(self):
        """Status 12: Completed (maximum iterations)."""
        assert CommentFromStatus(12) == 'Completed (maximum iterations)'

    @pytest.mark.unit
    def test_status_volatiles_escaped(self):
        """Status 15: Completed (volatiles escaped)."""
        assert CommentFromStatus(15) == 'Completed (volatiles escaped)'

    @pytest.mark.unit
    def test_status_generic_error(self):
        """Status 20: Generic error."""
        result = CommentFromStatus(20)
        assert 'Error' in result

    @pytest.mark.unit
    def test_status_interior_error(self):
        """Status 21: Interior model error."""
        assert 'Interior' in CommentFromStatus(21)

    @pytest.mark.unit
    def test_status_atmosphere_error(self):
        """Status 22: Atmosphere model error."""
        assert 'Atmosphere' in CommentFromStatus(22)

    @pytest.mark.unit
    def test_status_unknown(self):
        """Unknown status codes get UNHANDLED label."""
        result = CommentFromStatus(999)
        assert 'UNHANDLED' in result


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

    @pytest.mark.unit
    def test_recursive_get_nested(self):
        """Access nested key path."""
        d = {'a': {'b': {'c': 42}}}
        assert recursive_get(d, ['a', 'b', 'c']) == 42

    @pytest.mark.unit
    def test_recursive_get_missing_key(self):
        """Missing key raises TypeError (tries to subscript non-dict)."""
        d = {'a': {'b': 1}}
        # When we try ['a', 'b', 'c'], we get 1 at ['a', 'b'],
        # then try to subscript 1['c'] which raises TypeError
        with pytest.raises(TypeError):
            recursive_get(d, ['a', 'b', 'c'])

    @pytest.mark.unit
    def test_recursive_get_missing_intermediate(self):
        """Missing intermediate key raises KeyError."""
        d = {'a': {'b': 1}}
        with pytest.raises(KeyError):
            recursive_get(d, ['x', 'y', 'z'])

    @pytest.mark.unit
    def test_recursive_get_three_level_nesting(self):
        """Deeply nested access works correctly."""
        d = {'x': {'y': {'z': 123}}}
        assert recursive_get(d, ['x', 'y', 'z']) == 123


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
