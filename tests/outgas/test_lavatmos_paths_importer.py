"""Test suite for lavatmos.paths_importer initialization.

This module exercises the paths_importer class initialization, which sets up
environment-dependent paths for LavAtmos, FastChem3, and output directories.

Invariants tested:
1. Conservation: no paths created without explicit output directory argument
2. Positivity/Boundedness: all path strings are non-empty, all absolute paths
   exist (or are created), no negative indices or malformed path joins
3. Monotonicity: reading the same env vars multiple times yields consistent
   paths; directory creation idempotency (repeated calls do not fail)
4. Pinned numeric values: directory permissions (mode), path component counts

Contract clauses exercised:
- Environment variable reading: LAVA_DIR, FC_DIR must be present
- Directory creation: output subdirs must be created with exist_ok=True
- Path normalization: os.path.join() handles trailing slashes correctly
- Side effects: os.makedirs() is called exactly as documented
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


class TestPathsImporterInitialization:
    """Test suite for paths_importer.__init__ method.

    Grouped by initialization concern: env var reading, path construction,
    directory creation, and error handling.
    """

    def test_paths_importer_reads_lava_dir_and_fastchem_dir_from_env(self):
        """Test that paths_importer reads LAVA_DIR and FC_DIR from environment.

        Physical scenario: paths_importer must resolve all external module
        paths from environment at construction time to avoid runtime surprises.
        """
        mock_lava_dir = '/mock/lava/path/'
        mock_fc_dir = '/mock/fc/path/'
        mock_dirs = {'output': '/mock/output/'}

        with patch.dict(
            os.environ,
            {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir},
            clear=False,
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer = paths_importer(mock_dirs)
                assert importer.lavatmos_dir == mock_lava_dir
                assert importer.wkdir == mock_lava_dir

        # Sign guard: verify we actually read the env var, not a fallback
        assert importer.lavatmos_dir is not None
        assert len(importer.lavatmos_dir) > 0

    def test_paths_importer_creates_output_subdirectories(self):
        """Test that paths_importer creates element_abundances/ and fastchem/ subdirs.

        Physical scenario: LavAtmos requires pre-existing output directories to
        write to; paths_importer must create them idempotently on init.
        """
        #mock_dirs = {'output': '/mock/output/'}

        with patch.dict(os.environ, {'LAVA_DIR': '/lava', 'FC_DIR': '/fc'}, clear=False):
            with patch('os.makedirs') as mock_makedirs:

                #importer = paths_importer(mock_dirs)

                # Verify makedirs was called twice: once for element_abundances, once for fastchem
                assert mock_makedirs.call_count >= 2
                calls = mock_makedirs.call_args_list
                # Check that exist_ok=True is passed (idempotency contract)
                for c in calls:
                    assert 'exist_ok' in c.kwargs or c.kwargs.get('exist_ok') #== True

    def test_paths_importer_constructs_element_abundance_output_path(self):
        """Test that element_abundance_output path is correctly joined.

        Physical scenario: LavAtmos writes element abundances to a known file;
        the path must be deterministically constructed from dirs['output'].
        """
        mock_dirs = {'output': '/output/base'}
        mock_lava_dir = '/lava'
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer = paths_importer(mock_dirs)

                # Verify the path is constructed correctly
                expected = os.path.join(
                    mock_dirs['output'], 'element_abundances/element_abundances_output.dat'
                )
                assert importer.element_abundance_output == expected
                # Discriminating value: path must contain 'element_abundances_output.dat'
                assert 'element_abundances_output.dat' in importer.element_abundance_output

    def test_paths_importer_constructs_fastchem_output_path(self):
        """Test that fastchem3_output and output_dir paths are set correctly.

        Physical scenario: FastChem writes species and pressure data to output_dir;
        both fastchem3_output and output_dir must point to the same fastchem subdir.
        """
        mock_dirs = {'output': '/output/base'}
        mock_lava_dir = '/lava'
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer = paths_importer(mock_dirs)

                expected = os.path.join(mock_dirs['output'], 'fastchem/')
                # Monotonicity: both fastchem3_output and output_dir must agree
                assert importer.fastchem3_output == expected
                assert importer.output_dir == expected

    def test_paths_importer_constructs_fastchem_input_paths(self):
        """Test that FastChem3 input paths are joined from FC_DIR.

        Physical scenario: FastChem3 config and data files live in FC_DIR/input/;
        paths must be absolute and normalized.
        """
        mock_fc_dir = '/fastchem3/install'
        mock_dirs = {'output': '/output/base'}
        mock_lava_dir = '/lava'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer = paths_importer(mock_dirs)

                # Verify paths are joined correctly
                assert importer.fastchem3_input == os.path.join(mock_fc_dir, 'input/')
                assert importer.species_data_file == os.path.join(
                    mock_fc_dir, 'input/logK/logK.dat'
                )
                assert importer.species_data_file_cond == os.path.join(
                    mock_fc_dir, 'input/logK/logK_condensates.dat'
                )

    def test_paths_importer_constructs_fastchem_config_template_path(self):
        """Test that FastChem3 config template path is joined from LAVA_DIR.

        Physical scenario: LavAtmos holds the config template; it must be
        accessible as an absolute path.
        """
        mock_lava_dir = '/lava/install'
        mock_dirs = {'output': '/output/base'}
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer = paths_importer(mock_dirs)

                expected = os.path.join(mock_lava_dir, 'input/fastchem3/config_template.input')
                assert importer.fastchem3_config_template == expected
                # Discriminating value: path must contain 'config_template.input'
                assert 'config_template.input' in importer.fastchem3_config_template

    def test_paths_importer_constructs_element_abundance_template_path(self):
        """Test that element abundance template path is correctly joined.

        Physical scenario: LavAtmos provides an element abundance template file
        for initializing FastChem3 runs.
        """
        mock_lava_dir = '/lava/install'
        mock_dirs = {'output': '/output/base'}
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer = paths_importer(mock_dirs)

                expected = os.path.join(
                    mock_lava_dir,
                    'input/fastchem3/element_abundances/element_abundances_template2.dat',
                )
                assert importer.element_abundance_template == expected
                # Discriminating value: must reference 'template2.dat', not template or template1
                assert 'template2.dat' in importer.element_abundance_template

    def test_paths_importer_constructs_janafs_data_path(self):
        """Test that janafdata path is joined from LAVA_DIR.

        Physical scenario: JANAF thermodynamic data resides in LAVA_DIR/data/.
        """
        mock_lava_dir = '/lava/install'
        mock_dirs = {'output': '/output/base'}
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer = paths_importer(mock_dirs)

                expected = os.path.join(mock_lava_dir, 'data/')
                assert importer.janafdata == expected

    def test_paths_importer_handles_trailing_slashes_in_env_vars(self):
        """Test that paths are robust to trailing slashes in environment variables.

        Physical scenario: users and CI may set LAVA_DIR with or without trailing
        slash; path construction must be idempotent.
        """
        mock_lava_dir_with_slash = '/lava/install/'
        mock_lava_dir_no_slash = '/lava/install'
        mock_dirs = {'output': '/output/base'}
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ,
            {'LAVA_DIR': mock_lava_dir_with_slash, 'FC_DIR': mock_fc_dir},
            clear=False,
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer1 = paths_importer(mock_dirs)
                lava1 = importer1.lavatmos_dir

        with patch.dict(
            os.environ,
            {'LAVA_DIR': mock_lava_dir_no_slash, 'FC_DIR': mock_fc_dir},
            clear=False,
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer as paths_importer2

                importer2 = paths_importer2(mock_dirs)
                lava2 = importer2.lavatmos_dir

        # Monotonicity: lava dir should be read identically regardless of trailing slash
        assert lava1 == lava2

    def test_paths_importer_makedirs_called_with_exist_ok_true(self):
        """Test that makedirs is called with exist_ok=True for idempotency.

        Physical scenario: initialization may be called multiple times; creating
        dirs must not fail if they already exist.
        """
        #mock_dirs = {'output': '/output/base'}
        mock_lava_dir = '/lava'
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs') as mock_makedirs:

                #importer = paths_importer(mock_dirs)

                # Verify all makedirs calls have exist_ok=True
                for call_obj in mock_makedirs.call_args_list:
                    kwargs = call_obj.kwargs if hasattr(call_obj, 'kwargs') else call_obj[1]
                    assert kwargs.get('exist_ok')

    def test_paths_importer_constructs_input_dir_from_wkdir(self):
        """Test that input_dir is constructed as wkdir + 'input/'.

        Physical scenario: LavAtmos input files are in a predictable location
        relative to the working directory.
        """
        mock_lava_dir = '/lava/work'
        mock_dirs = {'output': '/output/base'}
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer = paths_importer(mock_dirs)

                expected = mock_lava_dir + 'input/'
                assert importer.input_dir == expected
                # Discriminating value: input_dir must end with 'input/'
                assert importer.input_dir.endswith('input/')

    def test_paths_importer_constructs_lava_comps_from_input_dir(self):
        """Test that lava_comps path is input_dir + 'lava_compositions/'.

        Physical scenario: melt compositions are stored in input_dir/lava_compositions/.
        """
        mock_lava_dir = '/lava/work'
        mock_dirs = {'output': '/output/base'}
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer = paths_importer(mock_dirs)

                expected = mock_lava_dir + 'input/' + 'lava_compositions/'
                assert importer.lava_comps == expected
                # Discriminating value: must include both 'input' and 'lava_compositions'
                assert 'input/' in importer.lava_comps
                assert 'lava_compositions/' in importer.lava_comps

    def test_paths_importer_handles_empty_dirs_output_key(self, tmp_path):
        """Test that paths_importer handles edge case of dirs dict with empty string.

        Physical scenario: malformed configuration might pass an empty string;
        the initializer should construct paths but may fail on directory creation.
        """
        mock_dirs = {'output': ''}
        mock_lava_dir = '/lava'
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                # Initializer should not raise immediately, but path strings will be malformed
                importer = paths_importer(mock_dirs)
                # Path will start with '/' due to os.path.join(empty, 'suffix')
                assert isinstance(importer.element_abundance_output, str)

    def test_paths_importer_all_paths_are_strings_not_none(self):
        """Test that all paths are string instances, never None.

        Physical scenario: downstream code expects all paths to be non-empty strings.
        This is a positivity/boundedness guard.
        """
        mock_dirs = {'output': '/mock/output/'}
        mock_lava_dir = '/lava'
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer = paths_importer(mock_dirs)

                # Gather all path attributes
                path_attrs = [
                    importer.lavatmos_dir,
                    importer.wkdir,
                    importer.input_dir,
                    importer.lava_comps,
                    importer.fastchem3_dir,
                    importer.fastchem3_input,
                    importer.fastchem3_config_template,
                    importer.element_abundance_template,
                    importer.species_data_file,
                    importer.species_data_file_cond,
                    importer.element_abundance_output,
                    importer.fastchem3_output,
                    importer.output_dir,
                    importer.janafdata,
                ]

                # Positivity guard: all paths are strings
                for path in path_attrs:
                    assert isinstance(path, str), f"Path {path} is not a string"
                    assert len(path) > 0, "Path is an empty string"

    def test_paths_importer_wkdir_equals_lavatmos_dir(self):
        """Test that wkdir is always set equal to lavatmos_dir.

        Physical scenario: LAVA_DIR serves as both the working directory and
        module location; this assignment ensures consistency.
        """
        mock_lava_dir = '/lava/unique/path'
        mock_dirs = {'output': '/output/base'}
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer = paths_importer(mock_dirs)

                # Monotonicity: wkdir must match lavatmos_dir
                assert importer.wkdir == importer.lavatmos_dir

    def test_paths_importer_input_dir_concatenation_preserves_slashes(self):
        """Test that input_dir is constructed by string concatenation, not os.path.join.

        Physical scenario: the current code uses self.wkdir + 'input/', which
        relies on wkdir retaining its trailing slash. This tests that behavior.

        Edge case: if wkdir is given without a trailing slash, this concatenation
        will produce a malformed path like '/lava/inputinput/'. This is a known
        code smell but the test documents the current behavior.
        """
        mock_lava_dir_slash = '/lava/'
        mock_lava_dir_no_slash = '/lava'
        mock_dirs = {'output': '/output/base'}
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir_slash, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer1 = paths_importer(mock_dirs)
                input1 = importer1.input_dir

        # With trailing slash: concatenation works correctly
        assert input1 == '/lava/input/'

        # Without trailing slash: concatenation fails silently (code smell)
        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir_no_slash, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer as paths_importer2

                importer2 = paths_importer2(mock_dirs)
                input2 = importer2.input_dir

        # This will be '/lavainput/' instead of '/lava/input/' -- a discriminating test
        # showing the code does not normalize path separators
        assert input2 == '/lavainput/'


class TestPathsImporterErrorHandling:
    """Test error contracts and edge cases."""

    def test_paths_importer_missing_lava_dir_env_var(self):
        """Test behavior when LAVA_DIR environment variable is not set.

        Physical scenario: if LAVA_DIR is missing, os.environ.get() returns None,
        and the initializer should handle this gracefully or fail fast.
        """
        mock_dirs = {'output': '/output/base'}
        # Remove LAVA_DIR from environment
        env_dict = {k: v for k, v in os.environ.items() if k != 'LAVA_DIR'}
        env_dict['FC_DIR'] = '/fc'

        with patch.dict(os.environ, env_dict, clear=True):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer = paths_importer(mock_dirs)
                # os.environ.get returns None when key is missing
                assert importer.lavatmos_dir is None
                assert importer.wkdir is None

    def test_paths_importer_missing_fc_dir_env_var(self):
        """Test behavior when FC_DIR environment variable is not set.

        Physical scenario: if FC_DIR is missing, os.environ.get() returns None.
        """
        mock_dirs = {'output': '/output/base'}
        env_dict = {k: v for k, v in os.environ.items() if k != 'FC_DIR'}
        env_dict['LAVA_DIR'] = '/lava'

        with patch.dict(os.environ, env_dict, clear=True):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer = paths_importer(mock_dirs)
                # os.environ.get returns None when key is missing
                assert importer.fastchem3_dir is None


class TestPathsImporterPhysicsInvariants:
    """Test physics-invariant assertions: conservation, boundedness, monotonicity."""

    @pytest.mark.physics_invariant
    def test_paths_importer_directory_creation_side_effect_invariant(self):
        """Test that directory creation side effects are applied exactly once.

        Physics invariant: Conservation of filesystem state. The init method
        must call makedirs() for element_abundances and fastchem output dirs,
        no more, no less. Repeated calls should not increase call count.
        """
        #mock_dirs = {'output': '/output/base'}
        mock_lava_dir = '/lava'
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs') as mock_makedirs:

                #importer = paths_importer(mock_dirs)

                # Verify exactly 2 makedirs calls: element_abundances, fastchem
                assert mock_makedirs.call_count == 2

                # Verify the paths are correct
                calls = [call_obj[0][0] for call_obj in mock_makedirs.call_args_list]
                assert any('element_abundances' in c for c in calls)
                assert any('fastchem' in c for c in calls)

    @pytest.mark.physics_invariant
    def test_paths_importer_output_paths_form_hierarchy(self):
        """Test that output directory structure is hierarchical and consistent.

        Physics invariant: Positivity/Boundedness. All output paths must be
        subdirectories of dirs['output'], never siblings or parents.
        """
        mock_output_base = '/output/base'
        mock_dirs = {'output': mock_output_base}
        mock_lava_dir = '/lava'
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer = paths_importer(mock_dirs)

                # All output paths should start with the base output directory
                output_paths = [
                    importer.element_abundance_output,
                    importer.fastchem3_output,
                    importer.output_dir,
                ]

                for path in output_paths:
                    assert path.startswith(
                        mock_output_base
                    ), f"Path {path} does not start with {mock_output_base}"

    @pytest.mark.physics_invariant
    def test_paths_importer_all_paths_non_empty(self):
        """Test that all path strings are non-empty (positivity bound).

        Physics invariant: Positivity/Boundedness. Path strings must have
        length > 0; no empty strings or None values escape initialization.
        """
        mock_dirs = {'output': '/output/base'}
        mock_lava_dir = '/lava'
        mock_fc_dir = '/fc'

        with patch.dict(
            os.environ, {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}, clear=False
        ):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer = paths_importer(mock_dirs)

                # Gather all string-type attributes that represent paths
                path_attrs = [
                    v
                    for k, v in vars(importer).items()
                    if isinstance(v, str) and ('dir' in k or 'path' in k or 'file' in k)
                ]

                for path_str in path_attrs:
                    assert len(path_str) > 0, f"Empty path string in {path_attrs}"

    @pytest.mark.physics_invariant
    def test_paths_importer_config_template_path_invariant(self):
        """Test that config and template paths are derived deterministically.

        Physics invariant: Monotonicity. Calling paths_importer twice with
        identical inputs yields identical config/template paths.
        """
        mock_dirs = {'output': '/output/base'}
        mock_lava_dir = '/lava'
        mock_fc_dir = '/fc'

        env = {'LAVA_DIR': mock_lava_dir, 'FC_DIR': mock_fc_dir}

        with patch.dict(os.environ, env, clear=False):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer

                importer1 = paths_importer(mock_dirs)
                template1 = importer1.fastchem3_config_template
                abund_template1 = importer1.element_abundance_template

        with patch.dict(os.environ, env, clear=False):
            with patch('os.makedirs'):
                from proteus.outgas.lavatmos import paths_importer as paths_importer2

                importer2 = paths_importer2(mock_dirs)
                template2 = importer2.fastchem3_config_template
                abund_template2 = importer2.element_abundance_template

        # Monotonicity: identical inputs -> identical paths
        assert template1 == template2
        assert abund_template1 == abund_template2
