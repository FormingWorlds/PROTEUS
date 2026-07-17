"""
Integration tests for PROTEUS observation wrapper and synthesis modules.

This module tests the integration between wrapper.py (the observation orchestrator)
and petitRADTRANS.py (the spectral synthesis engine).

**Features Tested**:
- calc_synthetic_spectra(): Synthetic spectrum computation (transit + eclipse)
- run_observe(): Complete observation pipeline
- Spectrum file I/O and data validation
- Error handling for missing atmosphere data

**Configuration**:
- Uses dummy.toml (all dummy modules for speed)
- Runs 2-3 timesteps to generate atmosphere output
- Validates output spectra format and physical ranges

**Runtime**: ~30-60s (depends on atmosphere computation)

**Documentation**:
- src/proteus/observe/wrapper.py
- src/proteus/observe/petitRADTRANS.py
- src/proteus/observe/common.py
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from helpers import PROTEUS_ROOT

# petitRADTRANS is an optional component and a machine without it is supported, so the
# absence of the package skips this file with a reason that names it. Anything else that
# goes wrong here is a failure: the point of these tests is to exercise the real
# synthesis, and a run that quietly declines to do so would report success having checked
# nothing.
pytest.importorskip('petitRADTRANS', reason='petitRADTRANS not installed (observe extra)')

from proteus import Proteus  # noqa: E402
from proteus.observe.common import (  # noqa: E402
    OBS_SOURCES,
    get_eclipse_fpath,
    get_transit_fpath,
    read_eclipse,
    read_transit,
)
from proteus.observe.wrapper import calc_synthetic_spectra, run_observe  # noqa: E402
from proteus.utils.prt_data import opacities_present  # noqa: E402

if TYPE_CHECKING:
    pass

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _require_checked(count: int, what: str) -> None:
    """Skip when a test found nothing to check, rather than let it report success.

    The tests below walk the observation sources and check whatever spectra the run
    produced. An atmosphere module that writes no profile leaves every source with
    nothing, so the loop bodies never execute and the test would end green having
    verified nothing at all. Saying so as a skip keeps the difference between "checked
    and correct" and "had nothing to check" visible in the run report.
    """
    if count == 0:
        pytest.skip(f'No {what} was produced; the run wrote no atmosphere profile')


@pytest.fixture(scope='module')
def dummy_run_with_observe():
    """
    Run PROTEUS with dummy config and observation module enabled.

    This fixture creates a complete PROTEUS simulation with the observation
    module enabled to generate transit/eclipse spectra. The output can be
    used for all observation-related tests in this module.

    The opacity tables are the one other thing that may legitimately be absent, and their
    absence skips. A failure of the run itself is raised, not skipped: these tests exist
    to exercise the real synthesis, so a fixture that swallowed the failure would leave
    every test in the file passing while checking nothing.

    Yields:
        tuple: (runner: Proteus, output_dir: str, hf_row: dict)
            - runner: The PROTEUS runner object
            - output_dir: Path to the output directory
            - hf_row: The last row from the helpfile (used for spectrum calculation)
    """
    if not opacities_present():
        pytest.skip('petitRADTRANS opacity tables not installed; see proteus.utils.prt_data')

    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load configuration with observe module enabled
        config_path = PROTEUS_ROOT / 'tests' / 'integration' / 'dummy.toml'

        runner = Proteus(config_path=config_path)

        # Override output path
        output_dir = str(Path(tmpdir) / f'observe_test_{unique_id}')
        runner.config.params.out.path = output_dir

        # Re-initialize directories after changing output path
        runner.init_directories()

        # Ensure observe module is enabled
        if not hasattr(runner.config, 'observe') or runner.config.observe is None:
            pytest.skip('Observe module not configured')

        # Ensure observe module is set to petitRADTRANS
        runner.config.observe.module = 'petitRADTRANS'

        # Reduce timesteps for speed (just 2-3 for atmosphere generation)
        runner.config.params.stop.time.maximum = 1e4  # years

        # Disable plotting and archive for speed
        runner.config.params.out.plot_mod = 0
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        # Run simulation to generate atmosphere output. A failure here is a failure of
        # the thing under test and is left to propagate.
        runner.start(resume=False, offline=True)

        assert runner.hf_all is not None and len(runner.hf_all) > 0, (
            'PROTEUS produced no helpfile rows, so there is no state to synthesise from'
        )

        hf_row = runner.hf_all.iloc[-1].to_dict()

        yield runner, output_dir, hf_row


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_observe_wrapper_calc_synthetic_spectra(dummy_run_with_observe):
    """Test calc_synthetic_spectra wrapper function.

    Validates that:
    - calc_synthetic_spectra successfully computes spectra for each source
    - Transit and eclipse spectra files are created
    - Spectrum data has expected format and physical ranges
    - No exceptions are raised during calculation
    """
    runner, output_dir, hf_row = dummy_run_with_observe

    config = runner.config

    # Call the wrapper function
    calc_synthetic_spectra(hf_row, config, runner.directories)

    # Verify output files were created for available sources
    observe_dir = Path(output_dir) / 'observe'
    assert observe_dir.exists(), f'Observe directory not created: {observe_dir}'

    # Check that transit/eclipse files exist for at least one source
    checked = 0
    for source in OBS_SOURCES:
        # Skip profile source if using dummy atmosphere module
        if source == 'profile' and config.atmos_clim.module == 'dummy':
            continue
        # Skip offchem source if no chemistry module
        if source == 'offchem' and config.atmos_chem.module is None:
            continue

        transit_file = Path(get_transit_fpath(output_dir, source, 'synthesis'))
        eclipse_file = Path(get_eclipse_fpath(output_dir, source, 'synthesis'))

        # At least one should exist
        if transit_file.exists() or eclipse_file.exists():
            checked += 1
            # Validate transit spectrum if it exists
            if transit_file.exists():
                transit_data = read_transit(output_dir, source, 'synthesis')
                assert transit_data is not None, f'Transit spectrum is None: {source}'
                assert len(transit_data) > 0, f'Transit spectrum is empty: {source}'
                assert (
                    'Wavelength/um' in transit_data.columns or list(transit_data.columns)[0]
                ), f'Transit spectrum missing wavelength column: {source}'

                # Validate physical ranges
                wavelengths = transit_data.iloc[:, 0].values
                depths = transit_data.iloc[:, 1].values
                assert np.all(wavelengths > 0), f'Negative wavelengths in transit: {source}'
                assert np.all(depths >= 0), (
                    f'Negative transit depths found (should be ppm): {source}'
                )
                assert np.all(np.isfinite(wavelengths)), (
                    f'Non-finite wavelengths in transit: {source}'
                )
                assert np.all(np.isfinite(depths)), f'Non-finite depths in transit: {source}'

            # Validate eclipse spectrum if it exists
            if eclipse_file.exists():
                eclipse_data = read_eclipse(output_dir, source, 'synthesis')
                assert eclipse_data is not None, f'Eclipse spectrum is None: {source}'
                assert len(eclipse_data) > 0, f'Eclipse spectrum is empty: {source}'
                assert (
                    'Wavelength/um' in eclipse_data.columns or list(eclipse_data.columns)[0]
                ), f'Eclipse spectrum missing wavelength column: {source}'

                # Validate physical ranges
                wavelengths = eclipse_data.iloc[:, 0].values
                depths = eclipse_data.iloc[:, 1].values
                assert np.all(wavelengths > 0), f'Negative wavelengths in eclipse: {source}'
                assert np.all(depths >= 0), (
                    f'Negative eclipse depths found (should be ppm): {source}'
                )
                assert np.all(np.isfinite(wavelengths)), (
                    f'Non-finite wavelengths in eclipse: {source}'
                )
                assert np.all(np.isfinite(depths)), f'Non-finite depths in eclipse: {source}'

    _require_checked(checked, 'spectrum')


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_observe_wrapper_run_observe(dummy_run_with_observe):
    """Test run_observe wrapper function.

    Validates that:
    - run_observe successfully orchestrates the full observation pipeline
    - Spectra are computed without errors
    - Output files are created in the correct location
    """
    runner, output_dir, hf_row = dummy_run_with_observe

    config = runner.config

    # Call the run_observe wrapper
    run_observe(hf_row, config, runner.directories)

    # Verify observe directory was created
    observe_dir = Path(output_dir) / 'observe'
    assert observe_dir.exists(), f'Observe directory not created: {observe_dir}'

    # Verify at least some spectrum files were created
    spectrum_files = list(observe_dir.glob('*_synthesis.csv'))
    if len(spectrum_files) == 0:
        pytest.skip('No spectrum files created; atmosphere profile data not available')


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_transit_depth_spectrum_format(dummy_run_with_observe):
    """Test that transit depth spectra have correct format and units.

    Validates that:
    - Wavelength is in micrometers (μm)
    - Transit depth is in ppm
    - Wavelength increases monotonically
    - Data is physically reasonable
    """
    runner, output_dir, hf_row = dummy_run_with_observe

    config = runner.config

    # Calculate spectra
    from proteus.observe.petitRADTRANS import transit_depth

    checked = 0
    for source in OBS_SOURCES:
        # Skip unsupported sources
        if source == 'profile' and config.atmos_clim.module == 'dummy':
            continue
        if source == 'offchem' and config.atmos_chem.module is None:
            continue

        try:
            result = transit_depth(hf_row, config, source, runner.directories)
            if result is None:
                continue
            checked += 1

            # Validate return format: [wavelength, depth]
            assert isinstance(result, np.ndarray), (
                f'transit_depth should return ndarray, got {type(result)}: {source}'
            )
            assert result.shape[1] == 2, (
                f'Expected 2 columns (wl, depth), got {result.shape[1]}: {source}'
            )

            wavelengths = result[:, 0]
            depths = result[:, 1]

            # Physical validation
            assert np.all(wavelengths > 0), f'Wavelengths should be positive: {source}'
            assert np.all(np.diff(wavelengths) > 0), (
                f'Wavelengths should increase monotonically: {source}'
            )
            assert np.all(depths > 0), f'Transit depths should be positive: {source}'
            assert np.all(depths < 1e6), f'Transit depths (ppm) should be < 1e6: {source}'

        except Exception as e:
            # Some sources might not be available, which is ok
            if 'Could not read atmosphere data' not in str(e):
                raise

    _require_checked(checked, 'transit spectrum')


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_eclipse_depth_spectrum_format(dummy_run_with_observe):
    """Test that eclipse depth spectra have correct format and units.

    Validates that:
    - Wavelength is in micrometers (μm)
    - Eclipse depth is in ppm
    - Wavelength increases monotonically
    - Data is physically reasonable
    """
    runner, output_dir, hf_row = dummy_run_with_observe

    config = runner.config

    # Calculate spectra
    from proteus.observe.petitRADTRANS import eclipse_depth

    checked = 0
    for source in OBS_SOURCES:
        # Skip unsupported sources
        if source == 'profile' and config.atmos_clim.module == 'dummy':
            continue
        if source == 'offchem' and config.atmos_chem.module is None:
            continue

        try:
            result = eclipse_depth(hf_row, config, source, runner.directories)
            if result is None:
                continue
            checked += 1

            # Validate return format: [wavelength, depth]
            assert isinstance(result, np.ndarray), (
                f'eclipse_depth should return ndarray, got {type(result)}: {source}'
            )
            assert result.shape[1] == 2, (
                f'Expected 2 columns (wl, depth), got {result.shape[1]}: {source}'
            )

            wavelengths = result[:, 0]
            depths = result[:, 1]

            # Physical validation
            assert np.all(wavelengths > 0), f'Wavelengths should be positive: {source}'
            assert np.all(np.diff(wavelengths) > 0), (
                f'Wavelengths should increase monotonically: {source}'
            )
            assert np.all(depths > 0), f'Eclipse depths should be positive: {source}'
            assert np.all(depths < 1e6), f'Eclipse depths (ppm) should be < 1e6: {source}'

        except Exception as e:
            # Some sources might not be available, which is ok
            if 'Could not read atmosphere data' not in str(e):
                raise

    _require_checked(checked, 'eclipse spectrum')


@pytest.mark.integration
def test_observe_files_persist(dummy_run_with_observe):
    """Test that observe output files persist on disk.

    Validates that:
    - Files are written correctly to disk
    - Files can be read back
    - Data persists across reads
    """
    runner, output_dir, hf_row = dummy_run_with_observe

    config = runner.config

    # Generate spectra
    calc_synthetic_spectra(hf_row, config, runner.directories)

    # Read files multiple times, ensure data is consistent
    spectra_data = {}
    for source in OBS_SOURCES:
        if source == 'profile' and config.atmos_clim.module == 'dummy':
            continue
        if source == 'offchem' and config.atmos_chem.module is None:
            continue

        transit_file = Path(get_transit_fpath(output_dir, source, 'synthesis'))
        if transit_file.exists():
            # Read twice and compare
            data1 = read_transit(output_dir, source, 'synthesis')
            data2 = read_transit(output_dir, source, 'synthesis')
            assert data1.equals(data2), f'Data inconsistency on re-read: {source}'
            spectra_data[source] = data1

    _require_checked(len(spectra_data), 'transit spectrum')


@pytest.mark.integration
def test_wrapper_invalid_module(dummy_run_with_observe):
    """A misspelled synthesis module is refused by name rather than falling back to the
    only module that exists, and the refusal happens before any other config is read.

    The message has to carry the offending name, since that is what tells the reader
    which config line to correct. The second call supplies a config holding nothing but
    the module name: it must still fail on the module rather than on the attributes it
    lacks, which pins the dispatch as the first thing the function does. That ordering is
    what makes the error survive a config that is broken in more than one way, instead of
    surfacing whichever problem the reader happens to reach first.
    """
    runner, _, hf_row = dummy_run_with_observe

    # Use a lightweight object so we can test wrapper error handling without
    # attrs validators rejecting assignment to an invalid module name first.
    class _Obj:
        pass

    bad_config = _Obj()
    bad_config.observe = _Obj()
    bad_config.atmos_clim = _Obj()
    bad_config.atmos_chem = _Obj()
    bad_config.observe.module = 'invalid_module'
    bad_config.atmos_clim.module = runner.config.atmos_clim.module
    bad_config.atmos_chem.module = runner.config.atmos_chem.module

    with pytest.raises(ValueError, match='Unknown synthesis module') as excinfo:
        calc_synthetic_spectra(hf_row, bad_config, runner.directories)
    assert 'invalid_module' in str(excinfo.value)

    # A config carrying only the module name still fails on the module, so the dispatch
    # runs before the source and spectrum-type lookups touch the other sections.
    bare_config = _Obj()
    bare_config.observe = _Obj()
    bare_config.observe.module = 'invalid_module'
    with pytest.raises(ValueError, match='Unknown synthesis module'):
        calc_synthetic_spectra(hf_row, bare_config, runner.directories)


@pytest.mark.integration
def test_observe_directory_structure(dummy_run_with_observe):
    """Test that observe output directory structure is created correctly.

    Validates that:
    - observe/ subdirectory is created
    - All spectrum files follow naming convention
    - Directory is accessible and writable
    """
    runner, output_dir, hf_row = dummy_run_with_observe

    config = runner.config

    # Generate spectra
    calc_synthetic_spectra(hf_row, config, runner.directories)

    observe_dir = Path(output_dir) / 'observe'
    assert observe_dir.exists(), 'observe/ subdirectory not created'
    assert observe_dir.is_dir(), 'observe path is not a directory'

    # Check file naming convention
    transit_files = list(observe_dir.glob('transit_*_synthesis.csv'))
    eclipse_files = list(observe_dir.glob('eclipse_*_synthesis.csv'))

    # Should have at least one set of files
    if len(transit_files) + len(eclipse_files) == 0:
        pytest.skip('No spectrum files created; atmosphere profile data not available')

    # Validate file naming matches expected pattern
    for f in transit_files:
        source = f.stem.replace('transit_', '').replace('_synthesis', '')
        assert source in OBS_SOURCES, f'Unknown source in filename: {source}'

    for f in eclipse_files:
        source = f.stem.replace('eclipse_', '').replace('_synthesis', '')
        assert source in OBS_SOURCES, f'Unknown source in filename: {source}'


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_transit_eclipse_depth_ratio(dummy_run_with_observe):
    """Test physical relationship between transit and eclipse depths.

    For a given wavelength and composition, validates that transit and eclipse
    depths maintain physically reasonable relationships.
    """
    runner, output_dir, hf_row = dummy_run_with_observe

    config = runner.config

    from proteus.observe.petitRADTRANS import eclipse_depth, transit_depth

    # Test for 'outgas' source (most basic, should always be available)
    try:
        transit_result = transit_depth(hf_row, config, 'outgas', runner.directories)
        eclipse_result = eclipse_depth(hf_row, config, 'outgas', runner.directories)

        _require_checked(
            int(transit_result is not None and eclipse_result is not None),
            'transit and eclipse spectrum pair',
        )

        if transit_result is not None and eclipse_result is not None:
            transit_wl = transit_result[:, 0]
            transit_depth = transit_result[:, 1]
            eclipse_wl = eclipse_result[:, 0]
            eclipse_depth = eclipse_result[:, 1]

            # Both should span similar wavelength ranges
            assert transit_wl.min() <= eclipse_wl.min() + 0.1 * abs(eclipse_wl.min()), (
                'Transit and eclipse wavelength ranges differ significantly'
            )
            assert transit_wl.max() >= eclipse_wl.max() - 0.1 * abs(eclipse_wl.max()), (
                'Transit and eclipse wavelength ranges differ significantly'
            )

    except Exception as e:
        if 'Could not read atmosphere data' not in str(e):
            raise
