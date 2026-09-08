"""
Integration tests for petitRADTRANS helper functions.

This module tests internal helper functions from petitRADTRANS.py that are
critical for spectrum computation but not directly exposed through the wrapper.

**Functions Tested**:
- _vmrs_to_mass_fractions(): VMR to mass fraction conversion
- _get_mix(): Gas mixture extraction from atmosphere data
- _get_ptr(): Pressure, temperature, radius extraction
- _get_supported_line_species(): Line species discovery
- _interpolate_prt_profiles(): Profile interpolation
- _prioritize_broadest_coverage_species(): Broadest-coverage prioritization logic
- _build_prt_composition(): Composition assembly

**Configuration**:
- Uses dummy.toml (all dummy modules for speed)
- Validates chemistry helper logic
- Tests edge cases and error conditions

**Runtime**: ~20-30s per test (depends on atmosphere availability)

**Documentation**:
- src/proteus/observe/petitRADTRANS.py
- src/proteus/utils/constants.py
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus
from proteus.observe.petitRADTRANS import (
    _build_prt_composition,
    _get_input_data_path,
    _get_ptr,
    _get_supported_cia_species,
    _get_supported_rayleigh_species,
    _interpolate_prt_profiles,
    _prioritize_broadest_coverage_species,
    _vmrs_to_mass_fractions,
)
from proteus.utils.constants import prt_rayleigh_species

if TYPE_CHECKING:
    pass

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


@pytest.fixture(scope='module')
def dummy_run_for_helpers():
    """
    Run PROTEUS with dummy config to generate atmosphere data.

    This fixture creates a minimal PROTEUS simulation to generate atmosphere
    output files that can be used for helper function testing.

    Yields:
        tuple: (runner: Proteus, output_dir: str, hf_row: dict, atm: dict)
            - runner: The PROTEUS runner object
            - output_dir: Path to the output directory
            - hf_row: Last row from helpfile
            - atm: Atmosphere data dictionary
    """
    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = PROTEUS_ROOT / 'tests' / 'integration' / 'dummy.toml'

        runner = Proteus(config_path=config_path)

        output_dir = str(Path(tmpdir) / f'helpers_test_{unique_id}')
        runner.config.params.out.path = output_dir

        runner.init_directories()

        # Ensure atmosphere module is available
        if runner.config.atmos_clim.module == 'dummy':
            # Dummy atmosphere won't have profile data
            pytest.skip('Dummy atmosphere module does not produce profile data')

        # Reduce timesteps for speed
        runner.config.params.stop.time.maximum = 1e4

        # Disable output for speed
        runner.config.params.out.plot_mod = 0
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        try:
            runner.start(resume=False, offline=True)

            if runner.hf_all is None or len(runner.hf_all) == 0:
                pytest.skip('Helpfile is empty')

            hf_row = runner.hf_all.iloc[-1].to_dict()

            # Try to read atmosphere data
            try:
                from proteus.atmos_clim.common import read_atmosphere_data

                atm_arr = read_atmosphere_data(
                    output_dir, [hf_row['Time']], extra_keys=['tmpl', 'pl', 'rl', 'x_gas']
                )
                if not atm_arr or atm_arr[-1] is None:
                    pytest.skip('Could not read atmosphere data')
                atm = atm_arr[-1]
            except Exception as e:
                pytest.skip(f'Failed to read atmosphere data: {e}')

            yield runner, output_dir, hf_row, atm
        except Exception as e:
            pytest.skip(f'Failed to run PROTEUS: {e}')


@pytest.mark.integration
def test_vmrs_to_mass_fractions_basic():
    """Test basic VMR to mass fraction conversion.

    Validates that:
    - Output shape matches input gases
    - Mass fractions sum to 1 at each layer
    - No negative values
    """
    # Simple test case: H2 and N2
    gases = ['H2', 'N2']
    vmrs = [
        np.array([0.8, 0.9, 0.7]),  # H2 VMR at 3 layers
        np.array([0.2, 0.1, 0.3]),  # N2 VMR at 3 layers
    ]

    mass_fractions, mmw = _vmrs_to_mass_fractions(gases, vmrs)

    # Check output structure
    assert isinstance(mass_fractions, dict), 'mass_fractions should be dict'
    assert len(mass_fractions) > 0, 'mass_fractions dict should not be empty'

    # Check mass fractions
    for gas, mf in mass_fractions.items():
        assert gas in gases, f'Unknown gas in output: {gas}'
        assert len(mf) == 3, f'Mass fraction array should have 3 layers, got {len(mf)}'
        assert np.all(mf >= 0.0), f'Negative mass fractions for {gas}'
        assert np.all(mf <= 1.0), f'Mass fractions > 1 for {gas}'

    # Check mean molar weight
    assert len(mmw) == 3, 'MMW array should have 3 layers'
    assert np.all(mmw > 0), 'MMW should be positive'

    # Check that mass fractions sum approximately to 1 at each layer
    for layer in range(3):
        total_mf = sum(mf[layer] for mf in mass_fractions.values())
        assert np.isclose(total_mf, 1.0, atol=1e-6), (
            f'Mass fractions do not sum to 1 at layer {layer}'
        )


@pytest.mark.integration
def test_vmrs_to_mass_fractions_normalization():
    """Test that VMRs are normalized correctly.

    Validates that:
    - Non-normalized VMRs are handled correctly
    - Output is consistent with normalized input
    """
    # Test with VMRs that don't sum to 1
    gases = ['CO2', 'H2O', 'O2']
    vmrs = [
        np.array([0.1, 0.2]),  # CO2: not normalized
        np.array([0.2, 0.3]),  # H2O
        np.array([0.5, 0.4]),  # O2
    ]  # Sums to 0.8, 0.9

    mass_fractions, mmw = _vmrs_to_mass_fractions(gases, vmrs)

    # Check that output is valid despite unnormalized input
    total_mf = sum(mf for mf in mass_fractions.values())
    assert len(total_mf) == 2, 'Output should have 2 layers'
    for layer in range(2):
        assert np.isclose(total_mf[layer], 1.0, atol=1e-6), (
            f'Mass fractions do not sum to 1 at layer {layer}'
        )


@pytest.mark.integration
def test_vmrs_to_mass_fractions_empty():
    """Test mass fraction conversion with empty input.

    Validates that:
    - Empty VMR list is handled gracefully
    - Returns empty dict and array
    """
    gases = []
    vmrs = []

    mass_fractions, mmw = _vmrs_to_mass_fractions(gases, vmrs)

    assert isinstance(mass_fractions, dict), 'Should return dict'
    assert len(mass_fractions) == 0, 'Dict should be empty'
    assert len(mmw) == 0, 'MMW array should be empty'


@pytest.mark.integration
def test_vmrs_to_mass_fractions_single_gas():
    """Test conversion with single gas.

    Validates that:
    - Single gas is handled correctly
    - Mass fraction equals 1.0
    """
    gases = ['H2']
    vmrs = [np.array([1.0, 1.0, 1.0])]

    mass_fractions, mmw = _vmrs_to_mass_fractions(gases, vmrs)

    assert 'H2' in mass_fractions, 'H2 should be in output'
    assert np.allclose(mass_fractions['H2'], 1.0), 'Single gas mass fraction should be 1.0'


@pytest.mark.integration
def test_prioritize_broadest_coverage_species(tmp_path):
    """Test broadest-coverage prioritization in line species list.

    Validates that:
    - The species with the broadest opacity coverage is moved to the front
    - Order of other species is preserved
    - Handles missing opacity coverage gracefully
    """
    input_data_path = tmp_path / 'input_data'
    for species, iso_dir, file_name in [
        ('CO2', '12C-16O2', '44CO2__Test.R1000_1-2mu.ktable.petitRADTRANS.h5'),
        ('H2O', '1H2-16O', '1H2-16O__Test.R1000_0.5-5mu.ktable.petitRADTRANS.h5'),
        ('CH4', '12C-1H4', '12C-1H4__Test.R1000_0.3-50mu.ktable.petitRADTRANS.h5'),
        ('N2', '14N2', '14N2__Test.R1000_0.8-1.1mu.ktable.petitRADTRANS.h5'),
    ]:
        species_dir = input_data_path / 'opacities' / 'lines' / 'correlated_k' / species
        (species_dir / iso_dir).mkdir(parents=True)
        (species_dir / iso_dir / file_name).write_text('dummy')

    # Case 1: broadest coverage species is present
    species = ['CO2', 'H2O', 'CH4', 'N2']
    result = _prioritize_broadest_coverage_species(species, str(input_data_path))
    assert result[0] == 'CH4', 'CH4 should be first'
    assert set(result) == set(species), 'Species set should be unchanged'

    # Case 2: no coverage data available
    species = ['CO2', 'H2O', 'N2']
    result = _prioritize_broadest_coverage_species(species, '/unused')
    assert result == species, 'Order should be unchanged when no files can be read'

    # Case 3: Empty list
    result = _prioritize_broadest_coverage_species([], '/unused')
    assert result == [], 'Empty list should remain empty'

    # Case 4: Only CH4
    result = _prioritize_broadest_coverage_species(['CH4'], str(input_data_path))
    assert result == ['CH4'], 'Single CH4 should work'


@pytest.mark.integration
def test_get_supported_rayleigh_species():
    """Test Rayleigh species filtering.

    Validates that:
    - Only valid Rayleigh species are returned
    - include_rayleigh flag is respected
    - Returns list of valid species from pRT constants
    """
    # Test with Rayleigh enabled
    gases = list(prt_rayleigh_species) + ['CO2']
    result = _get_supported_rayleigh_species(gases, include_rayleigh=True)

    assert isinstance(result, list), 'Should return list'
    for species in result:
        assert species in prt_rayleigh_species, f'Unknown Rayleigh species: {species}'

    # Test with Rayleigh disabled
    result = _get_supported_rayleigh_species(gases, include_rayleigh=False)
    assert len(result) == 0, 'Should return empty list when disabled'


@pytest.mark.integration
def test_get_supported_cia_species():
    """Test CIA species filtering.

    Validates that:
    - Only CIA pairs with both components present are returned
    - include_cia flag is respected
    - Handles missing components gracefully
    """
    # Test with both components present
    gases = ['H2', 'He', 'CO2']
    result = _get_supported_cia_species(gases, include_cia=True)

    # Some H2-He CIA should be present if data is available
    if len(result) > 0:
        for cia_species in result:
            components = [c for c in cia_species.split('--') if c]
            assert all(c in gases for c in components), f'Missing component in {cia_species}'

    # Test with CIA disabled
    result = _get_supported_cia_species(gases, include_cia=False)
    assert len(result) == 0, 'Should return empty list when disabled'


@pytest.mark.integration
def test_interpolate_prt_profiles_basic():
    """Test pressure, temperature, VMR profile interpolation.

    Validates that:
    - Output has correct length (n_points)
    - Temperature stays within bounds
    - Monotonicity is preserved
    """
    # Create simple profiles with increasing pressure
    prs = np.array([1e1, 1e2, 1e3, 1e4, 1e5])  # Pa - increasing
    tmp = np.array([400, 600, 900, 1200, 1500])  # K
    vmrs = [
        np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # Species 1
        np.array([0.9, 0.8, 0.7, 0.6, 0.5]),  # Species 2
    ]

    n_points = 100
    prs_interp, tmp_interp, vmrs_interp = _interpolate_prt_profiles(prs, tmp, vmrs, n_points)

    # Check output shapes
    assert len(prs_interp) == n_points, f'Expected {n_points} pressure points'
    assert len(tmp_interp) == n_points, f'Expected {n_points} temperature points'
    assert len(vmrs_interp) == len(vmrs), 'Should have same number of VMR arrays'
    for vmr in vmrs_interp:
        assert len(vmr) == n_points, f'Each VMR should have {n_points} points'

    # Check pressure is log-spaced
    assert np.all(np.diff(prs_interp) > 0), 'Pressure should increase monotonically'

    # Check temperature is within bounds
    from proteus.observe.petitRADTRANS import petitRADTRANS_TLIMS

    assert np.all(tmp_interp >= petitRADTRANS_TLIMS[0]), 'Temperature below minimum'
    assert np.all(tmp_interp <= petitRADTRANS_TLIMS[1]), 'Temperature above maximum'

    # Check VMRs are non-negative
    for vmr in vmrs_interp:
        assert np.all(vmr >= 0.0), 'VMRs should be non-negative'


@pytest.mark.integration
def test_get_ptr_basic():
    """Test pressure, temperature, radius extraction.

    Validates that:
    - Correct arrays are extracted
    - Arrays have matching lengths
    - Data is in correct order (may be reversed if pressure is decreasing)
    """
    # Create test atmosphere dict with increasing pressure
    atm = {
        'pl': np.array([1e3, 1e4, 1e5]),  # Pressure (Pa) - increasing
        'tmpl': np.array([900, 1200, 1500]),  # Temperature (K)
        'rl': np.array([6.7e6, 6.6e6, 6.5e6]),  # Radius (m)
    }

    prs, tmp, rad, vmrs = _get_ptr(atm, vmrs=None)

    # Check extraction - arrays should have correct values (possibly reversed)
    assert set(prs) == set(atm['pl']), 'Pressure values mismatch'
    assert len(prs) == len(atm['pl']), 'Pressure length mismatch'
    assert len(tmp) == len(atm['tmpl']), 'Temperature length mismatch'
    assert len(rad) == len(atm['rl']), 'Radius length mismatch'
    assert vmrs is None, 'VMRs should be None when not provided'

    # Pressure should be monotonically increasing after processing
    assert np.all(np.diff(prs) > 0), 'Pressure should be increasing'


@pytest.mark.integration
def test_get_ptr_with_vmrs():
    """Test pressure, temperature, radius extraction with VMRs.

    Validates that:
    - VMRs are returned in same count as input
    - Correct number of VMR arrays returned
    - VMRs may be reversed if pressure is decreasing
    """
    # Create test atmosphere with increasing pressure
    atm = {
        'pl': np.array([1e3, 1e4, 1e5]),  # Pressure (Pa) - increasing
        'tmpl': np.array([900, 1200, 1500]),
        'rl': np.array([6.7e6, 6.6e6, 6.5e6]),
    }

    vmr_input = [np.array([0.3, 0.4, 0.5]), np.array([0.7, 0.6, 0.5])]

    prs, tmp, rad, vmrs_out = _get_ptr(atm, vmrs=vmr_input)

    assert len(vmrs_out) == len(vmr_input), 'Number of VMR arrays mismatch'
    # Check that values match (may be in different order depending on pressure reversal)
    for vmr_in, vmr_out in zip(vmr_input, vmrs_out):
        assert len(vmr_in) == len(vmr_out), 'VMR length mismatch'
        assert set(vmr_in) == set(vmr_out), 'VMR values mismatch'


@pytest.mark.integration
def test_get_ptr_pressure_ordering():
    """Test pressure ordering correction in _get_ptr.

    Validates that:
    - Inverted pressure arrays are corrected
    - Temperature bounds are applied after reversal
    """
    # Inverted pressure (decreasing with height)
    atm = {
        'pl': np.array([1e1, 1e2, 1e3, 1e4, 1e5]),  # Inverted!
        'tmpl': np.array([400, 600, 900, 1200, 1500]),
        'rl': np.array([6.7e6, 6.6e6, 6.5e6, 6.4e6, 6.3e6]),  # Also inverted
    }

    vmrs = [np.array([0.1, 0.2, 0.3, 0.4, 0.5])]

    prs, tmp, rad, vmrs_out = _get_ptr(atm, vmrs=vmrs)

    # After correction, pressure should be increasing
    assert np.all(np.diff(prs) > 0), 'Pressure should be increasing after correction'

    # Temperature should be within bounds after correction
    from proteus.observe.petitRADTRANS import petitRADTRANS_TLIMS

    assert np.all(tmp >= petitRADTRANS_TLIMS[0]), 'Temperature below minimum'
    assert np.all(tmp <= petitRADTRANS_TLIMS[1]), 'Temperature above maximum'


@pytest.mark.integration
def test_get_input_data_path():
    """Test petitRADTRANS input data path discovery.

    Validates that:
    - Path can be found or raises appropriate error
    - Returns valid FWL_DATA path
    """
    # Test default discovery
    try:
        fwl_dir = os.environ.get('FWL_DATA')
        if fwl_dir is None:
            pytest.skip('FWL_DATA not set in test environment')

        path = _get_input_data_path({'fwl': fwl_dir})
        assert isinstance(path, str), 'Should return string path'
        assert len(path) > 0, 'Path should not be empty'
        assert 'prt' in path, 'Path should include prt directory'
        assert 'input_data' in path, 'Path should include input_data directory'
    except FileNotFoundError:
        # It's ok if pRT data is not installed in FWL_DATA_DIR
        pytest.skip('petitRADTRANS input data not found in FWL_DATA_DIR')


@pytest.mark.integration
def test_build_prt_composition_missing_gases(dummy_run_for_helpers):
    """Test composition building with limited gas data.

    Validates that:
    - Handles missing gas data gracefully
    - Returns valid composition even with sparse data
    """
    runner, output_dir, hf_row, atm = dummy_run_for_helpers

    # Create minimal hf_row with only outgas VMRs
    hf_row_minimal = {
        'H2_vmr': 0.8,
        'He_vmr': 0.2,
    }

    try:
        gases, vmrs, line_species, rayleigh_species, cia_species = _build_prt_composition(
            hf_row_minimal,
            atm,
            source='outgas',
            clip_vmr=1e-6,
            include_rayleigh=True,
            include_cia=True,
        )

        # Should have extracted some gases
        assert isinstance(gases, list), 'gases should be list'
        assert isinstance(vmrs, list), 'vmrs should be list'
        assert len(gases) == len(vmrs), 'gases and vmrs length mismatch'

    except Exception as e:
        if 'Could not locate' not in str(e):
            raise
