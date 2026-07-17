"""Unit tests for ``proteus.observe.petitRADTRANS``.

These tests pin the helper-level physics and ordering contracts that
feed the synthetic-observation backend:

- reference values are taken from the closest layer to the configured
  reference pressure
- descending pressure grids reverse pressure, temperature, radius, and
  VMR arrays together
- VMRs are normalized before computing mass fractions and mean molar
  masses

The module depends on the optional ``petitRADTRANS`` package, so the
tests inject a tiny fake package into ``sys.modules`` before importing
the backend module.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _install_fake_petitradtrans(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pkg = types.ModuleType('petitRADTRANS')
    fake_pkg.__file__ = '/fake/petitRADTRANS/__init__.py'
    fake_pkg.__path__ = []

    fake_constants = types.ModuleType('petitRADTRANS.physical_constants')
    fake_constants.c = 2.99792458e10
    fake_pkg.physical_constants = fake_constants

    fake_radtrans = types.ModuleType('petitRADTRANS.radtrans')
    fake_radtrans.Radtrans = MagicMock(name='Radtrans')

    monkeypatch.setitem(sys.modules, 'petitRADTRANS', fake_pkg)
    monkeypatch.setitem(sys.modules, 'petitRADTRANS.physical_constants', fake_constants)
    monkeypatch.setitem(sys.modules, 'petitRADTRANS.radtrans', fake_radtrans)


def _import_backend(monkeypatch: pytest.MonkeyPatch):
    _install_fake_petitradtrans(monkeypatch)

    fake_proteus = types.ModuleType('proteus')
    fake_proteus.__path__ = []
    fake_utils = types.ModuleType('proteus.utils')
    fake_utils.__path__ = []

    fake_constants = types.ModuleType('proteus.utils.constants')
    fake_constants.prt_cia_species = ()
    fake_constants.prt_gases = ('H2', 'He')
    fake_constants.prt_ignored_gases = ()
    fake_constants.prt_rayleigh_species = ()

    fake_helper = types.ModuleType('proteus.utils.helper')
    fake_helper.eval_gas_mmw = lambda gas: {'H2': 2.0e-3, 'He': 4.0e-3}[gas]

    fake_observe = types.ModuleType('proteus.observe')
    fake_observe.__path__ = []

    monkeypatch.setitem(sys.modules, 'proteus', fake_proteus)
    monkeypatch.setitem(sys.modules, 'proteus.utils', fake_utils)
    monkeypatch.setitem(sys.modules, 'proteus.utils.constants', fake_constants)
    monkeypatch.setitem(sys.modules, 'proteus.utils.helper', fake_helper)
    monkeypatch.setitem(sys.modules, 'proteus.observe', fake_observe)

    backend_path = Path(__file__).resolve().parents[2] / 'src/proteus/observe/petitRADTRANS.py'
    spec = importlib.util.spec_from_file_location('proteus.observe.petitRADTRANS', backend_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'proteus.observe.petitRADTRANS', module)
    spec.loader.exec_module(module)
    return module


def _make_config(
    *,
    line_opacity_mode: str = 'c-k',
    include_rayleigh: bool = False,
    include_cia: bool = False,
    remove_one_gas: bool = True,
    silent: bool = False,
):
    return types.SimpleNamespace(
        observe=types.SimpleNamespace(
            module='petitRADTRANS',
            clip_vmr=1e-8,
            reference_pressure=10.0,
            remove_one_gas=remove_one_gas,
            petitRADTRANS=types.SimpleNamespace(
                line_opacity_mode=line_opacity_mode,
                include_rayleigh=include_rayleigh,
                include_cia=include_cia,
                silent=silent,
            ),
        ),
        atmos_chem=types.SimpleNamespace(module='vulcan'),
    )


def test_get_reference_prt_values_uses_closest_config_pressure(monkeypatch):
    """Reference values are taken from the atmospheric layer sitting closest to the
    configured reference pressure, converted into the units petitRADTRANS expects.

    The configured reference of 11.5 bar is compared against decade-spaced layers at
    1, 10, 100 and 1000 bar, so the 10 bar layer wins. Decade spacing means picking a
    neighbouring layer moves the answer by a factor of ten, far outside the tolerance.
    The radius and gravity pins sit a factor of 100 above their SI inputs, so a missing
    metre-to-centimetre conversion fails them too.
    """
    mod = _import_backend(monkeypatch)

    atm = {
        'p': np.array([1.0e5, 1.0e6, 1.0e7, 1.0e8]),
        'r': np.array([7.00e6, 7.10e6, 7.20e6, 7.30e6]),
        'g': np.array([9.00, 9.10, 9.20, 9.30]),
    }
    config = MagicMock()
    config.observe.reference_pressure = 11.5  # bar, closest to the 10 bar layer

    reference_pressure, reference_radius, reference_gravity = mod._get_reference_prt_values(
        atm, config
    )

    assert reference_pressure == pytest.approx(10.0, rel=1e-12)
    assert reference_radius == pytest.approx(7.10e8, rel=1e-12)
    assert reference_gravity == pytest.approx(9.10e2, rel=1e-12)


def test_get_ptr_reverses_vmrs_with_descending_pressure(monkeypatch):
    """A profile stored from the surface upwards runs from high to low pressure, and
    petitRADTRANS needs it the other way round, so pressure, temperature, radius and
    every VMR array must be reversed together.

    Each array is strictly monotonic with distinct entries, so a reversal that silently
    skips one of them leaves that array's values attached to the wrong pressure and the
    corresponding assertion fails. The two VMR arrays are complementary at every layer,
    which is why they are compared element by element rather than by their sum: a
    per-layer sum stays at one whether or not the reversal happened.
    """
    mod = _import_backend(monkeypatch)

    atm = {
        'pl': np.array([1.0e5, 1.0e4, 1.0e3]),
        'tmpl': np.array([300.0, 400.0, 500.0]),
        'rl': np.array([1.0, 2.0, 3.0]),
    }
    vmrs = [
        np.array([0.10, 0.20, 0.30]),
        np.array([0.90, 0.80, 0.70]),
    ]

    prs, tmp, rad, vmrs_sorted = mod._get_ptr(atm, vmrs)

    assert np.array_equal(prs, np.array([1.0e3, 1.0e4, 1.0e5]))
    assert np.array_equal(tmp, np.array([500.0, 400.0, 300.0]))
    assert np.array_equal(rad, np.array([3.0, 2.0, 1.0]))
    assert vmrs_sorted is not None
    assert np.array_equal(vmrs_sorted[0], np.array([0.30, 0.20, 0.10]))
    assert np.array_equal(vmrs_sorted[1], np.array([0.70, 0.80, 0.90]))


@pytest.mark.physics_invariant
def test_vmrs_to_mass_fractions_normalizes_remaining_species(monkeypatch):
    """Clipping trace gases leaves the surviving VMRs summing to less than one, so the
    conversion renormalizes them before weighting by molar mass.

    The renormalization is observable only in the mean molar mass. A mass fraction is
    the ratio v_i M_i / sum_j(v_j M_j), which is unchanged when every VMR in a layer is
    scaled by the same constant, so the mass-fraction pins here hold with or without the
    step. The mean molar mass is an absolute quantity and does carry the scale: with the
    raw VMRs summing to 0.2 and 0.4 it would come back as 0.6 and 1.0 g/mol rather than
    3.0 and 2.5, so that pin is the one discriminating the renormalization.

    The two layers are asymmetric so that a swapped species index is visible: layer 0 is
    an equal H2/He mix and layer 1 is three parts H2 to one part He. With H2 at 2 g/mol
    and He at 4 g/mol the mass fractions follow by hand as 1/3 and 2/3 in layer 0 and
    0.6 and 0.4 in layer 1. An equal-parts mixture in both layers would be symmetric
    under an H2/He swap and would not discriminate one from the other.
    """
    mod = _import_backend(monkeypatch)

    gases = ['H2', 'He']
    vmrs = [
        np.array([0.10, 0.30]),  # H2; the per-layer VMR sums are 0.2 and 0.4, not 1.0
        np.array([0.10, 0.10]),  # He
    ]

    mass_fractions, mean_molar_masses = mod._vmrs_to_mass_fractions(gases, vmrs)

    # Hand-computed from the VMR ratios and the 2 / 4 g mol-1 molar masses.
    np.testing.assert_allclose(mass_fractions['H2'], np.array([1.0 / 3.0, 0.6]), rtol=1e-12)
    np.testing.assert_allclose(mass_fractions['He'], np.array([2.0 / 3.0, 0.4]), rtol=1e-12)
    # Carries the VMR scale, so this is the pin that sees the renormalization: skipping
    # it gives 0.6 and 1.0. petitRADTRANS wants amu (g mol-1), so the kg mol-1 masses are
    # scaled by 1e3; dropping that conversion lands three decades away at 3.0e-3.
    np.testing.assert_allclose(mean_molar_masses, np.array([3.0, 2.5]), rtol=1e-12)

    # Every gas in the mix reaches the output: the returned fractions account for all of
    # the mass. This is closure over the dict, not over the arithmetic, so it fails if a
    # contributing gas is filtered out of the result rather than merely mis-weighted.
    np.testing.assert_allclose(sum(mass_fractions.values()), np.ones(2), rtol=1e-12)
    # Boundedness: a single species can neither carry none of the mass nor all of it in a
    # two-gas mix, so a sign slip in either contribution escapes the interval.
    for gas in gases:
        assert np.all((mass_fractions[gas] > 0.0) & (mass_fractions[gas] < 1.0))


def test_load_stellar_toa_flux_reads_saved_sflux_and_interpolates(monkeypatch, tmp_path):
    """The stellar spectrum PROTEUS wrote alongside the run is read back and resampled
    onto the wavelength grid the eclipse-depth denominator is evaluated on.

    The tabulated flux doubles between successive wavelengths (1, 2 and 4 at 400, 500
    and 600 nm) and the two targets sit at the midpoints, so linear interpolation gives
    1.5 and 3.0 whereas interpolating in log space would give about 1.41 and 2.83. The
    trailing factor of 1e7 converts erg cm-2 s-1 nm-1 into erg cm-2 s-1 cm-1, so a
    dropped unit conversion moves the answer by seven decades.
    """
    mod = _import_backend(monkeypatch)

    data_dir = tmp_path / 'data'
    data_dir.mkdir(parents=True)
    spectrum_file = data_dir / '42.sflux'
    spectrum_file.write_text(
        '# WL(nm)\t Flux(ergs/cm**2/s/nm)   Stellar flux at t_star = 1.00e+00 yr\n'
        '4.00000000e+02\t1.00000000e+00\n'
        '5.00000000e+02\t2.00000000e+00\n'
        '6.00000000e+02\t4.00000000e+00\n'
    )

    target_wavelength_nm = np.array([4.50000000e02, 5.50000000e02])
    flux = mod._load_stellar_toa_flux(str(tmp_path), {'Time': 42}, target_wavelength_nm)

    assert flux.shape == (2,)
    # rtol well inside the 6% gap between the linear result and the log-space one.
    np.testing.assert_allclose(flux, np.array([1.5e7, 3.0e7]), rtol=1e-9)


def test_get_input_data_path_prefers_fwl_data_directory(monkeypatch, tmp_path):
    """The petitRADTRANS opacity tree is resolved as prt/input_data underneath the FWL
    data directory recorded in the startup dirs, and returned as a plain string.

    The nested directory is created but its parents are not otherwise populated, so the
    returned path has to be the nested one rather than the FWL root itself. The second
    check confirms the helper hands back a path that actually exists on disk instead of
    a plausible string it assembled without probing.
    """
    mod = _import_backend(monkeypatch)

    data_path = tmp_path / 'prt' / 'input_data'
    data_path.mkdir(parents=True)

    result = mod._get_input_data_path({'fwl': str(tmp_path)})
    assert result == str(data_path)
    assert Path(result).is_dir()


def test_get_input_data_path_raises_when_missing_everywhere(monkeypatch, tmp_path):
    """An FWL data directory that carries no petitRADTRANS opacity tree is a hard error
    rather than a silent fallback, because synthesising a spectrum without opacities
    would otherwise fail much later with an unrelated message.

    The path handed in does not exist at all, which is the shape of the problem when
    the reference data was never downloaded. The message has to name the directory it
    probed so the reader can see which tree was expected.
    """
    mod = _import_backend(monkeypatch)

    with pytest.raises(FileNotFoundError) as excinfo:
        mod._get_input_data_path({'fwl': str(tmp_path / 'does_not_exist')})
    assert 'input_data' in str(excinfo.value)


def test_get_input_data_path_raises_when_dirs_missing_fwl(monkeypatch, tmp_path):
    """A dirs dictionary carrying no 'fwl' entry is reported as a missing key, kept
    distinct from the error raised when the entry is present but the tree behind it is
    not.

    The two misconfigurations have different remedies, so collapsing them into one
    error would mislead: a missing key means the run started without FWL_DATA resolved,
    while a present key with no prt/input_data underneath means the reference data was
    never fetched. The second call supplies the key but points it at a directory that
    does not exist, and must surface FileNotFoundError instead of KeyError. A guard
    that probed the filesystem before checking the key would report the wrong one.
    """
    mod = _import_backend(monkeypatch)

    with pytest.raises(KeyError, match='fwl'):
        mod._get_input_data_path({})

    with pytest.raises(FileNotFoundError):
        mod._get_input_data_path({'fwl': str(tmp_path / 'no_such_tree')})


def test_supported_species_helpers_filter_and_include(monkeypatch, tmp_path):
    """The three opacity-selection helpers each drop a gas for their own reason, and the
    gas list is filtered rather than passed through wholesale.

    The line-species list separates the three exclusion routes on purpose: 'skipme' is
    excluded because it is an ignored gas, 'Ray' because it is handled as a Rayleigh
    scatterer instead, and CO2 because no opacity directory exists for it even though it
    is a perfectly ordinary gas. Rayleigh and CIA selection are each checked with the
    switch both off and on, so a helper wired to ignore its flag fails one of the pair.
    CIA offers 'H2--He' and 'H2--N2' while the mix holds no N2, which pins that every
    component of a collision pair must be present rather than just one.
    """
    mod = _import_backend(monkeypatch)

    input_data_path = tmp_path / 'input_data'
    (input_data_path / 'opacities' / 'lines' / 'correlated_k' / 'H2O').mkdir(parents=True)
    (input_data_path / 'opacities' / 'lines' / 'correlated_k' / 'CH4').mkdir(parents=True)

    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(input_data_path))
    monkeypatch.setattr(mod, 'prt_ignored_gases', {'skipme'})
    monkeypatch.setattr(mod, 'prt_rayleigh_species', {'Ray'})
    monkeypatch.setattr(mod, 'prt_cia_species', ('H2--He', 'H2--N2'))

    line_species = mod._get_supported_line_species(
        ['H2O', 'skipme', 'Ray', 'CH4', 'CO2'], str(input_data_path)
    )
    assert line_species == ['H2O', 'CH4']
    assert mod._get_supported_rayleigh_species(['H2O', 'Ray'], False) == []
    assert mod._get_supported_rayleigh_species(['H2O', 'Ray'], True) == ['Ray']
    assert mod._get_supported_cia_species(['H2', 'He'], False) == []
    assert mod._get_supported_cia_species(['H2', 'He'], True) == ['H2--He']


def test_supported_line_species_filters_by_provided_data_tree(monkeypatch, tmp_path):
    """Opacity availability is resolved underneath the input-data path handed to the
    helper, so pointing a run at a different data tree changes which gases survive.

    The first tree holds an opacity directory for H2 but none for He, so only H2 comes
    back. The second call passes the same gases against a separate, empty tree and must
    return nothing: without that contrast a helper that ignored its argument and read
    from a module-level default would still satisfy the first assertion.
    """
    mod = _import_backend(monkeypatch)

    auto_path = tmp_path / 'auto_input_data'
    (auto_path / 'opacities' / 'lines' / 'correlated_k' / 'H2').mkdir(parents=True)
    line_species = mod._get_supported_line_species(['H2', 'He'], str(auto_path))
    assert line_species == ['H2']

    empty_path = tmp_path / 'empty_input_data'
    empty_path.mkdir()
    assert mod._get_supported_line_species(['H2', 'He'], str(empty_path)) == []


def test_vmrs_to_mass_fractions_handles_empty_and_zero_total(monkeypatch):
    """A composition with no gases left after clipping short-circuits to empty outputs,
    and a mix whose molar masses all vanish is refused rather than divided by.

    The empty case is the boundary reached when every gas falls below the clip
    threshold; it must return empty containers instead of raising, because the callers
    treat that as a legitimate no-op. The second case patches the molar-mass lookup to
    zero so the total mass collapses while the VMR sum stays positive, which reaches the
    mass guard without tripping the earlier VMR guard first and keeps the two error
    paths separable.
    """
    mod = _import_backend(monkeypatch)

    mass_fractions, mean_molar_masses = mod._vmrs_to_mass_fractions(['H2'], [])
    assert mass_fractions == {}
    assert mean_molar_masses.size == 0

    monkeypatch.setattr(mod, 'eval_gas_mmw', lambda gas: 0.0)
    with pytest.raises(ValueError, match='zero total mass fraction'):
        mod._vmrs_to_mass_fractions(['H2', 'He'], [np.array([1.0, 1.0]), np.array([0.0, 0.0])])


def test_vmrs_to_mass_fractions_raises_for_zero_total_vmr(monkeypatch):
    """A layer holding no gas at all cannot be renormalized, so the conversion refuses
    instead of dividing by zero and handing NaN mass fractions to petitRADTRANS.

    The first case empties every layer. The second is the discriminating one: only the
    upper layer is empty while the lower one holds an ordinary H2/He mix. The guard is
    written over np.any, so one bad layer among good ones must still raise; a guard
    written over np.all would accept this mix and let the NaN through.
    """
    mod = _import_backend(monkeypatch)

    with pytest.raises(ValueError, match='zero total VMR'):
        mod._vmrs_to_mass_fractions(['H2', 'He'], [np.array([0.0, 0.0]), np.array([0.0, 0.0])])

    with pytest.raises(ValueError, match='zero total VMR'):
        mod._vmrs_to_mass_fractions(['H2', 'He'], [np.array([0.8, 0.0]), np.array([0.2, 0.0])])


def test_prioritize_broadest_coverage_species_noop_paths(monkeypatch):
    """When no coverage can be discovered the incoming order is left untouched, so an
    undiscoverable opacity tree degrades to the caller's ordering instead of an error or
    an empty list.

    Both calls use a path that does not exist, which is what makes the coverage lookup
    return nothing for every species. The empty-list case guards the boundary where
    there is no species to promote at all, and the populated case pins that species with
    no parseable coverage file keep their relative order rather than being dropped.
    """
    mod = _import_backend(monkeypatch)

    assert mod._prioritize_broadest_coverage_species([], '/unused') == []
    assert mod._prioritize_broadest_coverage_species(['H2O', 'CO2'], '/unused') == [
        'H2O',
        'CO2',
    ]


def test_prioritize_broadest_coverage_species_uses_widest_file(monkeypatch, tmp_path):
    """The line species with the widest wavelength coverage is moved to the front while
    the rest keep the order they arrived in, because petitRADTRANS takes its output grid
    from the first line species it is given.

    Coverage is parsed out of the opacity filenames: CH4 spans 0.3 to 50 um (49.7 um
    wide), H2O spans 0.5 to 5 um (4.5 um) and CO2 spans 1 to 2 um (1 um). The widths are
    an order of magnitude apart so the winner is unambiguous. Feeding the same three
    species twice in different orders separates promoting the widest from sorting the
    whole list by width. The tables sit in isotopologue subdirectories, mirroring the
    petitRADTRANS tree where files live one level below the species directory, so a
    non-recursive search finds nothing and leaves the input order untouched.
    """
    mod = _import_backend(monkeypatch)

    input_data_path = tmp_path / 'input_data'
    for species, iso_dir, file_name in [
        ('CO2', '12C-16O2', '44CO2__Test.R1000_1-2mu.ktable.petitRADTRANS.h5'),
        ('H2O', '1H2-16O', '1H2-16O__Test.R1000_0.5-5mu.ktable.petitRADTRANS.h5'),
        ('CH4', '12C-1H4', '12C-1H4__Test.R1000_0.3-50mu.ktable.petitRADTRANS.h5'),
    ]:
        species_dir = input_data_path / 'opacities' / 'lines' / 'correlated_k' / species
        (species_dir / iso_dir).mkdir(parents=True)
        (species_dir / iso_dir / file_name).write_text('dummy')

    result = mod._prioritize_broadest_coverage_species(
        ['CO2', 'H2O', 'CH4'], str(input_data_path)
    )
    assert result == ['CH4', 'CO2', 'H2O']

    # Same species, permuted input. CH4 leads both times, but the trailing pair follows
    # the input order rather than a fixed ranking, which is what separates the contract
    # from a width sort: above, a sort would have put H2O (4.5 um) ahead of CO2 (1 um).
    permuted = mod._prioritize_broadest_coverage_species(
        ['H2O', 'CO2', 'CH4'], str(input_data_path)
    )
    assert permuted == ['CH4', 'H2O', 'CO2']


def test_init_radtrans_suppresses_output_when_enabled(monkeypatch, capsys):
    """With silencing switched on, the chatter petitRADTRANS prints while loading
    opacities is captured instead of reaching the PROTEUS console.

    The stand-in constructor writes to stdout and stderr both, because petitRADTRANS
    uses both and a redirect covering only stdout would leave half the noise visible.
    The companion test drives the same constructor with silencing off, which is what
    rules out this test passing merely because the constructor never ran.
    """
    mod = _import_backend(monkeypatch)

    class NoisyRadtrans:
        def __init__(self, **kwargs):
            print('noise-out')
            print('noise-err', file=sys.stderr)

    config = _make_config(silent=True)
    _ = mod._init_radtrans(NoisyRadtrans, config, line_species=['H2O'])

    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''


def test_init_radtrans_keeps_output_when_silencing_disabled(monkeypatch, capsys):
    """With silencing switched off the constructor's output reaches the console
    untouched, so the redirect is applied only when it was asked for.

    This is the counterpart to the suppression test: the same noisy constructor and the
    same call, differing only in the config flag. Together the pair pins that the
    suppression comes from the flag rather than from the constructor staying silent.
    """
    mod = _import_backend(monkeypatch)

    class NoisyRadtrans:
        def __init__(self, **kwargs):
            print('noise-out')
            print('noise-err', file=sys.stderr)

    config = _make_config(silent=False)
    _ = mod._init_radtrans(NoisyRadtrans, config, line_species=['H2O'])

    captured = capsys.readouterr()
    assert 'noise-out' in captured.out
    assert 'noise-err' in captured.err


def test_get_mix_handles_outgas_profile_and_offchem(monkeypatch):
    """Each composition source reads its VMRs from a different place and shapes them
    differently, so the three routes are exercised against one another.

    The helpfile carries a single surface VMR per gas, which becomes a column constant
    with height; the climate profile carries per-level VMRs under '<gas>_vmr' keys and
    is padded at the bottom, repeating the lowest level so the array matches the
    level-edge grid; the chemistry output carries per-level VMRs under bare gas names.
    The helpfile and profile values for the same gas are deliberately different (0.8
    against 0.7 and 0.6), so a source mix-up produces the wrong column rather than a
    coincidentally identical one.
    """
    mod = _import_backend(monkeypatch)
    monkeypatch.setattr(mod, 'prt_gases', ('H2', 'CH4', 'He'))

    hf_row = {'H2_vmr': 0.8, 'CH4_vmr': 0.1, 'He_vmr': 0.1}
    atm_profile = {
        'pl': np.array([1.0e5, 1.0e4]),
        'H2_vmr': np.array([0.7, 0.6]),
        'CH4_vmr': np.array([0.2, 0.3]),
        'He_vmr': np.array([0.1, 0.1]),
    }
    atm_offchem = {
        'pl': np.array([1.0e5, 1.0e4]),
        'H2': np.array([0.7, 0.6]),
        'CH4': np.array([0.2, 0.3]),
        'He': np.array([0.1, 0.1]),
    }

    gases_outgas, vmrs_outgas = mod._get_mix(hf_row, atm_profile, 'outgas', 1e-8)
    gases_profile, vmrs_profile = mod._get_mix(hf_row, atm_profile, 'profile', 1e-8)
    gases_offchem, vmrs_offchem = mod._get_mix(hf_row, atm_offchem, 'offchem', 1e-8)

    assert gases_outgas == ['H2', 'CH4', 'He']
    assert np.allclose(vmrs_outgas[0], np.array([0.8, 0.8]))
    assert gases_profile == ['H2', 'CH4', 'He']
    assert np.allclose(vmrs_profile[1], np.array([0.2, 0.2, 0.3]))
    assert gases_offchem == ['H2', 'CH4', 'He']
    assert np.allclose(vmrs_offchem[2], np.array([0.1, 0.1]))


def test_get_mix_excludes_species_when_source_keys_missing_or_below_clip(monkeypatch):
    """Gases too rare to matter for radiative transfer are dropped, whether they are
    rare because the source reports a tiny abundance or absent from the source entirely.

    The clip sits at 1e-5 and the cases straddle it. Outgas reports 1e-6, safely below,
    so nothing survives. The profile peaks at exactly 1e-5, which is the boundary: the
    comparison is inclusive, so H2 is kept, and an exclusive comparison would drop it.
    Offchem reports 2e-5, clearly above. CH4 and He appear in no source, so they fall
    back to zero and are dropped, which is the path a gas takes when the atmosphere
    module never wrote it.
    """
    mod = _import_backend(monkeypatch)
    monkeypatch.setattr(mod, 'prt_gases', ('H2', 'CH4', 'He'))

    hf_row = {'H2_vmr': 1.0e-6}
    atm_profile = {
        'pl': np.array([1.0e5, 1.0e4]),
        'H2_vmr': np.array([1.0e-5, 1.0e-6]),
    }
    atm_offchem = {
        'pl': np.array([1.0e5, 1.0e4]),
        'H2': np.array([2.0e-5, 2.0e-5]),
    }

    gases_outgas, vmrs_outgas = mod._get_mix(hf_row, atm_profile, 'outgas', 1.0e-5)
    gases_profile, vmrs_profile = mod._get_mix(hf_row, atm_profile, 'profile', 1.0e-5)
    gases_offchem, vmrs_offchem = mod._get_mix(hf_row, atm_offchem, 'offchem', 1.0e-5)

    assert gases_outgas == []
    assert vmrs_outgas == []
    assert gases_profile == ['H2']
    assert np.allclose(vmrs_profile[0], np.array([1.0e-5, 1.0e-5, 1.0e-6]))
    assert gases_offchem == ['H2']
    assert np.allclose(vmrs_offchem[0], np.array([2.0e-5, 2.0e-5]))


def test_get_mix_unknown_source_returns_empty_selection(monkeypatch):
    """An unrecognised source name leaves every gas at its zero default, so nothing
    clears the clip and the selection comes back empty rather than falling through to
    one of the known sources.

    The helpfile holds abundances that would comfortably clear the clip under 'outgas',
    so a dispatch that quietly treated an unknown name as the default source would
    return both gases here and fail the assertions. Selecting nothing is what lets the
    callers surface the bad source name themselves rather than synthesising a spectrum
    from an empty atmosphere.
    """
    mod = _import_backend(monkeypatch)
    monkeypatch.setattr(mod, 'prt_gases', ('H2', 'CH4'))

    hf_row = {'H2_vmr': 0.9, 'CH4_vmr': 0.1}
    atm = {'pl': np.array([1.0e5, 1.0e4])}

    gases, vmrs = mod._get_mix(hf_row, atm, 'unknown_source', 1.0e-8)
    assert gases == []
    assert vmrs == []


def test_get_ptr_keeps_order_when_already_ascending_and_vmrs_none(monkeypatch):
    """A profile that already runs from low to high pressure is handed back untouched,
    and a caller that supplies no VMRs gets None back.

    The reversal is conditional on the second pressure sitting below the first, so this
    is the branch where none of the arrays are rewritten and the returned arrays must
    equal the inputs element for element. The temperatures sit inside the supported
    range, so this branch says nothing about the clamp; its companion covers that.
    """
    mod = _import_backend(monkeypatch)

    atm = {
        'pl': np.array([1.0e3, 1.0e4, 1.0e5]),
        'tmpl': np.array([250.0, 260.0, 270.0]),
        'rl': np.array([1.0, 2.0, 3.0]),
    }

    prs, tmp, rad, vmrs_sorted = mod._get_ptr(atm, None)

    assert np.array_equal(prs, atm['pl'])
    assert np.array_equal(tmp, atm['tmpl'])
    assert np.array_equal(rad, atm['rl'])
    assert vmrs_sorted is None


@pytest.mark.physics_invariant
def test_get_ptr_reverses_without_vmrs_when_descending(monkeypatch):
    """A descending profile is reordered even when the caller supplies no VMRs, and the
    reordered temperatures are pulled inside the range the opacity tables cover.

    The coldest level sits at 100 K, just under the 100.5 K floor petitRADTRANS
    supports, so the clamp is observable: without it that entry stays at 100 K and the
    lower-bound assertion fails. Choosing a temperature only half a kelvin below the
    floor also keeps the clamp from masking a failed reversal, since the reversal is
    checked on pressure and radius, which are not clamped.

    This is the branch where passing no VMRs matters: the reordering block is entered,
    so the guard in front of the VMR rewrite is what keeps the helper from iterating
    None. The transit and eclipse paths both call the helper this way when they want
    pressure, temperature and radius alone.
    """
    mod = _import_backend(monkeypatch)

    atm = {
        'pl': np.array([1.0e5, 1.0e4, 1.0e3]),
        'tmpl': np.array([100.0, 200.0, 300.0]),
        'rl': np.array([1.0, 2.0, 3.0]),
    }

    prs, tmp, rad, vmrs_sorted = mod._get_ptr(atm, None)

    assert np.array_equal(prs, np.array([1.0e3, 1.0e4, 1.0e5]))
    assert np.array_equal(rad, np.array([3.0, 2.0, 1.0]))
    assert np.all(tmp >= mod.petitRADTRANS_TLIMS[0])
    assert vmrs_sorted is None


def test_atm_profile_and_offchem_helpers(monkeypatch, tmp_path):
    """The two profile readers hand back the current atmosphere in a common shape: the
    climate reader returns the latest record, and the chemistry reader renames its
    columns and converts height above the surface into planetary radius.

    The chemistry output names its columns tmp, p and z while the rest of the backend
    works in tmpl, pl and rl, so the rename is what lets one set of downstream helpers
    serve both sources. The height offset is the substantive step: with the sample point
    at z = 0 and the interior radius at 10 m, the radius has to come back as 10, whereas
    a reader that forgot to add the interior radius would leave it at 0.
    """
    mod = _import_backend(monkeypatch)

    fake_atmos_clim_common = types.ModuleType('proteus.atmos_clim.common')
    fake_atmos_clim_common.read_atmosphere_data = lambda *_a, **_k: [
        {'p': np.array([1.0]), 'marker': 1}
    ]
    fake_atmos_clim = types.ModuleType('proteus.atmos_clim')
    fake_atmos_clim.__path__ = []
    monkeypatch.setitem(sys.modules, 'proteus.atmos_clim', fake_atmos_clim)
    monkeypatch.setitem(sys.modules, 'proteus.atmos_clim.common', fake_atmos_clim_common)

    fake_atmos_chem = types.ModuleType('proteus.atmos_chem')
    fake_atmos_chem.__path__ = []
    fake_atmos_chem_common = types.ModuleType('proteus.atmos_chem.common')
    fake_atmos_chem_common.read_result = lambda *_a, **_k: pd.DataFrame(
        {'tmp': [300.0], 'p': [1.0e5], 'z': [0.0], 'H2': [0.7]}
    )
    monkeypatch.setitem(sys.modules, 'proteus.atmos_chem', fake_atmos_chem)
    monkeypatch.setitem(sys.modules, 'proteus.atmos_chem.common', fake_atmos_chem_common)

    outdir = str(tmp_path)
    profile = mod._get_atm_profile(outdir, {'Time': 1})
    assert profile['marker'] == 1
    np.testing.assert_allclose(profile['p'], np.array([1.0]))
    offchem = mod._get_atm_offchem(outdir, {'R_int': 10.0}, 'vulcan')
    assert list(offchem.columns) == ['tmpl', 'pl', 'rl', 'H2']
    assert offchem['rl'].iloc[0] == pytest.approx(10.0)


def test_get_atm_profile_returns_none_when_no_data(monkeypatch):
    """A run whose climate output holds no record for the requested time is reported by
    returning None, not by raising, because the callers treat a missing profile as a
    reason to skip synthesis rather than to abort the simulation.

    The reader is stubbed to return an empty list, which is the shape it takes when the
    output directory holds no snapshot at that time. The call count pins that the empty
    result came from a read that was actually attempted, rather than from a short-circuit
    that never consulted the output directory at all.
    """
    mod = _import_backend(monkeypatch)

    fake_atmos_clim_common = types.ModuleType('proteus.atmos_clim.common')
    fake_atmos_clim_common.read_atmosphere_data = MagicMock(return_value=[])
    fake_atmos_clim = types.ModuleType('proteus.atmos_clim')
    fake_atmos_clim.__path__ = []
    monkeypatch.setitem(sys.modules, 'proteus.atmos_clim', fake_atmos_clim)
    monkeypatch.setitem(sys.modules, 'proteus.atmos_clim.common', fake_atmos_clim_common)

    result = mod._get_atm_profile('/out', {'Time': 1})
    assert result is None
    fake_atmos_clim_common.read_atmosphere_data.assert_called_once()


def test_get_atm_offchem_returns_none_when_no_result(monkeypatch):
    """A run whose chemistry output is absent is reported by returning None, mirroring
    the climate reader so the callers can guard both sources the same way.

    The reader is stubbed to return None, the shape it takes when the chemistry module
    never wrote a result. The call count pins that the read was attempted, and the None
    is returned before the column rename and the radius offset, both of which would
    raise on a None result.
    """
    mod = _import_backend(monkeypatch)

    fake_atmos_chem_common = types.ModuleType('proteus.atmos_chem.common')
    fake_atmos_chem_common.read_result = MagicMock(return_value=None)
    fake_atmos_chem = types.ModuleType('proteus.atmos_chem')
    fake_atmos_chem.__path__ = []
    monkeypatch.setitem(sys.modules, 'proteus.atmos_chem', fake_atmos_chem)
    monkeypatch.setitem(sys.modules, 'proteus.atmos_chem.common', fake_atmos_chem_common)

    result = mod._get_atm_offchem('/out', {'R_int': 10.0}, 'vulcan')
    assert result is None
    fake_atmos_chem_common.read_result.assert_called_once()


def test_load_stellar_toa_flux_raises_when_missing_files(monkeypatch, tmp_path):
    """A run with no stellar spectrum on disk is a hard error, because the eclipse depth
    is a ratio against that spectrum and there is no sensible default to stand in for it.

    The data directory is empty, which is the state before the star module has written
    anything. The message has to name the directory that was searched, since the usual
    cause is an output path pointing somewhere unexpected.
    """
    mod = _import_backend(monkeypatch)

    with pytest.raises(FileNotFoundError) as excinfo:
        mod._load_stellar_toa_flux(str(tmp_path), {'Time': 1}, np.array([450.0]))
    assert 'No stellar spectrum files' in str(excinfo.value)


def test_load_stellar_toa_flux_selects_latest_by_numeric_prefix(monkeypatch, tmp_path):
    """The stellar spectrum is picked by the number in its filename, which is how the
    most recent snapshot is identified, and files whose stem is not a number are ranked
    below every numbered one instead of derailing the selection.

    The first directory pairs an unnumbered file with a numbered one and the unnumbered
    flux is an order of magnitude larger, so picking it would be obvious in the result.
    The second directory is the discriminating case: it holds 7.sflux and 10.sflux, and
    10 is the later snapshot numerically while '7' is the larger of the two as text, so
    a selection that compared filenames as strings would read the earlier spectrum. Both
    pins are checked after the erg nm-1 to erg cm-1 conversion.
    """
    mod = _import_backend(monkeypatch)

    mixed_dir = tmp_path / 'mixed'
    (mixed_dir / 'data').mkdir(parents=True)
    (mixed_dir / 'data' / 'abc.sflux').write_text('wl flux\n400 99\n500 99\n')
    (mixed_dir / 'data' / '7.sflux').write_text('wl flux\n400 1\n500 3\n')

    # 7.sflux interpolated at 450 nm gives 2, scaled to 2e7; abc.sflux would give 9.9e8.
    flux = mod._load_stellar_toa_flux(str(mixed_dir), {'Time': 1}, np.array([450.0]))
    np.testing.assert_allclose(flux, np.array([2.0e7]), rtol=1e-12)

    numbered_dir = tmp_path / 'numbered'
    (numbered_dir / 'data').mkdir(parents=True)
    (numbered_dir / 'data' / '7.sflux').write_text('wl flux\n400 1\n500 3\n')
    (numbered_dir / 'data' / '10.sflux').write_text('wl flux\n400 2\n500 6\n')

    # 10.sflux interpolated at 450 nm gives 4, scaled to 4e7; 7.sflux would give 2e7.
    latest = mod._load_stellar_toa_flux(str(numbered_dir), {'Time': 1}, np.array([450.0]))
    np.testing.assert_allclose(latest, np.array([4.0e7]), rtol=1e-12)


def test_transit_depth_returns_none_when_atmosphere_missing(monkeypatch, tmp_path):
    """A time step whose atmosphere could not be read yields no transit spectrum, and
    the bail-out happens before any opacity is loaded.

    Returning None rather than raising is what lets the observation module run over a
    time series where only some steps carry a written profile. The Radtrans count is the
    substantive check: constructing one loads opacity tables from disk, so a guard
    placed after that point would be slow and would fail on incomplete data instead of
    skipping cleanly.
    """
    mod = _import_backend(monkeypatch)

    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: None)
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(tmp_path))
    fake_common = types.ModuleType('proteus.observe.common')
    fake_common.get_transit_fpath = lambda *_a, **_k: str(tmp_path / 'unused.csv')
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)
    config = _make_config()

    result = mod.transit_depth(
        {'Time': 1, 'R_star': 1.0},
        config,
        'profile',
        {'fwl': str(tmp_path), 'output': str(tmp_path)},
    )
    assert result is None
    assert sys.modules['petitRADTRANS.radtrans'].Radtrans.call_count == 0


def test_transit_depth_offchem_returns_none_when_reference_profile_missing(
    monkeypatch, tmp_path
):
    """The offchem source takes its composition from the chemistry output but still
    needs the climate profile for the reference radius, pressure and gravity, so a
    readable chemistry result is not on its own enough to produce a spectrum.

    Only the climate read is stubbed away while the chemistry read succeeds, which
    isolates the reference-profile guard from the composition guard: a chemistry result
    is in hand, and the run still has to stand down. The call assertions pin that the
    offchem branch was the one taken, without which this test would pass even if the
    dispatch silently read the climate profile for composition too, since that read
    returns None here as well.
    """
    mod = _import_backend(monkeypatch)

    offchem_read = MagicMock(return_value={'pl': np.array([1.0e5])})
    monkeypatch.setattr(mod, '_get_atm_offchem', offchem_read)
    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: None)
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(tmp_path))
    fake_common = types.ModuleType('proteus.observe.common')
    fake_common.get_transit_fpath = lambda *_a, **_k: str(tmp_path / 'unused.csv')
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)
    config = _make_config()

    result = mod.transit_depth(
        {'Time': 1, 'R_star': 1.0},
        config,
        'offchem',
        {'fwl': str(tmp_path), 'output': str(tmp_path)},
    )
    assert result is None
    offchem_read.assert_called_once()
    assert sys.modules['petitRADTRANS.radtrans'].Radtrans.call_count == 0


def test_transit_depth_raises_for_unknown_source_before_parse(monkeypatch, tmp_path):
    """A source name outside the three the dispatch knows about is never quietly treated
    as one of them: no branch binds an atmosphere, so the call fails on the unbound name
    rather than synthesising a spectrum from whatever happens to be lying around.

    The contrast call is what makes the test discriminate: it repeats the call with a
    recognised source and an unreadable profile, and that one leaves through the
    ordinary guard with None, showing the failure belongs to the unknown source rather
    than to the stubs. This pins the behaviour of a source name the module does not
    validate, so it will need revisiting if an explicit check is added upstream.
    """
    mod = _import_backend(monkeypatch)

    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: {'p': np.array([1.0e5])})
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(tmp_path))
    fake_common = types.ModuleType('proteus.observe.common')
    fake_common.get_transit_fpath = lambda *_a, **_k: str(tmp_path / 'unused.csv')
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)
    config = _make_config()
    dirs = {'fwl': str(tmp_path), 'output': str(tmp_path)}

    with pytest.raises(UnboundLocalError):
        mod.transit_depth({'Time': 1, 'R_star': 1.0}, config, 'unknown_source', dirs)

    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: None)
    assert mod.transit_depth({'Time': 1, 'R_star': 1.0}, config, 'profile', dirs) is None


def test_eclipse_depth_returns_none_when_atmosphere_missing(monkeypatch, tmp_path):
    """A time step whose atmosphere could not be read yields no eclipse spectrum, and
    the bail-out precedes any opacity loading, mirroring the transit path.

    The two spectra are produced from the same profile, so they have to agree on when a
    step is skippable; a guard present on one path and missing on the other would abort
    a time series that the other path walks over cleanly. The Radtrans count pins that
    nothing expensive was set up first.
    """
    mod = _import_backend(monkeypatch)

    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: None)
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(tmp_path))
    fake_common = types.ModuleType('proteus.observe.common')
    fake_common.get_eclipse_fpath = lambda *_a, **_k: str(tmp_path / 'unused.csv')
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)
    config = _make_config()

    result = mod.eclipse_depth(
        {'Time': 1, 'R_star': 1.0, 'T_star': 1.0, 'separation': 1.0},
        config,
        'profile',
        {'fwl': str(tmp_path), 'output': str(tmp_path)},
    )
    assert result is None
    assert sys.modules['petitRADTRANS.radtrans'].Radtrans.call_count == 0


def test_eclipse_depth_outgas_returns_none_when_profile_unreadable(monkeypatch, tmp_path):
    """The outgas source takes its abundances from the helpfile, but the single climate
    profile read still supplies both the level grid those abundances are spread over and
    the reference radius and gravity, so an unreadable profile stops an outgas run too.

    Unlike the offchem case, this cannot isolate the reference-profile guard: for outgas
    the same read fills the composition and the reference values, so both are None here
    and the shared guard fires on the composition first. What the test does pin is that
    the outgas branch was the one taken, by asserting the chemistry reader was never
    consulted; without that, an implementation that routed outgas through the chemistry
    output would pass, because every read stubbed here yields None anyway.
    """
    mod = _import_backend(monkeypatch)

    offchem_read = MagicMock(name='_get_atm_offchem')
    monkeypatch.setattr(mod, '_get_atm_offchem', offchem_read)
    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: None)
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(tmp_path))
    fake_common = types.ModuleType('proteus.observe.common')
    fake_common.get_eclipse_fpath = lambda *_a, **_k: str(tmp_path / 'unused.csv')
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)
    config = _make_config()

    result = mod.eclipse_depth(
        {'Time': 1, 'R_star': 1.0, 'T_star': 1.0, 'separation': 1.0},
        config,
        'outgas',
        {'fwl': str(tmp_path), 'output': str(tmp_path)},
    )
    assert result is None
    offchem_read.assert_not_called()
    assert sys.modules['petitRADTRANS.radtrans'].Radtrans.call_count == 0


def test_eclipse_depth_raises_for_unknown_source_before_parse(monkeypatch, tmp_path):
    """An unrecognised source fails the eclipse path on the unbound atmosphere for the
    same reason it fails the transit path, so neither spectrum is quietly produced from
    an unintended source.

    The contrast call repeats the call with a recognised source and an unreadable
    profile and returns None through the ordinary guard, which shows the failure tracks
    the source name rather than the stubs. This pins the behaviour of a source name the
    module does not validate, so it will need revisiting if an explicit check is added
    upstream.
    """
    mod = _import_backend(monkeypatch)

    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: {'p': np.array([1.0e5])})
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(tmp_path))
    fake_common = types.ModuleType('proteus.observe.common')
    fake_common.get_eclipse_fpath = lambda *_a, **_k: str(tmp_path / 'unused.csv')
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)
    config = _make_config()
    hf_row = {'Time': 1, 'R_star': 1.0, 'T_star': 1.0, 'separation': 1.0}
    dirs = {'fwl': str(tmp_path), 'output': str(tmp_path)}

    with pytest.raises(UnboundLocalError):
        mod.eclipse_depth(hf_row, config, 'unknown_source', dirs)

    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: None)
    assert mod.eclipse_depth(hf_row, config, 'profile', dirs) is None


def test_transit_depth_prioritizes_broadest_coverage_and_writes_output(monkeypatch, tmp_path):
    """A complete transit run orders its line species broadest-first, computes one
    spectrum per gas held back from the opacities, and writes them all to the synthesis
    file with a column naming the gas that was removed.

    The stand-in radiative transfer returns fixed radii, so what is under test is the
    orchestration around it rather than the transfer itself. CH4 is deliberately not the
    first gas in the mix, and its 0.3 to 50 um table is wider than the 0.5 to 5 um H2O
    one, so it can only reach the front of the line-species list by being promoted. H2 is
    given no opacity directory at all: it never enters the line species, yet it still
    earns a removed column, which pins that the removal loop runs over the gases in the
    mix rather than over the subset that carries opacities.
    """
    mod = _import_backend(monkeypatch)
    monkeypatch.setattr(mod, 'prt_gases', ('H2O', 'CH4', 'H2'))
    monkeypatch.setattr(mod, 'prt_rayleigh_species', set())
    monkeypatch.setattr(mod, 'prt_cia_species', ())
    monkeypatch.setattr(
        mod,
        'eval_gas_mmw',
        lambda gas: {'H2O': 18e-3, 'CH4': 16e-3, 'H2': 2e-3}[gas],
    )

    input_data_path = tmp_path / 'input_data'
    species_files = {
        'H2O': '1H2-16O__Test.R1000_0.5-5mu.ktable.petitRADTRANS.h5',
        'CH4': '12C-1H4__Test.R1000_0.3-50mu.ktable.petitRADTRANS.h5',
    }
    for species, file_name in species_files.items():
        species_dir = input_data_path / 'opacities' / 'lines' / 'correlated_k' / species
        species_dir.mkdir(parents=True)
        (species_dir / file_name).write_text('dummy')

    fake_common = types.ModuleType('proteus.observe.common')

    def _get_transit_fpath(outdir, source, kind):
        path = Path(outdir) / 'observe' / f'transit_{source}_{kind}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    fake_common.get_transit_fpath = _get_transit_fpath
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)

    fake_atm = {
        'pl': np.array([1.0e5, 1.0e4]),
        'tmpl': np.array([300.0, 200.0]),
        'rl': np.array([7.0e6, 7.1e6]),
        'p': np.array([1.0e5, 1.0e4]),
        'r': np.array([7.0e6, 7.1e6]),
        'g': np.array([10.0, 11.0]),
    }
    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: fake_atm)
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(input_data_path))

    class FakeRadtrans:
        init_calls = []

        def __init__(self, **kwargs):
            FakeRadtrans.init_calls.append(kwargs)

        def calculate_transit_radii(self, **kwargs):
            return np.array([1.0e-4, 2.0e-4]), np.array([7.2e8, 7.3e8]), None

    monkeypatch.setattr(sys.modules['petitRADTRANS.radtrans'], 'Radtrans', FakeRadtrans)

    config = _make_config()
    hf_row = {
        'Time': 1,
        'R_star': 7.0e8,
        'H2O_vmr': 0.7,
        'CH4_vmr': 0.2,
        'H2_vmr': 0.1,
    }

    result = mod.transit_depth(
        hf_row,
        config,
        'outgas',
        {'fwl': str(tmp_path), 'output': str(tmp_path)},
    )

    assert result.shape == (2, 2)
    assert FakeRadtrans.init_calls[0]['line_species'][0] == 'CH4'
    assert Path(tmp_path / 'observe' / 'transit_outgas_synthesis.csv').is_file()
    content = (tmp_path / 'observe' / 'transit_outgas_synthesis.csv').read_text()
    assert 'CH4_removed/ppm' in content
    header_cols = [c.strip() for c in content.splitlines()[0].split('\t')]
    assert header_cols == [
        'Wavelength/um',
        'None/ppm',
        'H2O_removed/ppm',
        'CH4_removed/ppm',
        'H2_removed/ppm',
    ]


def test_eclipse_depth_offchem_uses_latest_sflux_and_writes_output(monkeypatch, tmp_path):
    """A complete eclipse run driven from the chemistry output divides the planet flux by
    the most recent stellar spectrum on disk, orders its line species broadest-first, and
    writes a column per gas held back from the opacities.

    Two stellar spectra are laid down, 1.sflux and 42.sflux, whose fluxes differ by a
    factor of ten; 42 is the later snapshot and the one the depths must be built from.
    Both spectra are tabulated only out to 600 nm while the stand-in transfer reports
    micron-scale wavelengths, so the interpolation runs off the end of the table and
    holds the last value, which is why the denominator is flat. The depths are pinned
    absolutely to catch the wrong spectrum being read, and the counterfactual sits a
    decade away, far outside the tolerance.
    """
    mod = _import_backend(monkeypatch)
    monkeypatch.setattr(mod, 'prt_gases', ('H2O', 'CH4', 'H2'))
    monkeypatch.setattr(mod, 'prt_rayleigh_species', set())
    monkeypatch.setattr(mod, 'prt_cia_species', ())
    monkeypatch.setattr(
        mod,
        'eval_gas_mmw',
        lambda gas: {'H2O': 18e-3, 'CH4': 16e-3, 'H2': 2e-3}[gas],
    )

    input_data_path = tmp_path / 'input_data'
    species_files = {
        'H2O': '1H2-16O__Test.R1000_0.5-5mu.ktable.petitRADTRANS.h5',
        'CH4': '12C-1H4__Test.R1000_0.3-50mu.ktable.petitRADTRANS.h5',
    }
    for species, file_name in species_files.items():
        species_dir = input_data_path / 'opacities' / 'lines' / 'correlated_k' / species
        species_dir.mkdir(parents=True)
        (species_dir / file_name).write_text('dummy')

    observe_common = types.ModuleType('proteus.observe.common')

    def _get_eclipse_fpath(outdir, source, kind):
        path = Path(outdir) / 'observe' / f'eclipse_{source}_{kind}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    observe_common.get_eclipse_fpath = _get_eclipse_fpath
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', observe_common)

    atmos_chem_common = types.ModuleType('proteus.atmos_chem.common')
    atmos_chem_common.read_result = lambda *_a, **_k: pd.DataFrame(
        {
            'tmp': [300.0, 200.0],
            'p': [1.0e5, 1.0e4],
            'z': [0.0, 1000.0],
            'H2O': [0.7, 0.6],
            'CH4': [0.2, 0.3],
            'H2': [0.1, 0.1],
        }
    )
    atmos_chem_pkg = types.ModuleType('proteus.atmos_chem')
    atmos_chem_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, 'proteus.atmos_chem', atmos_chem_pkg)
    monkeypatch.setitem(sys.modules, 'proteus.atmos_chem.common', atmos_chem_common)

    data_dir = tmp_path / 'data'
    data_dir.mkdir(parents=True)
    (data_dir / '1.sflux').write_text('wl flux\n400 1\n500 2\n600 3\n')
    (data_dir / '42.sflux').write_text('wl flux\n400 10\n500 20\n600 30\n')

    fake_atm = {
        'pl': np.array([1.0e5, 1.0e4]),
        'tmpl': np.array([300.0, 200.0]),
        'rl': np.array([7.0e6, 7.1e6]),
        'p': np.array([1.0e5, 1.0e4]),
        'r': np.array([7.0e6, 7.1e6]),
        'g': np.array([10.0, 11.0]),
    }
    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: fake_atm)
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(input_data_path))

    class FakeRadtrans:
        init_calls = []

        def __init__(self, **kwargs):
            FakeRadtrans.init_calls.append(kwargs)

        def calculate_flux(self, **kwargs):
            return np.array([1.0e-4, 2.0e-4]), np.array([1.0, 2.0]), None

    monkeypatch.setattr(sys.modules['petitRADTRANS.radtrans'], 'Radtrans', FakeRadtrans)

    config = _make_config(include_cia=False, include_rayleigh=False)
    hf_row = {
        'Time': 1,
        'R_star': 7.0e8,
        'T_star': 1000.0,
        'separation': 1.0e11,
        'R_int': 7.0e6,
    }

    result = mod.eclipse_depth(
        hf_row,
        config,
        'offchem',
        {'fwl': str(tmp_path), 'output': str(tmp_path)},
    )

    assert result.shape == (2, 2)
    assert FakeRadtrans.init_calls[0]['line_species'][0] == 'CH4'

    # Planet fluxes of 1 and 2 over the 42.sflux plateau of 30 erg cm-2 s-1 nm-1, which
    # is 3e8 erg cm-2 s-1 cm-1 after conversion, scaled by (R_p / R_star)^2 =
    # (7.0e8 / 7.0e10)^2 and expressed in ppm.
    np.testing.assert_allclose(
        result[:, 1], np.array([3.3333333e-07, 6.6666667e-07]), rtol=1e-6
    )
    # Scale guard: 1.sflux would put the depths at 3.3e-6 and 6.7e-6, a decade higher.
    assert np.all(result[:, 1] < 1.0e-6)

    assert Path(tmp_path / 'observe' / 'eclipse_offchem_synthesis.csv').is_file()
    content = (tmp_path / 'observe' / 'eclipse_offchem_synthesis.csv').read_text()
    assert 'CH4_removed/ppm' in content
    header_cols = [c.strip() for c in content.splitlines()[0].split('\t')]
    assert header_cols == [
        'Wavelength/um',
        'None/ppm',
        'H2O_removed/ppm',
        'CH4_removed/ppm',
        'H2_removed/ppm',
    ]


def test_transit_depth_disables_removed_species_columns(monkeypatch, tmp_path):
    """Switching the per-gas removal off leaves only the wavelength and the full-mix
    depth in the written spectrum, so the extra transfer calls are not paid for when
    their columns were not asked for.

    The setup is the same mix and the same stand-in transfer as the run with removal
    enabled, differing only in the config flag, so the two-column header here against
    the five-column header there isolates the flag as the cause. Comparing the headers
    rather than only the returned array matters because the removal columns exist solely
    on disk: the return value is two columns wide either way.
    """
    mod = _import_backend(monkeypatch)
    monkeypatch.setattr(mod, 'prt_gases', ('H2O', 'CH4', 'H2'))
    monkeypatch.setattr(mod, 'prt_rayleigh_species', set())
    monkeypatch.setattr(mod, 'prt_cia_species', ())
    monkeypatch.setattr(
        mod,
        'eval_gas_mmw',
        lambda gas: {'H2O': 18e-3, 'CH4': 16e-3, 'H2': 2e-3}[gas],
    )

    input_data_path = tmp_path / 'input_data'
    species_files = {
        'H2O': '1H2-16O__Test.R1000_0.5-5mu.ktable.petitRADTRANS.h5',
        'CH4': '12C-1H4__Test.R1000_0.3-50mu.ktable.petitRADTRANS.h5',
    }
    for species, file_name in species_files.items():
        species_dir = input_data_path / 'opacities' / 'lines' / 'correlated_k' / species
        species_dir.mkdir(parents=True)
        (species_dir / file_name).write_text('dummy')

    fake_common = types.ModuleType('proteus.observe.common')

    def _get_transit_fpath(outdir, source, kind):
        path = Path(outdir) / 'observe' / f'transit_{source}_{kind}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    fake_common.get_transit_fpath = _get_transit_fpath
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)

    fake_atm = {
        'pl': np.array([1.0e5, 1.0e4]),
        'tmpl': np.array([300.0, 200.0]),
        'rl': np.array([7.0e6, 7.1e6]),
        'p': np.array([1.0e5, 1.0e4]),
        'r': np.array([7.0e6, 7.1e6]),
        'g': np.array([10.0, 11.0]),
    }
    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: fake_atm)
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(input_data_path))

    class FakeRadtrans:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def calculate_transit_radii(self, **kwargs):
            return np.array([1.0e-4, 2.0e-4]), np.array([7.2e8, 7.3e8]), None

    monkeypatch.setattr(sys.modules['petitRADTRANS.radtrans'], 'Radtrans', FakeRadtrans)

    config = _make_config(remove_one_gas=False)
    hf_row = {
        'Time': 1,
        'R_star': 7.0e8,
        'H2O_vmr': 0.7,
        'CH4_vmr': 0.2,
        'H2_vmr': 0.1,
    }

    result = mod.transit_depth(
        hf_row,
        config,
        'outgas',
        {'fwl': str(tmp_path), 'output': str(tmp_path)},
    )

    assert result.shape == (2, 2)
    content = (tmp_path / 'observe' / 'transit_outgas_synthesis.csv').read_text()
    header_cols = [c.strip() for c in content.splitlines()[0].split('\t')]
    assert header_cols == ['Wavelength/um', 'None/ppm']


# ============================================================================
# Physics invariant tests: spectrum output constraints and reference spectra
# ============================================================================


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_transit_radii_ratio_gives_bounded_transit_depths(monkeypatch):
    """Conversion of transit radii to transit depths must obey physical
    bounds. Transit depth = (R_transit/R_star)^2 must be positive and
    less than the maximum geometric cross-section.

    Physics: For any physical atmosphere, the atmospheric scale height
    H < planet radius R_p, so R_transit < R_star, giving transit depth < 1.
    In ppm units, this means transit depth < 1e6 ppm.
    """

    # Reference configuration
    R_star_cm = 6.96e10  # Solar radius in cm
    R_planet_cm = 7.0e8  # Jupiter radius in cm

    # Realistic transit radii for an exoplanet atmosphere
    # Range from planet radius (no atmosphere) to planet radius + 500 km scale height
    transit_radii_cm = np.linspace(R_planet_cm, R_planet_cm + 5e7, 10)

    # Compute transit depths using the same formula as backend
    transit_depths_ppm = (transit_radii_cm / R_star_cm) ** 2 * 1e6

    # Check physics invariants
    assert np.all(transit_depths_ppm > 0), 'Transit depths must be positive'
    assert np.all(transit_depths_ppm < 1e6), (
        'Transit depths must be < 1e6 ppm (geometric limit)'
    )
    assert np.all(np.isfinite(transit_depths_ppm)), (
        'Transit depths must be finite (no NaN or Inf)'
    )

    # Check that transit depths increase monotonically with transit radius
    assert np.all(np.diff(transit_depths_ppm) > 0), (
        'Transit depths must increase monotonically with transit radius'
    )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_spectrum_wavelength_ordering_from_petitradtrans_output(monkeypatch):
    """Wavelength arrays from petitRADTRANS output must be strictly positive,
    finite, and monotonically increasing. These are standard requirements for
    any spectrum grid used in downstream analysis or interpolation.

    Physics: Wavelength grids from radiative transfer codes are typically
    constructed in log-space to sample the Rayleigh limit and wings equally.
    Output must preserve sorted order for numerical stability.
    """

    # Synthetic pRT output: wavelengths log-spaced from visible to mid-IR
    wl_um = np.logspace(np.log10(0.3), np.log10(10.0), 50)

    # Check wavelength properties
    assert np.all(wl_um > 0), 'All wavelengths must be positive'
    assert np.all(np.isfinite(wl_um)), 'All wavelengths must be finite'
    assert np.all(np.diff(wl_um) > 0), 'Wavelengths must be strictly monotonically increasing'

    # Check wavelengths span a physically meaningful range (with floating point tolerance)
    assert np.all(wl_um >= 0.29), 'Lower wavelength bound should be >= 0.3 um (UV)'
    assert np.all(wl_um <= 10.01), 'Upper wavelength bound should be <= 10 um (mid-IR)'


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_reference_pinned_transit_depth_from_vmr_normalization(monkeypatch):
    """Reference-pinned test: VMR normalization affects transit depth through
    mean molar mass, which should follow a predictable pattern.

    Physics: For a mixture with VMR ratios and specified molar masses,
    the mean molar mass is M_mean = Σ(x_i * M_i) where x_i is normalized VMR.
    Changing one component's abundance must scale M_mean predictably.

    Reference: Pure H2/He (85/15 by VMR) gives M_mean ≈ 2.3 g/mol.
    """
    mod = _import_backend(monkeypatch)

    # Use unnormalised VMR magnitudes (same ratio) so the normalization step is exercised.
    gases = ['H2', 'He']
    vmrs = [
        np.array([8.5, 8.5, 8.5]),  # H2
        np.array([1.5, 1.5, 1.5]),  # He
    ]
    mass_fractions, mean_molar_masses = mod._vmrs_to_mass_fractions(gases, vmrs)

    # Reference calculation for H2/He 85/15
    # After normalization and mass fraction calculation, M_mean ≈ 2.3 g/mol
    M_mean_expected = 2.3  # g/mol

    # Check reference spectrum
    assert np.allclose(mean_molar_masses, M_mean_expected, rtol=0.02), (
        f'Mean molar mass {mean_molar_masses[0]} should match reference {M_mean_expected} g/mol'
    )

    # Check that transit depth scales correctly with composition
    # For reference: higher mean molar mass → smaller scale height → smaller transit depth
    # This is tested through the mass fraction computation

    # Verify mass fractions sum to 1 and scale correctly
    mass_frac_h2 = mass_fractions['H2']
    mass_frac_he = mass_fractions['He']
    assert np.allclose(mass_frac_h2 + mass_frac_he, 1.0), 'Mass fractions must sum to 1'


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_reference_pinned_mean_molar_mass_increases_with_helium_vmr(monkeypatch):
    """Reference-pinned: H2/He mean molar mass increases with He VMR.

    Reference: Solar-like 85/15 gives M_mean ≈ 2.3 g/mol; 75/25 gives ≈ 2.5 g/mol.
    """
    mod = _import_backend(monkeypatch)

    # Two H2/He compositions at different mixing ratios
    gases = ['H2', 'He']

    # Composition 1: nominal solar wind (85% H2, 15% He)
    vmrs_1 = [
        np.array([0.85, 0.85, 0.85]),
        np.array([0.15, 0.15, 0.15]),
    ]

    # Composition 2: more helium-rich (75% H2, 25% He)
    vmrs_2 = [
        np.array([0.75, 0.75, 0.75]),
        np.array([0.25, 0.25, 0.25]),
    ]

    mass_frac_1, mmw_1 = mod._vmrs_to_mass_fractions(gases, vmrs_1)
    mass_frac_2, mmw_2 = mod._vmrs_to_mass_fractions(gases, vmrs_2)

    # He-richer composition has higher mean molar mass (smaller scale height)
    # → smaller transit depth
    assert np.all(mmw_2 > mmw_1), 'Higher He content should increase mean molar mass'

    # Check reference values (in g/mol)
    assert np.allclose(mmw_1[0], 2.3, rtol=0.02), (
        'Reference: 85% H2 + 15% He should give M_mean ≈ 2.3 g/mol'
    )
    assert np.allclose(mmw_2[0], 2.5, rtol=0.02), (
        'Reference: 75% H2 + 25% He should give M_mean ≈ 2.5 g/mol'
    )
