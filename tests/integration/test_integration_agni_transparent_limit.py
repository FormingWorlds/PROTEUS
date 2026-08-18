# AGNI transparent-atmosphere solve against the Stefan-Boltzmann limit.
#
# Module under test: src/proteus/atmos_clim/agni.py
#
# Contract clauses exercised:
# - `update_agni_atmos` selects the transparent branch when P_surf falls
#   below `atmos_clim.agni.psurf_thresh`.
# - `run_agni` reports the top-of-atmosphere upward longwave flux as
#   `F_olr`, and the net upward flux as `F_atm`.
# - Transparent mode holds the column isothermal at the surface
#   temperature, so with no instellation a black surface emits the
#   analytical limit `F_olr = sigma * T_surf**4` and a surface of albedo a
#   emits `sigma * T_surf**4 * (1 - a * exp(-tau))`.
#
# Invariants asserted: analytical limit (Stefan-Boltzmann), positivity,
# monotonicity in T_surf, and radiative energy balance at zero
# instellation.
#
# Integration tier: the Julia runtime boot, the AGNI package load, and the
# SOCRATES spectral-file build dominate the wall time; each transparent
# solve itself is milliseconds.
#
# Documentation: For testing standards, see:
# - docs/How-to/testing.md
# - docs/Explanations/test_framework.md
# Validation record: docs/Validation/atmos_clim/agni.md
#
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus
from proteus.atmos_clim.agni import (
    activate_julia,
    init_agni_atmos,
    run_agni,
    update_agni_atmos,
)
from proteus.atmos_clim.common import get_spfile_path
from proteus.star.wrapper import write_spectrum
from proteus.utils.coupler import ZeroHelpfileRow

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

# Run the AGNI coupling only in nightly CI (requires compiled binaries)
RUN_NIGHTLY_SMOKE = os.environ.get('PROTEUS_CI_NIGHTLY', '0') == '1'

# CODATA 2018 Stefan-Boltzmann constant [W m-2 K-4]. AGNI carries the same
# value, so the analytical limit is pinned against the physical constant
# rather than against a repeat of the model's own arithmetic.
SIGMA_SB = 5.670374419e-8

# Surface temperatures spanning magma-ocean conditions. The 3x span gives a
# factor 81 in T**4 against 27 in T**3, which resolves the exponent far
# above any tolerance used here.
T_SURF_GRID = (1000.0, 1500.0, 2000.0, 3000.0)

# Transparent mode holds the column isothermal at T_surf, so a black surface
# recovers sigma T**4 whatever residual opacity remains: what the column
# absorbs it re-emits. The tolerance covers arithmetic only.
RTOL_GREYGAS = 1.0e-5

# The banded scheme evaluates the Planck function at band centres and
# truncates outside the spectral range, so it recovers the black-body flux
# to a few times 1e-5. The tolerance keeps a factor of about 6 in hand.
RTOL_SOCRATES = 5.0e-4


def _stellar_spectrum(n_points: int = 2000, t_star: float = 5772.0):
    """Build a black-body stellar spectrum in the units of a `.sflux` file.

    The spectrum exists only so AGNI can insert the solar and Planck blocks
    into its copy of the spectral file. Every quantity pinned by this module
    is longwave and is measured at zero instellation, so the shape of the
    stellar spectrum does not enter the result.

    Parameters
    ----------
        n_points : int
            Number of wavelength samples.
        t_star : float
            Stellar effective temperature [K].

    Returns
    ----------
        wl_arr : np.ndarray
            Wavelength [nm].
        fl_arr : np.ndarray
            Flux density [erg s-1 cm-2 nm-1].
    """
    h_planck = 6.62607015e-34  # [J s]
    c_light = 2.99792458e8  # [m s-1]
    k_boltz = 1.380649e-23  # [J K-1]

    wl_arr = np.logspace(np.log10(0.5), np.log10(1.0e6), n_points)  # nm
    wl_m = wl_arr * 1.0e-9

    # Planck radiance [W m-2 m-1 sr-1], then flux at the stellar surface.
    # The exponent is capped so the shortest wavelengths underflow to zero
    # instead of overflowing.
    exponent = np.minimum(h_planck * c_light / (wl_m * k_boltz * t_star), 700.0)
    radiance = (2.0 * h_planck * c_light**2 / wl_m**5) / np.expm1(exponent)
    flux_si = np.pi * radiance  # [W m-2 m-1]

    # 1 W m-2 m-1 = 1e-6 erg s-1 cm-2 nm-1, then dilute to 1 AU
    dilution = (6.957e8 / 1.495978707e11) ** 2
    return wl_arr, flux_si * 1.0e-6 * dilution


def _make_runner(tmpdir: str, *, spectral_file, surf_greyalbedo: float):
    """Configure a Proteus runner for a transparent AGNI solve.

    Parameters
    ----------
        tmpdir : str
            Directory that receives the run output tree.
        spectral_file : str or None
            `'greygas'` for the analytic two-stream scheme, or None to let
            AGNI build a banded file from FWL_DATA.
        surf_greyalbedo : float
            Grey surface albedo; the surface emissivity is one minus this.

    Returns
    ----------
        runner : Proteus
            Runner with directories initialised.
    """
    runner = Proteus(config_path=PROTEUS_ROOT / 'input' / 'dummy.toml')
    cfg = runner.config

    cfg.atmos_clim.module = 'agni'
    cfg.atmos_clim.agni.spectral_file = spectral_file
    cfg.atmos_clim.agni.surf_material = 'greybody'
    cfg.atmos_clim.agni.solve_energy = False
    cfg.atmos_clim.agni.chemistry = 'none'
    cfg.atmos_clim.agni.rainout = False
    cfg.atmos_clim.agni.oceans = False
    cfg.atmos_clim.surf_greyalbedo = surf_greyalbedo
    cfg.atmos_clim.surf_state = 'fixed'
    cfg.interior_energetics.module = 'dummy'
    cfg.interior_struct.module = 'dummy'
    cfg.params.out.path = str(Path(tmpdir) / 'output')
    cfg.params.out.plot_mod = 0
    cfg.params.out.write_mod = 0
    runner.init_directories()
    return runner


def _make_hf_row(t_surf: float, p_surf: float = 1.0e-3):
    """Build a helpfile row that drives AGNI into transparent mode.

    Parameters
    ----------
        t_surf : float
            Surface temperature [K].
        p_surf : float
            Surface pressure [bar], below `agni.psurf_thresh` so that the
            transparent branch is selected.

    Returns
    ----------
        hf_row : dict
            Helpfile row with the fields the AGNI wrapper reads.
    """
    hf_row = ZeroHelpfileRow()
    hf_row['Time'] = 0.0
    hf_row['T_surf'] = t_surf
    hf_row['T_magma'] = t_surf
    hf_row['P_surf'] = p_surf
    hf_row['gravity'] = 9.81
    hf_row['R_int'] = 6.371e6
    hf_row['R_xuv'] = 6.371e6
    hf_row['p_xuv'] = 1.0e-5
    hf_row['F_ins'] = 0.0
    hf_row['albedo_pl'] = 0.0
    hf_row['H2O_vmr'] = 1.0
    return hf_row


def _sweep_transparent(runner, t_surf_grid=T_SURF_GRID):
    """Solve a transparent atmosphere at each surface temperature.

    Parameters
    ----------
        runner : Proteus
            Runner returned by `_make_runner`.
        t_surf_grid : tuple of float
            Surface temperatures to solve [K].

    Returns
    ----------
        results : list of tuple
            One `(t_surf, output, state)` entry per temperature, where
            `state` copies the atmosphere arrays read by the assertions.
            The solver mutates one struct in place, so the arrays must be
            copied at each step rather than referenced afterwards.
    """
    dirs = runner.directories
    cfg = runner.config

    activate_julia(dirs, 0)
    atmos = init_agni_atmos(dirs, cfg, _make_hf_row(t_surf_grid[0]))
    assert bool(atmos.is_alloc), 'AGNI atmosphere struct was not allocated'

    results = []
    for t_surf in t_surf_grid:
        hf_row = _make_hf_row(t_surf)
        atmos = update_agni_atmos(atmos, hf_row, dirs, cfg)
        assert bool(atmos.transparent), (
            f'P_surf={hf_row["P_surf"]} bar is below '
            f'psurf_thresh={cfg.atmos_clim.agni.psurf_thresh} bar, '
            'so the transparent branch must be selected'
        )
        atmos, output = run_agni(atmos, 1, dirs, cfg, hf_row, write_data=False)
        state = {
            'flux_u_lw': np.array(atmos.flux_u_lw, copy=True),
            'tau_band': np.array(atmos.tau_band, copy=True),
            'nbands': int(atmos.nbands),
            'pl': np.array(atmos.pl, copy=True),
            'g': np.array(atmos.g, copy=True),
        }
        results.append((t_surf, output, state))

    return results


def _lw_column_tau(state, kappa_lw: float) -> float:
    """Column longwave optical depth to the surface, grey-gas scheme.

    The grey-gas scheme has no spectral file, so there are no band
    boundaries to cut at: `tau_band` stores the longwave and shortwave
    optical depths already summed into one column
    (`AGNI/src/energy/energy.jl`, `_radtrans_greygas!`). This mirrors that
    routine's own per-layer accumulation, `d_tau_lw = d_p * kappa_lw / g`,
    using the same cell-edge pressure and cell-centre gravity arrays, so it
    isolates the longwave term the scheme actually integrates rather than
    a value read back from the combined array.

    Parameters
    ----------
        state : dict
            One `_sweep_transparent` state entry; needs `pl` and `g`.
        kappa_lw : float
            Grey longwave opacity [m2 kg-1].

    Returns
    ----------
        tau_lw : float
            Column longwave optical depth from the top of the atmosphere
            to the surface.
    """
    d_p = np.diff(state['pl'])
    return float(np.sum(d_p * kappa_lw / state['g']))


def _assert_blackbody_limit(results, *, rtol: float):
    """Assert the Stefan-Boltzmann limit and its discrimination guards.

    Parameters
    ----------
        results : list of tuple
            Output of `_sweep_transparent`, solved above a black surface.
        rtol : float
            Relative tolerance on the pinned flux.
    """
    for t_surf, output, state in results:
        expected = SIGMA_SB * t_surf**4
        assert output['F_olr'] == pytest.approx(expected, rel=rtol), (
            f'transparent OLR at T_surf={t_surf} K must equal the black-body '
            f'emission {expected:.6e} W m-2'
        )

        # Sign guard: emitted flux leaves the planet in every case.
        assert output['F_olr'] > 0.0

        # Exponent guard: a T**3 law sits a factor T below the pinned value.
        assert abs(output['F_olr'] - SIGMA_SB * t_surf**3) > rtol * expected

        # An isothermal column re-emits what it absorbs, so above a black
        # surface the upward longwave flux does not change with height.
        lw_up = state['flux_u_lw']
        assert lw_up[0] == pytest.approx(lw_up[-1], rel=rtol)

        # No shortwave source, so the net upward flux is the emitted flux
        # and no reflected shortwave leaves the planet.
        assert output['F_atm'] == pytest.approx(output['F_olr'], rel=rtol)
        assert output['F_sct'] == pytest.approx(0.0, abs=1e-6)

    # Scale guard: a Kelvin-vs-Celsius or a W-vs-erg regression moves the
    # 1000 K flux off 5.67e4 W m-2 by orders of magnitude.
    assert 1.0e4 < results[0][1]['F_olr'] < 1.0e5

    # Monotonicity: hotter surfaces emit more, across the whole sweep.
    fluxes = [output['F_olr'] for _, output, _ in results]
    assert all(np.diff(fluxes) > 0.0)

    # Ratio pin: the 3x temperature span gives 81 under T**4 and 27 under
    # T**3, so the ratio alone separates the two laws.
    t_span = results[-1][0] / results[0][0]
    ratio = fluxes[-1] / fluxes[0]
    assert ratio == pytest.approx(t_span**4, rel=2.0 * rtol)
    assert abs(ratio - t_span**3) > 1.0


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@pytest.mark.skipif(
    not RUN_NIGHTLY_SMOKE,
    reason='AGNI coupling test requires Julia/AGNI binaries (nightly only)',
)
def test_transparent_greygas_olr_equals_blackbody_emission():
    """A transparent grey-gas atmosphere emits the black-body surface flux.

    Physical scenario: a bare rocky surface with a negligible atmosphere and
    no instellation. With the opacity removed the outgoing longwave
    radiation must equal the black-body emission of the surface,
    `F_olr = sigma * T_surf**4`, which is the analytical limit of any
    radiative transfer scheme.

    Analytical limit: Stefan-Boltzmann law with unit emissivity, pinned
    against the CODATA 2018 constant at four surface temperatures.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = _make_runner(tmpdir, spectral_file='greygas', surf_greyalbedo=0.0)
        results = _sweep_transparent(runner)

        # Headline pin at both ends of the sweep, 5.67e4 W m-2 at 1000 K and
        # 4.59e6 W m-2 at 3000 K.
        assert results[0][1]['F_olr'] == pytest.approx(
            SIGMA_SB * T_SURF_GRID[0] ** 4, rel=RTOL_GREYGAS
        )
        assert results[-1][1]['F_olr'] == pytest.approx(
            SIGMA_SB * T_SURF_GRID[-1] ** 4, rel=RTOL_GREYGAS
        )

        _assert_blackbody_limit(results, rtol=RTOL_GREYGAS)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@pytest.mark.skipif(
    not RUN_NIGHTLY_SMOKE,
    reason='AGNI coupling test requires Julia/AGNI binaries (nightly only)',
)
def test_transparent_greygas_olr_scales_with_surface_emissivity():
    """A grey surface emits the black-body flux reduced by its emissivity.

    Physical scenario: the same bare surface, now with a grey albedo of 0.3.
    Kirchhoff's law fixes the emissivity at 1 minus the albedo, so the
    surface emits 70 per cent of the black-body flux. Transparent mode holds
    the column isothermal at the surface temperature, so the closed form of
    the Schwarzschild solution for an isothermal slab of optical depth tau
    over that surface is `sigma T**4 [1 - a exp(-tau)]`: the attenuated
    surface beam plus the slab's own emission.

    Analytical limit: the isothermal-slab solution, which reduces to the
    grey-body flux as tau goes to zero and to the black-body flux as tau
    grows. The slab is longwave, so the pin uses the column's longwave
    optical depth alone; see `_lw_column_tau` for why `tau_band` cannot be
    read directly under the grey-gas scheme.
    """
    albedo = 0.3
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = _make_runner(tmpdir, spectral_file='greygas', surf_greyalbedo=albedo)
        kappa_lw = runner.config.atmos_clim.agni.grey_opacity_lw
        results = _sweep_transparent(runner)

        for t_surf, output, state in results:
            blackbody = SIGMA_SB * t_surf**4
            greybody = (1.0 - albedo) * blackbody

            tau = _lw_column_tau(state, kappa_lw)
            expected = blackbody * (1.0 - albedo * np.exp(-tau))

            assert output['F_olr'] == pytest.approx(expected, rel=1.0e-4), (
                f'grey surface at T_surf={t_surf} K under an isothermal slab '
                f'of tau={tau:.4e} must emit {expected:.6e} W m-2'
            )

            # The surface boundary condition itself is the exact grey-body
            # emission, with no contribution from the slab above it.
            assert state['flux_u_lw'][-1] == pytest.approx(greybody, rel=1.0e-9)

            # Guard against a dropped emissivity: the black-body flux is 43
            # per cent above the value measured here.
            assert abs(output['F_olr'] - blackbody) > 0.25 * output['F_olr']

            # Guard against a neglected slab: the bare grey-body flux lies
            # below the emitted flux by far more than the tolerance.
            assert output['F_olr'] - greybody > 1.0e-3 * greybody

            # Sign and ordering: emission leaves the planet, and the slab can
            # only add to the attenuated surface beam.
            assert greybody < output['F_olr'] < blackbody


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@pytest.mark.skipif(
    not RUN_NIGHTLY_SMOKE,
    reason='AGNI coupling test requires Julia/AGNI binaries (nightly only)',
)
def test_transparent_greygas_emissivity_pin_needs_longwave_only_tau():
    """The emissivity pin discriminates a longwave-only tau from the combined column.

    Physical scenario: the same isothermal grey surface as above, but with
    the shortwave grey opacity raised to match the longwave value. Zero
    instellation means no shortwave flux ever enters the column, so the
    correct answer is unchanged from the default-opacity case: `F_olr`
    still follows the longwave-only slab formula. Reading the combined
    `tau_band` column instead, which now carries a shortwave term of the
    same order as the longwave one, puts roughly twice the true optical
    depth into that formula, which misses the measured flux by far more
    than the pin's own tolerance even though the column stays optically
    thin.

    Discrimination: at the default opacities the shortwave term sits five
    orders of magnitude below the longwave one, so a combined read and a
    longwave-only read agree by coincidence. Here they diverge by
    construction, so only the longwave-only read satisfies the pin.
    """
    albedo = 0.3
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = _make_runner(tmpdir, spectral_file='greygas', surf_greyalbedo=albedo)
        kappa_lw = runner.config.atmos_clim.agni.grey_opacity_lw
        runner.config.atmos_clim.agni.grey_opacity_sw = kappa_lw
        results = _sweep_transparent(runner)

        for t_surf, output, state in results:
            blackbody = SIGMA_SB * t_surf**4

            tau_lw = _lw_column_tau(state, kappa_lw)
            expected_lw = blackbody * (1.0 - albedo * np.exp(-tau_lw))
            assert output['F_olr'] == pytest.approx(expected_lw, rel=RTOL_GREYGAS), (
                f'grey surface at T_surf={t_surf} K must still follow the '
                f'longwave-only slab formula once the shortwave opacity no '
                f'longer sits five orders of magnitude below the longwave one'
            )

            # The combined column carries the shortwave term too, and here
            # that puts roughly twice the true longwave depth into the slab
            # formula, which the pin's own tolerance must reject.
            tau_combined = float(state['tau_band'][-1, 0])
            assert tau_combined > 1.5 * tau_lw
            expected_combined = blackbody * (1.0 - albedo * np.exp(-tau_combined))
            assert output['F_olr'] != pytest.approx(expected_combined, rel=RTOL_GREYGAS)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@pytest.mark.skipif(
    not RUN_NIGHTLY_SMOKE,
    reason='AGNI coupling test requires Julia/AGNI binaries (nightly only)',
)
def test_transparent_banded_olr_recovers_blackbody_emission():
    """The banded radiative transfer chain recovers the black-body limit.

    Physical scenario: the same transparent surface, solved with the
    spectral file and the SOCRATES two-stream solver that coupled runs use.
    The band integration evaluates the Planck function at band centres and
    truncates outside the spectral range, so the recovered flux is close to
    but not identical with `sigma * T_surf**4`. Pinning that agreement
    validates the spectral file, the band integration, and the surface
    boundary condition together.

    Analytical limit: Stefan-Boltzmann law with unit emissivity, recovered
    to better than 1e-4 relative across 1000 K to 3000 K.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = _make_runner(tmpdir, spectral_file=None, surf_greyalbedo=0.0)

        spfile = get_spfile_path(runner.directories['fwl'], runner.config)
        if not os.path.isfile(spfile):
            pytest.skip(f'spectral file not present in FWL_DATA: {spfile}')

        # AGNI copies the spectral file and inserts the solar and Planck
        # blocks, which needs a stellar spectrum in the output tree.
        os.makedirs(runner.directories['output/data'], exist_ok=True)
        wl_arr, fl_arr = _stellar_spectrum()
        spectrum_row = {'Time': 0.0, 'age_star': 4.6e9}
        write_spectrum(wl_arr, fl_arr, spectrum_row, runner.directories['output'])

        results = _sweep_transparent(runner)

        # The banded solve must use more than one band, otherwise the grey
        # scheme has been selected and the chain under test is not exercised.
        assert results[0][2]['nbands'] > 1

        _assert_blackbody_limit(results, rtol=RTOL_SOCRATES)

        # The band discretisation is a deficit, never a surplus: truncation
        # and midpoint evaluation only lose flux against the exact integral.
        for t_surf, output, _ in results:
            assert output['F_olr'] < SIGMA_SB * t_surf**4
