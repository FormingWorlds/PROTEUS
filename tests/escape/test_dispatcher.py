"""Unit tests for ``proteus.escape.dispatcher``, the coupling to the ZEPHYRUS dispatcher.

The dispatcher itself is validated in ZEPHYRUS; these tests check what PROTEUS
adds around it: the unit conversion of the settings, the atmosphere column it
is given, the helpfile record of its verdict, and the hysteresis memory that
record carries from one step to the next.

Invariants checked: the per-element rates sum to the bulk rate; the rate of a
state the dispatcher labels energy-limited equals the energy-limited formula
at the photospheric radius; pressures reach ZEPHYRUS in Pa.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from proteus.config._escape import Escape, Zephyrus
from proteus.escape.dispatcher import (
    PROFILE_ISOTHERMAL,
    PROFILE_KEPT,
    PROFILE_SOLVE,
    REGIME_CODES,
    dispatch_settings,
    escape_inputs,
    escape_profile,
    run_dispatcher,
    store_result,
)
from proteus.utils.constants import M_sun, element_list, gas_list

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

M_EARTH = 5.972e24  # kg
R_EARTH = 6.371e6  # m
AU = 1.495978707e11  # m


def _config(**zephyrus):
    """Config stub carrying the escape block, the star mass and the profile top."""
    return SimpleNamespace(
        escape=Escape(module='zephyrus', zephyrus=Zephyrus(**zephyrus)),
        star=SimpleNamespace(mass=1.0),
        atmos_clim=SimpleNamespace(module='agni', p_top=1e-10),
    )


def _row(F_xuv=10.0, T_eqm=800.0):
    """Helpfile row of a 5 Earth-mass, 1.5 Earth-radius H2-He planet at 0.1 au."""
    row = {f'{g}_vmr': 0.0 for g in gas_list}
    row.update({f'{e}_kg_atm': 0.0 for e in element_list})
    row.update(
        {
            'M_planet': 5.0 * M_EARTH,
            'R_int': 1.5 * R_EARTH,
            'semimajorax': 0.1 * AU,
            'eccentricity': 0.0,
            'T_eqm': T_eqm,
            'F_xuv': F_xuv,
            'F_ins': 1.0e5,
            'F_int': 1.0,
            'age_star': 1.0e8,
            'P_surf': 100.0,
            'H2_vmr': 0.83,
            'He_vmr': 0.17,
            'H_kg_atm': 1.0e21,
            'He_kg_atm': 3.0e20,
            'R_xuv': 0.0,
            'p_xuv': 0.0,
        }
    )
    return row


def _atmos(row, current=True):
    """Atmosphere state holding an isothermal H2-He column at the row's T_eqm."""
    from zephyrus.profiles import isothermal_profile

    prof = isothermal_profile(
        row['M_planet'],
        row['R_int'],
        row['T_eqm'],
        {'H2': 0.83, 'He': 0.17},
        row['P_surf'] * 1e5,
        1e-5,
        n_levels=150,
    )
    column = {'p': prof.p, 'r': prof.r, 'T': prof.T, 'vmr': prof.vmr}
    return SimpleNamespace(escape_profile=column, escape_profile_current=current)


def test_settings_reach_zephyrus_in_pa_with_the_tidal_default_resolved():
    """Pressures are configured in bar and passed in Pa; an unset tidal flag is on.

    ``P_photo`` = 0.02 bar is 2000 Pa and ``Pxuv`` = 5e-5 bar is 5 Pa. Passing
    the bar value unconverted would put the photosphere at 0.02 Pa, five
    decades too high, which is the mistake this guards against.
    """
    st = dispatch_settings(Zephyrus())
    assert st.P_photo == pytest.approx(2000.0, rel=1e-12)
    assert st.P_base_fixed == pytest.approx(5.0, rel=1e-12)
    assert st.tidal is True
    assert abs(st.P_photo - 0.02) > 1e3
    assert st.T_anchor_decades == pytest.approx(3.0, rel=1e-15)
    st_el = dispatch_settings(Zephyrus(scheme='el', P_photo=0.1, tidal=False, T_exo=1500.0))
    assert st_el.P_photo == pytest.approx(1.0e4, rel=1e-12)
    assert st_el.tidal is False
    assert st_el.T_exo_value == pytest.approx(1500.0, rel=1e-12)


def test_regime_codes_cover_every_zephyrus_label_once():
    """Every label ZEPHYRUS can return has one nonzero helpfile code.

    Zero is the code of a row the dispatcher did not set, so a label that
    mapped to zero would read back as no previous branch and silently drop the
    hysteresis memory.
    """
    from zephyrus.dispatcher import REGIME_LABELS

    assert set(REGIME_CODES) == set(REGIME_LABELS)
    codes = list(REGIME_CODES.values())
    assert len(set(codes)) == len(codes)
    assert min(codes) >= 1


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_energy_limited_state_matches_the_energy_limited_formula(validation_plot):
    """A state the dispatcher labels energy-limited escapes at the energy-limited rate.

    Physical scenario: a 5 Earth-mass, 1.5 Earth-radius planet with an
    isothermal 800 K H2-He envelope at 0.1 au, with the photon and
    recombination limits off so that the wind rate is the energy-limited one.
    Reference: the released ``zephyrus.escape.EL_escape`` with ``scaling = 2``,
    epsilon pi R_xuv^2 F_xuv R_p / (G M_p), at the radius of the 0.02 bar
    level of the column (Erkaev et al. 2007, without the tidal factor). The
    per-element rates must sum to the bulk rate, and the rate must scale
    linearly with the XUV flux. Discrimination: the R_xuv^3 form, which uses
    the XUV radius in place of R_p, sits above it by R_xuv / R_p.
    """
    from zephyrus.escape import EL_escape

    config = _config(photon_limit=False, recombination_limit=False, tidal=False)
    fluxes = np.array([1.0, 10.0, 100.0])
    rates, refs, wrong = [], [], []
    for F in fluxes:
        row = _row(F_xuv=F)
        atmos = _atmos(row)
        mdot = run_dispatcher(config, row, atmos_o=atmos, dt=1.0e3)
        col = atmos.escape_profile
        r_xuv = np.interp(np.log(2000.0), np.log(col['p'][::-1]), col['r'][::-1])
        ref = EL_escape(False, row['semimajorax'], 0.0, row['M_planet'], M_sun, 0.1,
                        row['R_int'], r_xuv, F, scaling=2)  # fmt: skip
        assert row['esc_regime'] == REGIME_CODES['hydrodynamic:EL']
        assert mdot == pytest.approx(ref, rel=1e-12)
        assert row['esc_rate_total'] == pytest.approx(mdot, rel=1e-15)
        total = sum(row[f'esc_rate_{e}'] for e in element_list)
        assert total == pytest.approx(mdot, rel=1e-12)
        assert row['esc_rate_H'] > row['esc_rate_He'] > 0.0
        rates.append(mdot)
        refs.append(ref)
        wrong.append(ref * r_xuv / row['R_int'])
    assert rates[2] / rates[0] == pytest.approx(100.0, rel=1e-9)
    assert all(abs(w / r - 1.0) > 0.05 for w, r in zip(wrong, refs))

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(fluxes, refs, '-', label=r'$\epsilon \pi R_\mathrm{xuv}^2 F R_p / (G M_p)$')
        ax.loglog(fluxes, wrong, '--', label=r'$R_\mathrm{xuv}^3$ form')
        ax.loglog(fluxes, rates, 'o', label='Dispatcher through PROTEUS')
        ax.set_xlabel(r'XUV flux $F_\mathrm{xuv}$ [W m$^{-2}$]')
        ax.set_ylabel(r'Escape rate [kg s$^{-1}$]')
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'escape_dispatcher_energy_limited')


@pytest.mark.physics_invariant
def test_without_a_solved_column_escape_reads_an_isothermal_one():
    """With no converged climate column, an isothermal one at T_eqm stands in.

    The column starts at the surface pressure and stops at ``p_top`` or where
    the gas becomes unbound, whichever comes first; its temperature is T_eqm
    throughout and pressure falls while radius rises.
    """
    row = _row()
    profile, source = escape_profile(_config(), row, None)
    assert source == PROFILE_ISOTHERMAL
    assert profile.p[0] == pytest.approx(row['P_surf'] * 1e5, rel=1e-12)
    assert profile.r[0] == pytest.approx(row['R_int'], rel=1e-12)
    np.testing.assert_allclose(profile.T, row['T_eqm'], rtol=1e-12)
    assert np.all(np.diff(profile.p) < 0.0)
    assert np.all(np.diff(profile.r) > 0.0)
    # Mean particle mass of 83 % H2 and 17 % He, about 2.36 amu.
    assert profile.mmw[0] == pytest.approx(2.36 * 1.6605e-27, rel=5e-3)

    empty = {k: (0.0 if k.endswith('_vmr') else v) for k, v in row.items()}
    assert escape_profile(_config(), empty, None)[0] is None
    thin = dict(row, P_surf=1e-12)  # below the 1e-10 bar top
    assert escape_profile(_config(), thin, None)[0] is None


def test_column_source_is_recorded():
    """A column kept from an earlier solve is read, and recorded as such.

    The record lets a reader see from the helpfile alone which escape rates
    rest on a column the latest climate solve did not confirm.
    """
    row = _row()
    _, fresh = escape_profile(_config(), row, _atmos(row, current=True))
    _, kept = escape_profile(_config(), row, _atmos(row, current=False))
    assert (fresh, kept) == (PROFILE_SOLVE, PROFILE_KEPT)

    run_dispatcher(_config(), row, atmos_o=_atmos(row, current=False), dt=1.0)
    assert row['esc_profile_source'] == PROFILE_KEPT


def test_the_rate_branch_is_remembered_for_the_next_step():
    """The branch that set the rate is read back as the next step's hysteresis memory.

    A row the dispatcher has not set (code zero, as on the first step or on
    resume from an older helpfile) gives no memory, so the switch is sharp.
    """
    config = _config()
    row = _row()
    atmos = _atmos(row)
    first = escape_inputs(config, row, escape_profile(config, row, atmos)[0], 1.0)
    assert first.prev_regime is None

    run_dispatcher(config, row, atmos_o=atmos, dt=1.0)
    code = row['esc_rate_branch']
    label = {v: k for k, v in REGIME_CODES.items()}[int(code)]
    second = escape_inputs(config, row, escape_profile(config, row, atmos)[0], 1.0)
    assert second.prev_regime == label
    assert second.dt == pytest.approx(1.0 * 365.25 * 86400.0, rel=1e-12)


def test_a_rate_for_an_untracked_element_is_refused():
    """An element PROTEUS does not track has no inventory to debit, so it fails loudly.

    Dropping it would let the per-element rates fall short of the bulk rate
    without a trace. The row is left as it was.
    """
    result = SimpleNamespace(
        regime='hydrostatic',
        rate_branch='hydrostatic',
        mdot=2.0,
        per_species={'H': 1.0, 'Xx': 1.0},
        flags={},
        diagnostics={},
    )
    row = {'esc_rate_total': 0.0}
    with pytest.raises(ValueError, match='untracked elements') as err:
        store_result(row, result, PROFILE_SOLVE)
    assert "'Xx'" in str(err.value)
    assert "'H'" not in str(err.value)
    assert row == {'esc_rate_total': 0.0}


def test_diagnostics_the_dispatcher_did_not_evaluate_read_as_minus_one():
    """A missing or infinite diagnostic is written as -1, never as inf or nan.

    An infinite sonic-point Knudsen number (no wind evaluated) would otherwise
    reach the helpfile, where it breaks plots and comparisons.
    """
    result = SimpleNamespace(
        regime='boiloff',
        rate_branch='boiloff',
        mdot=5.0,
        per_species={'H': 5.0},
        flags={},
        diagnostics={
            'knudsen': {'kn_sc': float('inf')},
            'base_level': {'p_Pa': 1e-4, 'r_m': 2.0e7, 'clamp_decades': None},
        },
    )
    row = {'R_xuv': 0.0, 'p_xuv': 0.0}
    store_result(row, result, PROFILE_SOLVE)
    assert row['esc_kn_sonic'] == pytest.approx(-1.0)
    assert row['esc_T_exo'] == pytest.approx(-1.0)
    assert row['esc_base_clamp_dex'] == pytest.approx(0.0)
    assert row['p_xuv'] == pytest.approx(1e-9, rel=1e-12)
    assert row['R_xuv'] == pytest.approx(2.0e7, rel=1e-12)
    assert row['esc_regime'] == REGIME_CODES['boiloff']


def test_every_setting_reaches_zephyrus_with_its_own_value():
    """Each dispatcher setting is passed under its own name, with pressures in Pa.

    Every field takes a value distinct from its default and from the other
    fields of its type, so a swapped or hard-coded field fails.
    """
    z = Zephyrus(
        base_method='fixed_pressure',
        base_out_of_range='extend',
        P_photo=0.05,
        Pxuv=2e-4,
        kn_crit=0.7,
        kn_hysteresis=2.5,
        gate='plasma',
        efficiency=0.23,
        efficiency_mode='caldiroli',
        T_exo_mode='thermostat',
        T_exo=1234.0,
        cool_atomic=False,
        cool_co2_band=True,
        cool_o_finestructure=False,
        cool_recombination=True,
        fractionate=False,
        tidal=False,
        photon_limit=False,
        recombination_limit=True,
        nozzle_temperature='wind',
        residual_mode='luminosity_capped',
        lambda_crit=17.0,
        kzz=55.0,
        hydrostatic_levels_min=50,
        hydrostatic_levels_max=900,
        hydrostatic_rtol=0.03,
        T_anchor_decades=1.5,
    )
    st = dispatch_settings(z)
    expected = dict(
        base_method='fixed_pressure',
        base_out_of_range='extend',
        gate='plasma',
        efficiency_mode='caldiroli',
        T_exo_mode='thermostat',
        cool_atomic=False,
        cool_co2_band=True,
        cool_o_finestructure=False,
        cool_recombination=True,
        fractionate=False,
        tidal=False,
        photon_limit=False,
        recombination_limit=True,
        nozzle_temperature='wind',
        residual_mode='luminosity_capped',
        hydrostatic_levels_min=50,
        hydrostatic_levels_max=900,
    )
    for name, value in expected.items():
        assert getattr(st, name) == value, name
    floats = dict(
        P_photo=5000.0,
        P_base_fixed=20.0,
        kn_crit=0.7,
        kn_hysteresis=2.5,
        efficiency=0.23,
        T_exo_value=1234.0,
        lambda_crit=17.0,
        kzz=55.0,
        hydrostatic_rtol=0.03,
        T_anchor_decades=1.5,
    )
    for name, value in floats.items():
        assert getattr(st, name) == pytest.approx(value, rel=1e-12), name


def test_inputs_carry_the_planet_in_si():
    """The dispatcher inputs are the row's state in SI, with the fluxes floored.

    The stellar mass arrives in kg, the age and step in seconds, the
    bolometric flux from F_ins, the negative interior flux and a zero
    instellation at the floor, the eccentricity and opacity as given, and the
    reservoirs without the elements the atmosphere does not hold.
    """
    from proteus.escape.dispatcher import F_FLOOR

    config = _config(kappa_photo=3e-3)
    row = dict(_row(), eccentricity=0.2, F_int=-4.0, F_ins=0.0, age_star=2.0e8)
    profile, _source = escape_profile(config, row, _atmos(row))
    inp = escape_inputs(config, row, profile, 10.0)
    assert inp.M_star == pytest.approx(M_sun, rel=1e-12)
    assert inp.age == pytest.approx(2.0e8 * 365.25 * 86400.0, rel=1e-12)
    assert inp.dt == pytest.approx(10.0 * 365.25 * 86400.0, rel=1e-12)
    assert inp.F_bol == pytest.approx(F_FLOOR, rel=1e-12)
    assert inp.F_int == pytest.approx(F_FLOOR, rel=1e-12)
    assert inp.F_xuv == pytest.approx(10.0, rel=1e-12)
    assert inp.e == pytest.approx(0.2, rel=1e-12)
    assert inp.kappa_photo == pytest.approx(3e-3, rel=1e-12)
    assert set(inp.reservoirs) == {'H', 'He'}
    assert inp.reservoirs['H'] == pytest.approx(1.0e21, rel=1e-12)
    assert inp.M_p == pytest.approx(5.0 * M_EARTH, rel=1e-12)
    assert inp.R_p == pytest.approx(1.5 * R_EARTH, rel=1e-12)


@pytest.mark.physics_invariant
def test_tidal_eccentric_state_matches_the_tidal_energy_limited_formula():
    """With the tidal factor on an eccentric orbit, the rate is the tidal energy-limited one.

    Physical scenario: the energy-limited state of the reference test at
    e = 0.3, where the Hill radius at periapsis is about 27 planetary radii and
    the Erkaev tidal factor lowers the barrier by about 6%. The reference is
    ``EL_escape`` with the tidal factor. The stellar mass enters through the
    Hill radius, so a mass passed in solar masses instead of kg would put the
    factor at 1 and miss by about 6%.
    """
    from zephyrus.escape import EL_escape

    config = _config(photon_limit=False, recombination_limit=False, tidal=True)
    row = dict(_row(F_xuv=10.0), eccentricity=0.3)
    atmos = _atmos(row)
    mdot = run_dispatcher(config, row, atmos_o=atmos, dt=1.0e3)
    col = atmos.escape_profile
    r_xuv = np.interp(np.log(2000.0), np.log(col['p'][::-1]), col['r'][::-1])
    tidal = EL_escape(True, row['semimajorax'], 0.3, row['M_planet'], M_sun, 0.1,
                      row['R_int'], r_xuv, 10.0, scaling=2)  # fmt: skip
    plain = EL_escape(False, row['semimajorax'], 0.3, row['M_planet'], M_sun, 0.1,
                      row['R_int'], r_xuv, 10.0, scaling=2)  # fmt: skip
    assert row['esc_regime'] == REGIME_CODES['hydrodynamic:EL']
    assert mdot == pytest.approx(tidal, rel=1e-9)
    assert tidal / plain > 1.03


def test_a_renamed_state_records_its_rate_branch_and_every_diagnostic():
    """The row keeps the branch that set the rate apart from the label, with each diagnostic.

    A wind relabeled as Roche-lobe overflow keeps hydrodynamic:EL as its rate
    branch, which is the hysteresis memory of the next step. Each diagnostic
    takes a distinct value, so a crossed key fails.
    """
    result = SimpleNamespace(
        regime='roche_overflow',
        rate_branch='hydrodynamic:EL',
        mdot=7.0,
        per_species={'H': 4.0, 'O': 3.0},
        flags={},
        diagnostics={
            'knudsen': {'kn_sc': 0.11},
            'lambda_gate': 23.0,
            'hydrostatic': {
                'T_exo': 1100.0,
                'T_esc_neutral': 5300.0,
                'T_esc_plasma': 2650.0,
                'f_plus_exo': 0.004,
            },
            'base_level': {'p_Pa': 2.0e-4, 'r_m': 9.0e6, 'clamp_decades': 0.7},
        },
    )
    row = {}
    store_result(row, result, PROFILE_KEPT)
    assert row['esc_regime'] == REGIME_CODES['roche_overflow']
    assert row['esc_rate_branch'] == REGIME_CODES['hydrodynamic:EL']
    got = {k: row[k] for k in ('esc_kn_sonic', 'esc_lambda', 'esc_T_exo', 'esc_T_esc_neutral',
                               'esc_T_esc_plasma', 'esc_f_ion_exo', 'esc_base_clamp_dex')}  # fmt: skip
    want = dict(esc_kn_sonic=0.11, esc_lambda=23.0, esc_T_exo=1100.0, esc_T_esc_neutral=5300.0,
                esc_T_esc_plasma=2650.0, esc_f_ion_exo=0.004, esc_base_clamp_dex=0.7)  # fmt: skip
    assert got == pytest.approx(want, rel=1e-12)
    assert row['esc_profile_source'] == PROFILE_KEPT
    assert row['p_xuv'] == pytest.approx(2.0e-9, rel=1e-12)
    assert row['esc_rate_H'] + row['esc_rate_O'] == pytest.approx(7.0, rel=1e-12)

    no_base = SimpleNamespace(**{**vars(result), 'diagnostics': {}})
    kept = {'p_xuv': 5e-5, 'R_xuv': 7.0e6}
    store_result(kept, no_base, PROFILE_SOLVE)
    assert kept['p_xuv'] == pytest.approx(5e-5, rel=1e-15)
    assert kept['R_xuv'] == pytest.approx(7.0e6, rel=1e-15)


def test_no_atmosphere_or_a_broken_flux_means_no_escape_this_step(caplog):
    """A step with no column to escape from, or a non-finite flux, records no escape.

    Below the column top there is no atmosphere for the dispatcher to read,
    and a NaN flux means an upstream failure; either way the bulk and element
    rates are zero and the verdict codes read "not set". The NaN case warns.
    """
    import logging

    config = _config()
    thin = dict(_row(), P_surf=1e-12)
    assert run_dispatcher(config, thin, atmos_o=None, dt=1.0) == 0.0
    assert thin['esc_rate_total'] == 0.0 and thin['esc_rate_H'] == 0.0
    assert (thin['esc_regime'], thin['esc_profile_source']) == (0.0, 0.0)
    for key in ('F_xuv', 'F_ins', 'F_int'):
        broken = dict(_row(), **{key: float('nan')})
        with caplog.at_level(logging.WARNING):
            assert run_dispatcher(config, broken, atmos_o=_atmos(_row()), dt=1.0) == 0.0
        assert key in caplog.text
        assert broken['esc_rate_total'] == 0.0
        caplog.clear()
    # A finite row escapes, so the zeros above are the guard and not the planet.
    assert run_dispatcher(config, _row(), atmos_o=_atmos(_row()), dt=1.0) > 1.0
