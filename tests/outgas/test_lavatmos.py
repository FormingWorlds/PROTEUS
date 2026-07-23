"""Tests for proteus.outgas.lavatmos rock-vapour outgassing.

Covers run_vapourisation (the routine that combines the volatile
solve with the LavAtmos+FastChem rock-vapour re-equilibration into one
atmospheric composition) and set_magmaproperties. External LavAtmos / FastChem
calls are mocked at the narrowest scope: run_lavatmos is replaced by a fake that
writes a minimal boa_chem.dat, read_in_element_fracs_normalized returns a
controlled element-fraction dict, and vol_list / vap_list / element_list /
species_lib are replaced by small controlled sets.

Invariants exercised:
- atmospheric mass from the hydrostatic column relation M = P_surf * A / g;
- surface-pressure split P_vol + P_vap == P_surf with P_vap >= 0 (clamp);
- self-consistent atmospheric molar mass atm_kg_per_mol == mu_outgassed;
- outgassed rock-vapour mass M_vaps >= 0; combined VMRs in [0, 1];
- the temperature floor in set_magmaproperties.

See docs/How-to/testing.md and docs/Explanations/test_framework.md.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

import proteus.outgas.lavatmos as lavatmos_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Matches the constant used inside lavatmos.run_vapourisation so the
# in-test recomputation of the hydrostatic column mass agrees to float precision.
_G_CONST = 6.67430e-11


class _DummySpecies:
    """Stand-in for lavatmos.Species_db with a molar weight in g/mol."""

    def __init__(self, name, fc_name, weight):
        self.name = name
        self.fc_name = fc_name
        self.weight = weight


def _make_hf_row():
    """Minimal helpfile row with the keys run_vapourisation reads.

    R_int is stored in cm (as elsewhere in PROTEUS); M_planet and R_int are
    Earth-like so surface gravity is ~9.8 m/s^2 and the column mass is a
    physically plausible ~1e19 kg for a 2 bar atmosphere.
    """
    hf_row = {
        'H_kg_atm': 1.0,
        'C_kg_atm': 1.0,
        'N_kg_atm': 0.0,
        'S_kg_atm': 0.0,
        'O_kg_atm': 0.0,
        'P_surf': 1.0,  # bar (volatile-only, pre-vapour)
        'T_magma': 2000.0,  # K
        'atm_kg_per_mol': 0.029,  # kg/mol; must be overwritten by the vapour solve
        'M_atm': 1e18,  # kg (previous mass; must NOT be used by the new formula)
        'M_planet': 5.97e24,  # kg (Earth)
        'R_int': 6.37e8,  # cm (Earth radius)
        'M_vaps': 0.0,
    }
    for vol in ('CO2', 'H2O'):
        hf_row.setdefault(vol + '_kg_solid', 0.0)
        hf_row.setdefault(vol + '_kg_liquid', 0.0)
        hf_row.setdefault(vol + '_mol_solid', 0.0)
        hf_row.setdefault(vol + '_mol_liquid', 0.0)
    return hf_row


def _install_common_mocks(monkeypatch, element_fracs, env_dir):
    """Replace the module rosters and the element-fraction reader with small
    controlled values. vap_list=[H2O] (treated as
    the rock-vapour species here); element_list carries two rock-forming
    elements (Fe, Mg) so the M_vaps accumulation path is exercised. LAVA_DIR /
    FC_DIR are set to a temp dir so paths_importer builds without needing a real
    LavAtmos/FastChem install (CI runs without those env vars)."""
    monkeypatch.setenv('LAVA_DIR', str(env_dir))
    monkeypatch.setenv('FC_DIR', str(env_dir))
    monkeypatch.setattr(lavatmos_mod, 'vap_list', ['H2O'])
    monkeypatch.setattr(lavatmos_mod, 'noble_gases', [])
    monkeypatch.setattr(lavatmos_mod, 'element_list', ['H', 'C', 'N', 'S', 'O', 'Fe', 'Mg'])
    species_lib = {
        'H': _DummySpecies('H', 'H', 1.0),
        'C': _DummySpecies('C', 'C', 12.0),
        'N': _DummySpecies('N', 'N', 14.0),
        'S': _DummySpecies('S', 'S', 32.0),
        'O': _DummySpecies('O', 'O', 16.0),
        'Fe': _DummySpecies('Fe', 'Fe', 55.85),
        'Mg': _DummySpecies('Mg', 'Mg', 24.3),
        'CO2': _DummySpecies('CO2', 'C1O2', 44.0),
        'H2O': _DummySpecies('H2O', 'H2O1', 18.0),
        'O2': _DummySpecies('O2', 'O2', 32.0),
    }
    monkeypatch.setattr(lavatmos_mod, 'species_lib', species_lib)
    monkeypatch.setattr(
        lavatmos_mod, 'read_in_element_fracs_normalized', lambda path: dict(element_fracs)
    )


def _fake_run_lavatmos_factory(pbar, mu, co2_vmr, h2o_vmr, o2_vmr):
    """Return a run_lavatmos replacement that writes a minimal boa_chem.dat."""

    def _fake(dirs_arg, config, hf_row_arg, nfrac, first_iter):
        paths = lavatmos_mod.paths_importer(dirs_arg)
        boa_path = os.path.join(paths.fastchem3_output, 'boa_chem.dat')
        with open(boa_path, 'w') as f:
            f.write('Pbar mu C1O2 H2O1 O2\n')
            f.write(f'{pbar} {mu} {co2_vmr} {h2o_vmr} {o2_vmr}\n')

    return _fake


@pytest.mark.physics_invariant
def test_run_vapourisation_combines_into_single_composition(tmp_path, monkeypatch):
    """Volatile + rock-vapour results combine into one atmosphere with a
    hydrostatic-column mass, a self-consistent molar mass, and a P_vol/P_vap
    split that sums back to P_surf.

    The FastChem total is 2.0 bar (> the 1.0 bar volatile input), mu = 30 g/mol,
    and the rock elements Fe/Mg carry the vapour mass. Pins M_atm to P*A/g and
    guards against the previous pressure-ratio formula.
    """
    dirs = {'output': str(tmp_path)}
    hf_row = _make_hf_row()
    element_fracs = {'H': 0.0, 'C': 0.0, 'N': 0.0, 'S': 0.0, 'O': 0.1, 'Fe': 0.6, 'Mg': 0.3}
    _install_common_mocks(monkeypatch, element_fracs, tmp_path)
    monkeypatch.setattr(
        lavatmos_mod,
        'run_lavatmos',
        _fake_run_lavatmos_factory(pbar=2.0, mu=30.0, co2_vmr=0.5, h2o_vmr=0.4999, o2_vmr=1e-6),
    )

    lavatmos_mod.run_vapourisation(dirs, config=None, hf_row=hf_row, first_iter=True)

    # New total surface pressure comes straight from the FastChem table.
    assert hf_row['P_surf'] == pytest.approx(2.0, rel=1e-12)

    # Hydrostatic column mass M = P_surf * A / g, recomputed from the fixture.
    r_m = hf_row['R_int'] * 1e-2
    gravity = _G_CONST * hf_row['M_planet'] / r_m**2
    expected_M = (2.0 * 1e5) * (4.0 * np.pi * r_m**2) / gravity
    assert hf_row['M_atm'] == pytest.approx(expected_M, rel=1e-9)
    # Exponent/factor guard: the old pressure-ratio formula would give
    # M_atm_old * Pbar_new / P_surf_old = 1e18 * 2 = 2e18 kg, which is ~5x
    # smaller than the column mass (~1e19 kg); the two must not be confusable.
    assert abs(hf_row['M_atm'] - 2e18) > 5e18
    # Sign and scale guards (kg, Earth-like 2 bar atmosphere).
    assert hf_row['M_atm'] > 0.0
    assert 1e18 < hf_row['M_atm'] < 1e20

    # Pressure split: P_vap is the excess over the volatile input (2 - 1 = 1 bar),
    # both parts non-negative, and they sum back to the new total.
    assert hf_row['P_vap'] == pytest.approx(1.0, rel=1e-12)
    assert hf_row['P_vap'] >= 0.0
    assert hf_row['P_vol'] + hf_row['P_vap'] == pytest.approx(hf_row['P_surf'], rel=1e-12)

    # Molar mass updated to the combined FastChem value (30 g/mol -> 0.030 kg/mol),
    # not left at the stale volatile-only 0.029 kg/mol fixture value.
    assert hf_row['atm_kg_per_mol'] == pytest.approx(0.030, rel=1e-6)
    assert hf_row['atm_kg_per_mol'] != pytest.approx(0.029, rel=1e-3)

    # Combined VMRs read back from FastChem are valid mole fractions.
    for vmr in (hf_row['CO2_vmr'], hf_row['H2O_vmr']):
        assert 0.0 <= vmr <= 1.0
    assert hf_row['CO2_kg_atm'] > 0.0

    # Rock-vapour bookkeeping: M_vaps accumulates the non-volatile element mass
    # and is non-negative; Fe (a rock-former) gets an atmospheric mass.
    assert hf_row['M_vaps'] >= 0.0
    assert hf_row['Fe_kg_atm'] > 0.0

    # fO2 diagnostics recorded.
    assert 'log10_fO2_vapourise' in hf_row
    assert math.isfinite(hf_row['log10_fO2_shift_vapourise'])


@pytest.mark.physics_invariant
def test_run_vapourisation_clamps_negative_P_vap(tmp_path, monkeypatch):
    """When the FastChem total falls below the volatile input pressure, the
    rock-vapour pressure would be negative; it is clamped to zero and P_vol is
    derived so P_vol + P_vap still equals the (new) surface pressure.

    Edge case: volatile input 5 bar but FastChem total only 2 bar.
    """
    dirs = {'output': str(tmp_path)}
    hf_row = _make_hf_row()
    hf_row['P_surf'] = 5.0  # volatile input above the FastChem total
    element_fracs = {'H': 0.0, 'C': 0.0, 'N': 0.0, 'S': 0.0, 'O': 0.1, 'Fe': 0.6, 'Mg': 0.3}
    _install_common_mocks(monkeypatch, element_fracs, tmp_path)
    monkeypatch.setattr(
        lavatmos_mod,
        'run_lavatmos',
        _fake_run_lavatmos_factory(pbar=2.0, mu=30.0, co2_vmr=0.5, h2o_vmr=0.4999, o2_vmr=1e-6),
    )

    lavatmos_mod.run_vapourisation(dirs, config=None, hf_row=hf_row, first_iter=False)

    # P_vap clamped to zero, never negative.
    assert hf_row['P_vap'] == pytest.approx(0.0, abs=1e-30)
    assert hf_row['P_vap'] >= 0.0
    # Identity preserved against the NEW total surface pressure (2 bar).
    assert hf_row['P_surf'] == pytest.approx(2.0, rel=1e-12)
    assert hf_row['P_vol'] + hf_row['P_vap'] == pytest.approx(hf_row['P_surf'], rel=1e-12)


def test_run_vapourisation_missing_fastchem_output_raises(tmp_path, monkeypatch):
    """A missing FastChem output directory triggers UpdateStatusfile and a
    RuntimeError rather than a silent skip.

    Error-contract path: read_in_element_fracs_normalized is stubbed so the
    routine reaches the fastchem-output existence check, which is forced False.
    """
    dirs = {'output': str(tmp_path)}
    hf_row = _make_hf_row()
    element_fracs = {'H': 0.0, 'C': 0.0, 'N': 0.0, 'S': 0.0, 'O': 0.1, 'Fe': 0.6, 'Mg': 0.3}
    _install_common_mocks(monkeypatch, element_fracs, tmp_path)
    monkeypatch.setattr(lavatmos_mod, 'run_lavatmos', lambda *a, **k: None)
    # Force the fastchem-output directory check to fail.
    monkeypatch.setattr(lavatmos_mod.os.path, 'exists', lambda p: False)

    called = {'status': False}

    def _fake_update(dirs_arg, code):
        called['status'] = True

    monkeypatch.setattr(lavatmos_mod, 'UpdateStatusfile', _fake_update)

    with pytest.raises(RuntimeError, match='fastchem'):
        lavatmos_mod.run_vapourisation(dirs, config=None, hf_row=hf_row, first_iter=True)
    assert called['status'] is True


@pytest.mark.physics_invariant
def _magma_config(t_min=1500.0, melt_comp_name='BSE_palm'):
    """Config stub exposing the outgas.lavatmos fields set_magmaproperties reads."""
    return SimpleNamespace(
        outgas=SimpleNamespace(
            lavatmos=SimpleNamespace(T_min=t_min, melt_comp_name=melt_comp_name)
        )
    )


def test_set_magmaproperties_temperature_floor(monkeypatch):
    """set_magmaproperties clamps the surface temperature up to the configured
    floor, passes surface pressure through, and takes the melt composition name
    from config.

    T_magma above the floor is kept; below is raised to the floor. The floor is
    the configured value, not a hardcoded 1500 K: a 1200 K config floor keeps a
    1300 K magma but raises a 1000 K magma to 1200 K.
    """
    monkeypatch.setattr(
        lavatmos_mod, 'paths_importer', lambda dirs: SimpleNamespace(output_dir='/tmp/x')
    )
    dirs = {'output': '/tmp/x'}

    cfg = _magma_config(t_min=1500.0, melt_comp_name='BSE_palm')
    hot = lavatmos_mod.set_magmaproperties(
        config=cfg, hf_row={'T_magma': 2000.0, 'P_surf': 3.5}, volatile_comp={}, dirs=dirs
    )
    assert hot.T_surf == pytest.approx(2000.0, rel=1e-12)
    assert hot.P_volatile == pytest.approx(3.5, rel=1e-12)
    assert hot.melt_comp_name == 'BSE_palm'
    assert hot.T_surf > 0.0

    cold = lavatmos_mod.set_magmaproperties(
        config=cfg, hf_row={'T_magma': 1000.0, 'P_surf': 0.2}, volatile_comp={}, dirs=dirs
    )
    # Floored at the configured 1500 K, not the input 1000 K.
    assert cold.T_surf == pytest.approx(1500.0, rel=1e-12)
    assert cold.T_surf != pytest.approx(1000.0, rel=1e-3)
    assert cold.T_surf > 0.0

    # Floor is configurable: a 1200 K floor keeps 1300 K but raises 1000 K to 1200 K.
    cfg_low = _magma_config(t_min=1200.0, melt_comp_name='custom_melt')
    warm = lavatmos_mod.set_magmaproperties(
        config=cfg_low, hf_row={'T_magma': 1300.0, 'P_surf': 1.0}, volatile_comp={}, dirs=dirs
    )
    assert warm.T_surf == pytest.approx(1300.0, rel=1e-12)  # above the 1200 floor
    assert warm.melt_comp_name == 'custom_melt'  # honoured from config
    floored_low = lavatmos_mod.set_magmaproperties(
        config=cfg_low, hf_row={'T_magma': 1000.0, 'P_surf': 1.0}, volatile_comp={}, dirs=dirs
    )
    assert floored_low.T_surf == pytest.approx(1200.0, rel=1e-12)
    # Discrimination: a hardcoded-1500 regression would floor this to 1500, not 1200.
    assert floored_low.T_surf != pytest.approx(1500.0, rel=1e-3)


@pytest.mark.physics_invariant
def test_run_vapourisation_preserves_noble_gases(tmp_path, monkeypatch):
    """Noble gases pass through the rock-vapour step: they are read back from the
    combined FastChem equilibrium (not dropped) and excluded from the rock-vapour
    mass M_vaps (they are inert atmospheric gas, not vaporised rock). A noble
    absent from the FastChem output keeps its prior value rather than crashing.

    Edge case: He is emitted by FastChem (updated); Ar is not (guard keeps its
    prior value). M_vaps must count only the rock-forming Fe (+ extra O), never
    the noble mass.
    """
    dirs = {'output': str(tmp_path)}
    monkeypatch.setenv('LAVA_DIR', str(tmp_path))
    monkeypatch.setenv('FC_DIR', str(tmp_path))
    monkeypatch.setattr(lavatmos_mod, 'vap_list', ['Fe'])  # a rock-vapour species
    monkeypatch.setattr(lavatmos_mod, 'noble_gases', ['He', 'Ar'])
    monkeypatch.setattr(
        lavatmos_mod, 'element_list', ['H', 'C', 'N', 'S', 'O', 'Fe', 'He', 'Ar']
    )
    species_lib = {
        'H': _DummySpecies('H', 'H', 1.0),
        'C': _DummySpecies('C', 'C', 12.0),
        'N': _DummySpecies('N', 'N', 14.0),
        'S': _DummySpecies('S', 'S', 32.0),
        'O': _DummySpecies('O', 'O', 16.0),
        'Fe': _DummySpecies('Fe', 'Fe', 55.85),
        'He': _DummySpecies('He', 'He', 4.0),
        'Ar': _DummySpecies('Ar', 'Ar', 39.95),
        'CO2': _DummySpecies('CO2', 'C1O2', 44.0),
        'O2': _DummySpecies('O2', 'O2', 32.0),
    }
    monkeypatch.setattr(lavatmos_mod, 'species_lib', species_lib)
    # Nonzero He/Ar fractions so a regression that wrongly folded them into M_vaps
    # would be detectable (they contribute mass only if incorrectly counted).
    element_fracs = {
        'H': 0.0,
        'C': 0.0,
        'N': 0.0,
        'S': 0.0,
        'O': 0.2,
        'Fe': 0.7,
        'He': 0.05,
        'Ar': 0.05,
    }
    monkeypatch.setattr(
        lavatmos_mod, 'read_in_element_fracs_normalized', lambda p: dict(element_fracs)
    )

    def fake_run(dirs_arg, config, hf_row_arg, nfrac, first_iter):
        paths = lavatmos_mod.paths_importer(dirs_arg)
        with open(os.path.join(paths.fastchem3_output, 'boa_chem.dat'), 'w') as f:
            # Includes an He column but deliberately NO Ar column.
            f.write('Pbar mu C1O2 Fe He O2\n')
            f.write('2.0 30.0 0.29 0.2 0.5 1e-6\n')

    monkeypatch.setattr(lavatmos_mod, 'run_lavatmos', fake_run)

    hf_row = _make_hf_row()
    hf_row['He_kg_atm'] = 5.0e15
    hf_row['Ar_kg_atm'] = 1.0e15
    hf_row['He_vmr'] = 0.0
    hf_row['Ar_vmr'] = 0.123  # prior value; must survive (Ar not in FastChem output)
    for s in ('CO2', 'Fe', 'He'):
        for suff in ('_kg_solid', '_kg_liquid', '_mol_solid', '_mol_liquid'):
            hf_row.setdefault(s + suff, 0.0)

    lavatmos_mod.run_vapourisation(dirs, config=None, hf_row=hf_row, first_iter=True)

    # He is read back from the combined FastChem equilibrium: not dropped.
    assert hf_row['He_vmr'] == pytest.approx(0.5, rel=1e-9)
    assert hf_row['He_bar'] > 0.0
    assert hf_row['He_kg_atm'] > 0.0
    # He is atmospheric-only, so kg_total equals kg_atm (not zeroed like rock vapour).
    assert hf_row['He_kg_total'] == pytest.approx(hf_row['He_kg_atm'], rel=1e-12)
    # Ar absent from FastChem output: prior value preserved, no KeyError.
    assert hf_row['Ar_vmr'] == pytest.approx(0.123, rel=1e-12)

    # M_vaps counts only rock-forming Fe plus the extra outgassed O, never the
    # noble gases. Recompute the expected value from the same fractions/weights.
    mmw = sum(element_fracs[e] * species_lib[e].weight for e in element_fracs)
    m_atm = hf_row['M_atm']
    fe_term = element_fracs['Fe'] * m_atm * species_lib['Fe'].weight / mmw
    o_term = element_fracs['O'] * m_atm * species_lib['O'].weight / mmw  # O_kg_atm was 0
    expected_m_vaps = fe_term + o_term
    assert hf_row['M_vaps'] == pytest.approx(expected_m_vaps, rel=1e-9)
    # Discrimination: had He+Ar been folded in, M_vaps would rise by their mass
    # terms to essentially the full M_atm; the correct value is strictly below it.
    noble_term = (
        element_fracs['He'] * m_atm * species_lib['He'].weight / mmw
        + element_fracs['Ar'] * m_atm * species_lib['Ar'].weight / mmw
    )
    assert abs(hf_row['M_vaps'] - (expected_m_vaps + noble_term)) > 0.5 * noble_term
    assert hf_row['M_vaps'] >= 0.0
