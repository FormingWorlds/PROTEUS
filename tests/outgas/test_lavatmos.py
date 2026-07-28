"""Tests for proteus.outgas.lavatmos rock vapourisation.

Covers run_vapourisation (the routine that combines the volatile
solve with the LavAtmos+FastChem rock-vapour re-equilibration into one
atmospheric composition) and set_magmaproperties. External LavAtmos / FastChem
calls are mocked at the narrowest scope: run_lavatmos is replaced by a fake that
writes a minimal boa_chem.dat, read_in_element_fracs_normalized returns a
controlled element-fraction dict, and vap_list / element_list /
species_lib are replaced by small controlled sets.

Invariants exercised:
- atmospheric mass from the hydrostatic column relation M = P_surf * A / g;
- surface-pressure split P_vol + P_vap == P_surf with P_vap >= 0 (clamp);
- self-consistent atmospheric molar mass atm_kg_per_mol == the combined mu;
- vapourised rock mass M_vaps >= 0; combined VMRs in [0, 1];
- the temperature floor in set_magmaproperties.

See docs/How-to/testing.md and docs/Explanations/test_framework.md.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
from calliope.oxygen_fugacity import OxygenFugacity

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

    R_int is stored in metres and gravity in m/s^2, matching the helpfile
    schema. The values are Earth-like and mutually consistent
    (g = G*M_planet/R_int**2 to within 0.1%), so the hydrostatic column mass
    is a physically plausible ~1e19 kg for a 2 bar atmosphere.
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
        'R_int': 6.371e6,  # m (Earth radius)
        'gravity': 9.81,  # m s-2 (surface, set by the interior structure module)
        'M_vaps': 0.0,
    }
    # Mantle reservoirs for every species the stub FastChem tables emit (O2
    # included: it carries the vapour-step fO2 and so is always present).
    for vol in ('CO2', 'H2O', 'O2'):
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
    # gas_list must be mocked alongside the other rosters: the real one carries
    # monatomic Fe/Mg as gas species, which would route the rock-forming
    # elements down the gas branch and leave Fe_kg_atm unset here.
    monkeypatch.setattr(lavatmos_mod, 'gas_list', ['CO2', 'H2O', 'O2'])
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

    # Set up fake LavAtmos directory structure required by paths_importer
    env_dir = tmp_path
    (env_dir / 'input' / 'lava_compositions').mkdir(parents=True)

    monkeypatch.setenv('LAVA_DIR', str(env_dir))
    monkeypatch.setenv('FC_DIR', str(tmp_path / 'fastchem'))

    hf_row = _make_hf_row()
    element_fracs = {'H': 0.0, 'C': 0.0, 'N': 0.0, 'S': 0.0, 'O': 0.1, 'Fe': 0.6, 'Mg': 0.3}
    _install_common_mocks(monkeypatch, element_fracs, tmp_path)
    monkeypatch.setattr(
        lavatmos_mod,
        'run_lavatmos',
        _fake_run_lavatmos_factory(pbar=2.0, mu=30.0, co2_vmr=0.5, h2o_vmr=0.4999, o2_vmr=1e-6),
    )

    lavatmos_mod.run_vapourisation(dirs, config=_magma_config(), hf_row=hf_row, first_iter=True)

    # New total surface pressure comes straight from the FastChem table.
    assert hf_row['P_surf'] == pytest.approx(2.0, rel=1e-12)

    # Hydrostatic column mass M = P_surf * A / g, recomputed from the fixture.
    # The fixture's gravity must be the surface value the interior module wrote,
    # not one re-derived here, so check the two agree before pinning M_atm.
    r_m = hf_row['R_int']
    gravity = hf_row['gravity']
    assert gravity == pytest.approx(_G_CONST * hf_row['M_planet'] / r_m**2, rel=1e-3)
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

    # Derived fO2 is the O2 partial pressure of the COMBINED atmosphere:
    # the FastChem O2 mixing ratio times the total pressure P_vol + P_vap,
    # not times the rock-vapour part alone. With VMR(O2) = 1e-6 and 2 bar
    # total this is log10(2e-6) = -5.699 log10 bar.
    p_O2 = 1e-6 * (hf_row['P_vol'] + hf_row['P_vap'])
    assert hf_row['fO2_vapourise_derived'] == pytest.approx(np.log10(p_O2), rel=1e-12)
    # Discrimination: dropping the pressure factor (log10(1e-6) = -6.0) or using
    # the 1 bar rock-vapour part alone (also -6.0) both sit 0.30 dex away, far
    # outside the tolerance, so the total-pressure factor is really exercised.
    assert abs(hf_row['fO2_vapourise_derived'] - np.log10(1e-6)) > 0.2

    # The IW offset is the same quantity measured against the buffer at T_magma.
    iw_2000 = OxygenFugacity(model='oneill')(hf_row['T_magma'])
    assert hf_row['fO2_vapourise_shift_IW_derived'] == pytest.approx(
        np.log10(p_O2) - iw_2000, rel=1e-12
    )
    assert math.isfinite(hf_row['fO2_vapourise_shift_IW_derived'])
    # A 2 bar, 1e-6 VMR O2 atmosphere is many dex more oxidising than IW at
    # 2000 K, so the offset must be positive and large; a sign error would flip it.
    assert hf_row['fO2_vapourise_shift_IW_derived'] > 1.0


@pytest.mark.physics_invariant
def test_run_vapourisation_clamps_negative_P_vap(tmp_path, monkeypatch):
    """When the FastChem total falls below the volatile input pressure, the
    rock-vapour pressure would be negative; it is clamped to zero and P_vol is
    derived so P_vol + P_vap still equals the (new) surface pressure.

    Edge case: volatile input 5 bar but FastChem total only 2 bar.
    """
    dirs = {'output': str(tmp_path)}
    # Set up fake LavAtmos directory structure required by paths_importer
    env_dir = tmp_path
    (env_dir / 'input' / 'lava_compositions').mkdir(parents=True)

    monkeypatch.setenv('LAVA_DIR', str(env_dir))
    monkeypatch.setenv('FC_DIR', str(tmp_path / 'fastchem'))

    hf_row = _make_hf_row()
    hf_row['P_surf'] = 5.0  # volatile input above the FastChem total
    element_fracs = {'H': 0.0, 'C': 0.0, 'N': 0.0, 'S': 0.0, 'O': 0.1, 'Fe': 0.6, 'Mg': 0.3}
    _install_common_mocks(monkeypatch, element_fracs, tmp_path)
    monkeypatch.setattr(
        lavatmos_mod,
        'run_lavatmos',
        _fake_run_lavatmos_factory(pbar=2.0, mu=30.0, co2_vmr=0.5, h2o_vmr=0.4999, o2_vmr=1e-6),
    )

    lavatmos_mod.run_vapourisation(
        dirs, config=_magma_config(), hf_row=hf_row, first_iter=False
    )

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

    # Set up fake LavAtmos directory structure required by paths_importer
    env_dir = tmp_path
    (env_dir / 'input' / 'lava_compositions').mkdir(parents=True)

    monkeypatch.setenv('LAVA_DIR', str(env_dir))
    monkeypatch.setenv('FC_DIR', str(tmp_path / 'fastchem'))

    hf_row = _make_hf_row()
    element_fracs = {'H': 0.0, 'C': 0.0, 'N': 0.0, 'S': 0.0, 'O': 0.1, 'Fe': 0.6, 'Mg': 0.3}
    _install_common_mocks(monkeypatch, element_fracs, tmp_path)
    monkeypatch.setattr(lavatmos_mod, 'run_lavatmos', lambda *a, **k: None)

    # Force the fastchem-output directory check to fail, and only that check:
    # paths_importer validates the lava-compositions directory with the same
    # os.path.exists, so a blanket False would abort earlier and the test would
    # pass on the wrong exception.
    fc_output = os.path.join(str(tmp_path), 'fastchem', '')
    real_exists = os.path.exists

    def _exists(p):
        return False if str(p) == fc_output else real_exists(p)

    monkeypatch.setattr(lavatmos_mod.os.path, 'exists', _exists)

    called = {'status': False}

    def _fake_update(dirs_arg, code):
        called['status'] = True

    monkeypatch.setattr(lavatmos_mod, 'UpdateStatusfile', _fake_update)

    with pytest.raises(RuntimeError, match='fastchem'):
        lavatmos_mod.run_vapourisation(
            dirs, config=_magma_config(), hf_row=hf_row, first_iter=True
        )
    assert called['status'] is True


def _magma_config(t_min=1500.0, melt_comp_name='BSE_palm', fO2_buffer_model='oneill'):
    """Config stub exposing the outgas.lavatmos fields the vapour step reads."""
    return SimpleNamespace(
        outgas=SimpleNamespace(
            lavatmos=SimpleNamespace(
                T_min=t_min,
                melt_comp_name=melt_comp_name,
                fO2_buffer_model=fO2_buffer_model,
            )
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
    env_dir = tmp_path
    (env_dir / 'input' / 'lava_compositions').mkdir(parents=True)

    monkeypatch.setenv('LAVA_DIR', str(env_dir))
    monkeypatch.setenv('FC_DIR', str(tmp_path / 'fastchem'))
    monkeypatch.setattr(lavatmos_mod, 'vap_list', ['Fe'])  # a rock-vapour species
    monkeypatch.setattr(lavatmos_mod, 'noble_gases', ['He', 'Ar'])
    monkeypatch.setattr(lavatmos_mod, 'gas_list', ['CO2', 'O2', 'Fe', 'He', 'Ar'])
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

    # One atmosphere described twice, the way a real run describes it: as
    # FastChem species mixing ratios (used for the per-gas masses) and as
    # LavAtmos element number fractions (used for the per-element masses).
    # Deriving the second from the first keeps the two routes consistent, so
    # a noble gas gets the same mass either way. He and Ar carry nonzero
    # fractions, so a regression that folded them into M_vaps would show up.
    species_vmr = {'CO2': 0.20, 'Fe': 0.30, 'He': 0.30, 'Ar': 0.15, 'O2': 0.05}
    atoms_per_species = {
        'CO2': {'C': 1, 'O': 2},
        'Fe': {'Fe': 1},
        'He': {'He': 1},
        'Ar': {'Ar': 1},
        'O2': {'O': 2},
    }
    mu = sum(x * species_lib[s].weight for s, x in species_vmr.items())
    n_atoms = {}
    for s, x in species_vmr.items():
        for e, n in atoms_per_species[s].items():
            n_atoms[e] = n_atoms.get(e, 0.0) + n * x
    n_tot = sum(n_atoms.values())
    element_fracs = {e: 0.0 for e in ('H', 'C', 'N', 'S', 'O', 'Fe', 'He', 'Ar')}
    element_fracs.update({e: n / n_tot for e, n in n_atoms.items()})

    monkeypatch.setattr(
        lavatmos_mod, 'read_in_element_fracs_normalized', lambda p: dict(element_fracs)
    )

    def fake_run(dirs_arg, config, hf_row_arg, nfrac, first_iter):
        paths = lavatmos_mod.paths_importer(dirs_arg)
        with open(os.path.join(paths.fastchem3_output, 'boa_chem.dat'), 'w') as f:
            # Includes an He column but deliberately NO Ar column.
            f.write('Pbar mu C1O2 Fe He O2\n')
            f.write(
                '2.0 %r %r %r %r %r\n'
                % (
                    mu,
                    species_vmr['CO2'],
                    species_vmr['Fe'],
                    species_vmr['He'],
                    species_vmr['O2'],
                )
            )

    monkeypatch.setattr(lavatmos_mod, 'run_lavatmos', fake_run)

    hf_row = _make_hf_row()
    hf_row['He_kg_atm'] = 5.0e15
    hf_row['Ar_kg_atm'] = 1.0e15
    hf_row['He_vmr'] = 0.0
    hf_row['Ar_vmr'] = 0.123  # prior value; must survive (Ar not in FastChem output)
    for s in ('CO2', 'Fe', 'He'):
        for suff in ('_kg_solid', '_kg_liquid', '_mol_solid', '_mol_liquid'):
            hf_row.setdefault(s + suff, 0.0)

    lavatmos_mod.run_vapourisation(dirs, config=_magma_config(), hf_row=hf_row, first_iter=True)

    # He is read back from the combined FastChem equilibrium: not dropped.
    assert hf_row['He_vmr'] == pytest.approx(species_vmr['He'], rel=1e-9)
    assert hf_row['He_bar'] > 0.0
    assert hf_row['He_kg_atm'] > 0.0
    # He is atmospheric-only, so kg_total equals kg_atm (not zeroed like rock vapour).
    assert hf_row['He_kg_total'] == pytest.approx(hf_row['He_kg_atm'], rel=1e-12)
    # The species route (VMR/mu) and the element route (frac/mmw) describe the
    # same atmosphere, so both must give the same He mass. The element route
    # writes last, so a mismatch here means the two normalisations disagree.
    mmw_el = sum(element_fracs[e] * species_lib[e].weight for e in element_fracs)
    he_w = species_lib['He'].weight
    assert hf_row['He_kg_atm'] == pytest.approx(
        species_vmr['He'] * hf_row['M_atm'] * he_w / mu, rel=1e-12
    )
    assert hf_row['He_kg_atm'] == pytest.approx(
        element_fracs['He'] * hf_row['M_atm'] * he_w / mmw_el, rel=1e-12
    )
    # Ar absent from FastChem output: prior value preserved, no KeyError.
    assert hf_row['Ar_vmr'] == pytest.approx(0.123, rel=1e-12)

    # M_vaps counts only rock-forming Fe plus the vapourised O, never the
    # noble gases. Recompute the expected value from the same fractions/weights.
    mmw = mmw_el
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
