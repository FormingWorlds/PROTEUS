"""
Stub-contract tests for #653 (Mariana), #526 (core partitioning),
#432 (EOS corrections).

These tests lock the stub APIs so Mariana / #526 / #432 implementers
can see the exact contract their replacement must satisfy.

Plan v6 §6.
"""
from __future__ import annotations

import numpy as np
import pytest

from proteus.config._struct import MantleComp
from proteus.redox import IMPLEMENTATION_STATUS
from proteus.redox.partitioning import (
    EOSCorrection,
    Fe3EvolutionResult,
    MetalSilicatePartitionResult,
    advance_fe_reservoirs,
    apply_metal_silicate_partitioning,
    eos_density_correction,
    liquidus_mineral_at_node,
    metal_silicate_KD,
    mineral_melt_KD_fe,
)

# ==================================================================
# #653 — advance_fe_reservoirs stub contract
# ==================================================================


@pytest.mark.unit
def test_fe3_evolution_status_is_stub_bulk():
    assert IMPLEMENTATION_STATUS['fe3_evolution'] == 'stub-bulk'


@pytest.mark.unit
def test_advance_fe_reservoirs_identity_on_trivial_input():
    """Stub preserves n_Fe3_melt and n_Fe2_melt unchanged."""
    n = 4
    result = advance_fe_reservoirs(
        pressure_profile=np.linspace(1e9, 1e11, n),
        temperature_profile=np.linspace(2000, 4000, n),
        melt_fraction_profile=np.full(n, 0.5),
        melt_fraction_profile_prev=np.full(n, 0.5),
        cell_mass_profile=np.full(n, 1e22),
        n_Fe3_melt_prev=1.0e20,
        n_Fe2_melt_prev=9.0e20,
        n_Fe3_solid_cell_prev=np.zeros(n),
        n_Fe2_solid_cell_prev=np.zeros(n),
        mantle_comp=MantleComp(),
    )
    assert isinstance(result, Fe3EvolutionResult)
    assert result.n_Fe3_melt == 1.0e20
    assert result.n_Fe2_melt == 9.0e20
    assert result.Fe3_frac_bulk_melt == pytest.approx(0.1)
    assert result.dm_Fe0_to_core == 0.0
    assert np.all(result.dn_Fe3_solid_cell == 0.0)
    assert np.all(result.dn_Fe2_solid_cell == 0.0)


@pytest.mark.unit
def test_advance_fe_reservoirs_log10_fO2_surface_sensible():
    """Stub returns a finite warm-start fO2 from the dispatcher."""
    n = 4
    result = advance_fe_reservoirs(
        pressure_profile=np.linspace(1e9, 1e11, n),
        temperature_profile=np.linspace(2000, 4000, n),
        melt_fraction_profile=np.full(n, 0.9),  # MO-active
        melt_fraction_profile_prev=np.full(n, 0.9),
        cell_mass_profile=np.full(n, 1e22),
        n_Fe3_melt_prev=1.0e20,
        n_Fe2_melt_prev=9.0e20,
        n_Fe3_solid_cell_prev=np.zeros(n),
        n_Fe2_solid_cell_prev=np.zeros(n),
        mantle_comp=MantleComp(),
    )
    assert np.isfinite(result.log10_fO2_surface)
    assert -30 < result.log10_fO2_surface < 10


# ==================================================================
# #653 — mineral_melt_KD_fe + liquidus_mineral_at_node stubs
# ==================================================================


@pytest.mark.unit
def test_mineral_melt_KD_stub_identity():
    """Every mineral returns (D_Fe3, D_Fe2) = (1.0, 1.0)."""
    for mineral in ('sp', 'ol', 'cpx', 'opx', 'gt',
                    'wad', 'maj', 'ring', 'bg', 'mw'):
        d3, d2 = mineral_melt_KD_fe(
            mineral, pressure=10e9, temperature=2500.0,
        )
        assert d3 == 1.0, f'{mineral}: D_Fe3 = {d3}'
        assert d2 == 1.0, f'{mineral}: D_Fe2 = {d2}'


@pytest.mark.unit
def test_liquidus_mineral_stub_returns_bg():
    for P in (1e9, 10e9, 50e9, 120e9):
        for T in (1500.0, 3000.0, 5000.0):
            assert liquidus_mineral_at_node(
                pressure=P, temperature=T,
            ) == 'bg'


# ==================================================================
# #526 — metal-silicate stubs
# ==================================================================


@pytest.mark.unit
def test_metal_silicate_status_is_zeros():
    assert IMPLEMENTATION_STATUS['metal_silicate'] == 'zeros'


@pytest.mark.unit
def test_metal_silicate_KD_returns_zero_for_all_supported():
    for element in ('H', 'O', 'Si', 'C', 'S', 'N'):
        assert metal_silicate_KD(
            element, pressure=40e9, temperature=4000.0, fO2_dIW=-2.0,
        ) == 0.0


@pytest.mark.unit
def test_metal_silicate_KD_unknown_element_raises():
    with pytest.raises(KeyError):
        metal_silicate_KD(
            'He', pressure=40e9, temperature=4000.0, fO2_dIW=-2.0,
        )


@pytest.mark.unit
def test_apply_metal_silicate_partitioning_returns_zero_fluxes():
    result = apply_metal_silicate_partitioning({}, 1.0)
    assert isinstance(result, MetalSilicatePartitionResult)
    assert all(v == 0.0 for v in result.fluxes.values())
    assert result.dR_budget_mantle == 0.0
    assert result.dR_budget_core == 0.0


# ==================================================================
# #432 — EOS correction stub
# ==================================================================


@pytest.mark.unit
def test_eos_correction_status_is_identity():
    assert IMPLEMENTATION_STATUS['eos_correction'] == 'identity'


@pytest.mark.unit
def test_eos_correction_returns_identity_factors():
    """Stub returns mantle=1.0, core=1.0 for any input."""
    for fe3 in (0.0, 0.04, 0.10, 0.50):
        for h_wt in (0.0, 1.0, 5.0):
            result = eos_density_correction(
                Fe3_frac=fe3, core_H_wt=h_wt,
            )
            assert isinstance(result, EOSCorrection)
            assert result.mantle_factor == 1.0
            assert result.core_factor == 1.0
