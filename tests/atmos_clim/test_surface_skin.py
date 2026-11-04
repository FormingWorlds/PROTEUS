"""Tests for the conductive surface skin from saturated condensate layers.

Source under test: ``src/proteus/atmos_clim/surface_skin.py``. Invariants checked:

- a stack of layers conducts the series flux ``dT / sum(d_i / k_i)``: the closed form, a
  finite-volume steady conduction solve, a conductivity bounded by the layer values and
  independent of the stacking order;
- a layer is its condensed mass spread over ``4 pi R_int**2`` at its density;
- only a saturated or held condensate forms a layer, and a stack with a thermal
  resistance below ``SKIN_RESISTANCE_MIN`` leaves the surface fixed;
- numeric config values pass through unchanged.
"""

from __future__ import annotations

import itertools
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from proteus.atmos_clim.surface_skin import (
    agni_sol_type,
    combine_layers,
    condensate_layer_thickness,
    effective_surf_state,
    resolve_surface_skin,
    saturated_condensate_mass,
    skin_values,
    update_surface_skin,
)
from proteus.utils.constants import (
    SKIN_RESISTANCE_MIN,
    condensate_conductivity,
    condensate_density,
    condensate_list,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

R_MERCURY = 2.4397e6  # m


def _saturation_config():
    """Config namespace with the skin derived from saturation."""
    return SimpleNamespace(
        atmos_clim=SimpleNamespace(
            surface_d='saturation', surface_k='saturation', surf_state='skin', surf_state_int=2
        )
    )


def _numeric_config(surf_state='skin', d=0.01, k=2.0):
    """Config namespace with a fixed numeric skin [m, W m-1 K-1]."""
    codes = {'fixed': 1, 'skin': 2}
    return SimpleNamespace(
        atmos_clim=SimpleNamespace(
            surface_d=d, surface_k=k, surf_state=surf_state, surf_state_int=codes[surf_state]
        )
    )


def _hf_row(R_int=R_MERCURY, **condensates):
    """Helpfile row with no condensate except those given as c=(mass kg, activity)."""
    row = {'R_int': R_int}
    for c in condensate_list:
        mass, activity = condensates.get(c, (0.0, 0.0))
        row[f'{c}_cd_kg'] = mass
        row[f'{c}_cd_activity'] = activity
    return row


def _mass_for_thickness(c, d, R_int=R_MERCURY):
    """Mass [kg] of condensate c spread into a layer of thickness d [m]."""
    return d * condensate_density[c] * 4.0 * np.pi * R_int**2


@pytest.mark.physics_invariant
def test_combined_conductivity_is_series_mean_of_graphite_over_sulfur():
    """1 m of graphite (100 W/m/K) under 0.1 m of sulfur (0.27 W/m/K) acts as one
    1.1 m skin of k = 1.1 / (1/100 + 0.1/0.27) = 2.8919 W/m/K, the series value."""
    d, k = combine_layers({'C': 1.0, 'S': 0.1}, {'C': 100.0, 'S': 0.27})
    assert d == pytest.approx(1.1, rel=1e-12)
    assert k == pytest.approx(2.891918208373905, rel=1e-12)
    # The skin flux k dT / d equals the series flux dT / R for any dT
    resistance = 1.0 / 100.0 + 0.1 / 0.27  # m2 K W-1
    assert k * 700.0 / d == pytest.approx(700.0 / resistance, rel=1e-12)
    # Plausible wrong means: thickness-weighted arithmetic (parallel layers) 90.93,
    # unweighted harmonic 0.539, thickness-weighted geometric 58.4
    for wrong in (90.93363636363635, 0.5385459259998007, 58.41005496554968):
        assert abs(k - wrong) > 0.3 * k
    # Scale: the thin insulating sulfur sets k, far below graphite
    assert 0.27 < k < 10.0


@pytest.mark.physics_invariant
def test_stack_conductivity_is_bounded_and_independent_of_order():
    """Stacked in any order, 3 layers give the same d and k, with
    min(k_i) <= k <= max(k_i); one layer gives back its own conductivity."""
    thickness = {'C': 2.0, 'S': 0.05, 'H2O4S': 0.2}  # m
    results = set()
    for order in itertools.permutations(thickness):
        d, k = combine_layers({c: thickness[c] for c in order}, condensate_conductivity)
        results.add((round(d, 12), round(k, 12)))
        assert min(condensate_conductivity[c] for c in thickness) <= k
        assert k <= max(condensate_conductivity[c] for c in thickness)
    assert len(results) == 1
    d_one, k_one = combine_layers({'S': 0.3}, condensate_conductivity)
    assert d_one == pytest.approx(0.3, rel=1e-12)
    assert k_one == pytest.approx(condensate_conductivity['S'], rel=1e-12)


def test_combine_layers_rejects_an_empty_or_unphysical_stack():
    """An empty stack, a zero thickness or a non-positive conductivity is refused."""
    with pytest.raises(ValueError, match='empty'):
        combine_layers({}, condensate_conductivity)
    with pytest.raises(ValueError, match='positive'):
        combine_layers({'C': 0.0}, condensate_conductivity)
    with pytest.raises(ValueError, match='positive'):
        combine_layers({'C': 1.0}, {'C': -100.0})
    # A valid neighbour does not mask the bad layer
    with pytest.raises(ValueError, match='S'):
        combine_layers({'C': 1.0, 'S': -0.1}, condensate_conductivity)


@pytest.mark.physics_invariant
def test_layer_thickness_spreads_mass_over_the_whole_surface():
    """1e18 kg of graphite (2260 kg/m3) over Mercury (R = 2.4397e6 m) is a
    5.916 m layer: mass / (rho 4 pi R**2), not the pi R**2 disc (23.66 m)."""
    layers = condensate_layer_thickness(_hf_row(C=(1.0e18, 1.0)))
    assert set(layers) == {'C'}
    assert layers['C'] == pytest.approx(5.915736815008217, rel=1e-12)
    assert abs(layers['C'] - 23.66294726003287) > 10.0  # area of a disc
    # The thickness falls with the square of the radius
    doubled = condensate_layer_thickness(_hf_row(R_int=2 * R_MERCURY, C=(1.0e18, 1.0)))
    assert doubled['C'] == pytest.approx(layers['C'] / 4.0, rel=1e-12)


def test_layer_thickness_refuses_a_bad_radius_or_condensate_state():
    """A non-positive radius or a non-finite condensate mass raises."""
    with pytest.raises(ValueError, match='radius'):
        condensate_layer_thickness(_hf_row(R_int=0.0, C=(1.0e18, 1.0)))
    with pytest.raises(ValueError, match='Non-finite'):
        saturated_condensate_mass(_hf_row(C=(float('nan'), 1.0)))
    # An empty surface is valid and has no layer
    assert condensate_layer_thickness(_hf_row()) == {}


@pytest.mark.physics_invariant
def test_only_saturated_or_held_condensates_form_layers():
    """A saturated condensate (a >= 0.999) or a held mass above 1 kg forms a layer;
    the solver floor below saturation and a saturated zero mass do not."""
    row = _hf_row(
        C=(1.0e3, 1.0),  # saturated, small mass
        S=(1.0e-6, 0.5),  # floor mass below saturation
        H2O=(1.0e10, 0.0),  # held by prevent_desaturation at zero activity
        H2O4S=(0.0, 1.0),  # saturated without mass
    )
    masses = saturated_condensate_mass(row)
    assert masses == {'C': pytest.approx(1.0e3), 'H2O': pytest.approx(1.0e10)}
    # Just below the saturation activity a small mass is still the solver floor
    assert saturated_condensate_mass(_hf_row(C=(1.0e-3, 0.998))) == {}
    assert saturated_condensate_mass(_hf_row(C=(1.0e-3, 0.999))) == {'C': pytest.approx(1.0e-3)}


@pytest.mark.physics_invariant
def test_a_negligible_stack_leaves_the_surface_fixed():
    """A graphite layer with d/k just below SKIN_RESISTANCE_MIN (1e-5 m2 K/W, so
    d = 1 mm) keeps the surface fixed; just above it gives a skin with k = 100."""
    config = _saturation_config()
    d_min = SKIN_RESISTANCE_MIN * condensate_conductivity['C']  # m
    assert d_min == pytest.approx(1.0e-3, rel=1e-12)

    below = _hf_row(C=(_mass_for_thickness('C', 0.9 * d_min), 1.0))
    assert resolve_surface_skin(config, below) == ('fixed', 0.0, 0.0)

    state, d, k = resolve_surface_skin(
        config, _hf_row(C=(_mass_for_thickness('C', 1.1 * d_min), 1.0))
    )
    assert state == 'skin'
    assert d == pytest.approx(1.1e-3, rel=1e-10)
    assert k == pytest.approx(100.0, rel=1e-12)
    # No condensate at all is the fixed state too
    assert resolve_surface_skin(config, _hf_row()) == ('fixed', 0.0, 0.0)


def test_numeric_config_values_pass_through_unchanged():
    """With numeric surface_d and surface_k the condensates are ignored."""
    config = _numeric_config(surf_state='skin', d=0.05, k=3.0)
    row = _hf_row(C=(1.0e18, 1.0))
    assert resolve_surface_skin(config, row) == ('skin', 0.05, 3.0)
    assert skin_values(
        config, {'boundary_layer_thickness': 7.0, 'boundary_layer_conductivity': 9.0}
    ) == (0.05, 3.0)
    assert agni_sol_type(_numeric_config(surf_state='fixed'), row) == 1
    assert effective_surf_state(_numeric_config(surf_state='fixed'), row) == 'fixed'


@pytest.mark.physics_invariant
def test_update_surface_skin_records_the_skin_the_consumers_read(caplog):
    """The written boundary_layer_thickness and boundary_layer_conductivity set the
    state, the skin values and the AGNI code that the atmosphere modules then read;
    a fixed state writes zeros."""
    config = _saturation_config()
    row = _hf_row(C=(1.0e18, 1.0), S=(_mass_for_thickness('S', 0.1), 1.0))
    with caplog.at_level(logging.INFO, logger='fwl.proteus.atmos_clim.surface_skin'):
        assert update_surface_skin(config, row) == 'skin'
    d_row, k_row = row['boundary_layer_thickness'], row['boundary_layer_conductivity']
    assert d_row == pytest.approx(5.915736815008217 + 0.1, rel=1e-12)
    resistance = 5.915736815008217 / 100.0 + 0.1 / 0.269  # m2 K W-1
    assert k_row == pytest.approx(d_row / resistance, rel=1e-12)
    assert skin_values(config, row) == (d_row, k_row)
    assert effective_surf_state(config, row) == 'skin'
    assert agni_sol_type(config, row) == 2
    assert 'Graphite' in caplog.text and 'Sulfur' in caplog.text

    empty = _hf_row()
    assert update_surface_skin(config, empty) == 'fixed'
    assert (empty['boundary_layer_thickness'], empty['boundary_layer_conductivity']) == (
        0.0,
        0.0,
    )
    assert effective_surf_state(config, empty) == 'fixed'
    assert agni_sol_type(config, empty) == 1


def _finite_volume_stack(thickness, conductivity, order, T_bottom, T_top, cells=16):
    """Steady 1D conduction through stacked layers on a cell-centred finite-volume grid.

    Layer interfaces lie on cell faces and each face uses the series conductance of
    its two half cells. Returns the cell centres [m], their temperatures [K] and the
    upward flux through the bottom and the top face [W m-2].
    """
    widths = np.concatenate([np.full(cells, thickness[c] / cells) for c in order])
    k = np.concatenate([np.full(cells, conductivity[c]) for c in order])
    n = widths.size
    g_face = 1.0 / (widths[:-1] / (2 * k[:-1]) + widths[1:] / (2 * k[1:]))
    g_bottom, g_top = 2 * k[0] / widths[0], 2 * k[-1] / widths[-1]
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i, g in enumerate(g_face):
        A[i, i] += g
        A[i + 1, i + 1] += g
        A[i, i + 1] -= g
        A[i + 1, i] -= g
    A[0, 0] += g_bottom
    b[0] += g_bottom * T_bottom
    A[-1, -1] += g_top
    b[-1] += g_top * T_top
    T = np.linalg.solve(A, b)
    z = np.cumsum(widths) - widths / 2
    return z, T, g_bottom * (T_bottom - T[0]), g_top * (T[-1] - T_top)


@pytest.mark.physics_invariant
def test_combined_skin_conducts_the_flux_of_a_layered_conduction_solve():
    """Analytical law of steady 1D conduction through layers in series (Fourier's law
    with continuous temperature and flux at each interface): 2 m of graphite, 5 cm of
    sulfur and 20 cm of sulfuric acid at Mercury's radius between 1800 K magma and a
    1200 K surface. A finite-volume solve with the layer conductivities gives the flux
    of the combined skin, dT k / d, and the piecewise-linear analytic profile."""
    order = ('C', 'S', 'H2O4S')  # bottom to top, densest first
    thickness = {'C': 2.0, 'S': 0.05, 'H2O4S': 0.2}  # m
    row = _hf_row(**{c: (_mass_for_thickness(c, thickness[c]), 1.0) for c in order})
    state, d, k = resolve_surface_skin(_saturation_config(), row)
    assert state == 'skin'
    assert d == pytest.approx(2.25, rel=1e-12)

    T_bottom, T_top = 1800.0, 1200.0  # K
    z, T, q_bottom, q_top = _finite_volume_stack(
        thickness, condensate_conductivity, order, T_bottom, T_top
    )
    q_skin = k * (T_bottom - T_top) / d  # W m-2
    # Exact for piecewise-constant k up to round-off in the linear solve
    assert q_bottom == pytest.approx(q_skin, rel=1e-9)
    assert q_top == pytest.approx(q_bottom, rel=1e-9)  # no heat lost inside the stack
    # Analytic profile: linear in each layer, with the drop q d_i / k_i across it
    tops = np.cumsum([thickness[c] for c in order])
    T_faces = T_bottom - q_skin * np.cumsum(
        [thickness[c] / condensate_conductivity[c] for c in order]
    )
    T_exact = np.interp(z, np.concatenate([[0.0], tops]), np.concatenate([[T_bottom], T_faces]))
    np.testing.assert_allclose(T, T_exact, rtol=1e-9)
    assert T_faces[-1] == pytest.approx(T_top, rel=1e-12)
    # Parallel layers (thickness-weighted arithmetic k) would carry about 34 times more
    k_parallel = sum(thickness[c] * condensate_conductivity[c] for c in order) / d
    assert k_parallel * (T_bottom - T_top) / d > 10 * q_bottom
    assert 0.0 < q_bottom < (T_bottom - T_top) * condensate_conductivity['C'] / d


def test_numeric_skin_leaves_the_interior_thickness_and_records_the_conductivity():
    """With numeric values the skin step writes only boundary_layer_conductivity
    (3.0 W/m/K); boundary_layer_thickness stays what the interior wrote, here the
    boundary interior's own thermal boundary layer (40 m), not surface_d (0.05 m)."""
    config = _numeric_config(surf_state='fixed', d=0.05, k=3.0)
    row = _hf_row(C=(1.0e18, 1.0))
    row['boundary_layer_thickness'] = 40.0  # m
    assert update_surface_skin(config, row) == 'fixed'
    assert row['boundary_layer_thickness'] == pytest.approx(40.0, rel=1e-12)
    assert row['boundary_layer_conductivity'] == pytest.approx(3.0, rel=1e-12)
    # The atmosphere still reads the configured skin, not the interior column
    assert skin_values(config, row) == (0.05, 3.0)
