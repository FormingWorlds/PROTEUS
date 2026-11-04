"""Tests for the conductive surface skin from saturated condensate layers.

Source under test: ``src/proteus/atmos_clim/surface_skin.py``. Invariants checked:

- a stack of layers conducts the series flux ``dT / sum(d_i / k_i)``: the closed form, a
  finite-volume steady conduction solve, and a conductivity bounded by the layer values;
- a layer is its condensed mass spread over ``4 pi R_int**2`` at its density;
- only a saturated or held condensate forms a layer, and a stack with a thermal
  resistance below ``SKIN_RESISTANCE_MIN`` leaves the surface fixed;
- the skin step is the only writer of the skin columns, in every mode, and a skin that
  becomes negligible is cleared rather than left over from the previous iteration.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from proteus.atmos_clim.surface_skin import (
    agni_sol_type,
    condensate_layer_thickness,
    effective_surf_state,
    saturated_condensate_mass,
    update_surface_skin,
)
from proteus.utils.constants import (
    SKIN_RESISTANCE_MIN,
    condensate_conductivity,
    condensate_density,
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
    row = {'R_int': R_int, 'boundary_layer_thickness': 0.0, 'boundary_layer_conductivity': 0.0}
    for c in condensate_density:
        mass, activity = condensates.get(c, (0.0, 0.0))
        row[f'{c}_cd_kg'] = mass
        row[f'{c}_cd_activity'] = activity
    return row


def _mass_for_thickness(c, d, R_int=R_MERCURY):
    """Mass [kg] of condensate c spread into a layer of thickness d [m]."""
    return d * condensate_density[c] * 4.0 * np.pi * R_int**2


def _skin_of(**layers):
    """Run the skin step on layers given as c=thickness [m]; return the state and row."""
    row = _hf_row(**{c: (_mass_for_thickness(c, d), 1.0) for c, d in layers.items()})
    state = update_surface_skin(_saturation_config(), row)
    return state, row['boundary_layer_thickness'], row['boundary_layer_conductivity']


@pytest.mark.physics_invariant
def test_combined_conductivity_is_series_mean_of_graphite_over_sulfur():
    """1 m of graphite (100 W/m/K) under 0.1 m of sulfur (0.269 W/m/K) acts as one
    1.1 m skin of k = 1.1 / (1/100 + 0.1/0.269) = 2.8815 W/m/K, the series value."""
    state, d, k = _skin_of(C=1.0, S=0.1)
    assert state == 'skin'
    assert d == pytest.approx(1.1, rel=1e-12)
    assert k == pytest.approx(2.8814879735125136, rel=1e-12)
    # The skin flux k dT / d equals the series flux dT / R for any dT
    resistance = 1.0 / 100.0 + 0.1 / 0.269  # m2 K W-1
    assert k * 700.0 / d == pytest.approx(700.0 / resistance, rel=1e-12)
    # Plausible wrong means: thickness-weighted arithmetic (parallel layers) 90.93,
    # unweighted harmonic 0.537, thickness-weighted geometric 58.4
    for wrong in (90.93354545454544, 0.5365566625776661, 58.39035509323326):
        assert abs(k - wrong) > 0.3 * k
    assert 0.269 < k < 10.0


@pytest.mark.physics_invariant
def test_stack_conductivity_is_bounded_by_its_layers():
    """3 layers give a conductivity between the smallest and largest layer value, and
    one layer gives back its own conductivity."""
    _, d, k = _skin_of(C=2.0, S=0.05, H2O4S=0.2)
    assert d == pytest.approx(2.25, rel=1e-12)
    assert condensate_conductivity['S'] <= k <= condensate_conductivity['C']
    assert k == pytest.approx(2.578677535717329, rel=1e-12)
    _, d_one, k_one = _skin_of(S=0.3)
    assert d_one == pytest.approx(0.3, rel=1e-12)
    assert k_one == pytest.approx(condensate_conductivity['S'], rel=1e-12)


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
    assert condensate_layer_thickness(_hf_row()) == {}


def test_a_non_finite_condensate_state_is_reported_and_counted_as_no_layer(caplog):
    """A NaN mass or activity logs a warning and forms no layer, while a valid
    neighbour in the same row still does."""
    row = _hf_row(C=(float('nan'), 1.0), S=(1.0e3, float('inf')), H2O=(1.0e10, 0.0))
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.atmos_clim.surface_skin'):
        masses = saturated_condensate_mass(row)
    assert masses == {'H2O': pytest.approx(1.0e10)}
    assert 'Non-finite' in caplog.text and 'C' in caplog.text


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
    d_min = SKIN_RESISTANCE_MIN * condensate_conductivity['C']  # m
    assert d_min == pytest.approx(1.0e-3, rel=1e-12)
    assert _skin_of(C=0.9 * d_min) == ('fixed', 0.0, 0.0)
    state, d, k = _skin_of(C=1.1 * d_min)
    assert state == 'skin'
    assert d == pytest.approx(1.1e-3, rel=1e-10)
    assert k == pytest.approx(100.0, rel=1e-12)
    # No condensate at all is the fixed state too
    assert _skin_of() == ('fixed', 0.0, 0.0)


@pytest.mark.physics_invariant
def test_a_skin_that_becomes_negligible_is_cleared_from_the_row():
    """When the layer shrinks below the threshold, the skin step writes zeros over the
    previous iteration's skin, so the atmosphere does not keep solving with it."""
    config = _saturation_config()
    row = _hf_row(C=(1.0e18, 1.0))
    assert update_surface_skin(config, row) == 'skin'
    assert row['boundary_layer_thickness'] == pytest.approx(5.915736815008217, rel=1e-12)
    # The graphite re-vapourises: the row carries the old skin until the step runs
    row['C_cd_kg'], row['C_cd_activity'] = 1.0e-6, 0.2
    assert update_surface_skin(config, row) == 'fixed'
    assert (row['boundary_layer_thickness'], row['boundary_layer_conductivity']) == (0.0, 0.0)
    assert effective_surf_state(config, row) == 'fixed'
    assert agni_sol_type(config, row) == 1


def test_numeric_config_values_are_written_over_any_interior_value():
    """With numeric values the skin step writes the configured skin (0.05 m, 3 W/m/K)
    into both columns, over a value an interior wrote (40 m), whatever the condensates,
    and the state and the AGNI code are the configured ones."""
    config = _numeric_config(surf_state='fixed', d=0.05, k=3.0)
    row = _hf_row(C=(1.0e18, 1.0))
    row['boundary_layer_thickness'] = 40.0  # m, e.g. a thermal boundary layer
    assert update_surface_skin(config, row) == 'fixed'
    assert row['boundary_layer_thickness'] == pytest.approx(0.05, rel=1e-12)
    assert row['boundary_layer_conductivity'] == pytest.approx(3.0, rel=1e-12)
    assert effective_surf_state(config, row) == 'fixed'
    assert agni_sol_type(config, row) == 1
    assert agni_sol_type(_numeric_config(surf_state='skin'), row) == 2


@pytest.mark.physics_invariant
def test_update_surface_skin_records_the_skin_the_consumers_read(caplog):
    """The written boundary_layer_thickness and boundary_layer_conductivity set the
    state and the AGNI code that the atmosphere modules then read."""
    config = _saturation_config()
    row = _hf_row(C=(1.0e18, 1.0), S=(_mass_for_thickness('S', 0.1), 1.0))
    with caplog.at_level(logging.INFO, logger='fwl.proteus.atmos_clim.surface_skin'):
        assert update_surface_skin(config, row) == 'skin'
    d_row, k_row = row['boundary_layer_thickness'], row['boundary_layer_conductivity']
    assert d_row == pytest.approx(5.915736815008217 + 0.1, rel=1e-12)
    resistance = 5.915736815008217 / 100.0 + 0.1 / 0.269  # m2 K W-1
    assert k_row == pytest.approx(d_row / resistance, rel=1e-12)
    assert effective_surf_state(config, row) == 'skin'
    assert agni_sol_type(config, row) == 2
    assert 'skin' in caplog.text


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
    state, d, k = _skin_of(**thickness)
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
