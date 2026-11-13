"""
Unit tests for proteus.interior_energetics.resolved_mantle.

The module holds what the boundary energetics backend needs once it resolves
the mantle on an adiabat: the melt model
cached against the structure mesh, the melt curve an ODE right-hand side reads
the melt fraction from, and the radial shells handed to the tides modules.

Invariants and contract clauses exercised:
- The melt curve reduces to the linear melt fraction, and its response to
  1 / (T_liquidus - T_solidus), with no adiabatic gradient and
  pressure-independent melting curves (analytical limit).
- The latent heat drawn along any path closes on the change in melt fraction,
  because the response is the slope of the interpolant (conservation).
- The melt fraction is bounded in [0, 1] and non-decreasing in the magma
  temperature, and its response is non-negative.
- Each node of the curve integrates the adiabat at most once, and a mantle
  above its liquidus everywhere needs no integration at all.
- The cached melt model and curve survive steps and are rebuilt when the
  structure module rewrites the mesh.
- Shells conserve the mantle mass and the mass integral of any field, and the
  tidal heat is the rate times the mass summed over them.

See docs/How-to/testing.md and docs/Explanations/test_framework.md.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
from _adiabat_mesh import (
    ADIABAT_DTDP,
    ADIABAT_TLIQ,
    ADIABAT_TSOL,
    CURVE_DTDP,
    count_adiabat_solves,
    patch_adiabat_inputs,
    write_mantle_mesh,
)

from proteus.interior_energetics import adiabat
from proteus.interior_energetics.resolved_mantle import (
    MeltCurve,
    get_melt_curve,
    get_melt_model,
    mantle_shells,
    shell_average,
    shell_edges,
    shell_masses,
    tidal_power,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Earth-scale mantle shared with the boundary backend tests.
_R_INT = 6.371e6  # m
_R_CORE = 3.4e6  # m
_M_CORE = 2.0e24  # kg
_M_MANTLE = 4.0e24  # kg


def _hf_row():
    """Scalars the structure mesh is checked against."""
    return {'R_int': _R_INT, 'R_core': _R_CORE, 'M_int': _M_CORE + _M_MANTLE, 'M_core': _M_CORE}


def _config():
    """The configuration fields the melt model reads once its inputs are patched."""
    return SimpleNamespace(
        interior_struct=SimpleNamespace(module='zalmoxis'),
        interior_energetics=SimpleNamespace(module='boundary'),
    )


def _model(tmp_path, monkeypatch, dtdp=0.0, curve_dtdp=0.0, **mesh_kw):
    """Melt model on a freshly written mesh, with the gradient and curves patched."""
    write_mantle_mesh(tmp_path, _R_CORE, _R_INT, _M_MANTLE, **mesh_kw)
    patch_adiabat_inputs(monkeypatch, dtdp=dtdp, curve_dtdp=curve_dtdp)
    return adiabat.AdiabatMeltModel(_config(), str(tmp_path), _hf_row(), {})


def _touch_mesh(tmp_path):
    """Advance the mesh modification time, as a structure re-solve does."""
    mesh = os.path.join(str(tmp_path), 'data', 'zalmoxis_output.dat')
    stat = os.stat(mesh)
    os.utime(mesh, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


# =============================================================================
# Melt curve
# =============================================================================


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_melt_curve_reduces_to_the_linear_melt_fraction_without_a_gradient(
    tmp_path, monkeypatch
):
    """With no gradient and constant melting curves the curve is the linear melt model.

    Analytical limit: an isothermal mantle between a pressure-independent
    solidus and liquidus melts linearly, so the curve must return
    (T - T_sol) / (T_liq - T_sol) and a response of 1 / (T_liq - T_sol).
    1837.4 K sits between nodes, so a node lookup without interpolation, or a
    response read from the wrong segment, is caught.
    """
    curve = MeltCurve(_model(tmp_path, monkeypatch, rho_cmb=4800.0, rho_top=4800.0))
    span = ADIABAT_TLIQ - ADIABAT_TSOL
    tmp = 1837.4

    expected = (tmp - ADIABAT_TSOL) / span  # 0.3374
    assert curve.phi(tmp) == pytest.approx(expected, rel=1e-12)
    assert curve.dphi_dtmp(tmp) == pytest.approx(1.0 / span, rel=1e-12)
    # Factor guard: a response over half the span would read 2e-3, not 1e-3.
    assert abs(curve.dphi_dtmp(tmp) - 2.0 / span) > 0.5 / span
    # Unit guard: nodes read in Celsius would place the mantle 273 K colder,
    # at a melt fraction of 0.064 rather than 0.337.
    assert abs(curve.phi(tmp) - (tmp - 273.15 - ADIABAT_TSOL) / span) > 0.2
    # Sign guard: the mantle melts as it warms.
    assert curve.dphi_dtmp(tmp) > 0.0

    # Edge cases: outside the mushy zone there is nothing left to melt or freeze.
    assert curve.phi(ADIABAT_TSOL - 63.7) == pytest.approx(0.0, abs=1e-15)
    assert curve.dphi_dtmp(ADIABAT_TSOL - 63.7) == pytest.approx(0.0, abs=1e-15)
    assert curve.phi(ADIABAT_TLIQ + 63.7) == pytest.approx(1.0, abs=1e-15)
    assert curve.dphi_dtmp(ADIABAT_TLIQ + 63.7) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.physics_invariant
def test_melt_curve_latent_heat_closes_on_the_change_in_melt_fraction(tmp_path, monkeypatch):
    """The response integrated over any path equals the change in melt fraction.

    Conservation: the latent heat a solver draws while the mantle cools from
    T_a to T_b is L M times the response integrated over the path. For that to
    equal L M times the melt that froze, the response must be the slope of the
    interpolant itself. The path spans 37.3 K across node boundaries on a
    sloped adiabat, where the response varies from segment to segment.
    """
    model = _model(tmp_path, monkeypatch, dtdp=ADIABAT_DTDP, curve_dtdp=CURVE_DTDP)
    curve = MeltCurve(model)
    tmp_a, tmp_b = 1823.6, 1786.3

    # Break the path at every node so each piece lies within one segment.
    cuts = np.concatenate(([tmp_b], np.arange(math.ceil(tmp_b), tmp_a), [tmp_a]))
    drawn = sum(
        curve.dphi_dtmp(0.5 * (lo + hi)) * (hi - lo) for lo, hi in zip(cuts[:-1], cuts[1:])
    )
    frozen = curve.phi(tmp_a) - curve.phi(tmp_b)

    assert drawn == pytest.approx(frozen, rel=1e-10)
    # Sign guard: cooling freezes melt.
    assert frozen > 0.0
    # Scale guard: tens of kelvin of a several-hundred-kelvin mushy zone.
    assert 1e-3 < frozen < 0.2
    # The interpolant tracks the adiabat it was built from, to second order
    # in the node spacing.
    exact = model.state(tmp_a).phi_mass - model.state(tmp_b).phi_mass
    assert frozen == pytest.approx(exact, abs=1e-4)


@pytest.mark.physics_invariant
def test_melt_curve_is_bounded_and_does_not_decrease_with_temperature(tmp_path, monkeypatch):
    """The melt fraction stays in [0, 1] and rises with the magma temperature.

    Swept from a wholly solid mantle (800 K) to one above its hottest liquidus
    (6100 K) on a sloped adiabat, with temperatures between nodes so the
    interpolation is exercised throughout.
    """
    curve = MeltCurve(_model(tmp_path, monkeypatch, dtdp=ADIABAT_DTDP, curve_dtdp=CURVE_DTDP))
    tmps = np.linspace(800.3, 6100.7, 41)
    phis = np.array([curve.phi(t) for t in tmps])
    slopes = np.array([curve.dphi_dtmp(t) for t in tmps])

    assert np.all((phis >= 0.0) & (phis <= 1.0))
    assert np.all(np.diff(phis) >= -1e-12)
    assert np.all(slopes >= 0.0)
    # Edge cases: both end states are reached within the sweep.
    assert phis[0] == pytest.approx(0.0, abs=1e-15)
    assert phis[-1] == pytest.approx(1.0, abs=1e-15)
    # The melting curves rise with depth faster than the adiabat, so the
    # mantle is mushy over a range far wider than the 1000 K at the surface.
    mushy = tmps[(phis > 0.0) & (phis < 1.0)]
    assert mushy[-1] - mushy[0] > 1000.0


@pytest.mark.physics_invariant
def test_melt_curve_integrates_each_node_once(tmp_path, monkeypatch):
    """Repeated lookups reuse the nodes, and a molten mantle needs no adiabat.

    An ODE solver queries the curve many times within a narrow temperature
    window. Each node the window touches must be integrated once and only
    once. Above the hottest liquidus the mantle is wholly molten whatever the
    gradient, so no integration is needed.
    """
    curve = MeltCurve(_model(tmp_path, monkeypatch, dtdp=ADIABAT_DTDP, curve_dtdp=CURVE_DTDP))
    calls = count_adiabat_solves(monkeypatch)

    # Edge case: above every liquidus, molten without integrating.
    hottest = float(np.max(curve.model.tmp_liq))
    assert curve.phi(hottest + 0.5) == pytest.approx(1.0, abs=1e-15)
    assert curve.dphi_dtmp(hottest + 0.5) == pytest.approx(0.0, abs=1e-15)
    assert calls[0] == 0

    # 200 lookups between 1900.2 and 1902.9 K touch the nodes 1900 to 1903.
    for tmp in np.linspace(1900.2, 1902.9, 200):
        curve.phi(tmp)
        curve.dphi_dtmp(tmp)
    assert calls[0] == 4
    # Revisiting the window costs nothing more.
    curve.phi(1901.5)
    assert calls[0] == 4


def test_melt_curve_rejects_an_undefined_temperature_or_spacing(tmp_path, monkeypatch):
    """A NaN magma temperature or a non-positive node spacing raises.

    Error contract: neither has a melt fraction, and returning one would let a
    diverged solver carry on with a meaningless latent budget.
    """
    model = _model(tmp_path, monkeypatch)
    curve = MeltCurve(model)

    with pytest.raises(ValueError, match='NaN'):
        curve.phi(float('nan'))
    with pytest.raises(ValueError, match='NaN'):
        curve.dphi_dtmp(float('nan'))
    with pytest.raises(ValueError, match='spacing must be positive'):
        MeltCurve(model, dtmp=0.0)
    # The rejected calls left no node behind.
    assert curve._nodes == {}


def test_get_melt_curve_keeps_its_nodes_until_the_structure_is_resolved(tmp_path, monkeypatch):
    """The cached curve, and the nodes it has computed, last as long as the structure.

    Between structure solves every step reads the same curve, so a mantle
    cooling slowly revisits nodes it already holds. After a re-solve the curve
    must be rebuilt on the new melt model, or the interior would keep reading
    melt fractions from the mantle the structure module has replaced.
    """
    write_mantle_mesh(tmp_path, _R_CORE, _R_INT, _M_MANTLE)
    patch_adiabat_inputs(monkeypatch, dtdp=ADIABAT_DTDP, curve_dtdp=CURVE_DTDP)
    dirs = {'output': str(tmp_path)}
    interior_o = SimpleNamespace()

    first = get_melt_curve(_config(), dirs, _hf_row(), interior_o)
    phi_first = first.phi(1850.5)
    assert get_melt_curve(_config(), dirs, _hf_row(), interior_o) is first
    assert set(first._nodes) == {1850, 1851}

    # A denser lower mantle melts less by mass at the same anchor.
    write_mantle_mesh(tmp_path, _R_CORE, _R_INT, _M_MANTLE, rho_cmb=6500.0, rho_top=4000.0)
    _touch_mesh(tmp_path)
    second = get_melt_curve(_config(), dirs, _hf_row(), interior_o)

    assert second is not first
    assert second.model is interior_o._adiabat_melt_model
    assert second._nodes == {}
    assert second.phi(1850.5) < phi_first


def test_get_melt_model_is_rebuilt_when_the_structure_is_resolved(tmp_path, monkeypatch):
    """A rewritten mantle mesh replaces the cached melt model.

    Between structure solves the model is reused, so the gradient tables are
    read once. After a re-solve it has to be rebuilt, otherwise the interior
    would keep integrating on the mantle the structure module has replaced.
    """
    write_mantle_mesh(tmp_path, _R_CORE, _R_INT, _M_MANTLE)
    patch_adiabat_inputs(monkeypatch)
    dirs = {'output': str(tmp_path)}
    interior_o = SimpleNamespace()

    first = get_melt_model(_config(), dirs, _hf_row(), interior_o)
    assert get_melt_model(_config(), dirs, _hf_row(), interior_o) is first

    _, _, density = write_mantle_mesh(
        tmp_path, _R_CORE, _R_INT, _M_MANTLE, rho_cmb=6000.0, rho_top=4000.0
    )
    _touch_mesh(tmp_path)

    second = get_melt_model(_config(), dirs, _hf_row(), interior_o)
    assert second is not first
    # The rebuilt model carries the new density, not the cached one.
    assert second.profile.density[0] == pytest.approx(density[0], rel=1e-9)
    # The interpolation cache survives the rebuild, so the tables are not reread.
    assert second.eos_cache is first.eos_cache


# =============================================================================
# Shells and tidal heating
# =============================================================================


@pytest.mark.physics_invariant
def test_shell_average_is_exact_for_a_uniform_field_and_grid_independent(tmp_path, monkeypatch):
    """Shell means recover a uniform field exactly and conserve a varying one.

    A uniform field must average to itself in every shell, including shells
    whose edges fall between the mesh levels. For a field that varies, a
    coarser or finer grid redistributes the mass between shells but must not
    change the mantle-wide mass integral.
    """
    profile = _model(tmp_path, monkeypatch, n=37).profile

    # Edge case: a uniform field on shells that do not align with the mesh.
    edges = shell_edges(profile, 13)
    np.testing.assert_allclose(
        shell_average(profile, edges, np.full(profile.radius.size, 0.25)), 0.25, rtol=1e-12
    )

    # A field rising linearly in radius: its mass integral does not depend on
    # the shell grid it is averaged over.
    field = (profile.radius - profile.r_core) / (profile.r_int - profile.r_core)
    totals = []
    for n in (1, 7, 64):
        edges = shell_edges(profile, n)
        m = shell_masses(profile, edges)
        totals.append(float(np.sum(shell_average(profile, edges, field) * m)))
    assert totals[1] == pytest.approx(totals[0], rel=1e-12)
    assert totals[2] == pytest.approx(totals[0], rel=1e-12)
    # A single shell reduces to the mass-weighted mean of the field. The
    # shell volume growing as r**2 pulls it above the midpoint value of one
    # half; the denser base pulls it back below the 0.598 a uniform-density
    # mantle of this geometry gives, so dropping either weight is caught.
    single = shell_average(profile, shell_edges(profile, 1), field)[0]
    assert 0.55 < single < 0.59

    # Error contract: a mantle needs at least one shell.
    with pytest.raises(ValueError, match='at least one shell'):
        shell_edges(profile, 0)


@pytest.mark.physics_invariant
def test_mantle_shells_carry_the_mantle_mass_under_their_overburden(tmp_path, monkeypatch):
    """Shells sum to the mantle mass, and pressure falls outward through them.

    The mesh integrates to 3% more than the structure's mantle mass, within
    what the schema check allows, so the rescaling onto the structure's mass
    is exercised. Each shell's density is its mass over its own volume.
    """
    profile = _model(tmp_path, monkeypatch, mass_scale=1.03).profile
    shells = mantle_shells(profile, 9, _M_MANTLE)

    assert float(np.sum(shells.mass)) == pytest.approx(_M_MANTLE, rel=1e-12)
    volume = (4.0 * np.pi / 3.0) * np.diff(shells.radius**3)
    np.testing.assert_allclose(shells.density * volume, shells.mass, rtol=1e-12)
    # Monotonicity: overburden pressure falls toward the surface.
    assert np.all(np.diff(shells.pressure) < 0.0)
    # The base shell is denser than the top one, as the mesh is.
    assert shells.density[0] > shells.density[-1]

    # Error contract: a mantle without mass cannot be split.
    with pytest.raises(ValueError, match='must be positive'):
        mantle_shells(profile, 9, 0.0)


@pytest.mark.physics_invariant
def test_tidal_power_is_the_rate_times_the_mass_summed_over_the_shells():
    """The tidal heat deposited is each shell's rate times its mass, summed.

    Asymmetric rates on unequal shells, so averaging the rate before
    multiplying by the total mass (4.67e14 W) is told apart from the correct
    sum (7.0e14 W).
    """
    tides = np.array([1e-9, 3e-9, 0.0])  # W kg-1
    mass = np.array([1e23, 2e23, 5e22])  # kg

    power = tidal_power(tides, mass)
    assert power == pytest.approx(7.0e14, rel=1e-12)
    assert abs(power - np.mean(tides) * np.sum(mass)) > 1e14
    assert power > 0.0

    # Edge case: no tidal heating anywhere deposits nothing.
    assert tidal_power(np.zeros(3), mass) == pytest.approx(0.0, abs=1e-30)

    # Error contract: rates on a different radial grid cannot be summed.
    with pytest.raises(ValueError, match='disagree on the radial grid'):
        tidal_power(tides[:2], mass)
