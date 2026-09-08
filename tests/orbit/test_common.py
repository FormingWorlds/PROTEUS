"""Unit tests for ``proteus.orbit.common``: the planetary moment of
inertia and the ``Tides_t`` container shared by the star-planet and
planet-satellite tidal evolution models.

Exercises:

- ``get_C_planet``: pinned against the exact uniform-density-sphere
  moment of inertia ``C = (2/5) M R^2``, and the SPIDER array-reversal
  branch, which is shown to silently flip the sign of the result when
  the reversal is skipped for surface-first input.
- ``Tides_t``: interaction lookup/registration contract.

Anti-happy-path coverage:

- ``Tides_t.get`` on an unregistered (primary, perturber) pair must
  raise ``KeyError`` rather than returning a default.
- ``get_C_planet``'s SPIDER-reversal branch is exercised both correctly
  (module='spider' on surface-first input) and incorrectly (surface-first
  input without the flag), pinning that the wrong branch does not merely
  drift but flips the sign of C_planet outright.

The Kepler solver and Hansen-coefficient machinery previously covered
here now live in ``proteus.orbit.hansen``; see ``test_hansen.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from proteus.interior_energetics.common import Interior_t
from proteus.orbit.common import Tides_t, get_C_planet

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# get_C_planet
# ---------------------------------------------------------------------------


def _uniform_sphere_interior(R: float, rho0: float, nlev_b: int) -> tuple[Interior_t, float]:
    interior = Interior_t(nlev_b=nlev_b)
    interior.radius = np.linspace(0.0, R, nlev_b)
    interior.density = np.full(nlev_b - 1, rho0)
    M = (4.0 / 3.0) * np.pi * R**3 * rho0
    return interior, M


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_get_c_planet_matches_uniform_density_sphere_analytic_value():
    """For a spatially uniform density, the shell-sum in ``get_C_planet``
    reduces to the exact textbook moment of inertia of a solid sphere,
    ``C = (2/5) M R^2`` -- exactly, not just approximately, because the
    per-shell integral ``rho * (r1^5 - r0^5) / 5`` is exact for constant
    rho regardless of how finely the shells are spaced.
    """
    R, rho0 = 6.371e6, 5500.0
    interior, M = _uniform_sphere_interior(R, rho0, nlev_b=8)
    hf_row: dict = {'M_int': M, 'R_int': R}
    cfg = cast(Any, SimpleNamespace(interior_energetics=SimpleNamespace(module='aragog')))

    get_C_planet(hf_row, cfg, interior)

    expected = (2.0 / 5.0) * M * R**2
    assert hf_row['C_planet'] == pytest.approx(expected, rel=1e-12)
    # Discrimination guard: the classic wrong-prefactor bugs for a solid
    # sphere are 1/3 (thin shell) and 1/2 (disk); both are far outside a
    # 1e-6 relative window around 2/5.
    assert abs(hf_row['C_planet'] / (M * R**2) - 1.0 / 3.0) > 0.05
    assert abs(hf_row['C_planet'] / (M * R**2) - 1.0 / 2.0) > 0.1
    # Sanity/scale guard: C_factor for a uniform sphere is exactly 0.4,
    # comfortably inside the physically reasonable [0.2, 0.4] range for
    # real (centrally condensed) planets quoted in the source's own log
    # message.
    assert 0.2 < hf_row['C_planet'] / (M * R**2) <= 0.4


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_get_c_planet_spider_reversal_recovers_cmb_first_ordering():
    """SPIDER emits interior arrays surface-first; ``get_C_planet``
    reverses them when ``config.interior_energetics.module == 'spider'``
    so that index 0 lines up with the CMB, matching the ordering every
    other caller uses. This test pins that the reversal branch produces
    the SAME value as directly supplying CMB-first arrays, using a
    non-uniform (core-mantle-crust) density profile so that reversal
    order actually matters (a uniform profile would pass even with a
    silently broken reversal).
    """
    R = 6.371e6
    r_edges_cmb_first = np.array([0.0, 0.5 * R, 0.8 * R, R])
    rho_cmb_first = np.array([9000.0, 5000.0, 3000.0])  # core -> mantle -> crust

    r0, r1 = r_edges_cmb_first[:-1], r_edges_cmb_first[1:]
    expected_C = (8 * np.pi / 3.0) * np.sum(rho_cmb_first * (r1**5 - r0**5) / 5.0)
    M = np.sum(rho_cmb_first * (4.0 / 3.0 * np.pi * (r1**3 - r0**3)))

    def run(radius, density, module):
        interior = Interior_t(nlev_b=4)
        interior.radius = radius.copy()
        interior.density = density.copy()
        hf_row: dict = {'M_int': M, 'R_int': R}
        cfg = cast(Any, SimpleNamespace(interior_energetics=SimpleNamespace(module=module)))
        get_C_planet(hf_row, cfg, interior)
        return hf_row['C_planet']

    # Non-SPIDER caller supplying already CMB-first arrays: no reversal
    # needed, must match the independently hand-summed expected value.
    c_direct = run(r_edges_cmb_first, rho_cmb_first, module='aragog')
    assert c_direct == pytest.approx(expected_C, rel=1e-12)

    # SPIDER caller supplying surface-first arrays: the reversal branch
    # must recover the identical physical answer.
    c_spider = run(r_edges_cmb_first[::-1], rho_cmb_first[::-1], module='spider')
    assert c_spider == pytest.approx(expected_C, rel=1e-12)

    # Sign guard: a positive density profile must give a positive moment
    # of inertia under the correct (CMB-first) pairing.
    assert expected_C > 0.0


@pytest.mark.physics_invariant
def test_get_c_planet_without_reversal_flag_flips_sign_on_surface_first_input():
    """Discriminating negative case for the reversal branch above: if
    surface-first arrays are supplied WITHOUT setting
    ``module == 'spider'``, the shell pairing ``(r0, r1) = (r_edges[:-1],
    r_edges[1:])`` sees a descending radius array, so ``r1 < r0`` for
    every shell and ``(r1**5 - r0**5)`` is negative throughout. The
    result is not a small numerical drift -- it is the exact negative of
    the physically correct value, which is what makes this bug loud
    rather than silent, and is the reason a caller mismatching the
    module flag would be caught immediately rather than producing a
    plausible-looking wrong answer.
    """
    R = 6.371e6
    r_edges_cmb_first = np.array([0.0, 0.5 * R, 0.8 * R, R])
    rho_cmb_first = np.array([9000.0, 5000.0, 3000.0])
    r0, r1 = r_edges_cmb_first[:-1], r_edges_cmb_first[1:]
    expected_C = (8 * np.pi / 3.0) * np.sum(rho_cmb_first * (r1**5 - r0**5) / 5.0)
    M = np.sum(rho_cmb_first * (4.0 / 3.0 * np.pi * (r1**3 - r0**3)))

    interior = Interior_t(nlev_b=4)
    interior.radius = r_edges_cmb_first[::-1].copy()
    interior.density = rho_cmb_first[::-1].copy()
    hf_row: dict = {'M_int': M, 'R_int': R}
    cfg = cast(Any, SimpleNamespace(interior_energetics=SimpleNamespace(module='dummy')))

    get_C_planet(hf_row, cfg, interior)

    assert hf_row['C_planet'] == pytest.approx(-expected_C, rel=1e-12)
    assert hf_row['C_planet'] < 0.0


# ---------------------------------------------------------------------------
# Tides_t
# ---------------------------------------------------------------------------


def test_tides_t_get_raises_keyerror_for_unregistered_interaction():
    """Error-contract case: querying an interaction that was never
    ``add``-ed must raise ``KeyError`` rather than returning ``None`` or
    a default, since callers (``sp1d``) index straight into ``.nmk``
    without a None-check.
    """
    tides = Tides_t()
    with pytest.raises(KeyError):
        tides.get(primary='planet', perturber='star')
    # No side effect: a failed lookup must not have registered anything.
    assert tides.interactions == []


def test_tides_t_add_is_idempotent_for_the_same_pair():
    """Calling ``add`` twice for the same (primary, perturber) pair must
    return the SAME object both times, not create a duplicate
    interaction (``sp1d`` calls ``.get`` repeatedly assuming a single
    canonical entry per pair).
    """
    tides = Tides_t()
    first = tides.add(primary='planet', perturber='star')
    second = tides.add(primary='planet', perturber='star')
    assert first is second
    assert len(tides.interactions) == 1
    # A different perturber must create a genuinely distinct entry.
    third = tides.add(primary='planet', perturber='moon')
    assert third is not first
    assert len(tides.interactions) == 2
