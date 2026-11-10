"""
Unit tests for proteus.interior_energetics.adiabat: the depth-resolved melt state.

Covers the mantle profile handed over by the interior structure module, the
adiabat integrated down it from the magma temperature, the melt fraction
between the pressure-dependent solidus and liquidus, and the mass- and
volume-weighted aggregation of that profile.

Testing standards and documentation:
- docs/test_infrastructure.md: Test infrastructure overview
- docs/test_categorization.md: Test marker definitions
- docs/test_building.md: Best practices for test construction

Functions tested:
- read_mantle_mesh(): Load and check the structure module's mantle profile
- integrate_adiabat(): dT/dP = nabla_ad T / P down the mantle
- melt_profile(): Local melt fraction and the adiabat-liquidus crossing
- liquidus_crossing_radius(): Bracketed crossing of the liquidus
- volume_melt_fraction(), mass_melt_fraction(): Shell-weighted aggregation
- solve_melt_state(): The three chained together
- evaluate_melting_curve(): Melting curve sampled over the mantle pressures
- resolve_melting_dir(), load_melting_curves(): Melting curve resolution
- uses_paleos_melting_curves(), load_paleos_melting_curves(): The PALEOS
  curves used whenever the structure module is Zalmoxis
- zalmoxis_nabla_ad(), AdiabatMeltModel: The Zalmoxis-backed gradient

Invariants exercised: positivity of pressure, density and temperature
throughout the profile; boundedness of the melt fraction in [0, 1]; monotonic
increase of temperature with depth along an adiabat; the analytical isentrope
T = T_top (P/P_top)**nabla_ad at constant gradient; and the ordering
Phi_mass < Phi_vol for a mantle whose density increases inward; and the
PALEOS liquidus pinned against Fei et al. (2021) and Belonoshko et al. (2005),
with the solidus below it by the mushy-zone factor at every pressure.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from proteus.interior_energetics.adiabat import (
    MAX_LEVELS,
    AdiabatMeltModel,
    MantleProfile,
    evaluate_melting_curve,
    integrate_adiabat,
    liquidus_crossing_radius,
    load_melting_curves,
    load_paleos_melting_curves,
    mass_melt_fraction,
    melt_profile,
    mesh_identity,
    read_mantle_mesh,
    resolve_melting_dir,
    solve_melt_state,
    uses_paleos_melting_curves,
    volume_melt_fraction,
    zalmoxis_nabla_ad,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Earth-like mantle geometry, used so the pressures the melting curves are
# sampled at span the range the tabulated curves were fitted over.
R_CORE = 3.48e6
R_INT = 6.371e6
N_LEVELS = 60


def _make_arrays(rho_cmb: float = 5500.0, rho_top: float = 4300.0, n: int = N_LEVELS):
    """Mantle arrays with a density that increases inward, hydrostatic in P."""
    radius = np.linspace(R_CORE, R_INT, n)
    frac = (radius - R_CORE) / (R_INT - R_CORE)
    density = rho_cmb + (rho_top - rho_cmb) * frac
    gravity = np.full(n, 10.0)

    pressure = np.zeros(n)
    pressure[-1] = 1.0e5
    for i in range(n - 2, -1, -1):
        drho = 0.5 * (density[i] + density[i + 1])
        pressure[i] = pressure[i + 1] + drho * gravity[i] * (radius[i + 1] - radius[i])

    return radius, pressure, density, gravity


def _make_profile(**kwargs) -> MantleProfile:
    """A MantleProfile built directly, bypassing the on-disk mesh."""
    radius, pressure, density, gravity = _make_arrays(**kwargs)
    return MantleProfile(radius=radius, pressure=pressure, density=density, gravity=gravity)


def _make_log_pressure_profile(n: int = 200) -> MantleProfile:
    """A profile whose pressure is geometrically spaced from the CMB outward.

    Uniform steps in log pressure keep the first-order truncation of the
    explicit adiabat stepping the same at every level, which is what lets the
    analytical isentrope be pinned to a stated tolerance.
    """
    radius = np.linspace(R_CORE, R_INT, n)
    pressure = np.geomspace(1.4e11, 1.0e9, n)
    return MantleProfile(
        radius=radius,
        pressure=pressure,
        density=np.full(n, 4500.0),
        gravity=np.full(n, 10.0),
    )


def _write_mesh(outdir, radius, pressure, density, gravity, temperature=None):
    """Write a five-column mantle mesh in the structure module's format."""
    data_dir = os.path.join(str(outdir), 'data')
    os.makedirs(data_dir, exist_ok=True)
    if temperature is None:
        temperature = np.full(len(radius), 3000.0)
    path = os.path.join(data_dir, 'zalmoxis_output.dat')
    np.savetxt(
        path, np.column_stack([radius, pressure, density, gravity, temperature]), fmt='%.17e'
    )
    return path


def _mesh_config(
    melting_dir=None, module='dummy', mantle_eos='PALEOS:MgSiO3', mushy_zone_factor=0.8
):
    """A Config-shaped namespace with only the fields the module reads."""
    return SimpleNamespace(
        interior_struct=SimpleNamespace(
            module=module,
            melting_dir=melting_dir,
            zalmoxis=SimpleNamespace(
                mantle_eos=mantle_eos, mushy_zone_factor=mushy_zone_factor
            ),
        )
    )


def _hf_row(radius, density, m_core=1.9e24):
    """Helpfile row consistent with the mass the mesh itself integrates to."""
    m_mantle = float(np.trapezoid(4.0 * np.pi * radius**2 * density, radius))
    return {
        'R_int': float(radius[-1]),
        'R_core': float(radius[0]),
        'M_core': m_core,
        'M_int': m_mantle + m_core,
    }


# ============================================================================
# read_mantle_mesh
# ============================================================================


@pytest.mark.physics_invariant
def test_read_mantle_mesh_parses_the_structure_columns_in_order(tmp_path):
    """The mesh columns reach the fields the structure module documents.

    A swap between the density and gravity columns is the failure this guards:
    both are positive and smooth, so nothing downstream would complain, but the
    adiabat would then be integrated against a gravity of a few thousand. The
    density is deliberately non-uniform and an order of magnitude away from the
    gravity, so a swap moves every value far outside tolerance.
    """
    radius, pressure, density, gravity = _make_arrays()
    _write_mesh(tmp_path, radius, pressure, density, gravity)

    profile = read_mantle_mesh(_mesh_config(), str(tmp_path), _hf_row(radius, density))

    np.testing.assert_allclose(profile.density, density, rtol=1e-12)
    np.testing.assert_allclose(profile.gravity, gravity, rtol=1e-12)
    # The temperature column is dropped: it belongs to the structure solve, not
    # to the adiabat anchored at the evolving magma temperature.
    assert not hasattr(profile, 'temperature')
    # Positivity and ordering invariants across the whole profile.
    assert np.all(profile.pressure > 0.0)
    assert np.all(profile.density > 0.0)
    assert np.all(np.diff(profile.radius) > 0.0)
    assert np.all(np.diff(profile.pressure) < 0.0)
    assert profile.r_int == pytest.approx(R_INT, rel=1e-12)
    assert profile.r_core == pytest.approx(R_CORE, rel=1e-12)


def test_read_mantle_mesh_reports_a_missing_structure_handover(tmp_path):
    """A mesh the structure module never wrote is named in the error.

    The boundary energetics backend cannot invent a density profile, so the
    absence of the handover file has to surface rather than fall through to an
    assumed mantle.
    """
    with pytest.raises(FileNotFoundError) as excinfo:
        read_mantle_mesh(_mesh_config(), str(tmp_path), {'R_int': R_INT})

    message = str(excinfo.value)
    assert 'zalmoxis_output.dat' in message
    # The escape hatch is named so a user without the reference data can run.
    assert 'linear' in message


@pytest.mark.parametrize(
    'corruption',
    ['too_few_columns', 'descending_radius', 'negative_density', 'single_level'],
    ids=[
        'three_column_file',
        'radius_written_surface_first',
        'density_column_went_negative',
        'mesh_truncated_to_one_level',
    ],
)
def test_read_mantle_mesh_rejects_a_malformed_profile(tmp_path, corruption):
    """A mesh that cannot support an adiabat is refused, not silently used."""
    radius, pressure, density, gravity = _make_arrays()
    hf_row = _hf_row(radius, density)
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    path = data_dir / 'zalmoxis_output.dat'

    if corruption == 'too_few_columns':
        np.savetxt(path, np.column_stack([radius, pressure, density]), fmt='%.17e')
    elif corruption == 'descending_radius':
        cols = np.column_stack([radius, pressure, density, gravity, gravity])[::-1]
        np.savetxt(path, cols, fmt='%.17e')
    elif corruption == 'negative_density':
        density = density.copy()
        density[3] = -1.0
        _write_mesh(tmp_path, radius, pressure, density, gravity)
    else:
        cols = np.column_stack([radius, pressure, density, gravity, gravity])[-1:]
        np.savetxt(path, cols, fmt='%.17e')

    written = path.read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        read_mantle_mesh(_mesh_config(), str(tmp_path), hf_row)
    # No side effect: the refused mesh is left on disk as the structure wrote it.
    assert path.read_bytes() == written


def _with_unfilled_surface_node(radius, pressure, density, gravity):
    """Append the unfilled outer node a structure solve leaves as padding."""
    r_pad = radius[-1] + (radius[-1] - radius[-2])
    return (
        np.append(radius, r_pad),
        np.append(pressure, 0.0),
        np.append(density, 0.0),
        np.append(gravity, gravity[-1]),
    )


@pytest.mark.physics_invariant
def test_read_mantle_mesh_drops_an_unfilled_surface_node(tmp_path):
    """The unfilled outer node is padding, not a corrupt level.

    The damped-Picard structure solve reaches a zero surface-pressure target
    between its two outermost nodes on the first call, before outgassing, and
    writes the outer one with zero density and zero pressure. Rejecting it made
    every outer_solver='picard' run fail at its first interior call. It is
    dropped, and every level the solve actually filled survives unchanged.
    """
    radius, pressure, density, gravity = _make_arrays()
    padded = _with_unfilled_surface_node(radius, pressure, density, gravity)
    _write_mesh(tmp_path, *padded)

    profile = read_mantle_mesh(_mesh_config(), str(tmp_path), _hf_row(padded[0], padded[2]))

    assert len(profile.radius) == len(radius)
    np.testing.assert_allclose(profile.radius, radius, rtol=1e-12)
    np.testing.assert_allclose(profile.pressure, pressure, rtol=1e-12)
    np.testing.assert_allclose(profile.density, density, rtol=1e-12)
    assert np.all(profile.density > 0.0)
    assert np.all(profile.pressure > 0.0)
    assert profile.r_int == pytest.approx(R_INT, rel=1e-12)


def test_read_mantle_mesh_still_rejects_an_interior_zero_density(tmp_path):
    """Dropping surface padding does not reach inside the mantle.

    Only the trailing run of non-positive densities is padding. A zero density
    at an interior level is a genuinely broken profile and must still be
    refused, even when the mesh also carries a legitimate padding node.
    """
    radius, pressure, density, gravity = _make_arrays()
    density = density.copy()
    density[3] = 0.0
    padded = _with_unfilled_surface_node(radius, pressure, density, gravity)
    _write_mesh(tmp_path, *padded)

    with pytest.raises(ValueError, match='non-positive density') as excinfo:
        read_mantle_mesh(_mesh_config(), str(tmp_path), _hf_row(padded[0], padded[2]))
    # The error names the file, so the broken structure output can be found.
    assert 'zalmoxis_output.dat' in str(excinfo.value)


@pytest.mark.physics_invariant
def test_read_mantle_mesh_resamples_a_finely_resolved_profile(tmp_path):
    """A structure mesh finer than the adiabat cap is resampled, not truncated.

    The gradient lookup runs once per level, so a very fine mesh would dominate
    the cost of a step. Resampling has to preserve the end points and the
    monotonic pressure ordering, otherwise the crossing search would run on a
    profile that no longer reaches the core-mantle boundary.
    """
    n = MAX_LEVELS * 2
    radius, pressure, density, gravity = _make_arrays(n=n)
    _write_mesh(tmp_path, radius, pressure, density, gravity)

    profile = read_mantle_mesh(_mesh_config(), str(tmp_path), _hf_row(radius, density))

    assert len(profile.radius) == MAX_LEVELS
    assert profile.r_core == pytest.approx(R_CORE, rel=1e-12)
    assert profile.r_int == pytest.approx(R_INT, rel=1e-12)
    assert np.all(np.diff(profile.pressure) < 0.0)
    # The resampled pressure at the base still matches the fine mesh's, so the
    # deep mantle is not quietly decompressed by the resampling.
    assert profile.pressure[0] == pytest.approx(pressure[0], rel=1e-9)


def test_mesh_identity_tracks_a_structure_resolve(tmp_path):
    """Rewriting the mesh changes its identity, so a cached profile is dropped.

    A structure re-solve that left the cached profile in place would advance
    the interior on a frozen mantle, which is the silent failure this identity
    exists to catch.
    """
    radius, pressure, density, gravity = _make_arrays()
    assert mesh_identity(str(tmp_path)) == ()

    path = _write_mesh(tmp_path, radius, pressure, density, gravity)
    first = mesh_identity(str(tmp_path))
    assert first != ()

    _write_mesh(tmp_path, radius, pressure, density * 1.05, gravity)
    # Stamp an explicit modification time: the rewrite lands within the
    # filesystem's timestamp granularity of the first, and the two files are
    # the same size, so the identity would otherwise appear unchanged for a
    # reason that has nothing to do with the caching being tested.
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert mesh_identity(str(tmp_path)) != first


# ============================================================================
# integrate_adiabat
# ============================================================================


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_integrate_adiabat_recovers_the_constant_gradient_isentrope():
    """A constant adiabatic gradient integrates to T = T_top (P/P_top)**nabla.

    Analytical limit: dT/dP = nabla T / P separates to d(ln T) = nabla d(ln P),
    so the isentrope is a power law in pressure. The profile spans 1 GPa to
    140 GPa on a geometrically spaced pressure grid, so every step covers the
    same interval in log pressure and the first-order truncation of the
    explicit stepping is uniform and small. The pressure ratio of 140 resolves
    the exponent far above that error: nabla = 0.3 and nabla = 0.6 differ by a
    factor of 4.4 at the base.
    """
    profile = _make_log_pressure_profile()
    nabla = 0.3
    tmp_top = 2000.0

    temperature = integrate_adiabat(profile, tmp_top, lambda p, t: nabla)

    ratio = profile.pressure[0] / profile.pressure[-1]
    expected = tmp_top * ratio**nabla
    # rtol=2e-2: each step advances by nabla * dP / P evaluated at the shallower
    # level, which overstates the exponential increment by nabla * x**2 / 2 with
    # x = dlnP, accumulating to about 1.3 % over the 199 steps of this grid. The
    # closed form is the exact isentrope the stepping approximates.
    assert temperature[0] == pytest.approx(expected, rel=2e-2)

    # Exponent guard: a regression to nabla = 0.6 lands 4.4 times hotter.
    wrong = tmp_top * ratio ** (2.0 * nabla)
    assert abs(temperature[0] - wrong) > 2.0 * expected
    # Sign guard: the mantle is hotter at depth, never cooler.
    assert np.all(np.diff(temperature) < 0.0)
    # Scale guard: this isentrope reaches about 8800 K, not 880 K (a
    # Pa-versus-GPa slip) or 88000 K (a missing division by the pressure).
    assert 5.0e3 < temperature[0] < 2.0e4
    assert np.all(temperature > 0.0)


@pytest.mark.physics_invariant
def test_integrate_adiabat_is_isothermal_without_a_gradient():
    """A vanishing gradient leaves the mantle at the magma temperature.

    Limit-input behaviour: a gradient of zero, or one the equation of state
    cannot supply, must carry the anchor through rather than drift. The anchor
    itself is always preserved exactly.
    """
    profile = _make_profile()
    tmp_top = 1850.0

    flat = integrate_adiabat(profile, tmp_top, lambda p, t: 0.0)
    missing = integrate_adiabat(profile, tmp_top, lambda p, t: None)

    np.testing.assert_allclose(flat, tmp_top, rtol=1e-12)
    np.testing.assert_allclose(missing, tmp_top, rtol=1e-12)
    assert flat[-1] == pytest.approx(tmp_top, rel=1e-12)


def test_integrate_adiabat_refuses_a_non_physical_anchor():
    """A non-positive magma temperature is refused before it reaches the log-space stepping.

    The gradient form divides by the temperature, so a zero or negative anchor
    would propagate a meaningless profile into the melt fraction rather than
    failing where the mistake was made.
    """
    profile = _make_profile()

    with pytest.raises(ValueError, match='positive'):
        integrate_adiabat(profile, 0.0, lambda p, t: 0.3)
    with pytest.raises(ValueError, match='positive'):
        integrate_adiabat(profile, -1200.0, lambda p, t: 0.3)


# ============================================================================
# melt_profile and the liquidus crossing
# ============================================================================


@pytest.mark.physics_invariant
def test_melt_profile_is_bounded_and_brackets_the_crossing():
    """The melt fraction stays within [0, 1] and the front sits inside the mantle.

    The adiabat is warmer than the liquidus in the shallow mantle and cooler
    than it at depth, the ordering that produces a surface magma ocean over a
    solid lower mantle, so the crossing must fall strictly between the two
    bounding radii.
    """
    profile = _make_profile()
    n = len(profile.radius)
    frac = np.linspace(0.0, 1.0, n)

    # Melting curves rising inward faster than the adiabat, so the two cross.
    tmp_sol = 4200.0 - 2400.0 * frac
    tmp_liq = 4600.0 - 2400.0 * frac
    # Cooler than the liquidus at the base, warmer than it at the top, so a
    # magma ocean sits over a solid lower mantle and the crossing is interior.
    temperature = 3400.0 - 800.0 * frac

    phi, r_liq = melt_profile(profile, temperature, tmp_sol, tmp_liq)

    assert np.all(phi >= 0.0)
    assert np.all(phi <= 1.0)
    assert R_CORE < r_liq < R_INT
    # Melting increases outward here, so the shallow mantle is the molten part.
    assert phi[-1] > phi[0]
    assert phi[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'offset,expected_r,expected_phi',
    [(4000.0, R_CORE, 1.0), (-4000.0, R_INT, 0.0)],
    ids=['mantle_entirely_above_its_liquidus', 'mantle_entirely_below_its_solidus'],
)
def test_melt_profile_pins_the_front_at_the_bounding_radii(offset, expected_r, expected_phi):
    """A wholly molten or wholly solid mantle puts the front at a mantle boundary.

    Edge case: with no crossing inside the mantle the front has to sit at the
    core-mantle boundary or at the surface, and the melt fraction at the
    corresponding extreme, rather than at an interpolated radius that no level
    supports.
    """
    profile = _make_profile()
    n = len(profile.radius)
    frac = np.linspace(0.0, 1.0, n)
    tmp_sol = 4200.0 - 2400.0 * frac
    tmp_liq = 4600.0 - 2400.0 * frac
    temperature = tmp_liq + offset

    phi, r_liq = melt_profile(profile, temperature, tmp_sol, tmp_liq)

    assert r_liq == pytest.approx(expected_r, rel=1e-12)
    np.testing.assert_allclose(phi, expected_phi, atol=1e-12)
    assert volume_melt_fraction(profile, phi) == pytest.approx(expected_phi, abs=1e-12)


@pytest.mark.physics_invariant
def test_melt_profile_handles_a_degenerate_melting_curve():
    """A solidus and liquidus that coincide give a step, not a division by zero.

    Edge case at the limit of a vanishing mushy zone: the level is molten
    exactly at or above the coincident curve and solid below it, and no
    non-finite value may reach the aggregation.
    """
    profile = _make_profile()
    n = len(profile.radius)
    curve = np.full(n, 2500.0)
    warm = np.full(n, 2500.0 + 1.0)
    cold = np.full(n, 2500.0 - 1.0)

    phi_warm, _ = melt_profile(profile, warm, curve, curve)
    phi_cold, _ = melt_profile(profile, cold, curve, curve)
    phi_exact, _ = melt_profile(profile, curve.copy(), curve, curve)

    np.testing.assert_allclose(phi_warm, 1.0, atol=1e-12)
    np.testing.assert_allclose(phi_cold, 0.0, atol=1e-12)
    # A mantle sitting exactly on the coincident curve counts as molten.
    np.testing.assert_allclose(phi_exact, 1.0, atol=1e-12)
    assert np.all(np.isfinite(phi_warm))


def test_liquidus_crossing_radius_takes_the_shallowest_crossing():
    """With more than one crossing the base of the surface magma ocean wins.

    A melting curve that dips can put the adiabat back above the liquidus at
    depth. The physically meaningful front is the one bounding the connected
    molten layer below the surface, so the search runs inward from the top.
    """
    radius = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Molten at the surface, solid in between, molten again at depth.
    excess = np.array([10.0, 20.0, -5.0, 5.0, 15.0])

    r_liq = liquidus_crossing_radius(radius, excess)

    # Crossing between r=3 and r=4, at 3 + 5/10 = 3.5.
    assert r_liq == pytest.approx(3.5, rel=1e-12)
    # The deeper crossing between r=2 and r=3 is not the one reported.
    assert r_liq > 3.0


def test_liquidus_crossing_radius_handles_the_uncrossed_cases():
    """A profile with no crossing reports a mantle boundary, not an interpolation.

    Limit inputs: an adiabat above the liquidus everywhere leaves the front at
    the base of the mantle, and one below it at the top of the mantle, so the
    reported front stays inside the domain in both directions.
    """
    radius = np.array([1.0, 2.0, 3.0, 4.0])

    assert liquidus_crossing_radius(radius, np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(
        1.0, rel=1e-12
    )
    assert liquidus_crossing_radius(
        radius, np.array([-1.0, -2.0, -3.0, -4.0])
    ) == pytest.approx(4.0, rel=1e-12)


# ============================================================================
# Aggregation over the mantle shell
# ============================================================================


@pytest.mark.physics_invariant
def test_mass_melt_fraction_lags_volume_when_density_increases_inward():
    """A denser lower mantle makes the mass melt fraction the smaller of the two.

    The part that crystallises first is the deep mantle, and it is also the
    dense part, so weighting by density has to give a strictly lower melt
    fraction than weighting by volume alone. Dropping the density weighting
    would collapse the two, which is exactly what this discriminates.
    """
    profile = _make_profile(rho_cmb=5500.0, rho_top=4300.0)
    n = len(profile.radius)
    # Molten above the mid-mantle, solid below it.
    phi = np.where(profile.radius > 0.5 * (R_CORE + R_INT), 1.0, 0.0)

    phi_mass = mass_melt_fraction(profile, phi)
    phi_vol = volume_melt_fraction(profile, phi)

    assert phi_mass < phi_vol
    assert 0.0 < phi_mass < 1.0
    assert 0.0 < phi_vol < 1.0
    # The gap is resolvable, not a rounding artefact: a 28 % density contrast
    # over the shell moves the melt fraction by more than a percent.
    assert phi_vol - phi_mass > 1.0e-2
    assert len(phi) == n


@pytest.mark.physics_invariant
def test_melt_fractions_agree_on_a_uniform_mantle():
    """With uniform density the mass and volume melt fractions coincide.

    The density weighting cancels exactly when the density is constant, so this
    pins the aggregation against the closed-form shell integral and confirms
    the two forms share a normalisation.
    """
    profile = _make_profile(rho_cmb=4500.0, rho_top=4500.0)
    r_cut = 0.5 * (R_CORE + R_INT)
    phi = np.where(profile.radius > r_cut, 1.0, 0.0)

    phi_mass = mass_melt_fraction(profile, phi)
    phi_vol = volume_melt_fraction(profile, phi)

    assert phi_mass == pytest.approx(phi_vol, rel=1e-9)
    # Closed form for a sharp front at r_cut, discretised on the same grid.
    expected = (R_INT**3 - r_cut**3) / (R_INT**3 - R_CORE**3)
    # rtol=5e-2: the step in phi falls between grid levels, so the trapezoid
    # smooths it over one shell of the 60-level mesh.
    assert phi_vol == pytest.approx(expected, rel=5e-2)
    # Scale guard: a front at the mid-radius of an Earth-like mantle leaves
    # most of the volume molten, not a few percent.
    assert 0.5 < phi_vol < 0.95


@pytest.mark.physics_invariant
def test_solve_melt_state_chains_the_adiabat_into_the_melt_fraction():
    """The melt state rises monotonically with the magma temperature.

    Property-based check across the whole input range: a hotter mantle can only
    melt more of itself, the crystallisation front can only retreat inward, and
    both melt fractions stay bounded.
    """
    profile = _make_profile()
    n = len(profile.radius)
    frac = np.linspace(0.0, 1.0, n)
    tmp_sol = 4200.0 - 2400.0 * frac
    tmp_liq = 4600.0 - 2400.0 * frac

    states = [
        solve_melt_state(profile, tmp, lambda p, t: 0.25, tmp_sol, tmp_liq)
        for tmp in (1700.0, 2100.0, 2500.0, 2900.0)
    ]

    phis = [s.phi_vol for s in states]
    fronts = [s.r_liq for s in states]
    assert phis == sorted(phis)
    assert phis[0] < phis[-1]
    assert fronts == sorted(fronts, reverse=True)
    assert all(0.0 <= s.phi_mass <= 1.0 for s in states)
    assert all(np.all(s.temperature > 0.0) for s in states)


# ============================================================================
# Melting curve resolution
# ============================================================================


def test_resolve_melting_dir_falls_back_to_the_shipped_folder():
    """An unset melting curve folder resolves to the one shipped by default.

    A configuration that never mentions melting curves still gets a
    pressure-dependent melting model, rather than failing on a None path.
    """
    assert resolve_melting_dir(_mesh_config(melting_dir=None)) == 'Wolf_Bower+2018'
    assert resolve_melting_dir(_mesh_config(melting_dir='Monteux-600')) == 'Monteux-600'


def test_load_melting_curves_names_the_missing_file(monkeypatch, tmp_path):
    """A melting curve folder that is not on disk is reported with a fix.

    The error has to name both the download command and the melt model that
    needs no reference data, because an offline run has no other way forward.
    """
    monkeypatch.setattr('proteus.utils.data.FWL_DATA_DIR', tmp_path, raising=False)

    with pytest.raises(FileNotFoundError) as excinfo:
        load_melting_curves(_mesh_config(melting_dir='NotAFolder'))

    message = str(excinfo.value)
    assert 'proteus get interiordata' in message
    assert 'linear' in message


# Fei et al. (2021) pin their MgSiO3 liquidus to 6000 K at the 140 GPa
# core-mantle boundary of the Earth; Belonoshko et al. (2005) anchor the
# low-pressure branch at 1831 K at zero pressure. The tabulated Wolf and Bower
# (2018) liquidus reaches only about 4760 K at 140 GPa, so the two sets are
# told apart by far more than any tolerance used below.
P_FEI_ANCHOR = 140.0e9
T_FEI_ANCHOR = 6000.0
T_BELONOSHKO_ZERO = 1831.0
T_WOLF_BOWER_140GPA = 4760.0


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_paleos_melting_curves_follow_the_published_liquidus():
    """With the Zalmoxis structure module the mantle melts on the PALEOS curves.

    Published benchmark: the liquidus is Fei et al. (2021, Nat. Commun. 12,
    876) above 2.55 GPa and Belonoshko et al. (2005) below, and the solidus is
    the liquidus scaled by the mushy-zone factor. A factor of 0.75 is used
    rather than the default 0.8 so a solidus built on a hardcoded default would
    miss its pin.
    """
    pytest.importorskip('zalmoxis')

    config = _mesh_config(module='zalmoxis', mushy_zone_factor=0.75)
    solidus, liquidus = load_melting_curves(config)

    t_liq = float(liquidus(P_FEI_ANCHOR))
    # rel=1e-12: the Fei branch is exactly 6000 (P/140 GPa)**0.26, which is
    # 6000 K at the anchor with no fitting residual.
    assert t_liq == pytest.approx(T_FEI_ANCHOR, rel=1e-12)
    # One bar sits 1e-5 GPa above zero, which moves the Belonoshko branch by
    # 0.013 K; rel=1e-4 absorbs that and nothing larger.
    assert float(liquidus(1.0e5)) == pytest.approx(T_BELONOSHKO_ZERO, rel=1e-4)
    assert float(solidus(P_FEI_ANCHOR)) == pytest.approx(0.75 * T_FEI_ANCHOR, rel=1e-12)

    # The tabulated default folder must not leak back in: it is over 1200 K
    # cooler at the same pressure.
    assert abs(t_liq - T_WOLF_BOWER_140GPA) > 1000.0
    # Scale guard: a curve fed GPa where it expects Pa would return the
    # zero-pressure intercept (about 1831 K) here, not a lower-mantle value.
    assert 5000.0 < t_liq < 7000.0

    # The two branches meet at 2.55 GPa, so the melt fraction sees no jump at
    # the switch (both evaluate to 2117.8 K there).
    below, above = float(liquidus(2.5499e9)), float(liquidus(2.5501e9))
    assert above == pytest.approx(below, rel=1e-4)

    # Across the whole mantle range the solidus sits strictly below the
    # liquidus and both rise with pressure.
    pressure = np.geomspace(1.0e5, 1.0e12, 200)
    t_sol_all, t_liq_all = solidus(pressure), liquidus(pressure)
    assert np.all(t_sol_all > 0.0)
    assert np.all(t_sol_all < t_liq_all)
    assert np.all(np.diff(t_liq_all) > 0.0)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('mushy_zone_factor', [0.7, 1.0], ids=['widest_mushy_zone', 'sharp'])
def test_zalmoxis_structure_bypasses_the_configured_melting_dir(
    monkeypatch, tmp_path, mushy_zone_factor
):
    """A melting curve folder is not read when the structure module is Zalmoxis.

    The folder named here does not exist, so reading it would raise. Under the
    Zalmoxis structure module the curves come from PALEOS instead, at both
    ends of the allowed mushy-zone factor: 0.7 is the widest mushy zone and
    1.0 collapses the solidus onto the liquidus, the degenerate case the melt
    profile treats as a sharp phase boundary.
    """
    pytest.importorskip('zalmoxis')
    monkeypatch.setattr('proteus.utils.data.FWL_DATA_DIR', tmp_path, raising=False)

    config = _mesh_config(
        melting_dir='NotAFolder', module='zalmoxis', mushy_zone_factor=mushy_zone_factor
    )
    assert uses_paleos_melting_curves(config)
    solidus, liquidus = load_melting_curves(config)

    pressure = np.geomspace(1.0e9, 3.0e11, 50)
    ratio = solidus(pressure) / liquidus(pressure)
    np.testing.assert_allclose(ratio, mushy_zone_factor, rtol=1e-12)
    assert float(liquidus(P_FEI_ANCHOR)) == pytest.approx(T_FEI_ANCHOR, rel=1e-12)

    # Discrimination: the same folder under the dummy structure module is read,
    # and its absence is an error, so the bypass is gated on the module.
    config.interior_struct.module = 'dummy'
    assert not uses_paleos_melting_curves(config)
    with pytest.raises(FileNotFoundError):
        load_melting_curves(config)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('scale', [0.72, 1.25], ids=['lowered', 'raised'])
def test_paleos_melt_scale_moves_both_curves(scale):
    """`interior_energetics.paleos_melt_scale` multiplies solidus and liquidus.

    Both curves scale by the same factor, so the mushy-zone ratio is unchanged,
    and the Fei et al. (2021) anchor moves to scale x 6000 K. Without the key
    (a config that predates it) the curves are the unscaled ones.
    """
    pytest.importorskip('zalmoxis')

    base = _mesh_config(module='zalmoxis', mushy_zone_factor=0.75)
    sol0, liq0 = load_melting_curves(base)

    scaled = _mesh_config(module='zalmoxis', mushy_zone_factor=0.75)
    scaled.interior_energetics = SimpleNamespace(paleos_melt_scale=scale)
    sol1, liq1 = load_melting_curves(scaled)

    pressure = np.geomspace(1.0e5, 1.0e12, 100)
    np.testing.assert_allclose(liq1(pressure), scale * liq0(pressure), rtol=1e-12)
    np.testing.assert_allclose(sol1(pressure), scale * sol0(pressure), rtol=1e-12)
    np.testing.assert_allclose(sol1(pressure) / liq1(pressure), 0.75, rtol=1e-12)
    assert float(liq1(P_FEI_ANCHOR)) == pytest.approx(scale * T_FEI_ANCHOR, rel=1e-12)
    # The unscaled curve is untouched by building a scaled one.
    assert float(liq0(P_FEI_ANCHOR)) == pytest.approx(T_FEI_ANCHOR, rel=1e-12)


def test_paleos_melting_curves_report_a_missing_dependency(monkeypatch):
    """Without Zalmoxis the PALEOS curves fail with both ways forward named.

    Zalmoxis is an optional dependency. When it is missing, the error has to
    name it and the constant-temperature melt model that needs no curves.
    """
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.startswith('zalmoxis'):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', _blocked)

    with pytest.raises(RuntimeError) as excinfo:
        load_paleos_melting_curves(_mesh_config(module='zalmoxis'))

    message = str(excinfo.value)
    assert 'zalmoxis' in message.lower()
    assert 'melt_model = "linear"' in message


@pytest.mark.physics_invariant
def test_adiabat_melt_model_samples_the_paleos_curves_under_zalmoxis(monkeypatch, tmp_path):
    """The melt model built on a Zalmoxis structure uses PALEOS curves throughout.

    The curves sampled on the mantle levels and the curves handed to the
    two-phase adiabatic gradient must be the same PALEOS pair, or the gradient
    would weight solid and liquid by a different melt fraction from the one
    the melt profile reports.
    """
    zm = pytest.importorskip('zalmoxis.melting_curves')

    captured = {}

    def _fake_nabla(config, cache, solidus, liquidus):
        captured['curves'] = (solidus, liquidus)
        # 5 K/GPa, a silicate-melt adiabat: shallower than the PALEOS
        # liquidus, so a mantle molten at the top crystallises at depth.
        return lambda p, t: 5.0e-9 * p / t

    monkeypatch.setattr('proteus.interior_energetics.adiabat.zalmoxis_nabla_ad', _fake_nabla)

    radius, pressure, density, gravity = _make_arrays()
    _write_mesh(tmp_path, radius, pressure, density, gravity)
    config = _mesh_config(melting_dir='Monteux-600', module='zalmoxis')
    model = AdiabatMeltModel(config, str(tmp_path), _hf_row(radius, density), {})

    expected_liq = zm.paleos_liquidus(model.profile.pressure)
    np.testing.assert_allclose(model.tmp_liq, expected_liq, rtol=1e-12)
    np.testing.assert_allclose(model.tmp_sol, 0.8 * expected_liq, rtol=1e-12)

    sol_fn, liq_fn = captured['curves']
    assert float(liq_fn(P_FEI_ANCHOR)) == pytest.approx(T_FEI_ANCHOR, rel=1e-12)
    assert float(sol_fn(P_FEI_ANCHOR)) == pytest.approx(0.8 * T_FEI_ANCHOR, rel=1e-12)

    # Edge: a top temperature just above the surface liquidus but below the
    # deep liquidus leaves the mantle partly molten, bounded in [0, 1].
    state = model.state(float(model.tmp_liq[-1]) + 10.0)
    assert 0.0 < state.phi_mass < 1.0
    assert np.all((state.phi >= 0.0) & (state.phi <= 1.0))


def test_evaluate_melting_curve_clamps_beyond_the_tabulated_range(caplog):
    """Pressures past the top of the table are evaluated at its edge, and reported.

    A deep mantle reaches past the pressure range a melting curve was fitted
    over. Leaving those levels undefined would put a non-finite value into the
    melt fraction, so they are clamped, and the extrapolation is surfaced
    rather than hidden.
    """
    knots = np.array([0.0, 1.0e11])

    def curve(pressure):
        values = np.interp(pressure, knots, np.array([2000.0, 5000.0]))
        return np.where(
            (np.asarray(pressure) >= knots[0]) & (np.asarray(pressure) <= knots[-1]),
            values,
            np.nan,
        )

    curve.x = knots
    pressure = np.array([1.0e10, 5.0e10, 1.4e11])

    with caplog.at_level('WARNING'):
        values = evaluate_melting_curve(curve, pressure, 'liquidus')

    assert np.all(np.isfinite(values))
    # The out-of-table level takes the value at the top of the table.
    assert values[-1] == pytest.approx(5000.0, rel=1e-12)
    # The in-table levels are untouched by the clamp.
    assert values[0] == pytest.approx(2300.0, rel=1e-12)
    assert any('liquidus' in record.message for record in caplog.records)


def test_evaluate_melting_curve_refuses_an_unusable_curve():
    """A curve that is undefined everywhere fails instead of poisoning the melt fraction.

    Without tabulated knots there is nothing to clamp to, so a non-finite
    result has to raise where the data problem is, not several steps later as
    a NaN melt fraction.
    """
    pressure = np.array([1.0e10, 5.0e10])

    with pytest.raises(ValueError, match='melting_dir') as excinfo:
        evaluate_melting_curve(lambda p: np.full(len(p), np.nan), pressure, 'solidus')
    # The message names the curve and counts every undefined level.
    assert 'Mantle solidus returned no value at 2 of the mantle levels' in str(excinfo.value)


# ============================================================================
# Zalmoxis-backed gradient
# ============================================================================


def test_zalmoxis_nabla_ad_reports_a_missing_dependency(monkeypatch):
    """Without Zalmoxis the adiabat model fails with both ways forward named.

    Zalmoxis is an optional dependency, so the boundary backend has to say that
    the constant-temperature melt model needs neither it nor the reference
    data.
    """
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.startswith('zalmoxis'):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', _blocked)

    with pytest.raises(RuntimeError) as excinfo:
        zalmoxis_nabla_ad(_mesh_config(), {}, lambda p: 2000.0, lambda p: 2500.0)

    message = str(excinfo.value)
    assert 'zalmoxis' in message.lower()
    assert 'linear' in message


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_zalmoxis_gradient_is_in_the_silicate_range(tmp_path):
    """The gradient from the Zalmoxis equation of state matches silicate melt.

    Cross-implementation check against the adiabatic gradient Zalmoxis uses for
    its own structure adiabat. Silicate melt runs at a few kelvin per gigapascal
    at upper-mantle pressures and falls toward the base of the mantle as the
    bulk modulus stiffens; a gradient outside a few to a few tens of kelvin per
    gigapascal would mean the lookup returned a nabla_ad on the wrong
    normalisation, or a dT/dP that was never converted.
    """
    pytest.importorskip('zalmoxis')

    radius, pressure, density, gravity = _make_arrays()
    _write_mesh(tmp_path, radius, pressure, density, gravity)
    config = _mesh_config()

    try:
        model = AdiabatMeltModel(config, str(tmp_path), _hf_row(radius, density), {})
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f'Zalmoxis EOS or melting curve data unavailable: {exc}')

    gradients = []
    for pres, tmp in ((1.0e10, 2500.0), (5.0e10, 3500.0), (1.3e11, 4500.0)):
        nabla = model.nabla_ad(pres, tmp)
        dtdp_per_gpa = nabla * tmp / pres * 1e9
        gradients.append(dtdp_per_gpa)
        # Sign guard: an adiabat that cooled with depth would be convectively
        # meaningless.
        assert nabla > 0.0
        # Scale guard: 1e-3 K/GPa would be a Pa-versus-GPa slip, 1e3 K/GPa a
        # missing division by the pressure.
        assert 2.0 < dtdp_per_gpa < 40.0

    # The gradient falls with depth as the bulk modulus stiffens.
    assert gradients[0] > gradients[-1]

    # The solidus sits below the liquidus at every level of the real curves.
    assert np.all(model.tmp_sol < model.tmp_liq)
    # And both rise with depth, so the deep mantle is harder to melt.
    assert model.tmp_liq[0] > model.tmp_liq[-1]
