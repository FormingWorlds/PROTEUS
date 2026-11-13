"""Mantle mesh and melt-model stand-ins shared by the adiabat melt-model tests.

The boundary energetics backend resolves its melt fraction on a mantle
adiabat (`proteus.interior_energetics.adiabat`) and splits the mantle into
shells (`proteus.interior_energetics.resolved_mantle`). Its tests and those of
the two modules need the same two inputs: a structure mesh on disk, and
stand-ins for the Zalmoxis adiabatic gradient and the tabulated melting curves,
which need an optional dependency and reference data. Both live here so every
test runs on identical inputs.

See `docs/How-to/testing.md` and `docs/Explanations/test_framework.md`.
"""

from __future__ import annotations

import os

import numpy as np

# Melting temperatures used when the curves are held constant with pressure [K]
ADIABAT_TSOL = 1500.0
ADIABAT_TLIQ = 2500.0

# Adiabat and melting-curve slopes used by the sloped tests, in K per pascal.
# The melting curves rise with pressure faster than the adiabat, the ordering
# that puts a magma ocean over a solid lower mantle. At 15 and 20 K/GPa both
# are in the range the tabulated silicate curves occupy.
ADIABAT_DTDP = 1.5e-8
CURVE_DTDP = 2.0e-8


def write_mantle_mesh(
    outdir,
    r_core,
    r_int,
    mantle_mass,
    rho_cmb=5500.0,
    rho_top=4300.0,
    n=60,
    mass_scale=1.0,
):
    """Write a structure mesh for a mantle between `r_core` and `r_int`.

    The density varies linearly in radius from `rho_cmb` to `rho_top`, so with
    the defaults it increases inward and the mass- and volume-weighted melt
    fractions genuinely differ; equal values give a uniform mantle. The profile
    is scaled onto `mantle_mass` times `mass_scale`, keeping the density
    contrast, since the structure's schema check allows a few percent between
    the mesh and the scalar mantle mass. The pressure is hydrostatic under a
    uniform gravity of 10 m s-2, from one bar at the top.

    Returns
    -------
    radius, pressure, density : np.ndarray
        The mesh as written, ascending in radius.
    """
    radius = np.linspace(r_core, r_int, n)
    frac = (radius - r_core) / (r_int - r_core)
    density = rho_cmb + (rho_top - rho_cmb) * frac
    density *= (
        mass_scale * mantle_mass / float(np.trapezoid(4 * np.pi * radius**2 * density, radius))
    )

    gravity = np.full(n, 10.0)
    pressure = np.zeros(n)
    pressure[-1] = 1e5
    for i in range(n - 2, -1, -1):
        pressure[i] = pressure[i + 1] + 0.5 * (density[i] + density[i + 1]) * gravity[i] * (
            radius[i + 1] - radius[i]
        )

    data_dir = os.path.join(str(outdir), 'data')
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(
        os.path.join(data_dir, 'zalmoxis_output.dat'),
        np.column_stack([radius, pressure, density, gravity, np.full(n, 3000.0)]),
        fmt='%.17e',
    )
    return radius, pressure, density


def patch_adiabat_inputs(
    monkeypatch,
    dtdp=0.0,
    curve_dtdp=0.0,
    tsol=ADIABAT_TSOL,
    tliq=ADIABAT_TLIQ,
):
    """Stand in for the Zalmoxis gradient and the tabulated melting curves.

    A constant dT/dP is used rather than a constant nabla_ad so the adiabat
    stays in the temperature range a mantle occupies across six decades of
    pressure, as the tabulated gradient does. With both slopes at zero the
    mantle is isothermal at the anchor temperature and the melting curves are
    constants, which is the limit in which the depth-resolved model must agree
    with the constant-temperature one.
    """
    monkeypatch.setattr(
        'proteus.interior_energetics.adiabat.zalmoxis_nabla_ad',
        lambda config, cache, sol, liq: lambda p, t: dtdp * p / t,
    )
    monkeypatch.setattr(
        'proteus.interior_energetics.adiabat.load_melting_curves',
        lambda config: (
            lambda p: tsol + curve_dtdp * np.asarray(p),
            lambda p: tliq + curve_dtdp * np.asarray(p),
        ),
    )


def count_adiabat_solves(monkeypatch):
    """Count the adiabat integrations behind every melt-state evaluation.

    Returns a one-element list whose entry is incremented on each call, so a
    test can assert how many times a melt curve went back to the adiabat.
    """
    from proteus.interior_energetics import adiabat

    calls = [0]
    original = adiabat.solve_melt_state

    def _counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr('proteus.interior_energetics.adiabat.solve_melt_state', _counted)
    return calls
