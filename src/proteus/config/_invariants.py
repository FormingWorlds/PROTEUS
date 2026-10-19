# Run-row invariant configuration [invariants]
from __future__ import annotations

import math

from attrs import define, field
from attrs.validators import ge, gt, in_


def finite_float(instance, attribute, value) -> None:
    """Require a finite numeric value (excluding booleans)."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        raise ValueError(f"'{attribute.name}' must be a finite number, got {value!r}.")


def step_energy_smin_ge_base(instance, attribute, value) -> None:
    """Require S_min step energy tolerance to be at least step_energy_rtol."""
    if value < instance.step_energy_rtol:
        raise ValueError(
            f"'step_energy_smin_rtol' ({value}) must be >= 'step_energy_rtol' "
            f'({instance.step_energy_rtol}).'
        )


def desync_stop_ge_warn(instance, attribute, value) -> None:
    """Require structure mass desync stop tolerance to be at least the warning tolerance."""
    if value < instance.struct_mass_desync_warn_rtol:
        raise ValueError(
            f"'struct_mass_desync_rtol' ({value}) must be >= "
            f"'struct_mass_desync_warn_rtol' ({instance.struct_mass_desync_warn_rtol})."
        )


@define
class Invariants:
    """Run-row invariant verification settings and tolerances.

    Attributes
    ----------
    mode: str
        Enforcement mode for run-row invariants: ``'off'``, ``'report'``, or ``'stop'``.
        Energy checks are report-only in every active mode (status code 30 is reserved).
    energy_residual_rtol: float
        Relative tolerance on cumulative coupled energy residual against gross throughput [-],
        evaluated once prior post-init rows have non-zero gross throughput (report-only).
        Sized above the ``2.51e-4`` maximum gross-normalized residual across the 53 post-closure
        baseline corpus runs (coarser-step configs ``input/minimal.toml`` and ``minimal_quick``
        reach ``2.37e-3`` and ``4.69e-4``).
    step_energy_rtol: float
        Relative tolerance on per-step energy residual increment normalized by per-step gross flux
        when ``T_magma > T(P_surf, S_min) + 0.5 K`` (or for all rows when no P-S EOS table is
        present) [-] (report-only). In post-closure default runs outside the ``S_min`` temperature
        clip, the per-step increment has 99th percentile ``5.4e-4``, 99.9th percentile ``2.7e-2``,
        and maximum ``0.117``.
    step_energy_smin_rtol: float
        Relative tolerance on per-step energy residual increment normalized by per-step gross flux
        when ``T_magma <= T(P_surf, S_min) + 0.5 K`` [-] (report-only). Here ``T(P_surf, S_min)``
        is interpolated linearly in pressure at each row's ``P_surf`` on the ``S_min`` block of the
        run's cached P-S EOS table ``temperature_solid.dat`` (e.g. ``302.342 K`` in
        ``run_20261018_131253_ea68`` and ``323.208 K`` in ``tutorial_earth`` at ``1 bar`` for
        ``S_min = 106.116 J kg^-1 K^-1`` in the dynamic PALEOS-2phase ``MgSiO3`` tables, and
        ``260.0 K`` at ``P = 0`` for ``S_min = -84.786 J kg^-1 K^-1`` in the Wolf & Bower 2018
        1 TPa ``MgSiO3`` solid table; the raw P-T source ``paleos_mgsio3_tables_pt_proteus_solid.dat``
        has ``300.488 K`` at ``P = 2.660 GPa`` rather than surface pressure). The ``0.5 K`` margin
        bounds the smooth low-pressure P-S linear interpolation error (``<= 9.49e-3 K``) and
        static-mesh versus row ``P_surf`` drift (``-0.0042 K`` to ``+0.071 K`` across
        ``S_min``-clipped corpus rows; ``< 0.5 K`` for ``|P_mesh - P_surf| <= 1,748 bar``),
        remaining well below the first unclipped entropy-grid step (``+22.74 K`` to ``+30.00 K``).
        Sized above the ``4.39%`` maximum observed on the ``S_min`` temperature clip in
        post-closure runs.
    solver_residual_rtol: float
        Relative tolerance on the Aragog ODE solver internal energy residual normalized by gross
        throughput [-], evaluated once prior post-init rows have non-zero gross throughput
        (report-only) and sized above the ``2.46e-14`` maximum observed in converged post-closure
        runs.
    core_residual_rtol: float
        Relative tolerance on the CMB core cooling ledger residual [-], evaluated once prior
        post-init rows have non-zero gross throughput (report-only) and sized above floating-point
        round-off on the single-ODE core integration.
    struct_mass_desync_warn_rtol: float
        Warning threshold on ``struct_mass_desync_frac`` [-], sized above the ``4.71e-3`` maximum
        observed in post-closure runs.
    struct_mass_desync_rtol: float
        Stop threshold on ``struct_mass_desync_frac`` [-], sized at twice the ``5.0e-3`` warning
        threshold.
    interior_mass_target_rtol: float
        Relative tolerance on ``M_int`` versus ``M_int_target`` at a structure update row [-],
        sized above the ``2.05e-3`` maximum observed on Zalmoxis structure update rows.
    core_mass_target_rtol: float
        Relative tolerance on ``M_core`` versus ``M_core_target`` at a structure update row [-],
        sized at about ``1.6`` times the ``1.565e-2`` maximum radial-trapezoidal core mass
        difference on the default 1 M_Earth Zalmoxis structure run ``a627_default`` (99th
        percentile ``1.518e-2``, which lies outside the 68 finished post-closure default runs;
        ``1.004e-2`` maximum and ``9.79e-3`` 99th percentile across those 68 runs).
    element_closure_rtol: float
        Relative tolerance on per-element reservoir closure ``atm + magma + solid = total`` [-],
        sized at ``2.5`` times the default ``outgas.solver_rtol = 1.0e-4``.
        For ``outgas.module = 'calliope'``, the tolerance is ``outgas.mass_thresh + element_closure_rtol * max(total, 0)``
        (only as tight as ``outgas.mass_thresh``, which is larger than typical noble-gas inventories;
        a zero-budget leak is seen above ``mass_thresh``).
    element_closure_atol_kg: float
        Absolute floor on per-element reservoir closure when ``outgas.module != 'calliope'`` [kg],
        and lower-bound floor ``-element_closure_atol_kg`` on volatile and reservoir masses.
        For ``outgas.module = 'calliope'``, the closure check uses ``outgas.mass_thresh + element_closure_rtol * max(total, 0)``
        and is only as tight as ``outgas.mass_thresh`` (a zero-budget leak is seen above ``mass_thresh``).
    volatile_gas_sum_rtol: float
        Relative tolerance on volatile gas species sum versus ``M_vol_atm`` [-], matching
        ``coupler.assert_mass_conservation``. Helpfiles without ``M_vol_atm`` fail the
        ``M_vol_atm_species_sum`` check.
    planet_mass_cap_rtol: float
        Relative tolerance on ``M_atm <= M_planet`` [-], matching ``coupler.assert_mass_conservation``.
    escape_bound_factor: float
        Multiplicative bound on element loss relative to ``esc_kg_cumulative`` [-] when
        ``planet.fO2_source = 'from_O_budget'`` (report-only because sub-threshold element
        flooring in ``calc_new_elements`` is not credited to ``esc_kg_cumulative`` or ``M_desiccated``).
    escape_bound_atol_kg: float
        Absolute tolerance on element loss relative to ``esc_kg_cumulative`` [kg] when
        ``planet.fO2_source = 'from_O_budget'`` (report-only).
    t_magma_max: float
        Strict upper bound on ``T_magma`` [K] above physical magma-ocean temperatures.
    phi_global_atol: float
        Absolute tolerance around ``[0, 1]`` for ``Phi_global`` [-] for floating-point round-off.
    """

    mode: str = field(default='report', validator=in_(('off', 'report', 'stop')))

    energy_residual_rtol: float = field(default=2.6e-4, validator=(finite_float, gt(0.0)))
    step_energy_rtol: float = field(default=1.0e-2, validator=(finite_float, gt(0.0)))
    step_energy_smin_rtol: float = field(
        default=5.0e-2,
        validator=(finite_float, gt(0.0), step_energy_smin_ge_base),
    )
    solver_residual_rtol: float = field(default=1.0e-12, validator=(finite_float, gt(0.0)))
    core_residual_rtol: float = field(default=1.0e-4, validator=(finite_float, gt(0.0)))

    struct_mass_desync_warn_rtol: float = field(
        default=5.0e-3,
        validator=(finite_float, gt(0.0)),
    )
    struct_mass_desync_rtol: float = field(
        default=1.0e-2,
        validator=(finite_float, gt(0.0), desync_stop_ge_warn),
    )
    interior_mass_target_rtol: float = field(default=3.0e-3, validator=(finite_float, gt(0.0)))
    core_mass_target_rtol: float = field(default=2.5e-2, validator=(finite_float, gt(0.0)))

    element_closure_rtol: float = field(default=2.5e-4, validator=(finite_float, gt(0.0)))
    element_closure_atol_kg: float = field(default=1.0, validator=(finite_float, gt(0.0)))
    volatile_gas_sum_rtol: float = field(default=1.0e-6, validator=(finite_float, gt(0.0)))
    planet_mass_cap_rtol: float = field(default=1.0e-6, validator=(finite_float, ge(0.0)))
    escape_bound_factor: float = field(default=1.5, validator=(finite_float, ge(1.0)))
    escape_bound_atol_kg: float = field(default=1.0e3, validator=(finite_float, ge(0.0)))

    t_magma_max: float = field(default=1.0e6, validator=(finite_float, gt(0.0)))
    phi_global_atol: float = field(default=1.0e-6, validator=(finite_float, ge(0.0)))
