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
    energy_residual_rtol: float
        Relative tolerance on cumulative coupled energy residual against gross throughput [-].
        Sized above the ``2.51e-4`` maximum gross-normalized residual in post-closure runs
        after the first post-initialisation step on default time-step configurations.
    step_energy_rtol: float
        Relative tolerance on per-step energy residual increment normalized by per-step gross flux
        outside the ``301 K`` temperature clip [-]. In post-closure default runs outside the
        clip, the per-step increment has 99th percentile ``5.4e-4``, 99.9th percentile ``2.7e-2``,
        and maximum ``0.117``.
    step_energy_smin_rtol: float
        Relative tolerance on per-step energy residual increment normalized by per-step gross flux
        when ``T_surf <= 301 K`` or ``T_magma <= 301 K`` [-]. Sized above the ``4.39%`` maximum
        observed on the ``301 K`` temperature clip in post-closure runs.
    solver_residual_rtol: float
        Relative tolerance on the Aragog ODE solver internal energy residual [-], sized above the
        ``2.46e-14`` maximum observed in converged post-closure runs.
    core_residual_rtol: float
        Relative tolerance on the CMB core cooling ledger residual [-], sized above floating-point
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
        sized above the ``1.23e-2`` maximum radial-trapezoidal core mass difference on Zalmoxis
        meshes.
    element_closure_rtol: float
        Relative tolerance on per-element reservoir closure ``atm + magma + solid = total`` [-],
        sized at ``2.5`` times the default ``outgas.solver_rtol = 1.0e-4``.
        For ``outgas.module = 'calliope'``, the tolerance is ``outgas.mass_thresh + element_closure_rtol * total``
        (only as tight as ``outgas.mass_thresh``, which is larger than typical noble-gas inventories).
    element_closure_atol_kg: float
        Absolute floor on per-element reservoir closure for ``atmodeller`` and solid-trapping bookkeeping [kg],
        and lower-bound floor ``-element_closure_atol_kg`` on volatile and reservoir masses.
        For ``outgas.module = 'calliope'``, the closure check uses ``outgas.mass_thresh + element_closure_rtol * total``
        and is only as tight as ``outgas.mass_thresh``.
    volatile_gas_sum_rtol: float
        Relative tolerance on volatile gas species sum versus ``M_vol_atm`` [-], matching
        ``coupler.assert_mass_conservation``. Helpfiles without ``M_vol_atm`` fail the gas-sum check.
    planet_mass_cap_rtol: float
        Relative tolerance on ``M_atm <= M_planet`` [-], matching ``coupler.assert_mass_conservation``.
    escape_bound_factor: float
        Multiplicative bound on element loss relative to ``esc_kg_cumulative`` [-] when
        ``planet.fO2_source = 'from_O_budget'``.
    escape_bound_atol_kg: float
        Absolute tolerance on element loss relative to ``esc_kg_cumulative`` [kg] when
        ``planet.fO2_source = 'from_O_budget'``.
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
    core_mass_target_rtol: float = field(default=2.0e-2, validator=(finite_float, gt(0.0)))

    element_closure_rtol: float = field(default=2.5e-4, validator=(finite_float, gt(0.0)))
    element_closure_atol_kg: float = field(default=1.0, validator=(finite_float, gt(0.0)))
    volatile_gas_sum_rtol: float = field(default=1.0e-6, validator=(finite_float, gt(0.0)))
    planet_mass_cap_rtol: float = field(default=1.0e-6, validator=(finite_float, ge(0.0)))
    escape_bound_factor: float = field(default=1.5, validator=(finite_float, ge(1.0)))
    escape_bound_atol_kg: float = field(default=1.0e3, validator=(finite_float, ge(0.0)))

    t_magma_max: float = field(default=1.0e6, validator=(finite_float, gt(0.0)))
    phi_global_atol: float = field(default=1.0e-6, validator=(finite_float, ge(0.0)))
