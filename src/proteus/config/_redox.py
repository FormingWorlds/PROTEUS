"""
Unified-redox-module configuration (#57).

Declares the `[redox]` TOML section and its attrs dataclass. Defaults
to `mode = 'static'` so pre-plan TOMLs load unchanged: the solver is
disabled and `hf_row['fO2_shift_IW']` is pinned to the legacy
`config.outgas.fO2_shift_IW` value every step.

Evolving-redox is opt-in via `mode = 'fO2_init'` (seed from
`outgas.fO2_shift_IW`, then let the solver drift) or
`mode = 'composition'` (seed from `MantleComp.Fe3_frac` via the
chosen oxybarometer).

Plan: claude-config/plans/let-s-brainstorm-about-this-calm-canyon.md
v6 §3.5.
"""
from __future__ import annotations

from attrs import define, field, validators

# Allowed string tokens for the runtime dispatchers in
# `proteus.redox.buffers`. Kept here (duplicated from buffers.py) to
# avoid the circular import that would result from attrs validators
# trying to reach into the runtime dispatch table.
_OXYBAROMETERS = (
    'schaefer2024',
    'hirschmann2022',
    'sossi2020',
    'stagno2013_peridotite',
)
_BUFFERS = ('IW', 'QFM', 'NNO')
_MODES = ('static', 'fO2_init', 'composition')


@define
class Redox:
    """
    Top-level configuration for the unified-redox-module (#57).

    Attributes
    ----------
    mode : str
        Redox regime:
          - 'static' (default): solver disabled, `hf_row['fO2_shift_IW']`
            pinned to `config.outgas.fO2_shift_IW` for every step.
            Pre-plan behaviour preserved. RB diagnostics still written
            as zeros or passive post-hoc quantities; they do not feed
            back into the outgas/interior solve.
          - 'fO2_init': seed from `outgas.fO2_shift_IW` at t=0; the
            redox solver runs each step and updates `fO2_shift_IW`
            per the Evans/Hirschmann redox-budget bookkeeping.
          - 'composition': seed from `MantleComp.Fe3_frac` at t=0 by
            evaluating the configured `oxybarometer`; solver runs
            each step thereafter.
    rtol : float
        Relative tolerance for the outer Brent solver over ΔIW.
    atol : float
        Absolute tolerance for the outer Brent solver. Units: mol e-
        for the residual; must be several orders of magnitude below
        the signal (R_total ~ 1e23 mol e-).
    max_iter : int
        Maximum Brent iterations per step.
    bracket_halfwidth : float
        Initial half-width of the Brent bracket, in log10 fO2 units,
        centred on the warm start (previous step's ΔIW or Mariana's
        Schaefer-Eq-13 suggestion). The solver widens to ±2x on first
        failure before falling back to the previous ΔIW.
    oxybarometer : str
        Which Fe3+/FeT → log10 fO2 map is used in `composition` mode
        and as the default melt-surface oxybarometer when the
        magma-ocean regime is active (φ_max > φ_crit). Options:
        'schaefer2024', 'hirschmann2022', 'sossi2020',
        'stagno2013_peridotite'.
    buffer : str
        Reference buffer used to express `hf_row['fO2_shift_IW']`
        downstream. Options: 'IW', 'QFM', 'NNO'. 'IW' matches
        pre-plan PROTEUS convention.
    phi_crit : float
        Critical bulk-mantle melt fraction above which the melt-surface
        regime is MO-active (Schaefer Eq 13 domain). Below, the
        solver falls back to `stagno2013_peridotite`. Default 0.4
        follows Schaefer+24 §2.4.
    include_core : bool
        Include `R_budget_core` in the conservation closure. Disable
        for debugging.
    include_mantle : bool
        Include `R_budget_mantle` in the conservation closure.
    include_atm : bool
        Include `R_budget_atm` in the conservation closure.
    soft_conservation_tol : float
        Per-step ceiling on |ΔR_total − ΔR_escape − ΔR_dispro| as a
        fraction of |R_total|, above which `assert_redox_conserved`
        logs a warning. Default 1e-3.
    """

    mode: str = field(
        default='static',
        validator=validators.in_(_MODES),
    )

    rtol: float = field(default=1e-3, validator=validators.gt(0.0))
    atol: float = field(default=1e-4, validator=validators.gt(0.0))
    max_iter: int = field(default=20, validator=validators.gt(0))
    bracket_halfwidth: float = field(default=4.0, validator=validators.gt(0.0))

    oxybarometer: str = field(
        default='schaefer2024',
        validator=validators.in_(_OXYBAROMETERS),
    )
    buffer: str = field(
        default='IW',
        validator=validators.in_(_BUFFERS),
    )

    phi_crit: float = field(default=0.4, validator=validators.ge(0.0))

    include_core: bool = field(default=True)
    include_mantle: bool = field(default=True)
    include_atm: bool = field(default=True)

    soft_conservation_tol: float = field(default=1e-3, validator=validators.gt(0.0))
