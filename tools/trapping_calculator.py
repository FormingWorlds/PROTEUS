#!/usr/bin/env python3
"""Offline calculator for solid-phase volatile trapping in a crystallising mantle.

Reads a completed PROTEUS ``runtime_helpfile.csv`` and computes, step by step,
the volatile mass a crystallising mantle would sequester into the solid. This is
a post-processing diagnostic: it writes a table and a plot, and never feeds back
into a simulation. Nothing here reads or writes PROTEUS configuration, and no
``_kg_solid`` value reaches PROTEUS state. In the real implementation at each
timestep PROTEUS will be informed by the trapping module, resulting in modified
volatile inventories and a different thermal evolution. This in turn will affect
the trapping calculation, providing a feedback loop. The offline calculation here
is a diagnostic of the trapping process, not a simulation of the coupled system.

Trapping model
--------------
Each increment of crystallised mantle mass ``dM_RM`` is treated as a mixture of
true crystals (mass fraction ``1 - F_tl``) and trapped interstitial melt (mass
fraction ``F_tl``). Crystals accept a species at the crystal/melt partition
coefficient ``D_Z``; trapped melt carries the full melt concentration. The
effective partition coefficient is therefore

    D_eff = (1 - F_tl) * D_Z + F_tl

and the mass sequestered over one step is

    dm_trap = D_eff * C_Z * dM_RM,   with   C_Z = {s}_kg_liquid / M_mantle_liquid

where ``C_Z`` is the mass fraction of the species in the melt, so ``dm_trap``
carries units of kg. Only ``D_Z`` varies between species; ``F_tl`` is shared.
This is the constant-``F_tl`` form attributed by the task specification to Sim
et al. (2024), Eq. 6.

Trapping modes
--------------
The bracket above is evaluated once per species, with ``F_tl`` supplied by the
selected mode. The mass-balance algebra is identical across modes.

``none``
    No trapping. Every trapped mass is identically zero. The bracket is not
    evaluated at all: this is the absence of the process, not ``F_tl = 0``,
    which would still bury a species at ``D_Z``. The per-step table and totals
    are still written so the three modes produce directly comparable output.
``constant``
    ``F_tl`` is a fixed scalar, default 0.01, usual range 0.01 to 0.05. Sim et
    al. (2024) report a constant comparison case at ``F_tl = 0.01``.
``dynamic``
    ``F_tl`` is recomputed at every step from the secular cooling rate, after
    Sim et al. (2024), Eq. 7:

        F_tl(t) = -(phi_c * tau / DeltaT) * dT/dt

    clamped to ``[0, phi_c]``. See the provenance note below.
``both``
    Not a fourth prescription for ``F_tl``. It runs the ``constant`` and
    ``dynamic`` calculations over the same helpfile and reports them together:
    the per-step table gains a second block of rows, told apart by the existing
    ``mode`` column, and the plot draws both cumulative curves on one pair of
    axes. Nothing is averaged or otherwise
    combined between the two, and neither result differs from what the same
    mode produces on its own. The pairing is constant against dynamic because
    those are the two competing prescriptions; ``none`` is the zero baseline
    and is available through ``--compare-modes``.

Provenance of the dynamic form
------------------------------
Sim et al. (2024), Eq. 7 defines the trapped melt fraction in terms of the
disaggregation melt fraction ``phi_c``, the compaction time scale ``tau``, the
freezing-front-to-solidus temperature difference ``DeltaT``, and the secular
cooling rate ``dT/dt``. Those four terms, their symbols, and the values used
(``phi_c = 0.3``, ``tau = 1 Ma``, ``DeltaT = 100 C``) are quoted directly from
the paper's Section 2.2. The paper further states that the trapped melt fraction
is bounded above by the disaggregation melt fraction, and that the equation
breaks down beyond that limit.

The prefactor is checked against the model the paper was run with, published
at https://github.com/joycesim/MOE (commit e7edd1c, 2024-09-26). Line 1241 of
its ``mars_module.py`` reads

    self.Ftl[ii] = -self.phic * self.tau * self.dTdt[ii] / self.deltaT

which is the form coded here. That source sets ``phic = 0.3``,
``deltaT = 100`` K and a default ``tau = 3.15e13`` s, which is 0.999 Ma, and its
constant comparison branch sets ``Ftl = 0.01``. All three defaults below match
it. ``tau`` is carried in years here rather than seconds because the helpfile
Time column is in years; the product ``tau * dT/dt`` is dimensionless either
way.

Three differences from that source are deliberate:

* It clamps with a literal ``if Ftl >= 0.3``, so its bound does not follow a
  changed ``phi_c``. The clamp here tracks ``phi_c``, which is what both the
  paper and that source describe the bound as being.
* It has no lower clamp at all. Its ``dT/dt`` is the right-hand side of the
  cooling ODE for a monotonically cooling magma ocean, so the fraction never
  turns negative and the case never arises. A PROTEUS run can warm, so the
  lower clamp here is an addition; see the warming-step note below.
* It evaluates ``dT/dt`` analytically as that ODE right-hand side. A finished
  helpfile carries no ODE to evaluate, so the rate is estimated by differencing
  consecutive rows instead. The two converge as the step size falls, but on a
  coarse or uneven helpfile the difference is real.

On mass against volume: the paper's prose calls ``F_tl`` a trapped melt volume
fraction, while Eq. 6 uses it in a mass balance. The published source resolves
this in favour of mass, its concentrations satisfying ``M_Z = M_MO * C_Z``, so
the bracket multiplies mass fractions throughout. It is treated as a mass
fraction here on that basis.

Which temperature
-----------------
Sim et al. (2024) Section 2.1 models the thermal evolution of the magma ocean
"by balancing the changes in the potential temperature, T", and names ``dT/dt``
the secular cooling rate of that same potential temperature. The PROTEUS
helpfile analogue is ``T_pot``, the characteristic mantle potential temperature
in K, which every interior module writes. It is the default here. Use
``--temperature-column`` to difference a different column, for instance
``T_magma``, and note that the result is then no longer the paper's ``dT/dt``.

The lower clamp at zero is in neither the paper nor its published source. Sim
et al. model a monotonically cooling magma ocean, so warming steps never arise
and neither the paper nor the source prescribes anything for them. A PROTEUS
run can warm. A warming step here yields ``F_tl = 0``, so the step still buries
the species at ``D_Z``: no interstitial melt is retained, but crystal partitioning
continues. Warming steps are counted and reported separately from the clamp so the
choice is visible in the output.

Regime of validity
------------------
Sim et al. report that for Mars at ``tau = 1 Ma`` the residual mantle
crystallises fast enough throughout the evolution for the trapped melt to sit at
the disaggregation limit of 30%. A dynamic run that clamps on most of its steps
is reproducing that regime rather than misbehaving. The clamp counter is printed
for exactly this reason.

Partition coefficients
----------------------
Crystal/melt partition coefficients are defined for elements entering a crystal
lattice as point defects, not for intact molecules. Hydrogen enters nominally
anhydrous silicates as OH groups and proton substitutions (2H+ for Mg2+, H+ with
Al3+ for Si4+, 4H+ for Si4+; Keppler & Bolfan-Casanova 2006, Table 1). A
molecule such as CO, NH3 or SO2 cannot partition into a lattice as a molecule:
its carrier elements enter as chemically distinct defects (N3-, sulfide or
sulfate, carbonate or C-H) whose speciation is set by oxygen fugacity, not by
the gas-phase molecular identity. The coefficients measured in the literature
are therefore elemental. Aubaud, Hauri & Hirschmann (2004) report
D_H(olivine/melt) = 0.0017 +/- 0.0005, and the ``H2O`` entry below is
mass-fraction bookkeeping over hydrogen rather than molecular partitioning.

Every species other than H2O carries ``D_Z = 0``. That is a positive physical
statement, not a placeholder for a value still to be looked up: lattice
incorporation is negligible for everything except water, and no experimentally
determined crystal/melt coefficient exists for CO, NH3, SO2 or the other
molecular species, so supplying one would be unfounded. A species at
``D_Z = 0`` is still trapped, at ``F_tl * C_Z * dM_RM``, because trapped
interstitial melt physically carries whatever is dissolved in it, molecules
included. That term is species-agnostic and molecularly sound, so ``D_Z = 0``
removes the crystal term and not the trapping.

The H2O value is the olivine/melt hydrogen coefficient of Aubaud et al. (2004),
measured at 1 to 2 GPa and 1230 to 1380 C, so using it at magma-ocean pressures
is an extrapolation. Pyroxene values (opx 0.019, cpx 0.023; Aubaud et al. 2004,
2008) can be supplied as ``--d-z H2O=0.019`` for a sensitivity test.

The choice has limited leverage on the totals. The interstitial term dominates
the bracket throughout the regime, but by a margin that depends on where
``F_tl`` sits: since ``D_eff = (1 - F_tl) * D_Z + F_tl``, the crystal term adds
``(1 - F_tl) * D_Z / F_tl`` relative to the interstitial term, which for water
is 17% at ``F_tl = 0.01`` and 0.4% at the disaggregation limit ``F_tl = 0.3``.
Water is thus the one species whose ``D_Z`` matters at all, and mainly at the
low end of the constant-mode range; for every species at ``D_Z = 0`` the
leverage is exactly zero by construction and the trapped mass is set by ``F_tl``
alone.

Indexing convention
-------------------
``dM_RM`` at row ``i`` is ``M_mantle_solid[i] - M_mantle_solid[i-1]``, and it is
multiplied by the melt concentration at the same row ``i``, following the
specification. ``dT/dt`` at row ``i`` uses the same step, ``(T[i] - T[i-1]) /
(Time[i] - Time[i-1])``, so the trapped fraction and the crystallised mass refer
to the same interval. Row 0 has no preceding step and contributes nothing.

Caveats on dM_RM
----------------
``M_mantle_solid`` is a state variable, not a rate, so differencing it inherits
two artefacts of the coupled run. The interior wrapper clamps these masses at
zero and can pin ``Phi_global`` to its previous value under the prevent-warming
clamp, which yields ``dM_RM = 0`` steps that are numerical rather than physical.
A Zalmoxis structure re-solve also changes ``M_mantle`` itself, so a difference
taken across such a step mixes crystallisation with a remesh. Steps where
``M_mantle`` moved are counted and reported so the totals can be read with that
in mind.

Usage
-----
    python tools/trapping_calculator.py -i output/<run>/runtime_helpfile.csv
    python tools/trapping_calculator.py -i <helpfile> --mode constant --f-tl 0.03
    python tools/trapping_calculator.py -i <helpfile> --mode dynamic \
        --phi-c 0.3 --tau 1.0e6 --delta-t 100.0
    python tools/trapping_calculator.py -i <helpfile> --mode both --f-tl 0.01
    python tools/trapping_calculator.py -i <helpfile> --compare-modes

The plot drawn here carries trapped mass only. ``F_tl`` and the disaggregation
bound ``phi_c`` are written to the per-step table and plotted separately by
``tools/plot_trapping_fraction.py``, which draws them for one run or across
several.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# The physics is shared with the live coupling in the PROTEUS timestep loop.
# Both call these, so the offline diagnostic and the simulation cannot drift
# apart on the bracket, the prefactor, or the melt concentration.
from proteus.outgas.trapping import (
    effective_partition,
    melt_concentration,
    raw_trapped_fraction,
)

# Trapping modes that each produce one TrappingResult, in the order the
# comparison summary reports them.
TRAPPING_MODES = ('none', 'constant', 'dynamic')
# Runs constant and dynamic together. Deliberately kept out of TRAPPING_MODES:
# it yields a pair of results rather than one, so anything iterating the modes
# to build a single result per mode must not pick it up.
COMBINED_MODE = 'both'
# The two modes COMBINED_MODE pairs, in plotting and reporting order.
COMBINED_PAIR = ('constant', 'dynamic')
# What the command line accepts for --mode.
CLI_MODES = (*TRAPPING_MODES, COMBINED_MODE)
DEFAULT_MODE = 'constant'

# Trapped-melt fraction for the constant mode. Single scalar shared by every
# species. Sim et al. (2024) use 0.01 for their constant comparison case.
DEFAULT_F_TL = 0.01
# Advisory only: values outside this window draw a printed note, not an error.
F_TL_ADVISORY_RANGE = (0.01, 0.05)

# Dynamic-mode parameters, after Sim et al. (2024), Section 2.2.
# Disaggregation melt fraction [1]: the melt fraction at which grain-grain
# contiguity is lost. Doubles as the upper bound on F_tl.
DEFAULT_PHI_C = 0.3
# Compaction time scale [yr]. Sim et al. use 1 Ma in the main text, and
# estimate 10.8 Ma to 1.08 Ga for the martian magma ocean. The helpfile Time
# column is in years, so tau is carried in years to match.
DEFAULT_TAU_YR = 1.0e6
# Temperature difference between the freezing front and the solidus [K].
# Sim et al. restrict their calculations to 100 C, which as a difference is
# 100 K.
DEFAULT_DELTA_T_K = 100.0
# Column differenced for dT/dt. Sim et al. define dT/dt on the magma-ocean
# potential temperature, whose helpfile analogue is T_pot [K].
DEFAULT_TEMPERATURE_COLUMN = 'T_pot'

# Crystal/melt partition coefficients, one per species. The set is every
# volatile PROTEUS tracks in the helpfile: the CALLIOPE volatile species
# followed by the noble gases. Species listed here that the helpfile cannot
# supply are skipped with a note rather than failing the run, so the same set
# serves runs that outgassed only a subset.
#
# Only H2O is non-zero, and only because the measured coefficient is elemental
# (hydrogen in the olivine lattice) rather than molecular. Every other entry is
# zero as a physical statement about lattice incorporation, not as a missing
# value; the interstitial-melt term traps all of them regardless. See the
# module docstring, 'Partition coefficients'.
DEFAULT_D_Z: dict[str, float] = {
    'H2O': 0.0017,
    'CO2': 0.0,
    'O2': 0.0,
    'H2': 0.0,
    'CH4': 0.0,
    'CO': 0.0,
    'N2': 0.0,
    'NH3': 0.0,
    'S2': 0.0,
    'SO2': 0.0,
    'H2S': 0.0,
    'He': 0.0,
    'Ne': 0.0,
    'Ar': 0.0,
    'Kr': 0.0,
    'Xe': 0.0,
}

# Columns the calculation cannot proceed without.
REQUIRED_COLUMNS = ('Time', 'Phi_global', 'M_mantle_solid', 'M_mantle_liquid')
# Per-species columns, formatted with the species name.
SPECIES_COLUMNS = ('{}_kg_liquid', '{}_kg_total')

# Relative change in M_mantle above which a step is counted as a remesh.
REMESH_RTOL = 1.0e-6

# Wong colourblind-friendly palette.
WONG_COLOURS = (
    '#0072B2',
    '#D55E00',
    '#009E73',
    '#CC79A7',
    '#E69F00',
    '#56B4E9',
    '#F0E442',
    '#000000',
)

# Species drawn on the cumulative plot before the cap applies. The default is
# the palette length, so the drawn species always carry distinct colours. The
# full species set is roughly twice that, and a figure carrying all of it is
# unreadable whatever the styling, so the cap is on by default and the species
# it leaves out are named in a printed note. ``--plot-top 0`` lifts it.
DEFAULT_PLOT_TOP = len(WONG_COLOURS)
# Line styles advanced once per full colour cycle, so a run that lifts the cap
# still gets a unique (colour, style) pair per species instead of two curves
# drawn identically. Only the single-mode figure can use this: the combined
# figure spends line style on the mode.
SPECIES_STYLES = ('-', '--', '-.', ':')

DEFAULT_OUTPUT_DIR = Path('output_files') / 'trapping'


class MissingColumnError(ValueError):
    """Raised when the helpfile lacks a column the calculation requires."""


@dataclass
class DynamicTrapping:
    """Per-step trapped-melt fraction from the secular cooling rate.

    ``f_tl`` is the clamped series actually used by the mass balance;
    ``f_tl_raw`` is the unclamped formula output, kept so a run can be inspected
    for how far outside the valid window it sat.
    """

    column: str
    phi_c: float
    tau: float
    delta_t: float
    dt_dt: np.ndarray
    f_tl_raw: np.ndarray
    f_tl: np.ndarray
    n_clamped_high: int
    n_clamped_low: int
    n_warming: int
    n_unusable: int

    @property
    def n_clamped(self) -> int:
        """Steps where the formula left the ``[0, phi_c]`` window."""
        return self.n_clamped_high + self.n_clamped_low


@dataclass
class SpeciesResult:
    """Per-species trapping record over the whole helpfile."""

    name: str
    d_z: float
    d_eff: float | np.ndarray | None
    c_z: np.ndarray
    dm_trap: np.ndarray
    cumulative: np.ndarray
    n_supply_clamped: int
    inventory_final: float

    @property
    def total_trapped(self) -> float:
        """Cumulative trapped mass at the last step [kg]."""
        return float(self.cumulative[-1]) if self.cumulative.size else 0.0

    @property
    def inventory_fraction(self) -> float:
        """Trapped mass as a fraction of the final whole-planet inventory."""
        if self.inventory_final <= 0.0:
            return float('nan')
        return self.total_trapped / self.inventory_final


@dataclass
class TrappingResult:
    """Trapping over every requested species, plus the step counters."""

    mode: str
    f_tl: float | np.ndarray | None
    time: np.ndarray
    dm_rm: np.ndarray
    species: list[SpeciesResult]
    n_negative_dm: int
    n_zero_melt: int
    n_remesh: int
    skipped: list[str]
    dynamic: DynamicTrapping | None = None


@dataclass
class CombinedResult:
    """The constant- and dynamic-mode runs over one helpfile, kept side by side.

    This is a pair of independent results, not a blend of them: each field is
    exactly what the corresponding single-mode run produces. It exists so the
    two competing prescriptions for ``F_tl`` can be tabulated and drawn on the
    same axes without either being averaged into the other.
    """

    constant: TrappingResult
    dynamic: TrappingResult

    @property
    def mode(self) -> str:
        """Name of the combined mode, mirroring ``TrappingResult.mode``."""
        return COMBINED_MODE

    def results(self) -> dict[str, TrappingResult]:
        """The two results keyed by mode name, in ``COMBINED_PAIR`` order."""
        return {'constant': self.constant, 'dynamic': self.dynamic}


def validate_mode(mode: str) -> str:
    """Return the mode unchanged, or refuse an unknown one by name.

    ``both`` is refused here as well. It is a valid command-line mode but not a
    valid single-result mode, so routing it into the single-result path is a
    caller error rather than a typo, and the message says which call to make.
    """
    if mode == COMBINED_MODE:
        raise ValueError(
            f'Mode {COMBINED_MODE!r} produces one result per mode in '
            f'{COMBINED_PAIR}, not a single result; call compute_both() instead.'
        )
    if mode not in TRAPPING_MODES:
        raise ValueError(f'Unknown trapping mode {mode!r}; expected one of {TRAPPING_MODES}')
    return mode


def crystallised_mass(m_solid: np.ndarray) -> tuple[np.ndarray, int]:
    """Per-step growth of the solid mantle [kg], with remelting clamped to zero.

    Returns the clamped increments and the number of steps that were negative
    before clamping. A negative increment means the mantle remelted over that
    step, which releases trapped volatiles rather than burying them; the release
    branch is out of scope here, so those steps contribute nothing.
    """
    solid = np.asarray(m_solid, dtype=float)
    dm = np.zeros(solid.size, dtype=float)
    if solid.size > 1:
        dm[1:] = np.diff(solid)
    n_negative = int(np.count_nonzero(dm < 0.0))
    return np.maximum(dm, 0.0), n_negative


def cooling_rate(time: np.ndarray, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Secular cooling rate [K yr-1] per step, with a usability mask.

    ``dT/dt`` at row ``i`` differences rows ``i-1`` and ``i``, so it spans the
    same interval as ``dM_RM``. Sim et al. instead evaluate the cooling ODE
    right-hand side analytically; a finished helpfile carries no such ODE, and
    the two estimates converge as the step size falls. Row 0 has no preceding step. A step is unusable
    when the helpfile repeats or reverses a timestamp, or carries a non-finite
    temperature; PROTEUS writes duplicate ``Time`` values at restart, so the
    zero-length step is a real case rather than a defensive flourish. Unusable
    steps report a zero rate and are excluded from the counters.
    """
    t = np.asarray(time, dtype=float)
    temp = np.asarray(temperature, dtype=float)
    rate = np.zeros(t.size, dtype=float)
    usable = np.zeros(t.size, dtype=bool)
    if t.size > 1:
        dt = np.diff(t)
        d_temp = np.diff(temp)
        step_ok = np.isfinite(dt) & np.isfinite(d_temp) & (dt > 0.0)
        step_rate = np.zeros(dt.size, dtype=float)
        np.divide(d_temp, dt, out=step_rate, where=step_ok)
        rate[1:] = step_rate
        usable[1:] = step_ok
    return rate, usable


def dynamic_trapped_fraction(
    time: np.ndarray,
    temperature: np.ndarray,
    column: str = DEFAULT_TEMPERATURE_COLUMN,
    phi_c: float = DEFAULT_PHI_C,
    tau: float = DEFAULT_TAU_YR,
    delta_t: float = DEFAULT_DELTA_T_K,
) -> DynamicTrapping:
    """Trapped-melt fraction per step from the cooling rate, after Sim Eq. 7.

    ``F_tl = -(phi_c * tau / DeltaT) * dT/dt``, clamped to ``[0, phi_c]``. The
    upper bound is the paper's: beyond the disaggregation melt fraction the
    relation is stated to break down, so the clamp marks invalid input rather
    than merely capping an extreme. The lower bound appears in neither the paper
    nor its published source, and handles warming steps, which a monotonically
    cooling magma ocean never produces; see the module docstring.
    """
    dt_dt, usable = cooling_rate(time, temperature)
    f_tl_raw = raw_trapped_fraction(dt_dt, phi_c, tau, delta_t)
    f_tl = np.where(usable, np.clip(f_tl_raw, 0.0, phi_c), 0.0)
    return DynamicTrapping(
        column=column,
        phi_c=float(phi_c),
        tau=float(tau),
        delta_t=float(delta_t),
        dt_dt=dt_dt,
        f_tl_raw=f_tl_raw,
        f_tl=f_tl,
        n_clamped_high=int(np.count_nonzero(usable & (f_tl_raw > phi_c))),
        n_clamped_low=int(np.count_nonzero(usable & (f_tl_raw < 0.0))),
        n_warming=int(np.count_nonzero(usable & (dt_dt >= 0.0))),
        n_unusable=int(np.count_nonzero(~usable[1:])) if usable.size > 1 else 0,
    )


def count_remesh_steps(frame: pd.DataFrame, rtol: float = REMESH_RTOL) -> int:
    """Steps where the total mantle mass moved, so dM_RM is not crystallisation alone."""
    if 'M_mantle' not in frame.columns:
        return 0
    mantle = frame['M_mantle'].to_numpy(dtype=float)
    if mantle.size < 2:
        return 0
    previous = np.abs(mantle[:-1])
    scale = np.where(previous > 0.0, previous, 1.0)
    return int(np.count_nonzero(np.abs(np.diff(mantle)) / scale > rtol))


def usable_species(frame: pd.DataFrame, d_z: dict[str, float]) -> tuple[list[str], list[str]]:
    """Split the requested species into those the helpfile supports and those it does not."""
    usable: list[str] = []
    skipped: list[str] = []
    for name in d_z:
        needed = [pattern.format(name) for pattern in SPECIES_COLUMNS]
        if all(column in frame.columns for column in needed):
            usable.append(name)
        else:
            skipped.append(name)
    return usable, skipped


def trap_species(
    frame: pd.DataFrame,
    name: str,
    d_z: float,
    f_tl: float | np.ndarray | None,
    dm_rm: np.ndarray,
) -> SpeciesResult:
    """Trapped mass per step for one species, capped by the melt it can draw on.

    ``f_tl`` of ``None`` is the no-trapping mode: the bracket is not evaluated
    and the trapped mass is identically zero. Otherwise the same bracket runs
    whether ``f_tl`` is a scalar or a per-step array.

    The supply cap keeps a long step from burying more of a species than the
    melt actually holds, which the unbounded product allows when dM_RM is large.
    """
    kg_liquid = np.maximum(frame[f'{name}_kg_liquid'].to_numpy(dtype=float), 0.0)
    c_z = melt_concentration(kg_liquid, frame['M_mantle_liquid'].to_numpy(dtype=float))
    inventory = frame[f'{name}_kg_total'].to_numpy(dtype=float)

    if f_tl is None:
        d_eff: float | np.ndarray | None = None
        raw = np.zeros(c_z.size, dtype=float)
    else:
        d_eff = effective_partition(f_tl, d_z)
        raw = d_eff * c_z * dm_rm

    dm_trap = np.minimum(raw, kg_liquid)
    return SpeciesResult(
        name=name,
        d_z=float(d_z),
        d_eff=d_eff,
        c_z=c_z,
        dm_trap=dm_trap,
        cumulative=np.cumsum(dm_trap),
        n_supply_clamped=int(np.count_nonzero(raw > kg_liquid)),
        inventory_final=float(inventory[-1]) if inventory.size else 0.0,
    )


def compute_trapping(
    frame: pd.DataFrame,
    mode: str = DEFAULT_MODE,
    f_tl: float = DEFAULT_F_TL,
    d_z: dict[str, float] | None = None,
    phi_c: float = DEFAULT_PHI_C,
    tau: float = DEFAULT_TAU_YR,
    delta_t: float = DEFAULT_DELTA_T_K,
    temperature_column: str = DEFAULT_TEMPERATURE_COLUMN,
) -> TrappingResult:
    """Run the trapping calculation over every species the helpfile can supply.

    The mode selects only how ``F_tl`` is obtained. ``none`` supplies nothing
    and short-circuits to zero, ``constant`` supplies the scalar ``f_tl``, and
    ``dynamic`` supplies a per-step series from the cooling rate. The mass
    balance downstream is the same in all three cases.

    Raises
    ------
    ValueError
        If ``mode`` is not one of ``TRAPPING_MODES``.
    MissingColumnError
        If any of the columns in ``REQUIRED_COLUMNS`` is absent, or, in dynamic
        mode, if the temperature column is absent. Species-level columns are not
        fatal: a species whose columns are missing is skipped and reported in
        ``TrappingResult.skipped``.
    """
    validate_mode(mode)
    coefficients = dict(DEFAULT_D_Z if d_z is None else d_z)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise MissingColumnError(
            'Helpfile is missing required column(s): '
            + ', '.join(missing)
            + '. Expected the PROTEUS runtime_helpfile.csv schema, which supplies '
            + ', '.join(REQUIRED_COLUMNS)
            + '. Check that the file is a helpfile and not a plot or archive export.'
        )

    dynamic: DynamicTrapping | None = None
    if mode == 'none':
        supplied: float | np.ndarray | None = None
    elif mode == 'constant':
        supplied = float(f_tl)
    else:
        if temperature_column not in frame.columns:
            raise MissingColumnError(
                f'Dynamic mode differences {temperature_column!r} for dT/dt, and the '
                'helpfile does not carry it. Available temperature columns: '
                + ', '.join(sorted(c for c in frame.columns if c.startswith('T_')))
                + '. Pass --temperature-column to select one, or use --mode constant.'
            )
        dynamic = dynamic_trapped_fraction(
            frame['Time'].to_numpy(dtype=float),
            frame[temperature_column].to_numpy(dtype=float),
            column=temperature_column,
            phi_c=phi_c,
            tau=tau,
            delta_t=delta_t,
        )
        supplied = dynamic.f_tl

    dm_rm, n_negative = crystallised_mass(frame['M_mantle_solid'].to_numpy(dtype=float))
    melt = frame['M_mantle_liquid'].to_numpy(dtype=float)
    names, skipped = usable_species(frame, coefficients)
    return TrappingResult(
        mode=mode,
        f_tl=supplied,
        time=frame['Time'].to_numpy(dtype=float),
        dm_rm=dm_rm,
        species=[
            trap_species(frame, n, float(coefficients[n]), supplied, dm_rm) for n in names
        ],
        n_negative_dm=n_negative,
        n_zero_melt=int(np.count_nonzero(melt <= 0.0)),
        n_remesh=count_remesh_steps(frame),
        skipped=skipped,
        dynamic=dynamic,
    )


def compute_both(
    frame: pd.DataFrame,
    f_tl: float = DEFAULT_F_TL,
    d_z: dict[str, float] | None = None,
    phi_c: float = DEFAULT_PHI_C,
    tau: float = DEFAULT_TAU_YR,
    delta_t: float = DEFAULT_DELTA_T_K,
    temperature_column: str = DEFAULT_TEMPERATURE_COLUMN,
) -> CombinedResult:
    """Run the constant and dynamic modes over one helpfile and keep both.

    Every argument is forwarded unchanged, so each half is identical to what
    ``compute_trapping`` returns for that mode alone. Nothing is averaged
    between them.

    Raises
    ------
    MissingColumnError
        From the dynamic half if ``temperature_column`` is absent, or from
        either half if a required column is. The constant half is computed
        first, so a helpfile that supports constant but not dynamic still
        raises rather than silently returning half a result.
    """
    shared = {
        'f_tl': f_tl,
        'd_z': d_z,
        'phi_c': phi_c,
        'tau': tau,
        'delta_t': delta_t,
        'temperature_column': temperature_column,
    }
    return CombinedResult(
        constant=compute_trapping(frame, mode='constant', **shared),
        dynamic=compute_trapping(frame, mode='dynamic', **shared),
    )


def build_table(result: TrappingResult, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-step table: mode, time, crystallised mass, and per-species trapping.

    Dynamic mode adds the cooling rate and both the raw and clamped trapped
    fractions, so a run can be read back without recomputing them.
    """
    data: dict[str, np.ndarray | list[str]] = {
        'mode': [result.mode] * int(result.time.size),
        'Time': result.time,
        'dM_RM': result.dm_rm,
    }
    if frame is not None and 'Phi_global' in frame.columns:
        data['Phi_global'] = frame['Phi_global'].to_numpy(dtype=float)
    if result.dynamic is not None:
        data[f'{result.dynamic.column}_dT_dt'] = result.dynamic.dt_dt
        data['F_tl_raw'] = result.dynamic.f_tl_raw
        data['F_tl'] = result.dynamic.f_tl
    elif result.mode == 'constant' and result.f_tl is not None:
        data['F_tl'] = np.full(result.time.size, float(result.f_tl))
    else:
        data['F_tl'] = np.zeros(result.time.size, dtype=float)
    for species in result.species:
        data[f'C_Z_{species.name}'] = species.c_z
        data[f'dm_trap_{species.name}'] = species.dm_trap
        data[f'cum_trap_{species.name}'] = species.cumulative
    return pd.DataFrame(data)


def build_combined_table(
    combined: CombinedResult, frame: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Both modes' per-step tables stacked, told apart by the ``mode`` column.

    The two modes carry different diagnostic columns, since dynamic adds the
    cooling rate and the unclamped fraction, so the stacked frame is the union
    of both column sets and the constant rows leave the dynamic-only columns
    empty. Stacking rather than widening keeps every column name identical to
    the single-mode tables, so a reader that already parses one mode parses
    this one unchanged.
    """
    tables = [build_table(result, frame) for result in combined.results().values()]
    return pd.concat(tables, ignore_index=True, sort=False)


def read_helpfile(path: Path, sep: str | None = None) -> pd.DataFrame:
    """Read a helpfile, defaulting to the whitespace delimiter PROTEUS writes.

    PROTEUS writes ``runtime_helpfile.csv`` tab-separated despite the extension,
    so a comma reader returns a single-column frame. The delimiter is taken from
    the header line unless one is supplied.
    """
    if sep is None:
        header = path.read_text(encoding='utf-8', errors='replace').split('\n', 1)[0]
        sep = ',' if ('\t' not in header and ',' in header) else r'\s+'
    return pd.read_csv(path, sep=sep, engine='python')


def describe_f_tl(result: TrappingResult) -> str:
    """One-line description of where F_tl came from, for the summary and plot."""
    if result.mode == 'none':
        return 'none (no trapping)'
    if result.mode == 'constant':
        return f'constant, F_tl = {float(result.f_tl):g}'
    dynamic = result.dynamic
    if dynamic is None:
        return 'dynamic, F_tl unavailable'
    used = np.asarray(dynamic.f_tl, dtype=float)
    span = f'{used.min():.4g} to {used.max():.4g}' if used.size else 'no steps'
    return f'dynamic, F_tl in [{span}]'


def summarise(result: TrappingResult) -> str:
    """Human-readable summary of the totals and the step counters."""
    lines = [
        'Solid-phase volatile trapping (offline diagnostic)',
        f'  mode                       : {describe_f_tl(result)}',
        f'  steps read                 : {result.time.size}',
        f'  remelting steps clamped    : {result.n_negative_dm}',
        f'  steps with no melt left    : {result.n_zero_melt}',
        f'  steps with a mantle remesh : {result.n_remesh}',
    ]
    dynamic = result.dynamic
    if dynamic is not None:
        lines += [
            f'  dT/dt column               : {dynamic.column}',
            f'  phi_c, tau, DeltaT         : {dynamic.phi_c:g} [1], '
            f'{dynamic.tau:g} yr, {dynamic.delta_t:g} K',
            f'  F_tl clamped to phi_c      : {dynamic.n_clamped_high} step(s)',
            f'  F_tl clamped up to zero    : {dynamic.n_clamped_low} step(s)',
            f'  warming steps (dT/dt >= 0) : {dynamic.n_warming}',
            f'  steps with unusable dt     : {dynamic.n_unusable}',
        ]
    if result.skipped:
        lines.append(f'  skipped (no columns)       : {", ".join(result.skipped)}')
    # Over the full species set most entries trap nothing, either because the
    # run never outgassed them or because no melt was left to draw on. They are
    # grouped onto one line so the species that do trap stay readable. When
    # nothing traps anywhere the mode is the zero baseline itself, and every
    # species keeps its full block so that baseline is still reported in full.
    trapping = [s for s in result.species if s.total_trapped > 0.0]
    silent = [s for s in result.species if s.total_trapped <= 0.0]
    for species in trapping or result.species:
        if species.d_eff is None:
            d_eff_text = 'not evaluated'
        else:
            values = np.asarray(species.d_eff, dtype=float)
            d_eff_text = (
                f'{float(values):.6g}'
                if values.ndim == 0
                else f'{values.min():.6g} to {values.max():.6g}'
            )
        percent = 100.0 * species.inventory_fraction
        lines += [
            f'  {species.name}: D_Z = {species.d_z:g}, D_eff = {d_eff_text}',
            f'    trapped        = {species.total_trapped:.4e} kg',
            f'    of inventory   = {percent:.4f} % of {species.inventory_final:.4e} kg',
            f'    supply-clamped = {species.n_supply_clamped} step(s)',
        ]
    if trapping and silent:
        lines.append(f'  trapped nothing            : {", ".join(s.name for s in silent)}')
    return '\n'.join(lines)


def compare_modes(
    frame: pd.DataFrame,
    f_tl: float = DEFAULT_F_TL,
    d_z: dict[str, float] | None = None,
    phi_c: float = DEFAULT_PHI_C,
    tau: float = DEFAULT_TAU_YR,
    delta_t: float = DEFAULT_DELTA_T_K,
    temperature_column: str = DEFAULT_TEMPERATURE_COLUMN,
) -> dict[str, TrappingResult]:
    """Run every mode on one helpfile so the totals can be read side by side."""
    return {
        mode: compute_trapping(
            frame,
            mode=mode,
            f_tl=f_tl,
            d_z=d_z,
            phi_c=phi_c,
            tau=tau,
            delta_t=delta_t,
            temperature_column=temperature_column,
        )
        for mode in TRAPPING_MODES
    }


def summarise_comparison(
    results: dict[str, TrappingResult], modes: tuple[str, ...] | None = None
) -> str:
    """Totals per species per mode, with the ratio each mode traps against constant.

    ``modes`` selects which of ``results`` to report and in what order,
    defaulting to all three single-result modes. The ``both`` mode passes its
    own pair so the summary covers exactly the two runs it performed.
    """
    ordered = TRAPPING_MODES if modes is None else tuple(modes)
    names: list[str] = []
    for mode in ordered:
        for species in results[mode].species:
            if species.name not in names:
                names.append(species.name)

    lines = [
        '',
        'Trapped mass by mode [kg]',
        f'  {"species":<8}' + ''.join(f'{mode:>16}' for mode in ordered),
    ]
    for name in names:
        totals = []
        for mode in ordered:
            match = [s for s in results[mode].species if s.name == name]
            totals.append(match[0].total_trapped if match else float('nan'))
        lines.append(f'  {name:<8}' + ''.join(f'{value:>16.4e}' for value in totals))

    baseline = results.get('constant') if 'constant' in ordered else None
    if baseline is not None:
        lines.append('')
        lines.append('Relative to constant mode')
        for name in names:
            match = [s for s in baseline.species if s.name == name]
            reference = match[0].total_trapped if match else 0.0
            if reference <= 0.0:
                lines.append(f'  {name:<8} constant traps nothing; ratio undefined')
                continue
            ratios = []
            for mode in ordered:
                other = [s for s in results[mode].species if s.name == name]
                total = other[0].total_trapped if other else 0.0
                ratios.append(total / reference)
            lines.append(f'  {name:<8}' + ''.join(f'{value:>16.4f}' for value in ratios))
    return '\n'.join(lines)


def summarise_both(combined: CombinedResult) -> str:
    """Both modes' full summaries, then their totals side by side."""
    return '\n'.join(
        [
            summarise(combined.constant),
            '',
            summarise(combined.dynamic),
            summarise_comparison(combined.results(), modes=COMBINED_PAIR),
        ]
    )


# Line style per mode on the combined plot. Colour is spent on species, so the
# two modes have to be told apart some other way.
COMBINED_STYLES = {'constant': '-', 'dynamic': '--'}


def _styled_axes(figsize: tuple[float, float]):
    """Figure and axes with the shared styling, returned with the pyplot module.

    Both plot functions draw cumulative trapped mass against time on identical
    axes, so the styling lives here rather than being repeated and drifting.
    matplotlib is imported inside the function because the calculation itself
    does not need it.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans']}
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Cumulative trapped mass [kg]')
    ax.set_yscale('log')
    ax.tick_params(direction='in', which='both', top=True, right=True)
    return plt, fig, ax


def log_time_axis(ax, time: np.ndarray) -> None:
    """Put the time axis on a log scale, starting at the first positive sample.

    A helpfile starts at t = 0, which a log axis cannot show, so the lower limit
    is the first positive time rather than the first row. A run with no positive
    time at all is left on the linear default. Shared with
    ``tools/plot_trapping_fraction.py`` so the two tools put time on the same
    axis.
    """
    positive = np.asarray(time, dtype=float)
    positive = positive[positive > 0.0]
    if positive.size:
        ax.set_xscale('log')
        ax.set_xlim(left=float(positive.min()))


def rank_species(
    species: list[SpeciesResult], top: int = DEFAULT_PLOT_TOP
) -> tuple[list[SpeciesResult], list[SpeciesResult]]:
    """Split species into those a figure draws and those the cap leaves out.

    Species are ordered by trapped mass so the cap drops the smallest
    contributors rather than whichever happened to be declared last. A run in
    which nothing traps at all carries no ranking to apply, so declaration
    order is kept and the figure still shows the zero baseline. ``top`` of zero
    or less lifts the cap.
    """
    if any(entry.total_trapped > 0.0 for entry in species):
        ordered = sorted(species, key=lambda entry: entry.total_trapped, reverse=True)
    else:
        ordered = list(species)
    if top <= 0 or len(ordered) <= top:
        return ordered, []
    return ordered[:top], ordered[top:]


def species_style(index: int) -> tuple[str, str]:
    """Colour and line style for the index-th species drawn on a figure.

    The palette holds eight colours, so the line style advances once per full
    colour cycle and the pair stays unique up to
    ``len(WONG_COLOURS) * len(SPECIES_STYLES)`` species. Without this the
    modulo on the palette alone would silently draw two species identically.
    """
    colour = WONG_COLOURS[index % len(WONG_COLOURS)]
    style = SPECIES_STYLES[(index // len(WONG_COLOURS)) % len(SPECIES_STYLES)]
    return colour, style


def describe_omitted(drawn: list[SpeciesResult], omitted: list[SpeciesResult]) -> str:
    """Note naming the species a figure left out, or an empty string if none.

    A silently truncated figure would read as a complete one, so the omitted
    species are named rather than merely counted.
    """
    if not omitted:
        return ''
    total = len(drawn) + len(omitted)
    names = ', '.join(entry.name for entry in omitted)
    return (
        f'Note: plot shows {len(drawn)} of {total} species; '
        f'omitted (smallest trapped mass): {names}'
    )


def plot_cumulative(result: TrappingResult, path: Path, top: int = DEFAULT_PLOT_TOP) -> None:
    """Write the cumulative trapped mass against time, one line per species.

    Mass is the only quantity drawn. The trapped-melt fraction that produced it
    is written to the per-step table and plotted by
    ``tools/plot_trapping_fraction.py``, which keeps this figure to a single
    axis and a single unit.

    The species set spans every volatile the helpfile carries, so the figure is
    capped at the ``top`` largest contributors and the rest are named in a
    printed note and counted in the legend title. Lifting the cap keeps the
    curves distinguishable by advancing the line style once the palette has
    been spent.
    """
    plt, fig, ax = _styled_axes((6.6, 4.4))
    drawn, omitted = rank_species(result.species, top)
    for index, species in enumerate(drawn):
        colour, style = species_style(index)
        ax.plot(
            result.time,
            species.cumulative,
            lw=1.8,
            ls=style,
            color=colour,
            label=f'{species.name} (D_Z = {species.d_z:g})',
        )
    log_time_axis(ax, result.time)
    ax.set_title(f'Solid-phase trapping: {describe_f_tl(result)}')
    ax.legend(
        frameon=False,
        loc='upper left',
        bbox_to_anchor=(0.0, 1.0),
        title=f'+{len(omitted)} not shown' if omitted else None,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    note = describe_omitted(drawn, omitted)
    if note:
        print(note)


def plot_combined(combined: CombinedResult, path: Path, top: int = DEFAULT_PLOT_TOP) -> None:
    """Write both modes' cumulative trapped mass on one pair of axes.

    Colour separates species and line style separates modes, so the same
    species can be read across the two prescriptions without a second panel.
    Because that spends both visual channels, the legend is split in two: one
    keyed by colour for the species, one keyed by line style for the modes.
    Neither trapped-melt fraction is drawn here; both are plotted by
    ``tools/plot_trapping_fraction.py``.

    Line style being spent on the mode leaves colour as the only channel for
    species here, so the cap cannot exceed the palette length however ``top``
    is set; a larger request is reduced to it rather than wrapping the palette
    and drawing two species in the same colour. Species are ranked by their
    constant-mode trapped mass, and the ones left out are named in a printed
    note.
    """
    plt, fig, ax = _styled_axes((7.4, 4.6))
    from matplotlib.lines import Line2D

    limit = len(WONG_COLOURS) if top <= 0 else min(top, len(WONG_COLOURS))
    drawn, omitted = rank_species(combined.constant.species, limit)
    for index, reference in enumerate(drawn):
        colour = WONG_COLOURS[index]
        for mode, result in combined.results().items():
            match = [s for s in result.species if s.name == reference.name]
            if not match:
                continue
            ax.plot(
                result.time,
                match[0].cumulative,
                lw=1.8,
                ls=COMBINED_STYLES[mode],
                color=colour,
            )
    log_time_axis(ax, combined.constant.time)
    ax.set_title('Melt trapping: constant and dynamic modes')

    species_keys = [
        Line2D(
            [],
            [],
            lw=1.8,
            color=WONG_COLOURS[index],
            label=f'{species.name} (D_Z = {species.d_z:g})',
        )
        for index, species in enumerate(drawn)
    ]
    mode_keys = [
        Line2D(
            [], [], lw=1.8, color='black', ls=COMBINED_STYLES[mode], label=describe_f_tl(result)
        )
        for mode, result in combined.results().items()
    ]
    # Two legends on one axes: the first has to be added as an artist or the
    # second replaces it. The mode legend carries a translucent ground because
    # the curves it describes end in that corner of the axes. Keep the two
    # stacked in the upper-left so the mode legend sits directly above the
    # species legend without overlapping the plot.
    ax.add_artist(
        ax.legend(
            handles=species_keys,
            frameon=False,
            loc='upper left',
            bbox_to_anchor=(0, 0.99),
            title=f'+{len(omitted)} not shown' if omitted else None,
        )
    )
    ax.legend(
        handles=mode_keys,
        loc='upper left',
        bbox_to_anchor=(0, 0.46),
        fontsize='small',
        frameon=True,
        framealpha=0.85,
        edgecolor='none',
    )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    note = describe_omitted(drawn, omitted)
    if note:
        print(note)


def parse_d_z(items: list[str] | None) -> dict[str, float]:
    """Parse repeated ``SPECIES=VALUE`` arguments into a partition dictionary.

    Supplying any override replaces the default set outright, so a run can be
    restricted to a single species without inheriting the defaults.
    """
    if not items:
        return dict(DEFAULT_D_Z)
    coefficients: dict[str, float] = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f'Expected SPECIES=VALUE for --d-z, got {item!r}')
        name, _, value = item.partition('=')
        coefficients[name.strip()] = float(value)
    return coefficients


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the calculator."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        '-i', '--input', type=Path, required=True, help='Path to runtime_helpfile.csv'
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        default=None,
        help='Destination for the per-step table (default: mode/source-specific name)',
    )
    parser.add_argument(
        '--plot',
        type=Path,
        default=None,
        help='Destination for the cumulative-trapping plot (default: mode/source-specific name)',
    )
    parser.add_argument('--no-plot', action='store_true', help='Skip the plot')
    parser.add_argument(
        '--plot-top',
        type=int,
        default=DEFAULT_PLOT_TOP,
        metavar='N',
        help=f'Draw only the N species that trap the most mass; 0 draws every '
        f'species (default {DEFAULT_PLOT_TOP}, the palette length). The combined '
        f'mode cannot exceed the palette length, since line style is spent on '
        f'the mode there.',
    )
    parser.add_argument(
        '--mode',
        choices=CLI_MODES,
        default=DEFAULT_MODE,
        help=f'How F_tl is obtained; {COMBINED_MODE!r} runs constant and dynamic '
        f'together and reports both (default {DEFAULT_MODE})',
    )
    parser.add_argument(
        '--compare-modes',
        action='store_true',
        help='Also print the totals every mode would produce on this helpfile',
    )
    parser.add_argument(
        '--f-tl',
        type=float,
        default=DEFAULT_F_TL,
        help=f'Trapped-melt fraction for constant mode (default {DEFAULT_F_TL:g})',
    )
    parser.add_argument(
        '--phi-c',
        type=float,
        default=DEFAULT_PHI_C,
        help=f'Disaggregation melt fraction [1] (default {DEFAULT_PHI_C:g})',
    )
    parser.add_argument(
        '--tau',
        type=float,
        default=DEFAULT_TAU_YR,
        help=f'Compaction time scale [yr] (default {DEFAULT_TAU_YR:g})',
    )
    parser.add_argument(
        '--delta-t',
        type=float,
        default=DEFAULT_DELTA_T_K,
        help=f'Freezing front to solidus temperature difference [K] '
        f'(default {DEFAULT_DELTA_T_K:g})',
    )
    parser.add_argument(
        '--temperature-column',
        default=DEFAULT_TEMPERATURE_COLUMN,
        help=f'Helpfile column differenced for dT/dt (default {DEFAULT_TEMPERATURE_COLUMN})',
    )
    parser.add_argument(
        '--d-z',
        action='append',
        metavar='SPECIES=VALUE',
        help='Partition coefficient override; repeatable. Replaces the defaults.',
    )
    parser.add_argument('--sep', default=None, help='Column delimiter (default: auto)')
    return parser


NO_SPECIES_MESSAGE = (
    'No requested species has both _kg_liquid and _kg_total columns; nothing to do.'
)


def write_outputs(table: pd.DataFrame, draw, output_csv: Path, plot: Path, make_plot: bool):
    """Write the per-step table and, unless suppressed, the plot.

    ``draw`` is called with the plot path only when a plot is wanted, so the
    matplotlib import stays out of a ``--no-plot`` run.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    print(f'Wrote per-step table to {output_csv}')
    if make_plot:
        plot.parent.mkdir(parents=True, exist_ok=True)
        draw(plot)
        print(f'Wrote plot to {plot}')


def main(argv: list[str] | None = None) -> int:
    """Read a helpfile, compute trapping, and write the table, plot, and summary."""
    args = build_parser().parse_args(argv)
    # Include both the selected mode and the input source in default names so
    # repeated runs do not silently overwrite one another. Explicit paths remain
    # completely under the caller's control.
    source = args.input.stem or 'helpfile'
    output_stem = f'trapping_{args.mode}_{source}'
    output_csv = args.output_csv or DEFAULT_OUTPUT_DIR / f'{output_stem}_table.csv'
    plot = args.plot or DEFAULT_OUTPUT_DIR / f'{output_stem}_cumulative.png'
    low, high = F_TL_ADVISORY_RANGE
    # The advisory covers every mode that consults the scalar, which includes
    # the combined mode: its constant half uses exactly the same value.
    if args.mode in ('constant', COMBINED_MODE) and not low <= args.f_tl <= high:
        print(f'Note: F_tl = {args.f_tl:g} is outside the usual {low:g} to {high:g} range.')

    frame = read_helpfile(args.input, sep=args.sep)
    settings = {
        'f_tl': args.f_tl,
        'd_z': parse_d_z(args.d_z),
        'phi_c': args.phi_c,
        'tau': args.tau,
        'delta_t': args.delta_t,
        'temperature_column': args.temperature_column,
    }

    if args.mode == COMBINED_MODE:
        combined = compute_both(frame, **settings)
        if not combined.constant.species:
            print(NO_SPECIES_MESSAGE)
            return 1
        write_outputs(
            build_combined_table(combined, frame),
            lambda destination: plot_combined(combined, destination, args.plot_top),
            output_csv,
            plot,
            not args.no_plot,
        )
        print(summarise_both(combined))
    else:
        result = compute_trapping(frame, mode=args.mode, **settings)
        if not result.species:
            print(NO_SPECIES_MESSAGE)
            return 1
        write_outputs(
            build_table(result, frame),
            lambda destination: plot_cumulative(result, destination, args.plot_top),
            output_csv,
            plot,
            not args.no_plot,
        )
        print(summarise(result))

    if args.compare_modes:
        print(summarise_comparison(compare_modes(frame, **settings)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
