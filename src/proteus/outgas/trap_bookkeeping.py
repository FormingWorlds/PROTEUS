"""Reservoir bookkeeping shared by the trapping schemes.

Both trapping schemes, the freezing front of :mod:`proteus.outgas.trapping`
and the per-node layer scheme of :mod:`proteus.outgas.layer_trapping`, move
trapped mass between the melt and the solid reservoirs through these
functions, and publish their diagnostics through :func:`record_step`, so the
chemistry, escape, desiccation and the closure check treat the two schemes
alike. The reservoir rules are those stated in :mod:`proteus.outgas.trapping`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from proteus.outgas.common import VOLATILE_ELEMENT_STOICH, element_masses_from_species
from proteus.outgas.compaction import BRANCH_GUARD, BRANCH_NONE
from proteus.utils.constants import noble_gases, vol_element_list, vol_list
from proteus.utils.helper import eval_gas_mmw

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Dissolved species the step can bury: the volatiles and the noble gases.
TRAPPED_SPECIES = tuple(vol_list) + tuple(noble_gases)
# Every trapped reservoir once: the species, then the volatile elements.
TRAPPED_COLUMNS = (*TRAPPED_SPECIES, *vol_element_list)


@dataclass
class TrappingStep:
    """What one trapping step moved, and the guards it tripped.

    Attributes
    ----------
    mode : str
        Value of ``outgas.trap_mode`` for the step.
    dm_rm : float
        Mantle mass crystallised over the step [kg]; negative when it remelted.
    f_tl : float
        Trapped melt fraction applied, a mass fraction in [0, 1].
    melt_mass : float
        Melt mass the dissolved volatiles were last partitioned into [kg].
    trapped_kg : dict
        Mass buried per species [kg]; negative on a remelting step.
    remelted : bool
        Whether the step released trapped mass to the melt.
    supply_capped : list
        Species whose burial was capped at the mass the melt held.
    branch : int
        Regime code, one of the ``BRANCH_*`` constants of
        :mod:`proteus.outgas.compaction`.
    tau_d, tau_s, t_res : float
        Percolation time at the top of the front, matrix deformation time and
        residence time of a parcel in the front [yr]; NaN when not computed.
    n_front : int
        Number of nodes the front spans.
    l_front : float
        Front thickness [m].
    v_front : float
        Front speed [m s-1].
    front_courant : float
        Front advance over the step, in front thicknesses [1].
    w_matrix_over_vf : float
        Matrix speed at the top of the front over the front speed [1].
    guard_reason : str
        Why the front could not be integrated; empty when it could.
    n_exited : int
        Mush nodes that reached the porosity floor over the step (``layers``).
    frac_bound : float
        Mass share of those nodes held at the no-drainage bound [1]; NaN when
        none reached the floor.
    n_substeps : int
        Drainage sub-steps the layer scheme took over the step.
    """

    mode: str
    dm_rm: float
    f_tl: float
    melt_mass: float
    trapped_kg: dict[str, float] = field(default_factory=dict)
    remelted: bool = False
    supply_capped: list[str] = field(default_factory=list)
    branch: int = BRANCH_NONE
    tau_d: float = float('nan')
    tau_s: float = float('nan')
    t_res: float = float('nan')
    n_front: int = 0
    l_front: float = float('nan')
    v_front: float = float('nan')
    front_courant: float = float('nan')
    w_matrix_over_vf: float = float('nan')
    guard_reason: str = ''
    n_exited: int = 0
    frac_bound: float = float('nan')
    n_substeps: int = 0

    @property
    def total_trapped(self) -> float:
        """Net mass buried this step, summed over species [kg].

        Negative on a remelting step, which releases trapped mass to the melt.
        """
        return float(sum(self.trapped_kg.values()))


def melt_concentration(
    kg_liquid: float | np.ndarray, melt_mass: float | np.ndarray
) -> float | np.ndarray:
    """Species mass fraction in the melt [1]; zero wherever no melt remains.

    A fully crystallised mantle carries no melt to draw from, so the quotient is
    replaced by zero instead of dividing by zero.

    Parameters
    ----------
    kg_liquid : float or ndarray
        Mass of the species dissolved in the melt [kg]; negative values count as 0.
    melt_mass : float or ndarray
        Mass of the melt [kg].
    """
    kg = np.maximum(np.asarray(kg_liquid, dtype=float), 0.0)
    melt = np.asarray(melt_mass, dtype=float)
    c_z = np.zeros(np.shape(kg), dtype=float)
    np.divide(kg, melt, out=c_z, where=melt > 0.0)
    if np.ndim(kg_liquid) == 0:
        return float(c_z)
    return c_z


def critical_melt_fraction(config: Config) -> float:
    """Disaggregation melt fraction ``phi_c`` [1], the solver's rheological transition.

    The single definition of ``phi_c`` for the trapping step, so the top of the
    drainage front and the no-drainage bound cannot drift apart from each other
    or from the interior rheology.
    """
    return float(config.interior_energetics.rfront_loc)


def partition_coefficients(config: Config) -> dict[str, float]:
    """Crystal/melt partition coefficient per species, from config.

    Water is the only species with appreciable lattice incorporation; the rest
    default to zero and are buried through the interstitial-melt term alone.

    Returns
    -------
    dict
        ``outgas.D_const_<species>`` [1] per trapped species; 0 for a species,
        such as a noble gas, that has no such field.
    """
    return {s: float(getattr(config.outgas, f'D_const_{s}', 0.0)) for s in TRAPPED_SPECIES}


def trapped_kg(hf_row: dict, name: str) -> float:
    """Mass of one species or element that trapping has buried [kg], else zero."""
    mass = float(hf_row.get(f'{name}_kg_trapped', 0.0))
    if not np.isfinite(mass) or mass <= 0.0:
        return 0.0
    return mass


def shift_reservoir(hf_row: dict, key: str, delta: float) -> None:
    """Add ``delta`` to one reservoir, floored at zero; a non-finite value is left alone."""
    value = float(hf_row.get(key, 0.0))
    if np.isfinite(value):
        hf_row[key] = max(0.0, value + delta)


def reservoir_sum(hf_row: dict, name: str, unit: str) -> float:
    """Atmosphere plus melt plus solid for one species, in ``kg`` or ``mol``."""
    return sum(float(hf_row.get(f'{name}_{unit}_{r}', 0.0)) for r in ('atm', 'liquid', 'solid'))


def species_carried(hf_row: dict, species: str) -> bool:
    """Whether every element of ``species`` has a positive whole-planet total.

    An element the run does not carry can still show trace melt mass from the
    chemistry solve; burying it would create mass its total never held.
    """
    elements = VOLATILE_ELEMENT_STOICH.get(species, {species: 1})
    return all(float(hf_row.get(f'{e}_kg_total', 0.0)) > 0.0 for e in elements)


def apply_to_reservoirs(hf_row: dict, moved: dict[str, float]) -> None:
    """Move mass between the melt and the solid, per species and per element.

    A positive mass is buried: it leaves ``_kg_liquid`` for ``_kg_solid``, and
    is added to ``_kg_trapped``, the share of ``_kg_solid`` that trapping owns
    and that the chemistry solve must leave alone. A negative mass is released
    by remelting and moves back the other way.
    """
    for species, mass in moved.items():
        if mass != 0.0:
            move_mass(hf_row, species, mass)
    # A noble gas is its own element and has been moved with the species.
    for element, mass in element_masses_from_species(moved).items():
        if element in vol_element_list and mass != 0.0:
            move_mass(hf_row, element, mass)


def move_mass(hf_row: dict, name: str, mass: float) -> None:
    """Take ``mass`` of one species or element out of the melt into the solid.

    A species or a noble gas also carries mole columns [mol]. Those of the melt
    and the solid follow the masses, so a step with no chemistry solve after it
    (a frozen or desiccated mantle) leaves them equal to ``kg / mmw``. The move
    leaves ``_mol_total`` as it is. The volatile elements carry no mole columns.
    """
    liq = float(hf_row.get(f'{name}_kg_liquid', 0.0))
    sol = float(hf_row.get(f'{name}_kg_solid', 0.0))
    hf_row[f'{name}_kg_liquid'] = max(0.0, liq - mass)
    hf_row[f'{name}_kg_solid'] = max(0.0, sol + mass)
    hf_row[f'{name}_kg_trapped'] = max(0.0, trapped_kg(hf_row, name) + mass)
    if f'{name}_mol_liquid' in hf_row:
        mmw = eval_gas_mmw(name)
        hf_row[f'{name}_mol_liquid'] = hf_row[f'{name}_kg_liquid'] / mmw
        hf_row[f'{name}_mol_solid'] = hf_row[f'{name}_kg_solid'] / mmw


def record_step(hf_row: dict, step: TrappingStep) -> None:
    """Publish the step's diagnostics so a trajectory can be read back."""
    hf_row['trap_dM_RM'] = step.dm_rm
    hf_row['trap_F_tl'] = step.f_tl
    hf_row['trap_kg_step'] = step.total_trapped
    hf_row['trap_branch'] = float(step.branch)
    hf_row['trap_tau_D'] = step.tau_d
    hf_row['trap_tau_s'] = step.tau_s
    hf_row['trap_t_res'] = step.t_res
    hf_row['trap_n_front'] = float(step.n_front)
    hf_row['trap_L_front'] = step.l_front
    hf_row['trap_v_front'] = step.v_front
    hf_row['trap_front_courant'] = step.front_courant
    hf_row['trap_w_matrix_over_vf'] = step.w_matrix_over_vf
    hf_row['trap_n_exited'] = float(step.n_exited)
    hf_row['trap_frac_bound'] = step.frac_bound
    hf_row['trap_n_substeps'] = float(step.n_substeps)
    hf_row['trap_kg_cumulative'] = (
        float(hf_row.get('trap_kg_cumulative', 0.0)) + step.total_trapped
    )
    if step.total_trapped != 0.0:
        log.info(
            '    %s = %.3e kg  (F_tl = %.4g, dM_RM = %.3e kg)',
            'trapped ' if step.total_trapped > 0.0 else 'released',
            abs(step.total_trapped),
            step.f_tl,
            step.dm_rm,
        )
    if step.branch == BRANCH_GUARD and step.mode != 'layers':
        # The guard replaces the drainage integral by its no-drainage limit,
        # which can bury far more than the integral would, so every step that
        # takes it is reported with its cause; the layer step reports its own.
        log.warning(
            'Trapping: the freezing front could not be integrated (%s). F_tl = %.3f '
            'is the no-drainage upper bound, the entry porosity capped at the '
            'critical melt fraction; %.3e kg was %s under it this step.',
            step.guard_reason or 'cause not recorded',
            step.f_tl,
            abs(step.total_trapped),
            'released' if step.total_trapped < 0.0 else 'buried',
        )
