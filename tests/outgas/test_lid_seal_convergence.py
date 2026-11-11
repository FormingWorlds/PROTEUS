"""Convergence of the lid seal leak step against a stiff ODE reference.

Invariant checked: along a falling melt fraction, with escape off and H2O, CO2
and N2 dissolving, the end-point exponential step of ``leak_step`` gives the
atmospheric and sealed masses of ``dM_i/dt = -(1/tau) max(M_i - M_eq,i, 0)``
integrated by Radau, for tau from 1e2 to 1e7 yr: each error is below 1e-4 of
the inventory of its species, and the step is first order where it is short
against the leak. The reference evaluates the solubility laws written out here,
not the CALLIOPE classes.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from proteus.config._outgas import Calliope, LidSeal
from proteus.outgas.common import element_masses_from_species
from proteus.outgas.lid_seal import SEALED_ELEMENTS, SEALED_SPECIES, leak_step
from proteus.utils.constants import gas_list
from proteus.utils.helper import eval_gas_mmw

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

M_MANTLE, R_INT, GRAVITY = 4.0e24, 6.371e6, 9.81  # [kg], [m], [m s-2]
T_SEAL, FO2 = 1500.0, 2.0  # [K], [log10 IW offset] of the closing solve
D = {'H2O': 0.0017, 'CO2': 0.0, 'N2': 0.0}
ATM0 = {'H2O': 1.0e17, 'CO2': 5.0e16, 'N2': 1.0e16}  # [kg]
PHI0, PHI1, SPAN, STEPS = 0.19, 0.135, 4.3e4, 120  # melt fraction path over SPAN [yr]


def _x_eq(p, p_tot):
    """Dissolved mass fractions [1] at partial pressures p and total p_tot [bar].

    Sossi et al. (2023) for H2O, Dixon et al. (1995) for CO2 and Dasgupta et al.
    (2022) for N2, with the melt composition of CALLIOPE.
    """
    w = 3.8e-7 * p['CO2'] * np.exp(-23.0 * (p['CO2'] - 1.0) / (83.15 * T_SEAL))
    n2, tot = p['N2'] * 1e-4, max(p_tot * 1e-4, 1e-15)
    das = np.exp(4.67 + 7.11 * 0.56 - 13.06 * 0.11 - 120.67 * 0.01)
    return {
        'H2O': 524e-6 * np.sqrt(p['H2O']),
        'CO2': 1e-2 * 4400.0 * w / (36.6 - 44.0 * w),
        'N2': 1e-6 * (n2**0.5 * np.exp(5908.0 * tot**0.5 / T_SEAL - 1.6 * FO2) + n2 * das),
    }


def _floors(atm, phi):
    """M_eq of each species [kg] for atmospheric masses atm [kg] at melt fraction phi."""
    n = {s: atm[s] / eval_gas_mmw(s) for s in atm}
    p_tot = sum(atm.values()) * GRAVITY / (4.0 * np.pi * R_INT**2) / 1e5
    x = _x_eq({s: n[s] / sum(n.values()) * p_tot for s in atm}, p_tot)
    return {s: x[s] * M_MANTLE * (phi + D[s] * (1.0 - phi)) for s in atm}


def _phi(t):
    return PHI0 + (PHI1 - PHI0) * t / SPAN


def _closure():
    """Sealed masses [kg] at closure: the melt-only equilibrium of the closing solve."""
    x = _floors(ATM0, PHI0)
    return {s: x[s] * PHI0 / (PHI0 + D[s] * (1.0 - PHI0)) for s in ATM0}


def _code(tau, steps):
    """Atmospheric and sealed masses [kg] at SPAN from leak_step in equal steps."""
    outgas = SimpleNamespace(lid_seal=LidSeal(enabled=True, tau_leak=tau), T_floor=700.0)
    outgas.calliope = Calliope()
    for s in SEALED_SPECIES:
        setattr(outgas, f'D_const_{s}', D.get(s, 0.0))
    cfg = SimpleNamespace(outgas=outgas)
    sealed = _closure()
    row = {'M_mantle': M_MANTLE, 'R_int': R_INT, 'gravity': GRAVITY, 'esc_rate_total': 0.0}
    row.update({'seal_Tmagma_close': T_SEAL, 'fO2_shift_IW_derived': FO2})
    for s in gas_list:
        for r in ('atm', 'sealed'):
            row[f'{s}_kg_{r}'] = row[f'{s}_mol_{r}'] = 0.0
    for r, masses in (('atm', ATM0), ('sealed', sealed)):
        for s, kg in masses.items():
            row[f'{s}_kg_{r}'], row[f'{s}_mol_{r}'] = kg, kg / eval_gas_mmw(s)
    for e in SEALED_ELEMENTS:
        row[f'{e}_kg_atm'] = element_masses_from_species(ATM0).get(e, 0.0)
        row[f'{e}_kg_sealed'] = element_masses_from_species(sealed).get(e, 0.0)
    for t in np.linspace(0.0, SPAN, steps + 1)[1:]:
        row['Phi_global'] = _phi(t)
        leak_step(cfg, row, SPAN / steps)
    return np.array([row[f'{s}_kg_{r}'] for r in ('atm', 'sealed') for s in ATM0])


def _reference(tau):
    """The same masses from Radau on the continuous law (rtol 1e-11)."""
    names = list(ATM0)

    def rhs(t, y):
        floors = _floors(dict(zip(names, y[:3])), _phi(t))
        out = np.array([max(y[3 + i] - floors[s], 0.0) / tau for i, s in enumerate(names)])
        return np.concatenate([out, -out])

    y0 = np.array([ATM0[s] for s in names] + [_closure()[s] for s in names])
    sol = solve_ivp(rhs, (0.0, SPAN), y0, method='Radau', rtol=1e-11, atol=1e2)
    assert sol.success
    return sol.y[:, -1]


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@pytest.mark.parametrize('tau', [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7])
def test_leak_step_converges_to_the_continuous_law(tau):
    """End state of 120 and 480 steps against Radau. Each error over the
    inventory of its species (held plus atmosphere) is below 1e-4, and where
    the step is short against the leak (lam dt < 0.1) 4 times more steps cut
    the largest error by at least 3.5, the first order of the end-point step."""
    ref = _reference(tau)
    inventory = np.tile(ref[:3] + ref[3:], 2)
    err = {n: np.abs(_code(tau, n) - ref) / inventory for n in (STEPS, 4 * STEPS)}
    assert err[STEPS].max() < 1e-4
    if SPAN / STEPS / tau < 0.1:
        assert err[STEPS].max() / err[4 * STEPS].max() > 3.5
