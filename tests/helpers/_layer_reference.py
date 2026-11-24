"""Continuous drainage of one mush node whose solver porosity falls to zero.

The layer scheme drains the pore melt of a node over each coupling step and
holds it to the solver porosity, taken linear in time within the step. A node
that freezes within a few steps then keeps what the continuous problem gives:
pore melt that drains at its own relative rate, held to a porosity that falls
linearly in time, whose crystallised part keeps ``D`` times the concentration
of the melt it grew from and passes the rest of its volatiles on. This module
integrates that problem with SciPy's LSODA, with no coupling step and no
sub-step, for the validation cases of ``tests/outgas/test_node_record.py`` and
``tests/outgas/test_layer_trapping.py``.

Two regimes alternate. While the porosity falls faster than the melt drains,
the pore melt equals the porosity and crystallises at the difference of the
two rates. Once the melt drains faster, it falls below the porosity and drains
freely until the porosity catches up with it. Events on both conditions switch
between the regimes, so the integrator never steps across one.

Testing standards: docs/How-to/testing.md,
docs/Explanations/test_framework.md
"""

from __future__ import annotations

from collections.abc import Callable

from scipy.integrate import solve_ivp


def kept_share(
    rate_of: Callable[[float], float],
    porosity: float,
    duration: float,
    floor: float,
    entry: float,
    partition: float = 0.0,
    rtol: float = 1.0e-10,
) -> float:
    """Share of a node's buried volatiles it keeps until its porosity reaches ``floor``.

    The node holds the pore melt ``entry`` [1], at or above ``porosity`` [1],
    the solver porosity at the start, which falls linearly to 0 over
    ``duration`` [s]; the node is locked when the porosity reaches ``floor``
    [1]. ``rate_of(f)`` is the drainage rate of pore melt ``f`` relative to
    itself, the outflow over the melt volume [s-1]. Melt beyond the porosity
    crystallises in place, the excess over it at the start included, and its
    crystals keep ``partition`` [1] times its concentration; the rest of the
    volatiles stays in the pore melt and drains with it. Returns the volatiles
    in the pore melt and in those crystals at the floor, over the volatiles
    buried at the start [1].
    """
    slope = porosity / duration
    t_stop = duration * (1.0 - floor / porosity)
    grown = partition * (entry - porosity) / entry
    t, melt, pore, crystal = 0.0, porosity, 1.0 - grown, grown
    capped = slope > rate_of(porosity) * porosity
    for _ in range(1000):
        if t >= t_stop * (1.0 - 1.0e-12):
            return pore + crystal
        if capped:
            t, pore, crystal = _capped(
                rate_of, porosity, slope, partition, (t, t_stop), pore, crystal, rtol
            )
            melt, capped = porosity - slope * t, False
        else:
            t, melt, pore = _free(rate_of, porosity, slope, (t, t_stop), melt, pore, rtol)
            capped = True
    raise RuntimeError('the drainage reference switched regimes 1000 times')


def _capped(rate_of, porosity, slope, partition, span, pore, crystal, rtol):
    """Integrate while the pore melt is the porosity; stop where the melt drains faster."""

    def bound(t):
        return porosity - slope * t

    def rhs(t, y):
        p = bound(t)
        growth = max(slope - rate_of(p) * p, 0.0) * y[0] / p
        return [-rate_of(p) * y[0] - partition * growth, partition * growth]

    def leaves(t, y):
        p = bound(t)
        return slope - rate_of(p) * p

    leaves.terminal, leaves.direction = True, -1
    sol = solve_ivp(
        rhs, span, [pore, crystal], method='LSODA', rtol=rtol, atol=1e-14, events=leaves
    )
    return float(sol.t[-1]), float(sol.y[0, -1]), float(sol.y[1, -1])


def _free(rate_of, porosity, slope, span, melt, pore, rtol):
    """Integrate while the pore melt lies below the porosity; stop where they meet."""

    def rhs(t, y):
        r = rate_of(y[0])
        return [-r * y[0], -r * y[1]]

    def meets(t, y):
        return y[0] - (porosity - slope * t)

    meets.terminal, meets.direction = True, 1
    sol = solve_ivp(
        rhs, span, [melt, pore], method='LSODA', rtol=rtol, atol=1e-14, events=meets
    )
    return float(sol.t[-1]), float(sol.y[0, -1]), float(sol.y[1, -1])
