# Code shared by all interior modules
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import erf

from proteus.utils.constants import B_ein

if TYPE_CHECKING:
    from aragog.eos.entropy import EntropyEOS

    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


@dataclass
class rheo_t:
    dotl: float
    delta: float
    xi: float
    gamma: float
    phist: float


# Lookup parameters for rheological properties
#     Taken from Kervazo+21 (https://doi.org/10.1051/0004-6361/202039433).
#     Note that the phi_star value of 0.4 differs from their Table 3,
#     however, this value is required to replicate their Figure 2 with
#     a rheological transition centred at 30% melt fraction.
par_visc = rheo_t(1.0, 25.7, 1.17e-9, 5.0, 0.4)
par_shear = rheo_t(10.0, 2.10, 7.08e-7, 5.0, 0.4)
par_bulk = rheo_t(1e9, 2.62, 0.102, 5.0, 0.4)


# Evaluate big Phi at a given layer
def _bigphi(phi: float, par: rheo_t):
    return (1.0 - phi) / (1.0 - par.phist)


# Evaluate big F at a given layer
def _bigf(phi: float, par: rheo_t):
    numer = np.pi**0.5 * _bigphi(phi, par) * (1.0 + _bigphi(phi, par) ** par.gamma)
    denom = 2.0 * (1.0 - par.xi)
    return (1.0 - par.xi) * erf(numer / denom)


# Evaluate rheological parameter at a given layer
def eval_rheoparam(phi: float, which: str):
    match which:
        case 'visc':
            par = par_visc
        case 'shear':
            par = par_shear
        case 'bulk':
            par = par_bulk
        case _:
            raise ValueError(f"Invalid rheological parameter 'f{which}'")
    # Evaluate parameter
    numer = 1.0 + _bigphi(phi, par) ** par.delta
    denom = (1.0 - _bigf(phi, par)) ** (B_ein * (1 - par.phist))
    return par.dotl * numer / denom


def _verify_initial_entropy(
    eos: EntropyEOS,
    P: float,
    S_target: float,
    tsurf: float,
    source: str,
) -> None:
    """Verify the entropy IC by a single-table temperature round trip.

    The entropy IC comes from ``EntropyEOS.invert_temperature(P, tsurf)``,
    which finds ``S`` such that ``temperature_scalar(P, S) == tsurf`` by Brent
    root-finding. This helper closes the loop on the same table: it maps
    ``S_target`` back to temperature with the vectorised ``EntropyEOS.temperature``
    surface and checks that the recovered temperature matches ``tsurf``.

    The check uses one table only, so it is invariant to the entropy reference
    constant and to the choice of EOS tabulation. It exercises two independent
    lever-rule temperature implementations (``temperature`` vectorised vs
    ``temperature_scalar`` inside the inversion), so a divergence between them,
    a Brent solve that landed off the root, or a non-finite table lookup all
    raise here instead of seeding the interior with a bad entropy.

    Tolerances on the recovered-temperature residual ``|T_recovered - tsurf|``:
        - FAIL threshold : ``max(5.0 K, 2e-3 * tsurf)``
        - WARN threshold : ``max(1.0 K, 5e-4 * tsurf)``
    The residual is bounded by the Brent ``xtol`` (0.1 in ``S``) times the local
    ``dT/dS``, under 0.2 K on the L 98-59 d table (measured 0.01 to 0.04 K over
    2500 to 3700 K), so these thresholds clear the healthy floor by more than an
    order of magnitude while a mis-inverted IC (tens of K off) still FAILs.

    Parameters
    ----------
    eos : EntropyEOS
        The entropy EOS used for the inversion (the same table the interior uses).
    P : float
        Pressure at which ``S_target`` was inverted [Pa].
    S_target : float
        Entropy returned by the inversion path [J/kg/K].
    tsurf : float
        Temperature that was inverted to obtain ``S_target`` [K].
    source : str
        Name of the calling path (for log context).

    Raises
    ------
    RuntimeError
        If the recovered temperature is non-finite or the residual exceeds the
        FAIL threshold.
    """
    if tsurf <= 0.0:
        log.warning('Entropy IC round-trip check skipped: non-positive tsurf=%.3g K', tsurf)
        return

    P_clamped = max(eos.P_min, min(float(P), eos.P_max))
    # Non-tautology contract: this must use the vectorised ``temperature``
    # surface, which is a separate implementation from the ``temperature_scalar``
    # that ``invert_temperature`` uses internally. If a future refactor makes the
    # forward and inverse maps share intermediate state, the round trip degrades
    # to a trivial identity and stops discriminating.
    T_recovered = float(np.ravel(np.asarray(eos.temperature(P_clamped, S_target)))[0])

    if not np.isfinite(T_recovered):
        raise RuntimeError(
            f'Entropy IC round-trip check FAIL ({source}): the temperature table '
            f'returned a non-finite value for S={S_target:.1f} J/kg/K at '
            f'P={P_clamped:.2e} Pa. The EOS lookup is broken; investigate before running.'
        )

    residual = abs(T_recovered - tsurf)
    warn_k = max(1.0, 5e-4 * tsurf)
    fail_k = max(5.0, 2e-3 * tsurf)
    if residual <= warn_k:
        verdict = 'PASS'
    elif residual <= fail_k:
        verdict = 'WARN'
    else:
        verdict = 'FAIL'

    log.debug(
        'Entropy IC round-trip check (%s): tsurf=%.1f K -> S=%.1f J/kg/K -> '
        'T_recovered=%.3f K, residual=%.4f K, verdict=%s',
        source,
        tsurf,
        S_target,
        T_recovered,
        residual,
        verdict,
    )

    if verdict == 'WARN':
        log.warning(
            'Entropy IC round-trip check WARN (%s): residual=%.4f K exceeds the '
            'WARN threshold (%.2f K) but stays under the FAIL threshold (%.2f K). '
            'The inversion is loose; check the EOS table if this persists.',
            source,
            residual,
            warn_k,
            fail_k,
        )

    if verdict == 'FAIL':
        raise RuntimeError(
            f'Entropy IC round-trip check FAIL ({source}): recovered '
            f'T={T_recovered:.1f} K from S={S_target:.1f} J/kg/K disagrees with '
            f'tsurf={tsurf:.1f} K by {residual:.2f} K (> {fail_k:.2f} K). The '
            f'entropy inversion did not land on the temperature table; investigate '
            f'before running.'
        )


_EOS_CACHE: dict = {}
_EOS_CACHE_MAX = 4
_TABLE_SUPERLIQ_N_POINTS = 200
_TABLE_SUPERLIQ_N_BISECT = 60


class InitialConditionError(RuntimeError):
    """The requested initial condition does not exist for this planet and EOS.

    Raised by the liquidus_super solves when no fully molten initial state is
    reachable or the melting curve is undefined where it must be checked.
    Retrying the interior step cannot change the outcome, so interior
    wrappers must not treat it as a transient solver failure.
    """


def _load_entropy_eos(eos_dir: str) -> EntropyEOS:
    """Load (and memoise) the P-S EOS tables in ``eos_dir``.

    The cache key holds the resolved directory and the name, size and
    modification time of every file in it, so a regenerated table set is
    reloaded. At most 4 table sets are kept, the oldest evicted first.

    Parameters
    ----------
    eos_dir : str
        Directory holding the SPIDER-format P-S tables.

    Returns
    -------
    EntropyEOS
        Table interpolator with ``P_min``, ``P_max``, ``S_min`` and ``S_max``.

    Raises
    ------
    FileNotFoundError
        If ``eos_dir`` is not a directory.
    """
    if not os.path.isdir(eos_dir):
        raise FileNotFoundError(f'SPIDER EOS table directory not found: {eos_dir}')
    stamp = []
    for name in sorted(os.listdir(eos_dir)):
        st = os.stat(os.path.join(eos_dir, name))
        stamp.append((name, st.st_size, st.st_mtime_ns))
    key = (os.path.realpath(eos_dir), tuple(stamp))
    if key not in _EOS_CACHE:
        from aragog.eos.entropy import EntropyEOS

        for stale in [k for k in _EOS_CACHE if k[0] == key[0]]:
            del _EOS_CACHE[stale]
        while len(_EOS_CACHE) >= _EOS_CACHE_MAX:
            del _EOS_CACHE[next(iter(_EOS_CACHE))]
        _EOS_CACHE[key] = EntropyEOS(eos_dir)
    return _EOS_CACHE[key]


def solve_superliquidus_entropy_from_tables(
    config: Config,
    hf_row: dict | None,
    eos_dir: str,
) -> dict:
    """Solve the ``liquidus_super`` entropy on the in-memory P-S tables.

    Finds the smallest uniform specific entropy whose adiabat lies at least
    ``config.planet.delta_T_super`` above the liquidus at every pressure
    between the surface (1 bar) and the core-mantle boundary. The liquidus is
    the one the interior solver uses for its melt fraction, ``T(P, S_liq(P))``
    from the tables' ``liquidus_P-S.dat``; ``interior_struct.melting_dir`` is
    not read here. No Zalmoxis or PALEOS data are read.

    The highest usable entropy is the table maximum, lowered for
    ``planet.ini_dsdr < 0`` by the entropy that perturbation adds at the
    core-mantle boundary, so the deepest initial entropy stays inside the
    table. When the requested superheat is not reachable below that ceiling
    but the ceiling still clears the liquidus, the entropy is clamped to it
    and a warning names the requested and achieved superheat. When even the
    ceiling stays below the liquidus somewhere, no fully-molten initial
    condition exists in the table and a ``RuntimeError`` is raised instead.

    Parameters
    ----------
    config : Config
        PROTEUS configuration. Uses ``planet.delta_T_super``,
        ``planet.ini_dsdr``, and ``planet.mass_tot``,
        ``interior_struct.core_frac`` and ``interior_struct.core_frac_mode``
        for the Noack & Lasbleis (2020) P_cmb estimate only.
    hf_row : dict or None
        Helpfile row. ``hf_row['P_cmb']`` is used when populated; otherwise a
        Noack & Lasbleis (2020) mass-aware estimate is used. ``R_int`` and
        ``R_core`` set the mantle thickness for the ``ini_dsdr`` ceiling.
    eos_dir : str
        Directory holding the SPIDER-format P-S tables.

    Returns
    -------
    dict
        ``S_target`` [J/kg/K], ``surface_T`` [K], ``cmb_T`` [K],
        ``achieved_superheat`` [K], ``binding_P`` [Pa], ``P_cmb`` [Pa] and
        ``clamped`` (bool).

    Raises
    ------
    FileNotFoundError
        If ``eos_dir`` is not a directory.
    RuntimeError
        If the table liquidus is undefined at any evaluated pressure, if the
        table temperature is not finite at the highest usable entropy, or if
        no fully-molten initial condition is reachable below that entropy.
    """
    eos = _load_entropy_eos(eos_dir)
    delta = float(config.planet.delta_T_super)

    from proteus.utils.structure_estimate import resolve_P_cmb

    P_cmb, estimated = resolve_P_cmb(hf_row, config)
    if estimated:
        log.warning(
            'liquidus_super: hf_row["P_cmb"] not yet populated; using '
            'Noack & Lasbleis (2020) mass-aware fallback P_cmb=%.1f GPa '
            '(mass_tot=%.2f M_Earth).',
            P_cmb / 1e9,
            float(config.planet.mass_tot),
        )
    # The deep mantle past the table cannot be certified molten, so there is
    # no clip; the tolerance only absorbs rounding of the table's end pressure.
    if P_cmb > float(eos.P_max) * (1.0 + 1e-9):
        raise InitialConditionError(
            'liquidus_super: the core-mantle boundary pressure %.3g GPa is above '
            'the EOS table maximum %.3g GPa (table covers %.3g to %.3g GPa); the '
            'deepest mantle cannot be checked for melt.'
            % (
                P_cmb / 1e9,
                float(eos.P_max) / 1e9,
                float(eos.P_min) / 1e9,
                float(eos.P_max) / 1e9,
            )
        )
    P_cmb = min(P_cmb, float(eos.P_max))
    P = np.geomspace(max(1e5, float(eos.P_min)), P_cmb, _TABLE_SUPERLIQ_N_POINTS)
    # The margin kinks at the melt-table and liquidus-file pressure nodes, so
    # evaluate there too: a table step between two grid points is not missed.
    nodes = [np.asarray(eos._liquidus['P'], dtype=float)]
    melt_T = getattr(eos, '_tables', {}).get('temperature_melt')
    if melt_T is not None:
        nodes.append(np.asarray(melt_T['P'], dtype=float))
    nodes = np.concatenate(nodes)
    P = np.unique(np.concatenate([P, nodes[(nodes > P[0]) & (nodes < P[-1])]]))

    # "Fully molten" is judged against the solver's own phase boundary: the
    # melt fraction comes from liquidus_P-S.dat, so the reference is
    # T(P, S_liq(P)). liquidus_entropy holds its end value outside the file's
    # pressure range, so coverage is checked against that range explicitly.
    # The relative tolerance absorbs the rounding of the file's end pressures
    # against the table edge P_max. Below S_min, T(P, S_liq) is a clipped
    # table-edge value, not the liquidus, so it counts as undefined too.
    P_liq_file = np.asarray(eos._liquidus['P'], dtype=float)
    P_liq_lo, P_liq_hi = float(P_liq_file.min()), float(P_liq_file.max())
    S_liq = np.asarray(eos.liquidus_entropy(P), dtype=float)
    T_liq = np.asarray(eos.temperature(P, S_liq), dtype=float)
    covered = (
        (P >= P_liq_lo * (1.0 - 1e-9))
        & (P <= P_liq_hi * (1.0 + 1e-9))
        & (S_liq >= float(eos.S_min))
        & np.isfinite(S_liq)
        & np.isfinite(T_liq)
    )
    if not covered.all():
        missing = P[~covered]
        raise InitialConditionError(
            'liquidus_super: the P-S table liquidus is undefined at %d of %d '
            'pressures between %.3g and %.3g GPa (liquidus file covers %.3g to '
            '%.3g GPa, table entropy from %.0f J/kg/K); the superheat target '
            'cannot be evaluated there.'
            % (
                missing.size,
                P.size,
                float(missing.min()) / 1e9,
                float(missing.max()) / 1e9,
                P_liq_lo / 1e9,
                P_liq_hi / 1e9,
                float(eos.S_min),
            )
        )
    above = S_liq > float(eos.S_max)
    if above.any():
        # The melt starts above the highest tabulated entropy: T(P, S_liq)
        # would be a clipped table-edge value, and no adiabat in the table is
        # molten there.
        i = int(np.argmax(S_liq - float(eos.S_max)))
        raise InitialConditionError(
            'liquidus_super: no fully-molten initial condition is reachable within '
            f'the EOS table; the table liquidus entropy exceeds the table maximum '
            f'({float(eos.S_max):.1f} J/kg/K) at {int(above.sum())} of {P.size} '
            f'pressures, by up to {float(S_liq[i]) - float(eos.S_max):.1f} J/kg/K '
            f'at P={float(P[i]) / 1e9:.3g} GPa.'
        )

    def _probe(S: float) -> tuple[float, float]:
        # A NaN temperature is not molten: its margin is -inf, never +inf.
        T = np.asarray(eos.temperature(P, np.full_like(P, S)), dtype=float)
        margin = np.where(np.isfinite(T), T - T_liq, -np.inf)
        i = int(np.argmin(margin))
        return float(margin[i]), float(P[i])

    # With ini_dsdr < 0 the modules add |ini_dsdr| * (R_surf - r) to the
    # uniform S_target, so the deepest node sits that much above S_target;
    # lower the usable ceiling by the same amount to keep it inside the table.
    # Both solvers' meshes end at hf_row R_core on this path: Aragog's inner
    # radius is R_core, and SPIDER's coresize is R_core / R_int (from the dummy
    # structure's mesh file, or core_frac in radius mode for the spider
    # structure, which sets R_core = core_frac * R_int).
    S_lo, S_hi = float(eos.S_min), float(eos.S_max)
    ini_dsdr = float(config.planet.ini_dsdr)
    if ini_dsdr < 0:
        R_int = hf_row.get('R_int') if hf_row is not None else None
        R_core = hf_row.get('R_core') if hf_row is not None else None
        if R_int and R_core and np.isfinite(R_int - R_core) and R_int > R_core:
            S_hi -= -ini_dsdr * (float(R_int) - float(R_core))
        else:
            log.warning(
                'liquidus_super: hf_row has no mantle radii (R_int, R_core), so '
                'the ini_dsdr perturbation is not subtracted from the table '
                'entropy ceiling; the deepest initial entropy can exceed S_max.'
            )

    sh_hi, P_hi = _probe(S_hi)
    if not np.isfinite(sh_hi):
        raise InitialConditionError(
            'liquidus_super: the EOS table gives a non-finite temperature at '
            f'P={P_hi / 1e9:.3g} GPa even at the highest usable entropy '
            f'({S_hi:.1f} J/kg/K); the superheat target cannot be evaluated.'
        )
    if sh_hi < 0:
        raise InitialConditionError(
            'liquidus_super: no fully-molten initial condition is reachable within '
            f'the EOS table; even at the highest usable entropy ({S_hi:.1f} J/kg/K) '
            f'the adiabat is {-sh_hi:.0f} K below the liquidus at P={P_hi / 1e9:.3g} GPa.'
        )
    clamped = sh_hi < delta
    if clamped:
        S = S_hi
    elif _probe(S_lo)[0] >= delta:
        S = S_lo
    else:
        for _ in range(_TABLE_SUPERLIQ_N_BISECT):
            mid = 0.5 * (S_lo + S_hi)
            if _probe(mid)[0] >= delta:
                S_hi = mid
            else:
                S_lo = mid
        S = S_hi

    achieved, P_bind = _probe(S)
    T_prof = np.asarray(eos.temperature(P, np.full_like(P, S)), dtype=float)
    out = {
        'S_target': float(S),
        'surface_T': float(T_prof[0]),
        'cmb_T': float(T_prof[-1]),
        'achieved_superheat': achieved,
        'binding_P': P_bind,
        'P_cmb': P_cmb,
        'clamped': clamped,
    }
    if clamped:
        log.warning(
            'liquidus_super: the requested superheat of %.0f K is not reachable '
            'within the EOS table (highest usable entropy %.1f J/kg/K). The initial '
            'entropy is clamped to that value, giving %.0f K of superheat at '
            'P=%.3g GPa and a surface temperature of %.0f K.',
            delta,
            S,
            achieved,
            P_bind / 1e9,
            out['surface_T'],
        )
    else:
        log.info(
            'liquidus_super (P-S tables): S=%.1f J/kg/K, surface T=%.0f K, '
            'T_cmb=%.0f K, superheat %.0f K at P=%.3g GPa.',
            S,
            out['surface_T'],
            out['cmb_T'],
            achieved,
            P_bind / 1e9,
        )
    return out


def compute_initial_entropy(
    config: Config,
    hf_row: dict | None = None,
    fallback: float = 3200.0,
    spider_eos_dir: str | None = None,
) -> float:
    """Compute initial mantle entropy from planet temperature settings.

    Converts the initial surface temperature to specific entropy using
    the PALEOS EOS tables (via Zalmoxis). Both SPIDER and Aragog use this
    to derive a physically consistent initial condition from
    config.planet.tsurf_init (or the accretion-mode override).

    In ``liquidus_super`` mode, for every structure module, the entropy is
    solved on the interior P-S tables in ``spider_eos_dir`` against their own
    liquidus (see ``solve_superliquidus_entropy_from_tables``).

    Special case: when ``config.planet.temperature_mode == 'isentropic'``,
    the entropy is taken directly from ``config.planet.ini_entropy`` and no
    EOS lookup is performed. This matches the CHILI intercomparison protocol
    and lets the interior solver map S -> T(P) via its own EOS table without
    going through PALEOS or Zalmoxis.

    Parameters
    ----------
    config : Config
        PROTEUS configuration.
    hf_row : dict, optional
        Helpfile row. When provided, checks for T_surface_initial
        (computed by Zalmoxis accretion mode) to override tsurf_init.
    fallback : float
        Entropy value [J/kg/K] returned when PALEOS is unavailable.
    spider_eos_dir : str, optional
        Directory of the interior P-S tables; required in ``liquidus_super``
        mode.

    Returns
    -------
    float
        Initial specific entropy [J/kg/K].

    Raises
    ------
    InitialConditionError
        In ``liquidus_super`` mode, when no fully molten entropy exists in the
        tables, or, with the Zalmoxis structure, when ``spider_eos_dir`` is
        missing.
    FileNotFoundError
        In ``liquidus_super`` mode with another structure module, when no
        ``spider_eos_dir`` is given.
    """
    # Direct-entropy mode: skip all EOS lookups and return the user-set value.
    # This is the CHILI-compatible path for SPIDER (and, once the "self" mode
    # for Aragog lands, for Aragog too).
    if config.planet.temperature_mode == 'isentropic':
        S = float(config.planet.ini_entropy)
        log.info(
            'Initial entropy from planet.ini_entropy (isentropic mode): %.1f J/kg/K',
            S,
        )
        return S

    # liquidus_super: start the mantle on the coolest single adiabat that is
    # fully molten everywhere with delta_T_super of superheat above the
    # liquidus, at the most-constraining depth. The entropy is solved on the
    # interior solver's own P-S tables against their own liquidus, the curve
    # that sets its melt fraction, for every structure module. With the
    # Zalmoxis structure the PALEOS P-T adiabat (solve_superliquidus_adiabat)
    # only anchors the structure solve's temperature profile.
    if config.planet.temperature_mode == 'liquidus_super':
        if config.interior_struct.module == 'zalmoxis':
            if not spider_eos_dir or not os.path.isdir(spider_eos_dir):
                raise InitialConditionError(
                    "temperature_mode='liquidus_super' solves the initial entropy "
                    "on the run's P-S tables (dirs['spider_eos_dir']), but that "
                    f'directory is missing: {spider_eos_dir!r}.'
                )
        elif not spider_eos_dir:
            raise FileNotFoundError(
                "temperature_mode='liquidus_super' with "
                f"interior_struct.module='{config.interior_struct.module}' "
                'needs the interior P-S EOS tables, but no table directory '
                f'was provided ({spider_eos_dir!r}).'
            )
        return float(
            solve_superliquidus_entropy_from_tables(config, hf_row, spider_eos_dir)['S_target']
        )

    # adiabatic_from_cmb: invert (P_cmb, tcmb_init) -> S via the same entropy
    # tables the interior solver integrates with. Because S is conserved along
    # an adiabat, S(P_cmb, tcmb_init) = S(P_surf, T_surf), so this returns the
    # entropy that reproduces T(P_cmb) = tcmb_init when the solver unpacks the
    # IC. Use this mode to force a fully molten initial state by an explicit
    # user-set CMB temperature.
    cmb_mode = config.planet.temperature_mode == 'adiabatic_from_cmb'

    if cmb_mode:
        from proteus.utils.structure_estimate import resolve_P_cmb

        mode = config.planet.temperature_mode
        # First call: hf_row['P_cmb'] is not yet populated, so a Noack &
        # Lasbleis (2020) mass-aware estimate (within ~5% over 0.5-10 M_Earth)
        # stands in, keeping super-Earths off the Earth-like 135 GPa value.
        P_cmb, estimated = resolve_P_cmb(hf_row, config)
        if estimated:
            mtot = float(config.planet.mass_tot)
            struct_mod = getattr(config.interior_struct, 'module', 'unknown')
            log.warning(
                '%s: hf_row["P_cmb"] missing or non-positive; using '
                'Noack & Lasbleis (2020) mass-aware fallback P_cmb=%.1f GPa '
                '(mass_tot=%.2f M_Earth). '
                'interior_struct.module=%r does not populate P_cmb before '
                'the IC is set. For best accuracy switch to '
                "module='zalmoxis' so a real structure solve populates "
                'P_cmb before the interior step.',
                mode,
                P_cmb / 1e9,
                mtot,
                struct_mod,
            )

        # The adiabatic_from_cmb anchor temperature is the user-set absolute
        # value; S(P_cmb, tcmb_init) is inverted below.
        tcmb = float(config.planet.tcmb_init)

        # Preferred path: invert via the Aragog entropy tables (the same
        # tables the solver integrates with), so the resulting S yields
        # exactly tcmb_init at P_cmb when the IC is unpacked.
        if spider_eos_dir and os.path.isdir(spider_eos_dir):
            try:
                from aragog.eos.entropy import EntropyEOS

                eos = EntropyEOS(spider_eos_dir)
                if P_cmb > eos.P_max:
                    log.warning(
                        'CMB-anchored IC: P_cmb=%.0f GPa exceeds the entropy '
                        'table maximum (%.0f GPa). invert_temperature clamps P '
                        'to the table edge, so the returned entropy reproduces '
                        'tcmb at %.0f GPa, not at the true P_cmb. The initial '
                        'condition is approximate at this planet mass.',
                        P_cmb / 1e9,
                        eos.P_max / 1e9,
                        eos.P_max / 1e9,
                    )
                S_target = float(eos.invert_temperature(P_cmb, tcmb))
                log.info(
                    'Initial entropy from CMB-anchored P-S inversion: '
                    'P_cmb=%.2e Pa, tcmb=%.0f K -> S=%.1f J/kg/K',
                    P_cmb,
                    tcmb,
                    S_target,
                )
                return S_target
            except (ImportError, ValueError, FileNotFoundError) as e:
                log.warning(
                    'CMB-anchored P-S inversion failed (%s); '
                    'falling back to PALEOS-2phase lookup.',
                    e,
                )

        # Fallback: PALEOS-2phase phase-weighted entropy lookup at
        # (P_cmb, tcmb_init). Same path as compute_surface_entropy,
        # just at the CMB instead of the surface.
        try:
            from zalmoxis.eos_export import compute_surface_entropy

            from proteus.interior_struct.zalmoxis import (
                load_zalmoxis_material_dictionaries,
                load_zalmoxis_solidus_liquidus_functions,
                resolve_2phase_mgsio3_paths,
            )
        except (ImportError, ModuleNotFoundError):
            log.warning(
                'Zalmoxis not installed; using fallback S=%.1f J/kg/K for tcmb=%.0f K.',
                fallback,
                tcmb,
            )
            return fallback

        zalmoxis_cfg = getattr(config.interior_struct, 'zalmoxis', None)
        if zalmoxis_cfg is None:
            log.warning(
                '%s mode requires interior_struct.module="zalmoxis"; '
                'using fallback S=%.1f J/kg/K.',
                mode,
                fallback,
            )
            return fallback

        mat_dicts = load_zalmoxis_material_dictionaries()
        solid_eos, liquid_eos = resolve_2phase_mgsio3_paths(zalmoxis_cfg.mantle_eos, mat_dicts)
        eos_entry = mat_dicts.get(zalmoxis_cfg.mantle_eos, {})
        paleos_eos_file = eos_entry.get('eos_file', '') or solid_eos or ''

        melt_funcs = load_zalmoxis_solidus_liquidus_functions(zalmoxis_cfg.mantle_eos, config)
        sol_func = liq_func = None
        if melt_funcs is not None:
            sol_func, liq_func = melt_funcs

        result = compute_surface_entropy(
            eos_file=paleos_eos_file,
            T_surface=tcmb,
            P_surface=P_cmb,
            solidus_func=sol_func,
            liquidus_func=liq_func,
            solid_eos_file=solid_eos,
            liquid_eos_file=liquid_eos,
        )
        S_target = float(result['S_target'])
        log.info(
            'Initial entropy from CMB-anchored PALEOS-2phase lookup: '
            'P_cmb=%.2e Pa, tcmb=%.0f K -> S=%.1f J/kg/K',
            P_cmb,
            tcmb,
            S_target,
        )
        return S_target

    # Determine effective surface temperature
    tsurf = config.planet.tsurf_init
    if hf_row is not None:
        T_computed = hf_row.get('T_surface_initial', 0)
        if T_computed and T_computed > 0:
            log.info(
                'Overriding tsurf_init with accretion thermal state: %.0f K -> %.0f K',
                tsurf,
                T_computed,
            )
            tsurf = T_computed

    # Try P-S inversion first (same tables as the entropy solver).
    # This is the preferred path: it uses the exact same EOS tables
    # that SPIDER and Aragog use during time integration.
    if spider_eos_dir and os.path.isdir(spider_eos_dir):
        S_target: float | None = None
        try:
            from aragog.eos.entropy import EntropyEOS

            eos = EntropyEOS(spider_eos_dir)
            S_target = eos.invert_temperature(1e5, tsurf)
            log.debug(
                'Initial entropy from P-S inversion: tsurf=%.0f K -> S=%.1f J/kg/K',
                tsurf,
                S_target,
            )
        except (ImportError, ValueError, FileNotFoundError) as e:
            # Expected inversion failures: missing aragog module, out-of-range
            # target temperature, or missing table files. Fall through to the
            # Zalmoxis adiabat path below.
            log.warning('P-S inversion failed in common.py: %s', e)
            S_target = None

        if S_target is not None:
            # Round-trip self-consistency check on the same table. Raises
            # RuntimeError on FAIL; that is a genuine inversion/table drift
            # and MUST propagate up the stack. Do NOT catch this inside the
            # inversion try block, or FAIL verdicts get silently demoted.
            _verify_initial_entropy(eos, 1e5, S_target, tsurf, source='spider_eos_dir')
            return S_target

    # Import errors (broken Zalmoxis install) should propagate, not fall back
    # silently. Only catch expected failures (missing config, missing tables).
    try:
        from zalmoxis.eos_export import compute_entropy_adiabat

        from proteus.interior_struct.zalmoxis import (
            load_zalmoxis_material_dictionaries,
            load_zalmoxis_solidus_liquidus_functions,
            resolve_2phase_mgsio3_paths,
        )
    except (ImportError, ModuleNotFoundError):
        log.warning(
            'Zalmoxis not installed. Using fallback S=%.1f J/kg/K for tsurf=%.0f K.',
            fallback,
            tsurf,
        )
        return fallback

    try:
        # Guard: Zalmoxis config may be absent when interior_struct.module='spider'
        zalmoxis_cfg = getattr(config.interior_struct, 'zalmoxis', None)
        if zalmoxis_cfg is None:
            raise RuntimeError('Zalmoxis config not available')

        mat_dicts = load_zalmoxis_material_dictionaries()
        # API-aware 2-phase table lookup. Required before the eos_file
        # sentinel selection so 2-phase mantle EoS configs (no top-level
        # eos_file) can use the solid sub-table as the sentinel.
        solid_eos, liquid_eos = resolve_2phase_mgsio3_paths(zalmoxis_cfg.mantle_eos, mat_dicts)

        eos_entry = mat_dicts.get(zalmoxis_cfg.mantle_eos, {})
        paleos_eos_file = eos_entry.get('eos_file', '') or solid_eos or ''
        if not paleos_eos_file or not os.path.isfile(paleos_eos_file):
            raise FileNotFoundError(f'PALEOS table not found: {paleos_eos_file}')

        melt_funcs = load_zalmoxis_solidus_liquidus_functions(zalmoxis_cfg.mantle_eos, config)
        sol_func = liq_func = None
        if melt_funcs is not None:
            sol_func, liq_func = melt_funcs

        # P_cmb only controls the diagnostic T(P) profile grid, not S_target.
        # S_target = S(P_surface, T_surface) is independent of P_cmb.
        result = compute_entropy_adiabat(
            eos_file=paleos_eos_file,
            T_surface=tsurf,
            P_surface=1e5,  # 1 bar
            P_cmb=135e9,
            n_points=500,
            solidus_func=sol_func,
            liquidus_func=liq_func,
            solid_eos_file=solid_eos,
            liquid_eos_file=liquid_eos,
        )
        S_target = float(result['S_target'])
        log.info(
            'Initial entropy: tsurf=%.0f K -> S=%.1f J/kg/K (PALEOS)',
            tsurf,
            S_target,
        )
        return S_target

    except (RuntimeError, FileNotFoundError, KeyError, ValueError) as e:
        log.warning(
            'Could not compute entropy from PALEOS (%s). '
            'Using fallback S=%.1f J/kg/K for tsurf=%.0f K.',
            e,
            fallback,
            tsurf,
        )
        return fallback


# Path to location at which to save tidal heating array
def get_file_tides(outdir: str):
    return os.path.join(outdir, 'data', 'tides_recent.dat')


# Path to location at which to persist the stale-structure flag. The flag is a
# per-run interior-state bit that must survive a resume: a run that fell back to
# the previous structure on a Zalmoxis non-convergence, then crashed, must come
# back knowing its on-disk mesh is stale rather than assuming it is fresh.
def get_file_structure_stale(outdir: str):
    return os.path.join(outdir, 'data', 'structure_stale.dat')


# Structure for holding interior variables at the current time-step
class Interior_t:
    def __init__(self, nlev_b: int, spider_dir=None, eos_dir=None):
        # Initial condition flag  (-1: init, 1: start, 2: running)
        self.ic = -1

        # Current time step length [yr]
        self.dt = 1.0

        # Consecutive solver-failure counters, reset on each success.
        # Per-run state by construction: every run builds a fresh
        # Interior_t, so concurrent Proteus instances in one Python
        # process (grid ensembles, notebooks, test sessions) cannot
        # share failure streaks. The abort thresholds live in
        # interior_energetics/wrapper.py.
        self.zalmoxis_fail_count = 0
        self.spider_fail_count = 0
        self.aragog_fail_count = 0

        # True when the interior is running on a fallback (previous-step)
        # structure because the last Zalmoxis re-solve did not converge; set on
        # that fall-back and cleared on the next successful re-solve. Downstream
        # consumers (Aragog's stale-mesh visibility counter, the baseline-commit
        # guard) read it here rather than from hf_row so it stays out of the
        # floats-only helpfile schema. Persisted to disk (write_structure_stale)
        # and restored on resume (resume_structure_stale) so a crash right after
        # a fall-back does not resume believing the on-disk mesh is fresh.
        self.structure_stale = False

        # Cumulative SPIDER time [yr]. Used by the CVode failure
        # fallback path (wrapper.py) to keep bookkeeping consistent
        # during retries.
        self._spider_cumulative_time = 0.0

        # Stiffness-aware adaptive time-step state.
        #
        # When the interior solver reports a "slow down" decision
        # (or a solver retry), ``dt_hysteresis_remaining`` is set to
        # ``config.params.dt.hysteresis_iters`` and counts down by
        # one at every call to ``next_step``. While > 0, the
        # speed-up scale factor is replaced with the milder
        # ``config.params.dt.hysteresis_sfinc`` so the controller
        # cannot ramp dt straight back into the stiff cliff it just
        # escaped from.
        self.dt_hysteresis_remaining = 0

        # True when the most recent call to next_step() had its step size
        # clamped. For example, by `_estimate_bolscale()`.
        self.timestep_clamped = False

        # Largest step [yr] the next call to next_step() may return, set by an
        # escape step whose loss hit the per-step cap. It is the step length
        # that would have put the same escape rate exactly at the cap, so the
        # overshoot is not repeated at the same size. Infinite when the most
        # recent escape step was not capped.
        self.escape_dt_limit = float('inf')

        # Lookup data for SPIDER (P-S tables, used by E_th and
        # melt-volume bookkeeping). Each is a (nS, nP, 3) array, the
        # third channel being the SI value of the quantity.
        self.lookup_rho_melt = None
        self.lookup_cp_solid = None
        self.lookup_cp_melt = None
        if spider_dir:
            resolved_eos = eos_dir or 'WolfBower2018_MgSiO3'
            self.lookup_rho_melt = self._load_ps_table(
                spider_dir, resolved_eos, 'density_melt.dat'
            )
            self.lookup_cp_solid = self._load_ps_table(
                spider_dir, resolved_eos, 'heat_capacity_solid.dat'
            )
            self.lookup_cp_melt = self._load_ps_table(
                spider_dir, resolved_eos, 'heat_capacity_melt.dat'
            )

        self.aragog_solver = None

        # Counter for consecutive Aragog steps integrated on a stale
        # Zalmoxis structure (i.e. with self.structure_stale=True
        # set by a Zalmoxis fall-back). Resets to 0 on every successful
        # structure refresh. Used by setup_or_update_solver to log how
        # long Aragog has been running on a frozen mesh, surfacing the
        # silent-stale-mesh failure mode. The hard-fail threshold lives
        # separately in the wrapper's _ZALMOXIS_MAX_CONSECUTIVE_FAILS
        # budget; this counter is for visibility, not enforcement.
        self._stale_struct_steps = 0

        # Simulation-time of the last SUCCESSFUL Zalmoxis structure
        # refresh. Distinct from the main-loop's `last_struct_time`
        # (which is reset on every call regardless of success). Drives
        # the stale-aware ceiling trigger in
        # update_structure_from_interior, which fires when
        # Time - last_successful_struct_time >= update_stale_ceiling.
        # Without this tracker, a fall-back keeps the trigger clock
        # ticking past the next ceiling and Aragog can advance through
        # an entire structure-update window with a frozen mesh. Set to
        # -inf at __init__; updated to hf_row['Time'] on every Zalmoxis
        # success (wrapper.py).
        self.last_successful_struct_time = float('-inf')

        # Number of levels
        self.nlev_b = int(nlev_b)
        self.nlev_s = self.nlev_b - 1

        # Arrays of interior properties at CURRENT  time-step.
        #    Radius has a length N+1. All others have length  N.
        self.radius = np.zeros(self.nlev_b)  # Radius [m].
        self.tides = np.zeros(self.nlev_s)  # Tidal power density [W kg-1].
        self.phi = np.zeros(self.nlev_s)  # Melt fraction.
        self.visc = np.zeros(self.nlev_s)  # Viscosity [Pa s].
        self.density = np.zeros(self.nlev_s)  # Mass density [kg m-3]
        self.mass = np.zeros(self.nlev_s)  # Mass of shell [kg]
        self.shear = np.zeros(self.nlev_s)  # Shear modulus [Pa]
        self.bulk = np.zeros(self.nlev_s)  # Bulk modulus [Pa]
        self.pres = np.zeros(self.nlev_s)  # Pressure [Pa]
        self.temp = np.zeros(self.nlev_s)  # Temperature [K]

    def _load_ps_table(self, spider_dir: str, eos_dir: str, filename: str) -> np.ndarray | None:
        """Load a SPIDER-format P-S lookup table.

        Used for ``density_melt.dat`` (volumetric melt fraction) and
        for ``heat_capacity_solid.dat`` / ``heat_capacity_melt.dat``
        (E_th computation in :func:`spider.ReadSPIDER`). All three
        files share the same on-disk layout.

        Search order: FWL_DATA dynamic EOS directory first, then
        SPIDER's bundled ``lookup_data/1TPa-dK09-elec-free``.

        Parameters
        ----------
        spider_dir : str
            Path to SPIDER installation directory.
        eos_dir : str
            Name of the dynamic EOS folder (e.g. 'WolfBower2018_MgSiO3').
        filename : str
            Bare filename of the P-S table to load.

        Returns
        -------
        np.ndarray or None
            Array of shape (nS, nP, 3) with columns (P, S, value) in
            SI units, or ``None`` if the file is not found.
        """
        fwl_data = os.environ.get('FWL_DATA', '')
        fwl_path = os.path.join(
            fwl_data,
            'interior_lookup_tables',
            'EOS',
            'dynamic',
            eos_dir,
            'P-S',
            filename,
        )
        local_path = os.path.join(spider_dir, 'lookup_data', '1TPa-dK09-elec-free', filename)

        if os.path.isfile(fwl_path):
            filepath = fwl_path
        elif os.path.isfile(local_path):
            filepath = local_path
        else:
            log.warning('%s not found for SPIDER P-S lookup', filename)
            return None

        data = np.genfromtxt(filepath)

        # Read dimensions from header: "# 5 nP nS"
        with open(filepath) as f:
            header = f.readline().strip().lstrip('#').split()
        nP = int(header[1])
        nS = int(header[2])

        # Read scale factors from line 5: "# P_scale S_scale value_scale"
        with open(filepath) as f:
            for _ in range(5):
                line = f.readline()
        scales = line.strip().lstrip('#').split()
        sfact = np.array([float(scales[0]), float(scales[1]), float(scales[2])])

        scaled = data * sfact
        return scaled.reshape(nS, nP, 3)

    def resume_tides(self, outdir: str):
        # Read tidal heating array from file, when resuming from disk.

        # If tides_recent.dat exists, we must be resuming a simulation from a
        #     previous state on the disk. Load the `tides` and `phi` arrays
        #     into this object. If we do not do this, tidal heating will be zero
        #     during the first iteration after model is resumed.

        file_tides = get_file_tides(outdir)

        # If the file is missing, then something has gone wrong
        if not os.path.exists(file_tides):
            log.warning('Cannot find tides file to resume from')
            return

        # Read the file
        data = np.loadtxt(file_tides)
        if self.nlev_s == 1:
            # dummy interior
            self.phi = np.array([data[0]])
            self.tides = np.array([data[1]])
        else:
            # resolved interior
            self.phi = np.array(data[:, 0])
            self.tides = np.array(data[:, 1])

        # Check the length
        if len(self.phi) != self.nlev_s:
            log.error('Array length mismatch when reading old tidal data')

    def write_tides(self, outdir: str):
        # Write tidal heating array to file.
        with open(get_file_tides(outdir), 'w') as hdl:
            # header information
            hdl.write('# 3 %d \n' % self.nlev_s)
            hdl.write('# Melt fraction, Tidal heating [W/kg] \n')
            hdl.write('# 1.0 1.0 \n')
            # for each level...
            for i in range(self.nlev_s):
                hdl.write('%.7e %.7e \n' % (self.phi[i], self.tides[i]))

    def write_structure_stale(self, outdir: str):
        # Persist the stale-structure flag so a resumed run recovers it. Written
        # whenever the flag changes (the Zalmoxis success and fall-back paths in
        # wrapper.py), so the on-disk value tracks the in-memory one. A write
        # failure (disk full, absent data/ directory) must not abort the run or,
        # on the fall-back path, skip the mesh and zalmoxis_output.dat rollback
        # that follows this call: the flag is only a resume-visibility bit, so
        # log and continue rather than propagate.
        try:
            with open(get_file_structure_stale(outdir), 'w') as hdl:
                hdl.write('%d\n' % (1 if self.structure_stale else 0))
        except OSError as exc:
            log.warning('Could not persist stale-structure flag file: %s', exc)

    def resume_structure_stale(self, outdir: str):
        # Restore the stale-structure flag from disk on resume. A missing file
        # means no fall-back was recorded, so the mesh
        # is fresh and the flag stays False. A malformed file is treated the
        # same way rather than aborting the resume over a visibility bit.
        path = get_file_structure_stale(outdir)
        if not os.path.exists(path):
            self.structure_stale = False
            return
        try:
            with open(path) as hdl:
                self.structure_stale = bool(int(hdl.read().strip()))
        except (ValueError, OSError):
            log.warning('Could not parse stale-structure flag file; assuming fresh')
            self.structure_stale = False

    def update_rheology(self, visc: bool = False):
        # Update shear and bulk moduli arrays based on the melt fraction at each layer.
        for i, p in enumerate(self.phi):
            self.shear[i] = eval_rheoparam(p, 'shear')
            self.bulk[i] = eval_rheoparam(p, 'bulk')
            if visc:
                self.visc[i] = eval_rheoparam(p, 'visc')


def get_C_planet(hf_row: dict, config: Config, interior_o: Interior_t):
    """Compute the planet's principal moment of inertia (C_int) based on the interior structure.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        config : Config
            Model configuration.
        interior_o : Interior_t
            Interior object containing interior arrays
    """
    # Calculate the planet's principal moment of inertia (C_planet)
    # Assuming a spherically symmetric mass distribution, we can use the formula:
    # C = (8/3) * pi * integral_0^R (rho(r) * r^4 dr)
    # where rho(r) is the density profile and R is the radius of the planet.

    # Get the radial grid and density profile from the interior object
    arr_keys = ('density', 'radius')
    lov = {k: np.array(getattr(interior_o, k), copy=True, dtype=float) for k in arr_keys}
    core_density = hf_row.get('core_density', None)
    if core_density is None:
        core_density = config.interior_struct.core_density
        # Warn user
        log.warning(
            'core_density not found in hf_row; using config value: %.3e kg/m^3',
            core_density,
        )

    # Reverse arrays if using SPIDER
    #  Such that i=0 is at the CMB
    if config.interior_energetics.module == 'spider':
        for k in arr_keys:
            lov[k] = lov[k][::-1]

    # Include the core density as the innermost layer
    r_edges = np.concatenate(([0.0], lov['radius']))
    rho = np.concatenate(([core_density], lov['density']))

    r0 = r_edges[:-1]
    r1 = r_edges[1:]

    integral = np.sum(rho * (r1**5 - r0**5) / 5.0)

    C_planet = (8 * np.pi / 3.0) * integral

    # Store C_planet in the helpfile row for later use
    hf_row['C_int'] = C_planet

    # Check if C_planet is physically reasonable
    C_factor_planet = C_planet / (hf_row['M_int'] * hf_row['R_int'] ** 2)
    log.info(
        f'Computed C_planet: {C_planet:.3e} kg.m^2, C_factor_planet: {C_factor_planet:.3f}'
    )
