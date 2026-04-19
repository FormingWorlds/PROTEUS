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
    config: Config,
    S_target: float,
    tsurf: float,
    source: str,
) -> None:
    """Cross-check a P-S-inverted entropy value against an independent PALEOS adiabat.

    The primary entropy IC path (both SPIDER via this module and Aragog via
    ``AragogRunner._set_entropy_ic``) inverts the P-S temperature table with
    ``EntropyEOS.invert_temperature``. This helper provides an orthogonal
    cross-check by calling ``zalmoxis.eos_export.compute_entropy_adiabat``,
    which constructs a PALEOS adiabat by stepping ``dT/dP|_S`` from the
    surface. The two code paths use the same underlying EOS tables but via
    different algorithms, so agreement confirms neither the inversion nor
    the adiabat integrator has drifted.

    Verdict thresholds (relative to ``S_target``):
        - PASS if abs((S_adiabat - S_target)/S_target) * 100 <= 1.0 %
        - WARN at 1-5 %  (log warning, do not modify S_target)
        - FAIL > 5 % raises RuntimeError (genuine divergence)

    The cross-check is a no-op for configs that cannot supply a PALEOS EOS
    file (no Zalmoxis installed, dummy structure, non-PALEOS mantle EOS).
    It intentionally never swallows ``AttributeError``/``TypeError`` so that
    stale solver APIs fail loudly.

    Parameters
    ----------
    config : Config
        PROTEUS configuration.
    S_target : float
        Entropy returned by the primary inversion path [J/kg/K].
    tsurf : float
        Surface temperature that was inverted to obtain ``S_target`` [K].
    source : str
        Name of the calling path (for log context).

    Raises
    ------
    RuntimeError
        If the cross-check FAILs (> 5 % discrepancy at the surface node).
    """
    try:
        from zalmoxis.eos_export import compute_surface_entropy

        from proteus.interior_struct.zalmoxis import (
            load_zalmoxis_material_dictionaries,
            load_zalmoxis_solidus_liquidus_functions,
            resolve_2phase_mgsio3_paths,
        )
    except (ImportError, ModuleNotFoundError) as e:
        log.debug('Entropy IC cross-check skipped: zalmoxis unavailable (%s)', e)
        return

    zalmoxis_cfg = getattr(config.interior_struct, 'zalmoxis', None)
    if zalmoxis_cfg is None:
        log.debug('Entropy IC cross-check skipped: no Zalmoxis config')
        return

    try:
        mat_dicts = load_zalmoxis_material_dictionaries()
        solid_eos, liquid_eos = resolve_2phase_mgsio3_paths(
            zalmoxis_cfg.mantle_eos, mat_dicts
        )
        eos_entry = mat_dicts.get(zalmoxis_cfg.mantle_eos, {})
        paleos_eos_file = eos_entry.get('eos_file', '') or solid_eos or ''
        if not paleos_eos_file or not os.path.isfile(paleos_eos_file):
            log.debug(
                'Entropy IC cross-check skipped: PALEOS file not found (%s)',
                paleos_eos_file,
            )
            return

        melt_funcs = load_zalmoxis_solidus_liquidus_functions(zalmoxis_cfg.mantle_eos, config)
        sol_func = liq_func = None
        if melt_funcs is not None:
            sol_func, liq_func = melt_funcs

        # Surface-only lookup. We do NOT integrate the full adiabat for the
        # cross-check because:
        # (a) the cross-check only needs scalar S(P_surface, T_surface) to
        #     compare against the primary EntropyEOS.invert_temperature call
        # (b) the full-adiabat integrator's bracket expansion can overshoot
        #     into the PALEOS non-converged region (MgSiO3 vapour regime at
        #     low P / high T, ~100% NaN there) and crash with a ValueError
        #     from brentq. This is exactly the bug that made the cross-check
        #     effectively dead on production runs before this fix.
        result = compute_surface_entropy(
            eos_file=paleos_eos_file,
            T_surface=tsurf,
            P_surface=1e5,
            solidus_func=sol_func,
            liquidus_func=liq_func,
            solid_eos_file=solid_eos,
            liquid_eos_file=liquid_eos,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        log.warning('Entropy IC cross-check skipped (expected error: %s)', e)
        return

    S_adiabat = float(result['S_target'])
    if S_target == 0.0:
        log.warning('Entropy IC cross-check skipped: S_target is zero')
        return

    rel_diff = abs(S_adiabat - S_target) / abs(S_target) * 100.0

    WARN_PCT = 1.0
    FAIL_PCT = 5.0
    if rel_diff <= WARN_PCT:
        verdict = 'PASS'
    elif rel_diff <= FAIL_PCT:
        verdict = 'WARN'
    else:
        verdict = 'FAIL'

    log.info(
        'Entropy IC cross-check (%s): S_inversion=%.1f J/kg/K vs '
        'S_adiabat=%.1f J/kg/K, diff=%.3f%%, verdict=%s',
        source,
        S_target,
        S_adiabat,
        rel_diff,
        verdict,
    )

    if verdict == 'FAIL':
        raise RuntimeError(
            f'Entropy IC cross-check FAIL: S_inversion={S_target:.1f} '
            f'vs S_adiabat={S_adiabat:.1f} ({rel_diff:.2f}% > {FAIL_PCT}%). '
            f'Primary inversion path disagrees with PALEOS adiabat by more '
            f'than the allowed tolerance. Investigate before running.'
        )


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

    Returns
    -------
    float
        Initial specific entropy [J/kg/K].
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

    # CMB-anchored adiabat: invert (P_cmb, tcmb_init) -> S via the same
    # entropy tables the interior solver integrates with. Because S is
    # conserved along an adiabat, S(P_cmb, T_cmb) = S(P_surf, T_surf), so
    # this returns exactly the entropy that produces T(P_cmb) = tcmb_init
    # when the solver unpacks the IC. Use this mode when the surface-
    # anchored adiabat under the current EOS would land in the mushy zone
    # at IC and you want to force a fully molten initial state.
    if config.planet.temperature_mode == 'adiabatic_from_cmb':
        tcmb = float(config.planet.tcmb_init)
        P_cmb = None
        if hf_row is not None:
            P_cmb = hf_row.get('P_cmb', None)
        if not P_cmb or P_cmb <= 0:
            log.warning(
                'adiabatic_from_cmb: hf_row["P_cmb"] missing or non-positive, '
                'falling back to Earth-like 135 GPa. The structure solve must '
                'run before compute_initial_entropy for the proper P_cmb to '
                'be available.',
            )
            P_cmb = 135e9

        # Preferred path: invert via the Aragog entropy tables (the same
        # tables the solver integrates with), so the resulting S yields
        # exactly tcmb_init at P_cmb when the IC is unpacked.
        if spider_eos_dir and os.path.isdir(spider_eos_dir):
            try:
                from aragog.eos.entropy import EntropyEOS

                eos = EntropyEOS(spider_eos_dir)
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
                    'falling back to PALEOS-2phase lookup.', e,
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
                'adiabatic_from_cmb requires interior_struct.module="zalmoxis"; '
                'using fallback S=%.1f J/kg/K.', fallback,
            )
            return fallback

        mat_dicts = load_zalmoxis_material_dictionaries()
        solid_eos, liquid_eos = resolve_2phase_mgsio3_paths(
            zalmoxis_cfg.mantle_eos, mat_dicts
        )
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
            log.info(
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
            # Cross-check against an independent PALEOS adiabat. Raises
            # RuntimeError on FAIL (> 5% disagreement) — that is a genuine
            # code-path divergence and MUST propagate up the stack. Do
            # NOT catch this inside the inversion try block, or FAIL
            # verdicts get silently demoted to warnings.
            _verify_initial_entropy(config, S_target, tsurf, source='spider_eos_dir')
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
        solid_eos, liquid_eos = resolve_2phase_mgsio3_paths(
            zalmoxis_cfg.mantle_eos, mat_dicts
        )

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


# Structure for holding interior variables at the current time-step
class Interior_t:
    def __init__(self, nlev_b: int, spider_dir=None, eos_dir=None):
        # Initial condition flag  (-1: init, 1: start, 2: running)
        self.ic = -1

        # Current time step length [yr]
        self.dt = 1.0

        # Cumulative SPIDER time [yr]. Used by the CVode failure
        # fallback path (wrapper.py) to keep bookkeeping consistent
        # during retries.
        self._spider_cumulative_time = 0.0

        # Stiffness-aware adaptive time-step state (2026-04-09).
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

        # Solver-reported stiffness score, 0 = easy, 1 = saturated.
        # Set by AragogRunner / AragogJAXRunner from the number of
        # internal ODE substeps per coupling call, normalised to a
        # "typical" baseline. Currently consumed only for logging;
        # future work: feed into the hysteresis trigger condition.
        self.solver_stiffness = 0.0

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

    def _load_ps_table(
        self, spider_dir: str, eos_dir: str, filename: str
    ) -> np.ndarray | None:
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
        local_path = os.path.join(
            spider_dir, 'lookup_data', '1TPa-dK09-elec-free', filename
        )

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

    def print(self):
        log.info('Printing interior arrays....')
        for attr in ('radius', 'tides', 'phi', 'visc', 'density', 'mass', 'shear', 'bulk'):
            log.info('    %s = %s' % (attr, str(getattr(self, attr))))

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

    def update_rheology(self, visc: bool = False):
        # Update shear and bulk moduli arrays based on the melt fraction at each layer.
        for i, p in enumerate(self.phi):
            self.shear[i] = eval_rheoparam(p, 'shear')
            self.bulk[i] = eval_rheoparam(p, 'bulk')
            if visc:
                self.visc[i] = eval_rheoparam(p, 'visc')
