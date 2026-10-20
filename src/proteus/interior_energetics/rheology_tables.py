"""Self-consistent-scheme tables for the elastic moduli of a crystal-melt mush.

The effective moduli follow the self-consistent scheme (SCS) of Berryman (1980,
J. Acoust. Soc. Am. 68, 1820) for spherical solid grains with spherical inclusions.
With the solid bulk modulus as the unit, the SCS depends only on the solid Poisson
ratio, the inclusion moduli relative to the solid bulk modulus, and the inclusion
volume fraction ``v``. Three tables are built on that dimensionless grid:

- drained: empty pores, gives ``kappa_d / kappa_sol`` and ``mu_d / kappa_sol``;
- saturated: inviscid melt (``kappa_liq``, 0), gives ``kappa / kappa_sol``;
- unrelaxed: melt at (``kappa_liq``, ``G_inf``), gives ``mu / kappa_sol``.

The tables are generated once per grid and solver identity, written to
``<FWL_DATA>/interior_lookup_tables/rheology_scs`` (or ``$PROTEUS_RHEOLOGY_CACHE_DIR``)
and memoised in memory. Spherical grains mean a crystal aspect ratio of 1, the
value at which the maximum packing fraction of Mader et al. (2013, eq. 49) is
``phi_m1`` (:func:`max_packing_fraction`).
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import os
import zipfile
from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from proteus.utils.helper import resolve_fwl_data_dir

log = logging.getLogger('fwl.' + __name__)

CACHE_ENV = 'PROTEUS_RHEOLOGY_CACHE_DIR'
CACHE_SUBDIR = Path('interior_lookup_tables') / 'rheology_scs'
TABLE_FORMAT = 1  # bump when the stored arrays change meaning
COLLAPSE_FRAC = 1e-9  # shear modulus below this fraction of mu_sol counts as collapsed
TOL = 1e-12  # absolute fixed-point residual, in units of kappa_sol and mu_sol
ITMAX = 500  # Newton iterations per volume fraction


@dataclass(frozen=True)
class ScsTableGrid:
    """Axes of the SCS tables (all dimensionless).

    Attributes
    ----------
    nu_min, nu_max, n_nu : float, float, int
        Range and number of nodes of the solid Poisson ratio.
    kliq_min, kliq_max, n_kliq : float, float, int
        Range and number of nodes of ``kappa_liq / kappa_sol``.
    gliq_min, gliq_max, n_gliq : float, float, int
        Range and number of log-spaced nodes of ``G_inf / kappa_sol``.
    n_v : int
        Number of nodes of the inclusion volume fraction on [0, 1].
    """

    nu_min: float = 0.15
    nu_max: float = 0.45
    n_nu: int = 31
    kliq_min: float = 0.05
    kliq_max: float = 1.5
    n_kliq: int = 30
    gliq_min: float = 1e-3
    gliq_max: float = 1.0
    n_gliq: int = 13
    n_v: int = 201

    def __post_init__(self):
        if not 0.0 < self.nu_min < self.nu_max < 0.5:
            raise ValueError('ScsTableGrid needs 0 < nu_min < nu_max < 0.5')
        if not 0.0 < self.kliq_min < self.kliq_max:
            raise ValueError('ScsTableGrid needs 0 < kliq_min < kliq_max')
        if not 0.0 < self.gliq_min < self.gliq_max:
            raise ValueError('ScsTableGrid needs 0 < gliq_min < gliq_max')
        if min(self.n_nu, self.n_kliq, self.n_gliq) < 2 or self.n_v < 3:
            raise ValueError('ScsTableGrid needs at least 2 nodes per axis and 3 in v')

    def axes(self) -> dict[str, np.ndarray]:
        """Node arrays: ``nu``, ``kliq``, ``log_gliq`` (log10) and ``v``."""
        return {
            'nu': np.linspace(self.nu_min, self.nu_max, self.n_nu),
            'kliq': np.linspace(self.kliq_min, self.kliq_max, self.n_kliq),
            'log_gliq': np.linspace(
                np.log10(self.gliq_min), np.log10(self.gliq_max), self.n_gliq
            ),
            'v': np.linspace(0.0, 1.0, self.n_v),
        }

    def hash_short(self) -> str:
        """12-character digest of the grid definition."""
        return hashlib.sha1(repr(astuple(self)).encode()).hexdigest()[:12]


def mu_over_kappa(nu):
    """Shear-to-bulk modulus ratio of an isotropic solid with Poisson ratio ``nu``."""
    nu = np.asarray(nu, dtype=float)
    return 3.0 * (1.0 - 2.0 * nu) / (2.0 * (1.0 + nu))


def poisson_ratio(kappa, mu):
    """Poisson ratio from the bulk and shear moduli (any common unit)."""
    kappa, mu = np.asarray(kappa, dtype=float), np.asarray(mu, dtype=float)
    return (3.0 * kappa - 2.0 * mu) / (2.0 * (3.0 * kappa + mu))


def max_packing_fraction(phi_m1: float, width: float, aspect_ratio: float) -> float:
    """Maximum crystal packing fraction, Mader et al. (2013, eq. 49).

    ``phi_m = phi_m1 exp[-(log10 r_p)^2 / (2 b^2)]`` with crystal aspect ratio ``r_p``
    and width ``b``; equals ``phi_m1`` for equant crystals (``r_p = 1``).
    """
    if aspect_ratio <= 0 or width <= 0:
        raise ValueError('aspect_ratio and width must be positive')
    return float(phi_m1 * np.exp(-(np.log10(aspect_ratio) ** 2) / (2.0 * width**2)))


def _scs_map(k, m, v, mu_s, k_i, mu_i):
    """One evaluation of the SCS update for spheres in units of kappa_sol."""
    z = m / 6.0 * (9.0 * k + 8.0 * m) / (k + 2.0 * m)
    p_s = (k + 4.0 / 3.0 * m) / (1.0 + 4.0 / 3.0 * m)
    q_s = (m + z) / (mu_s + z)
    p_i = (k + 4.0 / 3.0 * m) / (k_i + 4.0 / 3.0 * m)
    q_i = (m + z) / (mu_i + z)
    k_new = ((1.0 - v) * p_s + v * k_i * p_i) / ((1.0 - v) * p_s + v * p_i)
    m_new = ((1.0 - v) * mu_s * q_s + v * mu_i * q_i) / ((1.0 - v) * q_s + v * q_i)
    return k_new, m_new


def _newton_step(k, m, v, mu_s, k_i, mu_i):
    """Newton step on x = T(x) with a forward-difference Jacobian; damped step where it fails."""
    tk, tm = _scs_map(k, m, v, mu_s, k_i, mu_i)
    gk, gm = tk - k, tm - m
    hk, hm = 1e-7 * np.maximum(k, 1e-6), 1e-7 * np.maximum(m, COLLAPSE_FRAC * mu_s)
    a1, b1 = _scs_map(k + hk, m, v, mu_s, k_i, mu_i)
    a2, b2 = _scs_map(k, m + hm, v, mu_s, k_i, mu_i)
    j11, j21 = (a1 - tk) / hk - 1.0, (b1 - tm) / hk
    j12, j22 = (a2 - tk) / hm, (b2 - tm) / hm - 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        det = j11 * j22 - j12 * j21
        k_new = k + (-gk * j22 + gm * j12) / det
        m_new = m + (-gm * j11 + gk * j21) / det
    bad = ~np.isfinite(k_new) | ~np.isfinite(m_new) | (k_new < 0) | (m_new < 0)
    k_new = np.where(bad, k + 0.5 * gk, k_new)
    m_new = np.where(bad, m + 0.5 * gm, m_new)
    converged = (np.abs(gk) < TOL) & (np.abs(gm) < TOL * mu_s)
    return k_new, m_new, converged


def solve_scs(v_grid, mu_s, k_i, mu_i):
    """SCS moduli of spherical grains with spherical inclusions, in units of kappa_sol.

    Solves Berryman's SCS for every parameter set by continuation in ``v`` from the
    pure solid. From the fraction at which the shear modulus falls below
    ``COLLAPSE_FRAC * mu_s`` on, it is 0 and the bulk modulus is the Reuss average
    (0 for empty pores).

    Parameters
    ----------
    v_grid : array_like
        Increasing inclusion volume fractions starting at 0.
    mu_s, k_i, mu_i : array_like
        Solid shear modulus, inclusion bulk and shear moduli, all over kappa_sol;
        broadcast to one parameter set per element.

    Returns
    -------
    k_out, mu_out : ndarray
        Shape ``(n_sets, len(v_grid))``.

    Raises
    ------
    RuntimeError
        If a parameter set does not converge within ``ITMAX`` iterations.
    """
    mu_s, k_i, mu_i = (
        np.ravel(x)
        for x in np.broadcast_arrays(*(np.asarray(y, float) for y in (mu_s, k_i, mu_i)))
    )
    k, m = np.ones_like(mu_s), mu_s.copy()
    gone = np.zeros(mu_s.shape, dtype=bool)
    k_out, mu_out = np.zeros((mu_s.size, len(v_grid))), np.zeros((mu_s.size, len(v_grid)))
    for j, v in enumerate(np.asarray(v_grid, dtype=float)):
        a = ~gone
        kk, mm = k[a], m[a]
        args = (v, mu_s[a], k_i[a], mu_i[a])
        for _ in range(ITMAX):
            k_new, m_new, done = _newton_step(kk, mm, *args)
            busy = ~done & (mm >= 0.1 * COLLAPSE_FRAC * mu_s[a])
            if not busy.any():
                break
            kk, mm = np.where(busy, k_new, kk), np.where(busy, m_new, mm)
        else:
            raise RuntimeError(f'SCS did not converge at v = {v:.4f}')
        k[a], m[a] = kk, mm
        gone |= m < COLLAPSE_FRAC * mu_s
        with np.errstate(divide='ignore'):
            reuss = np.where(k_i > 0, 1.0 / (v / np.where(k_i > 0, k_i, 1.0) + (1.0 - v)), 0.0)
        k[gone], m[gone] = reuss[gone], 0.0
        k_out[:, j], mu_out[:, j] = k, m
    return k_out, mu_out


def build_tables(grid: ScsTableGrid) -> dict[str, np.ndarray]:
    """Solve the three SCS tables on ``grid`` (see the module docstring)."""
    ax = grid.axes()
    mu_s = mu_over_kappa(ax['nu'])
    kd, mud = solve_scs(ax['v'], mu_s, 0.0, 0.0)
    ns, nk, ng = mu_s[:, None], ax['kliq'][None, :], 10.0 ** ax['log_gliq']
    ksat, _ = solve_scs(ax['v'], ns, nk, 0.0)
    _, muu = solve_scs(ax['v'], ns[..., None], nk[..., None], ng[None, None, :])
    return {
        'drained_kappa': kd,
        'drained_mu': mud,
        'saturated_kappa': ksat.reshape(grid.n_nu, grid.n_kliq, grid.n_v),
        'unrelaxed_mu': muu.reshape(grid.n_nu, grid.n_kliq, grid.n_gliq, grid.n_v),
    }


@functools.lru_cache(maxsize=1)
def solver_identity() -> str:
    """Digest of the solver source and table format; part of the cache key."""
    src = ''.join(
        inspect.getsource(f) for f in (_scs_map, _newton_step, solve_scs, build_tables)
    )
    src += f'{TABLE_FORMAT}|{COLLAPSE_FRAC}|{TOL}'
    return hashlib.sha1(src.encode()).hexdigest()[:12]


def cache_dir() -> Path:
    """Table directory: ``$PROTEUS_RHEOLOGY_CACHE_DIR`` or the FWL_DATA default."""
    env = os.environ.get(CACHE_ENV)
    return Path(env).expanduser() if env else resolve_fwl_data_dir() / CACHE_SUBDIR


def table_path(grid: ScsTableGrid, directory: Path | None = None) -> Path:
    """Cache file of ``grid`` for the current solver identity."""
    directory = cache_dir() if directory is None else Path(directory)
    return directory / f'scs_{grid.hash_short()}_{solver_identity()}.npz'


def _write_atomic(path: Path, arrays: dict) -> None:
    """Write ``arrays`` to ``path`` through a temporary file, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def _read_valid(path: Path, grid: ScsTableGrid) -> dict | None:
    """Arrays of ``path`` if it exists and was built for ``grid`` by this solver, else None."""
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as exc:
        log.warning('Rheology table %s unreadable (%s); regenerating', path, exc)
        return None
    if (
        str(arrays.get('grid_hash')) != grid.hash_short()
        or str(arrays.get('solver')) != solver_identity()
    ):
        return None
    return arrays


def load_tables(grid: ScsTableGrid | None = None, directory=None, force=False) -> ScsTables:
    """SCS tables for ``grid``: memoised, read from the cache, or generated on a miss.

    Parameters
    ----------
    grid : ScsTableGrid, optional
        Table axes; the default grid when None.
    directory : str or Path, optional
        Cache directory; :func:`cache_dir` when None.
    force : bool
        Regenerate even when a valid cache file exists.
    """
    grid = ScsTableGrid() if grid is None else grid
    path = table_path(grid, directory)
    if force:
        _load_cached.cache_clear()
        _generate(path, grid)
    return _load_cached(path, grid)


def _generate(path: Path, grid: ScsTableGrid) -> dict:
    log.info('Generating rheology SCS tables %s', path)
    arrays = build_tables(grid) | {'grid_hash': grid.hash_short(), 'solver': solver_identity()}
    _write_atomic(path, arrays)
    return arrays


@functools.lru_cache(maxsize=4)
def _load_cached(path: Path, grid: ScsTableGrid) -> ScsTables:
    arrays = _read_valid(path, grid)
    if arrays is None:
        arrays = _generate(path, grid)
    else:
        log.debug('Rheology SCS tables cache hit: %s', path)
    return ScsTables(grid, arrays)


def _collapse_points(v, mu, mu_s):
    """Volume fraction where each shear-modulus row reaches 0, from its last two positive nodes.

    The SCS shear modulus falls linearly to its collapse, so the linear extrapolation
    recovers it between nodes (1/2 for drained spheres).
    """
    out = np.empty(len(mu))
    for i, (row, ref) in enumerate(zip(mu, np.ravel(mu_s))):
        j = np.flatnonzero(row > 1e-4 * ref)[-1]
        if j + 1 >= len(v):
            out[i] = v[-1]
            continue
        slope = (row[j] - row[j - 1]) / (v[j] - v[j - 1])
        out[i] = v[j] - row[j] / slope
    return out


class ScsTables:
    """Linear interpolators on the SCS tables; every modulus is in units of kappa_sol."""

    def __init__(self, grid: ScsTableGrid, arrays: dict):
        self.grid = grid
        ax = grid.axes()
        self._axes = ax
        nv, kl, lg, v = ax['nu'], ax['kliq'], ax['log_gliq'], ax['v']
        opt = {'method': 'linear', 'bounds_error': True}
        self._kd = RegularGridInterpolator((nv, v), arrays['drained_kappa'], **opt)
        self._mud = RegularGridInterpolator((nv, v), arrays['drained_mu'], **opt)
        self._ksat = RegularGridInterpolator((nv, kl, v), arrays['saturated_kappa'], **opt)
        # log interpolation: at large v the unrelaxed modulus scales with G_inf
        log_muu = np.log10(arrays['unrelaxed_mu'])
        self._log_muu = RegularGridInterpolator((nv, kl, lg, v), log_muu, **opt)
        self.drained_collapse = _collapse_points(v, arrays['drained_mu'], mu_over_kappa(nv))

    def _check(self, name, x, lo, hi):
        x = np.asarray(x, dtype=float)
        if np.any(~np.isfinite(x)) or np.any(x < lo) or np.any(x > hi):
            raise ValueError(f'{name} outside the rheology table range [{lo:g}, {hi:g}]')
        return x

    def _eval(self, interp, *cols):
        """Evaluate ``interp`` on broadcast inputs; the result has their broadcast shape."""
        cols = np.broadcast_arrays(*cols)
        pts = np.stack([c.ravel() for c in cols], axis=-1)
        return interp(pts).reshape(cols[0].shape)

    def _points(self, v, nu, kliq=None, gliq=None):
        g = self.grid
        cols = [self._check('nu', nu, g.nu_min, g.nu_max)]
        if kliq is not None:
            cols.append(self._check('kappa_liq/kappa_sol', kliq, g.kliq_min, g.kliq_max))
        if gliq is not None:
            cols.append(np.log10(self._check('G_inf/kappa_sol', gliq, g.gliq_min, g.gliq_max)))
        cols.append(self._check('v', v, 0.0, 1.0))
        return cols

    def drained(self, v, nu):
        """Drained bulk and shear moduli over kappa_sol at inclusion fraction ``v``."""
        pts = self._points(v, nu)
        return self._eval(self._kd, *pts), self._eval(self._mud, *pts)

    def saturated_kappa(self, v, nu, kliq):
        """Bulk modulus over kappa_sol with inviscid melt inclusions."""
        return self._eval(self._ksat, *self._points(v, nu, kliq))

    def unrelaxed_mu(self, v, nu, kliq, gliq):
        """Shear modulus over kappa_sol with melt inclusions at (kappa_liq, G_inf)."""
        return 10.0 ** self._eval(self._log_muu, *self._points(v, nu, kliq, gliq))

    def framework(self, v, v_crit, nu):
        """Drained moduli with the collapse moved from the SCS value to ``v_crit``.

        The volume axis is stretched, ``v' = v v_c0 / v_crit`` with ``v_c0`` the
        drained SCS collapse (1/2 for spheres); both moduli are 0 for ``v >= v_crit``.
        """
        v = self._check('v', v, 0.0, 1.0)
        v_crit = self._check('v_crit', v_crit, 1e-6, 1.0)
        v_c0 = np.interp(nu, self._axes['nu'], self.drained_collapse)
        v_s = np.minimum(v * v_c0 / v_crit, 1.0)
        k, m = self.drained(v_s, nu)
        cut = v >= v_crit
        return np.where(cut, 0.0, k), np.where(cut, 0.0, m)
