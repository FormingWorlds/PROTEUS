"""End-member properties of the MgSiO3 crystal-melt mush from the PALEOS unified table.

For every node of the PALEOS MgSiO3 unified P-T table (Attia et al.) this module derives

- the solid density, phase and adiabatic bulk modulus ``K_S = (d ln rho / dP)_s^-1``,
  differenced between neighbouring pressure rows at the same entropy and within the same
  PALEOS phase, so no difference is taken across a polymorph boundary;
- the solid shear modulus from the finite-strain law of Stixrude and Lithgow-Bertelloni
  (2005, GJI 162, 610, eq. 33), evaluated with the Eulerian strain from the PALEOS
  density and the PALEOS reference volume of the phase, and the shear and Debye
  parameters of Stixrude and Lithgow-Bertelloni (2011, GJI 184, 1180, Table A1);
- the liquid density and ``K_S`` on the melt side of the melting curve.

The derived arrays are cached next to the SCS tables (:func:`rheology_tables.cache_dir`),
keyed by the PALEOS file stamp and the code that builds them.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import os
from pathlib import Path

import numpy as np
from scipy.constants import R as R_GAS

from proteus.interior_energetics.rheology_tables import cache_dir, write_npz_atomic

log = logging.getLogger('fwl.' + __name__)

M_MGSIO3 = 0.100387  # kg/mol per MgSiO3 formula unit
N_ATOMS = 5  # atoms per MgSiO3 formula unit, for the Debye energy
T_REF = 300.0  # K, reference temperature of the SLB thermal term
PHASES = ('en', 'lpcen', 'hpcen', 'brg', 'ppv', 'liquid')  # codes 0..5; -1 = no data
LIQUID = PHASES.index('liquid')

# PALEOS reference volume [m3/mol per formula unit], K0 [Pa] and K0' of each solid phase
PALEOS_EOS = {
    'en': (31.350e-6, 106.2e9, 7.800),
    'lpcen': (31.280e-6, 112.8e9, 6.270),
    'hpcen': (30.310e-6, 128.4e9, 4.520),
    'brg': (24.408e-6, 262.3e9, 4.044),
    'ppv': (24.724e-6, 205.4e9, 5.069),
}
# SLB (2011) Table A1: G0 [Pa], G0', eta_S0, theta0 [K], gamma0, q0 (en, cen, hpcen, mgpv, mppv)
SLB_SHEAR = {
    'en': (77.0e9, 1.5, 2.5, 812.0, 0.78, 3.4),
    'lpcen': (81.0e9, 1.7, 1.7, 805.0, 0.96, 1.5),
    'hpcen': (88.0e9, 1.8, 2.1, 824.0, 1.12, 0.2),
    'brg': (173.0e9, 1.7, 2.6, 905.0, 1.57, 1.1),
    'ppv': (150.0e9, 2.0, 1.2, 855.0, 1.89, 1.1),
}
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_CHUNK = 20000


def phase_code(label) -> int:
    """PALEOS phase label (``solid-en``, ..., ``liquid``) to its index in ``PHASES``."""
    s = str(label).strip().lower()
    s = s[len('solid-') :] if s.startswith('solid-') else s
    return PHASES.index(s) if s in PHASES else -1


def debye_energy(theta, temp):
    """Debye vibrational energy per formula unit [J/mol] without the zero-point term."""
    theta, temp = np.broadcast_arrays(np.asarray(theta, float), np.asarray(temp, float))
    y = (theta / temp)[..., None]
    x = 0.5 * y * (_GL_X + 1.0)
    integral = 0.5 * y[..., 0] * np.sum(_GL_W * x**3 / np.expm1(x), axis=-1)
    return 9.0 * N_ATOMS * R_GAS * temp * (temp / theta) ** 3 * integral


def slb_shear_modulus(rho, temp, phase: str):
    """Solid shear modulus [Pa]: SLB (2005) eq. 33 with the PALEOS strain of ``phase``.

    ``G = (1+2f)^(5/2) [G0 + (3 K0 G0' - 5 G0) f + (6 K0 G0' - 24 K0 - 14 G0 + 9/2 K0 K0') f^2]
    - eta_S rho Delta U_q / M``, with ``f = ((rho V0 / M)^(2/3) - 1) / 2``.
    """
    v0, k0, k0p = PALEOS_EOS[phase]
    g0, g0p, eta0, theta0, gamma0, q0 = SLB_SHEAR[phase]
    rho, temp = np.asarray(rho, float), np.asarray(temp, float)
    f = 0.5 * ((rho * v0 / M_MGSIO3) ** (2.0 / 3.0) - 1.0)
    a1, a2 = 6.0 * gamma0, -12.0 * gamma0 + 36.0 * gamma0**2 - 18.0 * q0 * gamma0
    freq2 = 1.0 + a1 * f + 0.5 * a2 * f * f
    theta = theta0 * np.sqrt(np.where(freq2 > 0, freq2, np.nan))
    gamma = (1.0 + 2.0 * f) * (a1 + a2 * f) / (6.0 * freq2)
    eta_s = -gamma + (1.0 + 2.0 * f) ** 2 * (gamma0 + eta0) / freq2
    d_u = debye_energy(theta, temp) - debye_energy(theta, T_REF)
    cold = (1.0 + 2.0 * f) ** 2.5 * (
        g0
        + (3.0 * k0 * g0p - 5.0 * g0) * f
        + (6.0 * k0 * g0p - 24.0 * k0 - 14.0 * g0 + 4.5 * k0 * k0p) * f * f
    )
    return cold - eta_s * d_u * rho / M_MGSIO3


def read_paleos_grid(path: Path) -> dict:
    """PALEOS unified table on its regular (log10 P, log10 T) grid; missing nodes are NaN."""
    import pandas as pd

    cols = ['P', 'T', 'rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad', 'phase']
    tab = pd.read_csv(path, sep=r'\s+', comment='#', header=None, names=cols)
    lp, lt = (
        np.round(np.log10(tab['P'].to_numpy()), 9),
        np.round(np.log10(tab['T'].to_numpy()), 9),
    )
    log_p, log_t = np.unique(lp), np.unique(lt)
    i, j = np.searchsorted(log_p, lp), np.searchsorted(log_t, lt)
    grid = {'log_p': log_p, 'log_t': log_t}
    for name in ('rho', 's'):
        arr = np.full((log_p.size, log_t.size), np.nan)
        arr[i, j] = tab[name].to_numpy(float)
        grid[name] = arr
    code = np.full((log_p.size, log_t.size), -1, dtype=np.int8)
    code[i, j] = [phase_code(x) for x in tab['phase'].to_numpy()]
    grid['code'] = code
    return grid


def _lnrho_along(grid, k, c, s0):
    """ln rho of phase ``c`` on pressure row ``k`` at entropies ``s0``; NaN off its range."""
    if k < 0 or k >= grid['code'].shape[0]:
        return np.full(s0.shape, np.nan)
    sel = grid['code'][k] == c
    if sel.sum() < 2:
        return np.full(s0.shape, np.nan)
    s, lr = grid['s'][k, sel], np.log(grid['rho'][k, sel])
    o = np.argsort(s)
    return np.interp(s0, s[o], lr[o], left=np.nan, right=np.nan)


def _dlnrho_dp(p, k, lr0, nb):
    """(d ln rho / dP)_s at row ``k`` from same-phase neighbours ``nb[-2..2]``."""
    pm, p0, pp = (p[min(max(k + d, 0), p.size - 1)] for d in (-1, 0, 1))
    h1, h2 = p0 - pm, pp - p0
    central = (nb[1] - lr0) * h1 / (h2 * (h1 + h2)) + (lr0 - nb[-1]) * h2 / (h1 * (h1 + h2))
    out = np.where(np.isfinite(nb[1]) & np.isfinite(nb[-1]), central, np.nan)
    for side in (1, -1):
        k1, k2 = min(max(k + side, 0), p.size - 1), min(max(k + 2 * side, 0), p.size - 1)
        x1, x2 = p[k1] - p0, p[k2] - p0
        with np.errstate(divide='ignore', invalid='ignore'):
            three = (
                lr0 * (-x1 - x2) / (x1 * x2)
                + nb[side] * x2 / (x1 * (x2 - x1))
                - nb[2 * side] * x1 / (x2 * (x2 - x1))
            )
            two = (nb[side] - lr0) / x1
        one_sided = np.where(np.isfinite(nb[2 * side]), three, two)
        out = np.where(np.isfinite(out) | ~np.isfinite(nb[side]), out, one_sided)
    return out


def adiabatic_bulk_grid(grid) -> np.ndarray:
    """K_S [Pa] at every node from same-phase, same-entropy neighbouring pressure rows."""
    p = 10.0 ** grid['log_p']
    out = np.full(grid['rho'].shape, np.nan)
    for k in range(p.size):
        for c in np.unique(grid['code'][k]):
            if c < 0:
                continue
            idx = np.flatnonzero(grid['code'][k] == c)
            s0, lr0 = grid['s'][k, idx], np.log(grid['rho'][k, idx])
            nb = {d: _lnrho_along(grid, k + d, c, s0) for d in (-2, -1, 1, 2)}
            dlr = _dlnrho_dp(p, k, lr0, nb)
            with np.errstate(divide='ignore', invalid='ignore'):
                out[k, idx] = np.where(dlr > 0, 1.0 / dlr, np.nan)
    return out


def shear_grid(grid) -> np.ndarray:
    """SLB solid shear modulus [Pa] at every solid node (NaN elsewhere)."""
    temp = np.broadcast_to(10.0 ** grid['log_t'], grid['rho'].shape)
    out = np.full(grid['rho'].shape, np.nan)
    for c, name in enumerate(PHASES[:LIQUID]):
        idx = np.flatnonzero(grid['code'].ravel() == c)
        for lo in range(0, idx.size, _CHUNK):
            sl = idx[lo : lo + _CHUNK]
            out.flat[sl] = slb_shear_modulus(grid['rho'].flat[sl], temp.flat[sl], name)
    return out


def _melting_row(code_row, *arrays):
    """Solid-side and melt-side node indices of one pressure row (-1 when absent)."""
    solid = np.flatnonzero((code_row >= 0) & (code_row < LIQUID))
    if solid.size == 0:
        return -1, -1
    j_s = solid[-1]
    liquid = np.flatnonzero(code_row == LIQUID)
    liquid = liquid[liquid > j_s]
    j_l = liquid[0] if liquid.size else -1
    # step at most two cells into the phase where the boundary node has no K_S
    for _ in range(2):
        if all(np.isfinite(a[j_s]) for a in arrays) or j_s - 1 not in solid:
            break
        j_s -= 1
    return j_s, j_l


def build_endmembers(grid) -> dict:
    """Arrays stored in the cache: solid grids and melting-curve end members per pressure."""
    kappa = adiabatic_bulk_grid(grid)
    mu = np.where((grid['code'] >= 0) & (grid['code'] < LIQUID), shear_grid(grid), np.nan)
    solid = (grid['code'] >= 0) & (grid['code'] < LIQUID)
    n = grid['log_p'].size
    melt = {
        k: np.full(n, np.nan)
        for k in ('rho_sol', 'kappa_sol', 'mu_sol', 'rho_liq', 'kappa_liq', 'T_melt')
    }
    for k in range(n):
        j_s, j_l = _melting_row(grid['code'][k], kappa[k], mu[k])
        if j_s >= 0:
            melt['rho_sol'][k], melt['kappa_sol'][k] = grid['rho'][k, j_s], kappa[k, j_s]
            melt['mu_sol'][k], melt['T_melt'][k] = mu[k, j_s], 10.0 ** grid['log_t'][j_s]
        if j_l >= 0:
            melt['rho_liq'][k], melt['kappa_liq'][k] = grid['rho'][k, j_l], kappa[k, j_l]
    return {
        'log_p': grid['log_p'],
        'log_t': grid['log_t'],
        'rho_solid': np.where(solid, grid['rho'], np.nan),
        'kappa_solid': np.where(solid, kappa, np.nan),
        'mu_solid': mu,
    } | {f'melt_{k}': v for k, v in melt.items()}


@functools.lru_cache(maxsize=1)
def code_identity() -> str:
    """Digest of the end-member builders and constants; part of the cache key."""
    funcs = (debye_energy, slb_shear_modulus, read_paleos_grid, _lnrho_along, _dlnrho_dp)
    funcs += (adiabatic_bulk_grid, shear_grid, _melting_row, build_endmembers)
    src = ''.join(inspect.getsource(f) for f in funcs) + repr((PALEOS_EOS, SLB_SHEAR, M_MGSIO3))
    return hashlib.sha1(src.encode()).hexdigest()[:12]


def _file_identity(path: Path) -> str:
    st = os.stat(path)
    key = f'{os.path.realpath(path)}|{st.st_size}|{st.st_mtime_ns}'
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def endmember_cache_path(paleos_path: Path, directory: Path | None = None) -> Path:
    """Cache file of the end members derived from ``paleos_path``."""
    directory = cache_dir() if directory is None else Path(directory)
    return directory / f'paleos_endmembers_{_file_identity(paleos_path)}_{code_identity()}.npz'


@functools.lru_cache(maxsize=2)
def load_endmembers(paleos_path: Path, directory: Path | None = None) -> PaleosEndMembers:
    """End members from the cache, or built from the PALEOS table and cached on a miss."""
    paleos_path = Path(paleos_path)
    if not paleos_path.is_file():
        raise FileNotFoundError(f'PALEOS MgSiO3 unified table not found: {paleos_path}')
    path = endmember_cache_path(paleos_path, directory)
    arrays = None
    if path.is_file():
        try:
            with np.load(path) as data:
                arrays = {k: data[k] for k in data.files}
        except (OSError, ValueError, EOFError, KeyError) as exc:
            log.warning('Rheology end-member cache %s unreadable (%s); rebuilding', path, exc)
    if arrays is None:
        log.info('Deriving mush end members from %s', paleos_path)
        arrays = build_endmembers(read_paleos_grid(paleos_path))
        write_npz_atomic(path, arrays)
    return PaleosEndMembers(arrays)


def paleos_mgsio3_path() -> Path:
    """Path of the PALEOS MgSiO3 unified table in the FWL data tree."""
    from proteus.data import EOS_PALEOS_MGSIO3_UNIFIED, dataset_dir
    from proteus.utils.helper import resolve_fwl_data_dir

    root = dataset_dir(EOS_PALEOS_MGSIO3_UNIFIED, data_root=resolve_fwl_data_dir())
    return root / 'paleos_mgsio3_eos_table_pt.dat'


class PaleosEndMembers:
    """Interpolated end members; pressures in Pa, temperatures in K, moduli in Pa."""

    def __init__(self, arrays: dict):
        self._a = arrays
        self.log_p, self.log_t = arrays['log_p'], arrays['log_t']

    def melting(self, pres) -> dict:
        """Solid-side and melt-side properties at the melting curve, linear in log10 P."""
        lp = np.clip(np.log10(np.asarray(pres, float)), self.log_p[0], self.log_p[-1])
        out = {}
        for key in ('rho_sol', 'kappa_sol', 'mu_sol', 'rho_liq', 'kappa_liq', 'T_melt'):
            y = self._a[f'melt_{key}']
            ok = np.isfinite(y)
            out[key] = np.interp(lp, self.log_p[ok], y[ok])
        return out

    def solid(self, pres, temp) -> dict:
        """Solid density, K_S and shear modulus at (P, T), bilinear in (log10 P, log10 T).

        Corners without solid data are dropped and the weights renormalised; where no
        corner is solid (T above the solid range) the melting-curve solid side is used.
        """
        lp = np.clip(np.log10(np.asarray(pres, float)), self.log_p[0], self.log_p[-1])
        lt = np.clip(np.log10(np.asarray(temp, float)), self.log_t[0], self.log_t[-1])
        lp, lt = np.broadcast_arrays(lp, lt)
        i = np.clip(np.searchsorted(self.log_p, lp) - 1, 0, self.log_p.size - 2)
        j = np.clip(np.searchsorted(self.log_t, lt) - 1, 0, self.log_t.size - 2)
        wp = (lp - self.log_p[i]) / (self.log_p[i + 1] - self.log_p[i])
        wt = (lt - self.log_t[j]) / (self.log_t[j + 1] - self.log_t[j])
        fallback = self.melting(10.0**lp)
        out = {}
        for key, mkey in (('rho', 'rho_sol'), ('kappa', 'kappa_sol'), ('mu', 'mu_sol')):
            grid = self._a[f'{key}_solid']
            num, den = np.zeros(lp.shape), np.zeros(lp.shape)
            for di, dj, w in ((0, 0, (1 - wp) * (1 - wt)), (1, 0, wp * (1 - wt)),
                              (0, 1, (1 - wp) * wt), (1, 1, wp * wt)):  # fmt: skip
                val = grid[i + di, j + dj]
                ok = np.isfinite(val)
                num += np.where(ok, w * np.where(ok, val, 0.0), 0.0)
                den += np.where(ok, w, 0.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                out[key] = np.where(den > 0, num / den, fallback[mkey])
        return out
