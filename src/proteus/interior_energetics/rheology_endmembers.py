"""End-member properties of the MgSiO3 crystal-melt mush from the PALEOS unified table.

For every node of the PALEOS MgSiO3 unified P-T table (Attia et al.) this module derives

- the solid density, phase and adiabatic bulk modulus ``K_S = (d ln rho / dP)_s^-1``,
  differenced between neighbouring pressure rows at the same entropy and within the same
  PALEOS phase, so no difference is taken across a polymorph boundary;
- the solid shear modulus from the finite-strain law of Stixrude and Lithgow-Bertelloni
  (2005, GJI 162, 610, eq. 33), evaluated with the Eulerian strain from the PALEOS
  density and the PALEOS reference volume of the phase, and the shear and Debye
  parameters of Stixrude and Lithgow-Bertelloni (2011, GJI 184, 1180, Table A1);
- the liquid density and ``K_S`` on the melt side of the melting curve, and the
  unrelaxed melt shear modulus there (:func:`melt_shear_modulus`; the constant ``G_inf``
  of the config, ``interior_energetics.rheology.melt_shear_modulus``).

The derived arrays are cached under ``FWL_DATA/interior_lookup_tables/rheology`` (or
``$PROTEUS_RHEOLOGY_CACHE_DIR``, :func:`cache_dir`), keyed by the PALEOS file stamp, the code
that builds them and the melt shear modulus.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import os
from pathlib import Path

import numpy as np

from proteus.utils.helper import resolve_fwl_data_dir

log = logging.getLogger('fwl.' + __name__)

CACHE_ENV = 'PROTEUS_RHEOLOGY_CACHE_DIR'
CACHE_SUBDIR = Path('interior_lookup_tables') / 'rheology'

M_MGSIO3 = 0.100387  # kg/mol per MgSiO3 formula unit
N_ATOMS = 5  # atoms per MgSiO3 formula unit, for the Debye energy
R_GAS = 8.314462618  # J/(mol K)
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
    # The Debye energy is the integral of the Debye function, which is evaluated with Gauss-Legendre quadrature
    y = (theta / temp)[..., None]
    x = 0.5 * y * (_GL_X + 1.0)
    integral = 0.5 * y[..., 0] * np.sum(_GL_W * x**3 / np.expm1(x), axis=-1)
    return 9.0 * N_ATOMS * R_GAS * temp * (temp / theta) ** 3 * integral


def slb_shear_modulus(rho, temp, phase: str):
    """Solid shear modulus [Pa]: Stixrude & Lithgow-Bertelloni (2005) eq. 33 with the
    PALEOS strain of ``phase``."""
    # PALEOS density determines the strain, while the SLB parameters determine how that
    # strain and the thermal state affect shear rigidity
    v0, k0, k0p = PALEOS_EOS[phase]
    g0, g0p, eta0, theta0, gamma0, q0 = SLB_SHEAR[phase]

    rho, temp = np.asarray(rho, float), np.asarray(temp, float)

    # Calculate Eulerian finite strain; SLB (2005) eq. 24
    f = 0.5 * ((rho * v0 / M_MGSIO3) ** (2.0 / 3.0) - 1.0)

    # SLB (2005) eq. 47
    a1, a2 = 6.0 * gamma0, -12.0 * gamma0 + 36.0 * gamma0**2 - 18.0 * q0 * gamma0
    # SLB (2005) eq. 44
    freq2 = 1.0 + a1 * f + 0.5 * a2 * f * f
    # Strain-dependent Debye temperature
    theta = theta0 * np.sqrt(np.where(freq2 > 0, freq2, np.nan))

    # SLB (2005) eq. 44
    gamma = (1.0 + 2.0 * f) * (a1 + a2 * f) / (6.0 * freq2)
    # SLB (2005) eq. 46
    eta_s = -gamma + (1.0 + 2.0 * f) ** 2 * (gamma0 + eta0) / freq2
    # Compute the thermal contribution to the shear modulus as difference in enthalpy
    d_u = debye_energy(theta, temp) - debye_energy(theta, T_REF)

    # Cold contribution to the shear modulus; SLB (2005) eq. 33
    cold = (1.0 + 2.0 * f) ** 2.5 * (
        g0
        + (3.0 * k0 * g0p - 5.0 * g0) * f
        + (6.0 * k0 * g0p - 24.0 * k0 - 14.0 * g0 + 4.5 * k0 * k0p) * f * f
    )
    return cold - eta_s * d_u * rho / M_MGSIO3


def read_paleos_grid(path: Path) -> dict:
    """PALEOS unified table on its regular (log10 P, log10 T) grid; missing nodes are NaN."""
    import pandas as pd

    # Read the PALEOS table into a DataFrame, then pivot it into a dict of 2D arrays
    cols = ['P', 'T', 'rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad', 'phase']
    tab = pd.read_csv(path, sep=r'\s+', comment='#', header=None, names=cols)
    lp, lt = (
        np.round(np.log10(tab['P'].to_numpy()), 9),
        np.round(np.log10(tab['T'].to_numpy()), 9),
    )
    log_p, log_t = np.unique(lp), np.unique(lt)

    # Build the 2D arrays of rho and s on the (log P, log T) grid, with NaN where missing
    i, j = np.searchsorted(log_p, lp), np.searchsorted(log_t, lt)
    grid = {'log_p': log_p, 'log_t': log_t}
    for name in ('rho', 's'):
        arr = np.full((log_p.size, log_t.size), np.nan)
        arr[i, j] = tab[name].to_numpy(float)
        grid[name] = arr

    # Build the 2D array of phase codes on the (log P, log T) grid, with -1 where missing
    code = np.full((log_p.size, log_t.size), -1, dtype=np.int8)
    code[i, j] = [phase_code(x) for x in tab['phase'].to_numpy()]
    grid['code'] = code
    return grid


def _lnrho_along(grid, k, c, s0):
    """ln rho of phase ``c`` on pressure row ``k`` at entropies ``s0``; NaN off its range."""
    # The PALEOS table is on a regular (log10 P, log10 T) grid, but the entropy of a phase is
    # not monotonic in T, so we interpolate ln rho along the same-phase nodes of the pressure
    # row, and return NaN where the requested entropy is outside the range of that phase.
    if k < 0 or k >= grid['code'].shape[0]:
        return np.full(s0.shape, np.nan)
    # Find the indices of the nodes with phase code ``c`` in pressure row ``k``
    sel = grid['code'][k] == c
    # If there are fewer than two nodes with that phase code, we cannot interpolate
    if sel.sum() < 2:
        return np.full(s0.shape, np.nan)
    # Interpolate ln rho along the same-phase nodes of the pressure row, and return NaN where
    # the requested entropy is outside the range of that phase
    s, lr = grid['s'][k, sel], np.log(grid['rho'][k, sel])
    o = np.argsort(s)
    return np.interp(s0, s[o], lr[o], left=np.nan, right=np.nan)


def _dlnrho_dp(p, k, lr0, nb):
    """Estimate (d ln rho / dP)_s at pressure row k."""
    # Get the pressures at the previous row, current row and next row.
    # Clamp indices to the valid range so we do not index outside p.
    pm, p0, pp = (p[min(max(k + d, 0), p.size - 1)] for d in (-1, 0, 1))
    # Pressure spacing on each side of the current row.
    h1 = p0 - pm
    h2 = pp - p0
    # Central finite difference for nonuniform pressure spacing.
    # nb[-1] is the previous row; nb[1] is the next row.
    with np.errstate(divide='ignore', invalid='ignore'):
        central = (nb[1] - lr0) * h1 / (h2 * (h1 + h2)) + (lr0 - nb[-1]) * h2 / (h1 * (h1 + h2))
    # Accept the central estimate only where both adjacent densities are finite.
    out = np.where(
        np.isfinite(nb[1]) & np.isfinite(nb[-1]),
        central,
        np.nan,
    )
    # If the central estimate is unavailable, try one-sided estimates.
    for side in (1, -1):
        # Indices of the first and second neighbours on this side.
        k1 = min(max(k + side, 0), p.size - 1)
        k2 = min(max(k + 2 * side, 0), p.size - 1)
        # Their pressure coordinates relative to the current row.
        x1 = p[k1] - p0
        x2 = p[k2] - p0
        # Compute the one-sided derivative estimates
        with np.errstate(divide='ignore', invalid='ignore'):
            # Second-order one-sided derivative.
            three = (
                lr0 * (-x1 - x2) / (x1 * x2)
                + nb[side] * x2 / (x1 * (x2 - x1))
                - nb[2 * side] * x1 / (x2 * (x2 - x1))
            )
            # First-order one-sided derivative.
            two = (nb[side] - lr0) / x1
        # Use the second-order estimate where the second neighbour's
        # log-density is finite; otherwise use the first-order estimate.
        one_sided = np.where(
            np.isfinite(nb[2 * side]),
            three,
            two,
        )
        # Fill only entries that still lack a derivative.
        out = np.where(
            np.isfinite(out) | ~np.isfinite(nb[side]),
            out,
            one_sided,
        )
    # Any entry without sufficient valid neighbours remains NaN.
    return out


def adiabatic_bulk_grid(grid) -> np.ndarray:
    """K_S [Pa] at every node from same-phase, same-entropy neighbouring pressure rows."""
    # The adiabatic bulk modulus is K_S = (d ln rho / dP)_s^-1, differenced between
    # neighbouring pressure rows at the same entropy and within the same PALEOS phase
    p = 10.0 ** grid['log_p']
    out = np.full(grid['rho'].shape, np.nan)

    # For each pressure row, find the same-phase neighbours and compute K_S at every node
    for k in range(p.size):
        # For each phase code in this pressure row, find the indices of the nodes with that code
        for c in np.unique(grid['code'][k]):
            # Skip the missing code (-1) and the liquid code (LIQUID), which has no K_S
            if c < 0:
                continue
            # Find the indices of the nodes with this phase code in this pressure row
            idx = np.flatnonzero(grid['code'][k] == c)
            # Compute the adiabatic bulk modulus at every node with this phase code
            s0, lr0 = grid['s'][k, idx], np.log(grid['rho'][k, idx])
            # Get the ln rho of the same phase at the neighbouring pressure rows
            nb = {d: _lnrho_along(grid, k + d, c, s0) for d in (-2, -1, 1, 2)}
            # Compute the derivative (d ln rho / dP)_s at every node with this phase code
            dlr = _dlnrho_dp(p, k, lr0, nb)
            with np.errstate(divide='ignore', invalid='ignore'):
                out[k, idx] = np.where(dlr > 0, 1.0 / dlr, np.nan)
    return out


def shear_grid(grid) -> np.ndarray:
    """SLB solid shear modulus [Pa] at every solid node (NaN elsewhere)."""
    temp = np.broadcast_to(10.0 ** grid['log_t'], grid['rho'].shape)
    out = np.full(grid['rho'].shape, np.nan)
    # Compute the shear modulus at every solid node
    for c, name in enumerate(PHASES[:LIQUID]):
        # Find the indices of the nodes with this phase code in the entire grid
        idx = np.flatnonzero(grid['code'].ravel() == c)
        # Compute the shear modulus in chunks to avoid excessive memory usage
        for lo in range(0, idx.size, _CHUNK):
            sl = idx[lo : lo + _CHUNK]
            # Compute the shear modulus at every solid node with this phase code
            out.flat[sl] = slb_shear_modulus(grid['rho'].flat[sl], temp.flat[sl], name)
    return out


def _melting_row(code_row, *arrays):
    """Solid-side and melt-side node indices of one pressure row (-1 when absent)."""
    # Find the last solid node and the first melt node in this pressure row
    solid = np.flatnonzero((code_row >= 0) & (code_row < LIQUID))
    if solid.size == 0:
        return -1, -1
    # Step at most two cells into the solid phase where the boundary node has no K_S
    j_s = solid[-1]

    # Find the first liquid node after the last solid node in this pressure row
    liquid = np.flatnonzero(code_row == LIQUID)
    liquid = liquid[liquid > j_s]
    j_l = liquid[0] if liquid.size else -1
    # step at most two cells into the phase where the boundary node has no K_S
    for _ in range(2):
        if all(np.isfinite(a[j_s]) for a in arrays) or j_s - 1 not in solid:
            break
        j_s -= 1
    return j_s, j_l


def build_endmembers(grid, mu_liq) -> dict:
    """Arrays stored in the cache: solid grids and melting-curve end members per pressure.

    This function takes the PALEOS table on its regular (log10 P, log10 T) grid and derives the solid
    density, adiabatic bulk modulus and shear modulus at every node, and the solid-side and melt-side
    properties at the melting curve. The melt-side shear modulus is not derived from PALEOS, but is
    taken from the config (``interior_energetics.rheology.melt_shear_modulus``) when the end members
    are used in the rheology model.
    """
    # The adiabatic bulk modulus is derived from the PALEOS density and entropy
    kappa = adiabatic_bulk_grid(grid)

    # The solid shear modulus is derived from the PALEOS density and the SLB parameters of the phase
    solid = (grid['code'] >= 0) & (grid['code'] < LIQUID)
    mu = np.where(solid, shear_grid(grid), np.nan)

    # The melt-side shear modulus is the constant ``mu_liq`` of the config
    mu = np.where(grid['code'] == LIQUID, mu_liq, mu)

    # The melting curve is derived from the solid and melt-side properties
    n = grid['log_p'].size
    # Initialize the melt-side and solid-side properties at the melting curve to NaN
    melt = {
        k: np.full(n, np.nan)
        for k in ('rho_sol', 'kappa_sol', 'mu_sol', 'rho_liq', 'kappa_liq', 'mu_liq')
    }
    # For each pressure row, find the last solid node and the first melt node, and copy their properties
    for k in range(n):
        # Get the indices of the last solid node and the first melt node in this pressure row
        j_s, j_l = _melting_row(grid['code'][k], kappa[k], mu[k])
        # Copy the properties of the last solid node to the solid-side end members
        if j_s >= 0:
            melt['rho_sol'][k] = grid['rho'][k, j_s]
            melt['kappa_sol'][k] = kappa[k, j_s]
            melt['mu_sol'][k] = mu[k, j_s]
        # Copy the properties of the first melt node to the melt-side end members
        if j_l >= 0:
            melt['rho_liq'][k] = grid['rho'][k, j_l]
            melt['kappa_liq'][k] = kappa[k, j_l]
            melt['mu_liq'][k] = mu[k, j_l]
    return {
        'log_p': grid['log_p'],
        'log_t': grid['log_t'],
        'rho_solid': np.where(solid, grid['rho'], np.nan),
        'kappa_solid': np.where(solid, kappa, np.nan),
        # liquid nodes stay NaN, so a layer above the solid range takes the melting-curve solid side
        'mu_solid': np.where(solid, mu, np.nan),
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


def cache_dir() -> Path:
    """Cache directory: ``$PROTEUS_RHEOLOGY_CACHE_DIR`` or the FWL_DATA default."""
    env = os.environ.get(CACHE_ENV)
    return Path(env).expanduser() if env else resolve_fwl_data_dir() / CACHE_SUBDIR


def write_npz_atomic(path: Path, arrays: dict) -> None:
    """Write ``arrays`` to ``path`` through a temporary file, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def endmember_cache_path(
    paleos_path: Path, mu_liq: float, directory: Path | None = None
) -> Path:
    """Cache file of the end members derived from ``paleos_path`` with melt shear modulus ``mu_liq`` [Pa]."""
    directory = cache_dir() if directory is None else Path(directory)
    stem = f'paleos_endmembers_{_file_identity(paleos_path)}_{code_identity()}_mu{float(mu_liq):.6e}'
    return directory / f'{stem}.npz'


@functools.lru_cache(maxsize=2)
def load_endmembers(
    paleos_path: Path, mu_liq: float, directory: Path | None = None
) -> PaleosEndMembers:
    """End members from the cache, or built from the PALEOS table and cached on a miss.

    ``mu_liq`` is the unrelaxed melt shear modulus [Pa] of the config. It is stored in the
    melt-side end members, so it is part of the cache key: a cache built with another value
    is never read.
    """
    paleos_path = Path(paleos_path)
    if not paleos_path.is_file():
        raise FileNotFoundError(f'PALEOS MgSiO3 unified table not found: {paleos_path}')

    # Cache file path is keyed by the PALEOS file stamp, the builder code and mu_liq
    path = endmember_cache_path(paleos_path, mu_liq, directory)

    # Load the cache if it exists, otherwise build the end members and write the cache
    arrays = None
    if path.is_file():
        # Try to load the cache, but rebuild it if it is unreadable
        try:
            with np.load(path) as data:
                arrays = {k: data[k] for k in data.files}
        except (OSError, ValueError, EOFError, KeyError) as exc:
            log.warning('Rheology end-member cache %s unreadable (%s); rebuilding', path, exc)

    # If the cache was not loaded, build the end members from the PALEOS table and write the cache
    if arrays is None:
        log.info('Deriving mush end members from %s', paleos_path)
        arrays = build_endmembers(read_paleos_grid(paleos_path), mu_liq)
        write_npz_atomic(path, arrays)
    return PaleosEndMembers(arrays)


def paleos_mgsio3_path() -> Path:
    """Path of the PALEOS MgSiO3 unified table in the FWL data tree.

    ``$PROTEUS_PALEOS_MGSIO3_TABLE`` overrides it with another table file.
    """
    override = os.environ.get('PROTEUS_PALEOS_MGSIO3_TABLE')
    if override:
        return Path(override).expanduser()

    from proteus.data import EOS_PALEOS_MGSIO3_UNIFIED, dataset_dir
    from proteus.utils.helper import resolve_fwl_data_dir

    root = dataset_dir(EOS_PALEOS_MGSIO3_UNIFIED, data_root=resolve_fwl_data_dir())
    return root / 'paleos_mgsio3_eos_table_pt.dat'


class PaleosEndMembers:
    """Interpolated end members; pressures in Pa, temperatures in K, moduli in Pa."""

    def __init__(self, arrays: dict):
        # Cache the arrays across log P and log T
        self._a = arrays
        self.log_p, self.log_t = arrays['log_p'], arrays['log_t']

    def melting(self, pres) -> dict:
        """Solid-side and melt-side properties at the melting curve, linear in log10 P."""
        # The melting curve is the last solid node and the first melt node of each pressure
        lp = np.clip(np.log10(np.asarray(pres, float)), self.log_p[0], self.log_p[-1])

        # Interpolate the melting-curve end members along log10 P; NaN where no solid or melt
        out = {}
        for key in ('rho_sol', 'kappa_sol', 'mu_sol', 'rho_liq', 'kappa_liq', 'mu_liq'):
            y = self._a[f'melt_{key}']
            ok = np.isfinite(y)
            out[key] = np.interp(lp, self.log_p[ok], y[ok])
        return out

    def solid(self, pres, temp) -> dict:
        """Solid density, K_S and shear modulus at (P, T), bilinear in (log10 P, log10 T).

        Corners without solid data are dropped and the weights renormalised; where no
        corner is solid (T above the solid range) the melting-curve solid side is used.
        """
        # Select the log10 P and log10 T coordinates, clipped to the grid range
        lp = np.clip(np.log10(np.asarray(pres, float)), self.log_p[0], self.log_p[-1])
        lt = np.clip(np.log10(np.asarray(temp, float)), self.log_t[0], self.log_t[-1])
        lp, lt = np.broadcast_arrays(lp, lt)
        i = np.clip(np.searchsorted(self.log_p, lp) - 1, 0, self.log_p.size - 2)
        j = np.clip(np.searchsorted(self.log_t, lt) - 1, 0, self.log_t.size - 2)

        # Compute the bilinear interpolation weights in (log P, log T)
        wp = (lp - self.log_p[i]) / (self.log_p[i + 1] - self.log_p[i])
        wt = (lt - self.log_t[j]) / (self.log_t[j + 1] - self.log_t[j])

        # If no corner has solid data, use the melting-curve solid side as a fallback
        fallback = self.melting(10.0**lp)

        # Compute the bilinear interpolation of the solid properties at the four corners of the grid cell
        out = {}
        for key, mkey in (('rho', 'rho_sol'), ('kappa', 'kappa_sol'), ('mu', 'mu_sol')):
            # The solid properties are stored in the cache as 2D arrays on the (log P, log T) grid
            grid = self._a[f'{key}_solid']
            num, den = np.zeros(lp.shape), np.zeros(lp.shape)
            # Compute the weighted sum of the four corners of the grid cell, ignoring NaN values
            for di, dj, w in ((0, 0, (1 - wp) * (1 - wt)), (1, 0, wp * (1 - wt)),
                              (0, 1, (1 - wp) * wt), (1, 1, wp * wt)):  # fmt: skip
                val = grid[i + di, j + dj]
                ok = np.isfinite(val)
                num += np.where(ok, w * np.where(ok, val, 0.0), 0.0)
                den += np.where(ok, w, 0.0)
            # Normalize the weighted sum by the sum of the weights
            with np.errstate(invalid='ignore', divide='ignore'):
                out[key] = np.where(den > 0, num / den, fallback[mkey])
        return out
