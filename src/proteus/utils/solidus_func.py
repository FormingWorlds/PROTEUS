# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Generate melting curves in pressure–temperature (P–T) and pressure–entropy (P–S)
# space for several literature parametrizations of peridotite / silicate melting.

# This version is designed to work with the legacy EOS lookup tables:

#     - temperature_solid.dat
#     - temperature_melt.dat

# These tables provide temperature as a function of entropy and pressure,
# T(S, P), on structured grids. The script therefore:

# 1. Builds solidus and liquidus curves in P–T space from literature fits.
# 2. Converts those curves into P–S space by inverting the EOS relation T(S, P).
# 3. Resamples the solidus and liquidus entropy curves onto a common pressure grid.
# 4. Saves both the P–T and P–S versions to disk.

# Usage
# -----
# Export one model with a dedicated shortcut:

#     python solidus_func.py --katz2003 --X-h2o 30
#     python solidus_func.py --lin2024 --fO2 -4

# Export one model by explicit internal name:

#     python solidus_func.py --model wolf_bower_2018

# Export all supported models:

#     python solidus_func.py --all

# Notes
# -----
# - Pressure is handled internally in GPa for the melting-curve parametrizations,
#   unless a given published fit is explicitly defined in Pa and converted.
# - The EOS tables are assumed to have the SPIDER-style format with scaling factors
#   in the header.
# - The exported entropy files use the same scaled SPIDER-like header so they can
#   be re-used by downstream tools expecting that format.

# References included here
# ------------------------
# - Andrault et al. (2011), DOI: 10.1016/j.epsl.2011.02.006
# - Monteux et al. (2016), DOI: 10.1016/j.epsl.2016.05.010
# - Lin et al. (2024), DOI: 10.1038/s41561-024-01495-1
# - Hirschmann (2000), DOI: 10.1029/2000GC000070
# - Katz et al. (2003), DOI: 10.1029/2002GC000433
# - Stixrude (2014), DOI: 10.1098/rsta.2013.0076
# - Fei et al. (2021), DOI: 10.1038/s41467-021-21170-y
# - Belonoshko et al. (2005), DOI: 10.1103/PhysRevLett.94.195701
# - Fiquet et al. (2010), DOI: 10.1126/science.1192448
# """

# from __future__ import annotations

# import argparse
# import os
# from pathlib import Path

# import numpy as np
# import platformdirs
# from scipy.interpolate import RegularGridInterpolator, interp1d

# # =============================================================================
# # DEFAULT OUTPUT LOCATION
# # =============================================================================

# FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', platformdirs.user_data_dir('fwl_data')))

# MELTING_DIR = FWL_DATA_DIR / 'interior_lookup_tables' / 'Melting_curves'


# # =============================================================================
# # GENERAL HELPERS
# # =============================================================================


# def make_pressure_grid(Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500) -> np.ndarray:
#     r"""
#     Create a uniformly sampled pressure grid in GPa.

#     Parameters
#     ----------
#     Pmin : float, optional
#         Minimum pressure in GPa.
#     Pmax : float, optional
#         Maximum pressure in GPa.
#     n : int, optional
#         Number of grid points.

#     Returns
#     -------
#     np.ndarray
#         One-dimensional pressure array in GPa.
#     """
#     return np.linspace(Pmin, Pmax, n)


# def solidus_from_liquidus_stixrude(T_liq: np.ndarray) -> np.ndarray:
#     r"""
#     Estimate the solidus from a liquidus using the Stixrude ratio.
#     """
#     return T_liq / (1.0 - np.log(0.79))


# def liquidus_from_solidus_stixrude(T_sol: np.ndarray) -> np.ndarray:
#     r"""
#     Estimate the liquidus from a solidus using the inverse Stixrude ratio.
#     """
#     return T_sol * (1.0 - np.log(0.79))


# def _fmt_range(arr: np.ndarray, fmt: str = '.3f') -> str:
#     """
#     Format the finite minimum and maximum of an array as a string.
#     """
#     arr = np.asarray(arr, dtype=float)
#     mask = np.isfinite(arr)

#     if not np.any(mask):
#         return '[nan, nan]'

#     amin = np.min(arr[mask])
#     amax = np.max(arr[mask])
#     return f'[{amin:{fmt}}, {amax:{fmt}}]'


# DISPLAY_NAMES = {
#     'andrault_2011': 'Andrault et al. (2011)',
#     'monteux_2016': 'Monteux et al. (2016)',
#     'wolf_bower_2018': 'Wolf & Bower (2018)',
#     'katz_2003': 'Katz et al. (2003)',
#     'fei_2021': 'Fei et al. (2021)',
#     'belonoshko_2005': 'Belonoshko et al. (2005)',
#     'fiquet_2010': 'Fiquet et al. (2010)',
#     'hirschmann_2000': 'Hirschmann (2000)',
#     'stixrude_2014': 'Stixrude (2014)',
#     'lin_2024': 'Lin et al. (2024)',
# }


# def print_model_summary(
#     model_name: str,
#     P_sol: np.ndarray,
#     T_sol: np.ndarray,
#     P_liq: np.ndarray,
#     T_liq: np.ndarray,
#     P_common: np.ndarray,
#     S_sol_common: np.ndarray,
#     S_liq_common: np.ndarray,
# ):
#     """
#     Print a compact summary of the exported P-T and P-S ranges for one model.
#     """
#     label = DISPLAY_NAMES.get(model_name, model_name)
#     print(f'{label}:')
#     print(
#         f'  P-T solidus   : P_range = {_fmt_range(P_sol)} GPa, T_range = {_fmt_range(T_sol)} K'
#     )
#     print(
#         f'  P-T liquidus  : P_range = {_fmt_range(P_liq)} GPa, T_range = {_fmt_range(T_liq)} K'
#     )
#     print(
#         f'  P-S solidus   : P_range = {_fmt_range(P_common)} GPa, S_range = {_fmt_range(S_sol_common)} J kg^-1 K^-1'
#     )
#     print(
#         f'  P-S liquidus  : P_range = {_fmt_range(P_common)} GPa, S_range = {_fmt_range(S_liq_common)} J kg^-1 K^-1'
#     )
#     print()


# # =============================================================================
# # LITERATURE MELTING CURVES
# # =============================================================================


# def andrault_2011(kind: str = 'solidus', Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500):
#     r"""
#     Melting curve from Andrault et al. (2011).
#     """
#     P = make_pressure_grid(Pmin, Pmax, n)
#     P_pa = P * 1e9

#     if kind == 'solidus':
#         T0, a, c = 2045, 92e9, 1.3
#     elif kind == 'liquidus':
#         T0, a, c = 1940, 29e9, 1.9
#     else:
#         raise ValueError("kind must be 'solidus' or 'liquidus'")

#     T = T0 * ((P_pa / a) + 1.0) ** (1.0 / c)
#     return P, T


# def fei_2021(kind: str = 'liquidus', Pmin: float = 1.0, Pmax: float = 1000.0, n: int = 500):
#     r"""
#     Melting curve based on Fei et al. (2021).
#     """
#     P = make_pressure_grid(Pmin, Pmax, n)
#     T_liq = 6000.0 * (P / 140.0) ** 0.26

#     if kind == 'liquidus':
#         T = T_liq
#     elif kind == 'solidus':
#         T = solidus_from_liquidus_stixrude(T_liq)
#     else:
#         raise ValueError("kind must be 'solidus' or 'liquidus'")

#     return P, T


# def belonoshko_2005(
#     kind: str = 'liquidus', Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500
# ):
#     r"""
#     Melting curve based on Belonoshko et al. (2005).
#     """
#     P = make_pressure_grid(Pmin, Pmax, n)
#     T_liq = 1831.0 * (1.0 + P / 4.6) ** 0.33

#     if kind == 'liquidus':
#         T = T_liq
#     elif kind == 'solidus':
#         T = solidus_from_liquidus_stixrude(T_liq)
#     else:
#         raise ValueError("kind must be 'solidus' or 'liquidus'")

#     return P, T


# def fiquet_2010(kind: str = 'liquidus', Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500):
#     r"""
#     Melting curve based on Fiquet et al. (2010).
#     """
#     P = make_pressure_grid(Pmin, Pmax, n)
#     T_liq = np.zeros_like(P, dtype=float)

#     low = P <= 20.0
#     high = P > 20.0

#     # Original high-pressure constant is given in Pa in the paper.
#     # Here pressure is in GPa, so 4.054e6 Pa -> 0.004054 GPa.
#     T_liq[low] = 1982.1 * ((P[low] / 6.594) + 1.0) ** (1.0 / 5.374)
#     T_liq[high] = 78.74 * ((P[high] / 0.004056) + 1.0) ** (1.0 / 2.44)

#     if kind == 'liquidus':
#         T = T_liq
#     elif kind == 'solidus':
#         T = solidus_from_liquidus_stixrude(T_liq)
#     else:
#         raise ValueError("kind must be 'solidus' or 'liquidus'")

#     return P, T


# def monteux_2016(kind: str = 'solidus', Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500):
#     r"""
#     Melting curve from Monteux et al. (2016).
#     """
#     P = make_pressure_grid(Pmin, Pmax, n)
#     P_pa = P * 1e9

#     params = {
#         'solidus': {
#             'low': {'T0': 1661.2, 'a': 1.336e9, 'c': 7.437},
#             'high': {'T0': 2081.8, 'a': 101.69e9, 'c': 1.226},
#         },
#         'liquidus': {
#             'low': {'T0': 1982.1, 'a': 6.594e9, 'c': 5.374},
#             'high': {'T0': 2006.8, 'a': 34.65e9, 'c': 1.844},
#         },
#     }

#     if kind not in params:
#         raise ValueError("kind must be 'solidus' or 'liquidus'")

#     p = params[kind]
#     T = np.zeros_like(P_pa, dtype=float)

#     mask_low = P_pa <= 20e9
#     mask_high = P_pa > 20e9

#     T[mask_low] = p['low']['T0'] * ((P_pa[mask_low] / p['low']['a']) + 1.0) ** (
#         1.0 / p['low']['c']
#     )
#     T[mask_high] = p['high']['T0'] * ((P_pa[mask_high] / p['high']['a']) + 1.0) ** (
#         1.0 / p['high']['c']
#     )

#     return P, T


# def hirschmann_2000(kind: str = 'solidus', Pmin: float = 0.0, Pmax: float = 10.0, n: int = 500):
#     r"""
#     Melting curve from Hirschmann (2000).
#     """
#     P = make_pressure_grid(Pmin, Pmax, n)

#     coeffs = {
#         'solidus': (1085.7, 132.9, -5.1),
#         'liquidus': (1475.0, 80.0, -3.2),
#     }

#     if kind not in coeffs:
#         raise ValueError("kind must be 'solidus' or 'liquidus'")

#     A1, A2, A3 = coeffs[kind]
#     T_c = A1 + A2 * P + A3 * P**2
#     T = T_c + 273.15

#     return P, T


# def stixrude_2014(
#     kind: str = 'liquidus', Pmin: float = 1.0, Pmax: float = 1000.0, n: int = 500
# ):
#     r"""
#     Melting curve based on Stixrude (2014).
#     """
#     P = make_pressure_grid(Pmin, Pmax, n)
#     T_liq = 5400.0 * (P / 140.0) ** 0.480

#     if kind == 'liquidus':
#         T = T_liq
#     elif kind == 'solidus':
#         T = solidus_from_liquidus_stixrude(T_liq)
#     else:
#         raise ValueError("kind must be 'solidus' or 'liquidus'")

#     return P, T


# def wolf_bower_2018(
#     kind: str = 'solidus', Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500
# ):
#     r"""
#     Piecewise melting curve based on Wolf & Bower (2018) style fits.
#     """
#     P = make_pressure_grid(Pmin, Pmax, n)

#     params = {
#         'solidus': (
#             7.696777581585296,
#             870.4767697319186,
#             101.52655163737373,
#             15.959022187236807,
#             3.090844734784906,
#             1417.4258954709148,
#         ),
#         'liquidus': (
#             8.864665249317456,
#             408.58442302949794,
#             46.288444869816615,
#             17.549174419770257,
#             3.679647802112376,
#             2019.967799687511,
#         ),
#     }

#     if kind not in params:
#         raise ValueError("kind must be 'solidus' or 'liquidus'")

#     cp1, cp2, s1, s2, s3, intercept = params[kind]

#     c1 = intercept
#     c2 = c1 + (s1 - s2) * cp1
#     c3 = c2 + (s2 - s3) * cp2

#     T = np.zeros_like(P, dtype=float)

#     m1 = P < cp1
#     m2 = (P >= cp1) & (P < cp2)
#     m3 = P >= cp2

#     T[m1] = s1 * P[m1] + c1
#     T[m2] = s2 * P[m2] + c2
#     T[m3] = s3 * P[m3] + c3

#     return P, T


# def katz_2003(
#     kind: str = 'solidus',
#     X_h2o: float = 30.0,
#     Pmin: float = 0.0,
#     Pmax: float = 1000.0,
#     n: int = 500,
# ):
#     r"""
#     Hydrous melting-curve correction following Katz et al. (2003).
#     """
#     gamma = 0.75
#     K = 43.0

#     if kind not in {'solidus', 'liquidus'}:
#         raise ValueError("kind must be 'solidus' or 'liquidus'")

#     P, T_anhydrous = wolf_bower_2018(kind=kind, Pmin=Pmin, Pmax=Pmax, n=n)
#     delta_T = K * X_h2o**gamma
#     T = T_anhydrous - delta_T

#     return P, T


# def lin_2024(
#     kind: str = 'solidus',
#     fO2: float = -4.0,
#     Pmin: float = 0.0,
#     Pmax: float = 1000.0,
#     n: int = 500,
# ):
#     r"""
#     Oxygen-fugacity-dependent solidus following Lin et al. (2024).
#     """
#     P, T_anhydrous = wolf_bower_2018(kind='solidus', Pmin=Pmin, Pmax=Pmax, n=n)

#     delta_T = (340.0 / 3.2) * (2.0 - fO2)
#     T_sol = T_anhydrous + delta_T

#     if kind == 'solidus':
#         T = T_sol
#     elif kind == 'liquidus':
#         T = liquidus_from_solidus_stixrude(T_sol)
#     else:
#         raise ValueError("kind must be 'solidus' or 'liquidus'")

#     return P, T


# # =============================================================================
# # PHYSICAL-INTERVAL FILTER
# # =============================================================================


# def truncate_to_physical_interval(func):
#     r"""
#     Wrap a melting-curve function so only the main interval with
#     T_sol < T_liq is retained.
#     """

#     def wrapped(kind='solidus', Pmin=0.0, Pmax=1000.0, n=2000, **kwargs):
#         P_sol, T_sol = func(kind='solidus', Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)
#         P_liq, T_liq = func(kind='liquidus', Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)

#         good = T_sol < T_liq
#         idx = np.where(good)[0]

#         if len(idx) == 0:
#             raise ValueError(f'{func.__name__}: no physical interval where solidus < liquidus')

#         splits = np.where(np.diff(idx) > 1)[0] + 1
#         blocks = np.split(idx, splits)
#         main_block = max(blocks, key=len)

#         if kind == 'solidus':
#             return P_sol[main_block], T_sol[main_block]
#         if kind == 'liquidus':
#             return P_liq[main_block], T_liq[main_block]
#         raise ValueError("kind must be 'solidus' or 'liquidus'")

#     return wrapped


# andrault_2011_cut = truncate_to_physical_interval(andrault_2011)
# monteux_2016_cut = truncate_to_physical_interval(monteux_2016)
# wolf_bower_2018_cut = truncate_to_physical_interval(wolf_bower_2018)
# katz_2003_cut = truncate_to_physical_interval(katz_2003)


# # =============================================================================
# # MODEL DISPATCHER
# # =============================================================================

# SUPPORTED_MODELS = [
#     'andrault_2011',
#     'monteux_2016',
#     'wolf_bower_2018',
#     'katz_2003',
#     'fei_2021',
#     'belonoshko_2005',
#     'fiquet_2010',
#     'hirschmann_2000',
#     'stixrude_2014',
#     'lin_2024',
# ]


# def get_melting_curves(
#     model_name: str, Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 2000, **kwargs
# ):
#     r"""
#     Return solidus and liquidus curves for a given model.
#     """
#     models = {
#         'andrault_2011': andrault_2011_cut,
#         'monteux_2016': monteux_2016_cut,
#         'wolf_bower_2018': wolf_bower_2018_cut,
#         'katz_2003': katz_2003_cut,
#         'fei_2021': fei_2021,
#         'belonoshko_2005': belonoshko_2005,
#         'fiquet_2010': fiquet_2010,
#         'hirschmann_2000': hirschmann_2000,
#         'stixrude_2014': stixrude_2014,
#         'lin_2024': lin_2024,
#     }

#     if model_name not in models:
#         raise ValueError(f'Unknown model: {model_name}')

#     func = models[model_name]
#     P_sol, T_sol = func(kind='solidus', Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)
#     P_liq, T_liq = func(kind='liquidus', Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)

#     return P_sol, T_sol, P_liq, T_liq


# # =============================================================================
# # OUTPUT HELPERS
# # =============================================================================


# def save_PT_table(path: Path, P_gpa: np.ndarray, T_k: np.ndarray):
#     r"""
#     Save a pressure-temperature table to disk.
#     """
#     data = np.column_stack([P_gpa, T_k])
#     np.savetxt(path, data, fmt='%.18e %.18e', header='pressure temperature', comments='#')


# # =============================================================================
# # EOS LOOKUP TABLE SETTINGS
# # =============================================================================

# SCRIPT_DIR = Path(__file__).resolve().parent
# spider_dir = (SCRIPT_DIR / '../../../SPIDER').resolve()

# eos_solid_path = spider_dir / 'lookup_data' / '1TPa-dK09-elec-free' / 'temperature_solid.dat'
# eos_liquid_path = spider_dir / 'lookup_data' / '1TPa-dK09-elec-free' / 'temperature_melt.dat'

# if not eos_solid_path.exists():
#     raise FileNotFoundError(f'Missing EOS file: {eos_solid_path}')

# if not eos_liquid_path.exists():
#     raise FileNotFoundError(f'Missing EOS file: {eos_liquid_path}')

# nP = 2020
# nS_solid = 125
# nS_liquid = 95
# skip_header = 5

# SCALE_P_EOS = 1e9
# SCALE_T_EOS = 1.0
# SCALE_S_SOLID_EOS = 4.82426684604467e6
# SCALE_S_LIQUID_EOS = 4.805046659407042e6

# SCALE_P_OUT = 1_000_000_000.0
# SCALE_S_OUT = 4_824_266.84604467

# COMMON_HEADER = '\n'.join(
#     [
#         '# 5 2000',
#         '# Pressure, Entropy, Quantity',
#         '# column * scaling factor should be SI units',
#         '# scaling factors (constant) for each column given on line below',
#         '# 1000000000.0 4824266.84604467',
#     ]
# )


# def load_eos_T_of_SP(eos_path: Path, nS: int, scale_S_axis: float):
#     r"""
#     Load an EOS lookup table and build an interpolator for T(S, P).
#     """
#     raw = np.genfromtxt(eos_path, skip_header=skip_header)
#     resh = raw.reshape((nS, nP, 3))

#     P_axis_GPa = resh[0, :, 0] * SCALE_P_EOS / 1e9
#     S_axis = resh[:, 0, 1] * scale_S_axis
#     T_grid = resh[:, :, 2] * SCALE_T_EOS

#     T_interp = RegularGridInterpolator(
#         (S_axis, P_axis_GPa),
#         T_grid,
#         bounds_error=False,
#         fill_value=np.nan,
#     )
#     return S_axis, P_axis_GPa, T_interp


# def invert_to_entropy_along_profile(
#     P_gpa: np.ndarray, T_k: np.ndarray, S_axis: np.ndarray, T_of_SP
# ):
#     r"""
#     Convert a P-T curve into a P-S curve by inverting T(S, P).
#     """
#     S_out = np.full_like(T_k, np.nan, dtype=float)

#     for i, (P_i_gpa, T_i) in enumerate(zip(P_gpa, T_k)):
#         P_col = np.full_like(S_axis, P_i_gpa)
#         T_vals = T_of_SP(np.column_stack([S_axis, P_col]))

#         valid = np.isfinite(T_vals)
#         if np.count_nonzero(valid) < 2:
#             continue

#         Tv = T_vals[valid]
#         Sv = S_axis[valid]

#         order = np.argsort(Tv)
#         T_sorted = Tv[order]
#         S_sorted = Sv[order]

#         T_unique, idx_unique = np.unique(T_sorted, return_index=True)
#         S_unique = S_sorted[idx_unique]

#         if len(T_unique) < 2:
#             continue

#         if T_i < T_unique[0] or T_i > T_unique[-1]:
#             continue

#         try:
#             f = interp1d(T_unique, S_unique, kind='linear', assume_sorted=True)
#             S_out[i] = float(f(T_i))
#         except Exception:
#             try:
#                 f = interp1d(T_unique, S_unique, kind='nearest', assume_sorted=True)
#                 S_out[i] = float(f(T_i))
#             except Exception:
#                 continue

#     return S_out


# def build_common_entropy_grid(
#     P_sol: np.ndarray,
#     S_sol: np.ndarray,
#     P_liq: np.ndarray,
#     S_liq: np.ndarray,
#     n_common: int | None = None,
# ):
#     r"""
#     Resample solidus and liquidus entropy curves onto a shared pressure grid.
#     """
#     mask_sol = np.isfinite(S_sol)
#     mask_liq = np.isfinite(S_liq)

#     P_sol_v = np.asarray(P_sol[mask_sol], dtype=float)
#     S_sol_v = np.asarray(S_sol[mask_sol], dtype=float)
#     P_liq_v = np.asarray(P_liq[mask_liq], dtype=float)
#     S_liq_v = np.asarray(S_liq[mask_liq], dtype=float)

#     if len(P_sol_v) < 2 or len(P_liq_v) < 2:
#         return np.array([]), np.array([]), np.array([])

#     Pmin_common = max(np.min(P_sol_v), np.min(P_liq_v))
#     Pmax_common = min(np.max(P_sol_v), np.max(P_liq_v))

#     if (
#         not np.isfinite(Pmin_common)
#         or not np.isfinite(Pmax_common)
#         or Pmax_common <= Pmin_common
#     ):
#         return np.array([]), np.array([]), np.array([])

#     if n_common is None:
#         n_common = min(len(P_sol_v), len(P_liq_v))

#     order_sol = np.argsort(P_sol_v)
#     order_liq = np.argsort(P_liq_v)

#     P_sol_s = P_sol_v[order_sol]
#     S_sol_s = S_sol_v[order_sol]
#     P_liq_s = P_liq_v[order_liq]
#     S_liq_s = S_liq_v[order_liq]

#     P_sol_u, idx_sol = np.unique(P_sol_s, return_index=True)
#     S_sol_u = S_sol_s[idx_sol]

#     P_liq_u, idx_liq = np.unique(P_liq_s, return_index=True)
#     S_liq_u = S_liq_s[idx_liq]

#     if len(P_sol_u) < 2 or len(P_liq_u) < 2:
#         return np.array([]), np.array([]), np.array([])

#     P_common = np.linspace(Pmin_common, Pmax_common, n_common)

#     f_sol = interp1d(
#         P_sol_u,
#         S_sol_u,
#         kind='linear',
#         bounds_error=False,
#         fill_value=np.nan,
#         assume_sorted=True,
#     )
#     f_liq = interp1d(
#         P_liq_u,
#         S_liq_u,
#         kind='linear',
#         bounds_error=False,
#         fill_value=np.nan,
#         assume_sorted=True,
#     )

#     S_sol_common = f_sol(P_common)
#     S_liq_common = f_liq(P_common)

#     mask = np.isfinite(S_sol_common) & np.isfinite(S_liq_common)
#     return P_common[mask], S_sol_common[mask], S_liq_common[mask]


# def save_entropy_table_with_header(path: Path, P_gpa: np.ndarray, S_jpk: np.ndarray):
#     r"""
#     Save a pressure-entropy table in SPIDER-style scaled format.
#     """
#     P_pa = P_gpa * 1e9
#     data = np.column_stack([P_pa / SCALE_P_OUT, S_jpk / SCALE_S_OUT])
#     np.savetxt(path, data, fmt='%.18e %.18e', header=COMMON_HEADER, comments='')


# S_axis_solid, P_axis_solid, T_of_SP_solid = load_eos_T_of_SP(
#     eos_solid_path, nS_solid, SCALE_S_SOLID_EOS
# )
# S_axis_liquid, P_axis_liquid, T_of_SP_liquid = load_eos_T_of_SP(
#     eos_liquid_path, nS_liquid, SCALE_S_LIQUID_EOS
# )


# # =============================================================================
# # MAIN EXPORTER
# # =============================================================================


# def export_model_curves(
#     model_name: str,
#     out_root: Path | str = MELTING_DIR,
#     Pmin: float = 0.0,
#     Pmax: float = 1000.0,
#     n: int = 2000,
#     **kwargs,
# ):
#     r"""
#     Export one melting model in both P-T and P-S space.
#     """
#     out_dir = Path(out_root) / model_name
#     out_dir.mkdir(parents=True, exist_ok=True)

#     P_sol, T_sol, P_liq, T_liq = get_melting_curves(
#         model_name, Pmin=Pmin, Pmax=Pmax, n=n, **kwargs
#     )

#     save_PT_table(out_dir / 'solidus_P-T.dat', P_sol, T_sol)
#     save_PT_table(out_dir / 'liquidus_P-T.dat', P_liq, T_liq)

#     S_sol = invert_to_entropy_along_profile(P_sol, T_sol, S_axis_solid, T_of_SP_solid)
#     S_liq = invert_to_entropy_along_profile(P_liq, T_liq, S_axis_liquid, T_of_SP_liquid)

#     P_common, S_sol_common, S_liq_common = build_common_entropy_grid(
#         P_sol, S_sol, P_liq, S_liq, n_common=n
#     )

#     save_entropy_table_with_header(
#         out_dir / 'solidus_P-S.dat',
#         P_common,
#         S_sol_common,
#     )

#     save_entropy_table_with_header(
#         out_dir / 'liquidus_P-S.dat',
#         P_common,
#         S_liq_common,
#     )

#     print_model_summary(
#         model_name,
#         P_sol,
#         T_sol,
#         P_liq,
#         T_liq,
#         P_common,
#         S_sol_common,
#         S_liq_common,
#     )

#     print(f'  Saved to      : {out_dir.resolve()}')
#     print()

#     return {
#         'P_sol': P_sol,
#         'T_sol': T_sol,
#         'P_liq': P_liq,
#         'T_liq': T_liq,
#         'S_sol': S_sol,
#         'S_liq': S_liq,
#         'P_entropy_common': P_common,
#         'S_sol_common': S_sol_common,
#         'S_liq_common': S_liq_common,
#     }


# # =============================================================================
# # BATCH EXPORTER
# # =============================================================================


# def export_all_models(out_root: Path | str = MELTING_DIR, n: int = 2000):
#     r"""
#     Export all supported melting parametrizations.
#     """
#     for model in SUPPORTED_MODELS:
#         if model == 'katz_2003':
#             _ = export_model_curves(model, out_root=out_root, n=n, X_h2o=30.0)
#         elif model == 'lin_2024':
#             _ = export_model_curves(model, out_root=out_root, n=n, fO2=-4.0)
#         elif model == 'hirschmann_2000':
#             _ = export_model_curves(model, out_root=out_root, n=n, Pmax=10.0)
#         elif model == 'fei_2021':
#             _ = export_model_curves(model, out_root=out_root, n=n, Pmin=1.0)
#         elif model == 'stixrude_2014':
#             _ = export_model_curves(model, out_root=out_root, n=n, Pmin=1.0)
#         else:
#             _ = export_model_curves(model, out_root=out_root, n=n)


# # =============================================================================
# # COMMAND-LINE INTERFACE
# # =============================================================================


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description=(
#             'Export solidus and liquidus melting curves in P-T and P-S space '
#             'for one or more literature parametrizations.'
#         ),
#         epilog=(
#             'Examples:\n'
#             '  python solidus_func.py --all\n'
#             '  python solidus_func.py --katz2003 --X-h2o 30\n'
#             '  python solidus_func.py --lin2024 --fO2 -4\n'
#             '  python solidus_func.py --model wolf_bower_2018\n'
#         ),
#         formatter_class=argparse.RawTextHelpFormatter,
#     )

#     parser.add_argument(
#         '--all',
#         action='store_true',
#         help='Export all supported models.',
#     )

#     parser.add_argument(
#         '--model',
#         type=str,
#         default=None,
#         choices=SUPPORTED_MODELS,
#         help='Export a single model by internal name.',
#     )

#     parser.add_argument(
#         '--katz2003',
#         action='store_true',
#         help='Export Katz et al. (2003). Requires --X-h2o.',
#     )
#     parser.add_argument(
#         '--lin2024',
#         action='store_true',
#         help='Export Lin et al. (2024). Requires --fO2.',
#     )
#     parser.add_argument(
#         '--wolfbower2018',
#         action='store_true',
#         help='Export Wolf & Bower (2018).',
#     )
#     parser.add_argument(
#         '--andrault2011',
#         action='store_true',
#         help='Export Andrault et al. (2011).',
#     )
#     parser.add_argument(
#         '--monteux2016',
#         action='store_true',
#         help='Export Monteux et al. (2016).',
#     )
#     parser.add_argument(
#         '--fei2021',
#         action='store_true',
#         help='Export Fei et al. (2021).',
#     )
#     parser.add_argument(
#         '--belonoshko2005',
#         action='store_true',
#         help='Export Belonoshko et al. (2005).',
#     )
#     parser.add_argument(
#         '--fiquet2010',
#         action='store_true',
#         help='Export Fiquet et al. (2010).',
#     )
#     parser.add_argument(
#         '--hirschmann2000',
#         action='store_true',
#         help='Export Hirschmann (2000).',
#     )
#     parser.add_argument(
#         '--stixrude2014',
#         action='store_true',
#         help='Export Stixrude (2014).',
#     )

#     parser.add_argument(
#         '--out-root',
#         type=str,
#         default=str(MELTING_DIR),
#         help='Root directory where output folders will be created.',
#     )

#     parser.add_argument(
#         '--Pmin',
#         type=float,
#         default=0.0,
#         help='Minimum pressure in GPa.',
#     )
#     parser.add_argument(
#         '--Pmax',
#         type=float,
#         default=1000.0,
#         help='Maximum pressure in GPa.',
#     )
#     parser.add_argument(
#         '-n',
#         type=int,
#         default=2000,
#         help='Number of pressure samples.',
#     )

#     parser.add_argument(
#         '--X-h2o',
#         dest='X_h2o',
#         type=float,
#         default=None,
#         help='Water content parameter for Katz (2003). Required for --katz2003.',
#     )
#     parser.add_argument(
#         '--fO2',
#         type=float,
#         default=None,
#         help='Oxygen fugacity offset for Lin (2024). Required for --lin2024.',
#     )

#     return parser.parse_args()


# def resolve_requested_model(args) -> str | None:
#     """
#     Resolve which single-model shortcut flag was requested.
#     """
#     shortcut_map = {
#         'katz2003': 'katz_2003',
#         'lin2024': 'lin_2024',
#         'wolfbower2018': 'wolf_bower_2018',
#         'andrault2011': 'andrault_2011',
#         'monteux2016': 'monteux_2016',
#         'fei2021': 'fei_2021',
#         'belonoshko2005': 'belonoshko_2005',
#         'fiquet2010': 'fiquet_2010',
#         'hirschmann2000': 'hirschmann_2000',
#         'stixrude2014': 'stixrude_2014',
#     }

#     chosen = [model for flag, model in shortcut_map.items() if getattr(args, flag)]

#     if len(chosen) > 1:
#         raise SystemExit('Error: please select only one model shortcut flag at a time.')

#     if len(chosen) == 1:
#         return chosen[0]

#     return None


# def export_one_model_from_cli(model_name: str, args):
#     """
#     Export a single model, applying model-specific defaults and enforcing
#     required parameters.
#     """
#     kwargs = {}
#     Pmin = args.Pmin
#     Pmax = args.Pmax
#     n = args.n

#     if model_name == 'katz_2003':
#         if args.X_h2o is None:
#             raise SystemExit(
#                 'Error: --X-h2o is required when using Katz (2003).\n'
#                 'Example: python solidus_func.py --katz2003 --X-h2o 30'
#             )
#         kwargs['X_h2o'] = args.X_h2o

#     elif model_name == 'lin_2024':
#         if args.fO2 is None:
#             raise SystemExit(
#                 'Error: --fO2 is required when using Lin (2024).\n'
#                 'Example: python solidus_func.py --lin2024 --fO2 -4'
#             )
#         kwargs['fO2'] = args.fO2

#     elif model_name == 'hirschmann_2000':
#         if args.Pmax == 1000.0:
#             Pmax = 10.0

#     elif model_name == 'fei_2021':
#         if args.Pmin == 0.0:
#             Pmin = 1.0

#     elif model_name == 'stixrude_2014':
#         if args.Pmin == 0.0:
#             Pmin = 1.0

#     _ = export_model_curves(
#         model_name,
#         out_root=args.out_root,
#         Pmin=Pmin,
#         Pmax=Pmax,
#         n=n,
#         **kwargs,
#     )


# def main():
#     args = parse_args()

#     shortcut_model = resolve_requested_model(args)
#     explicit_model = args.model

#     if args.all:
#         if explicit_model is not None or shortcut_model is not None:
#             raise SystemExit(
#                 'Error: please use either --all or a single model selection, not both.'
#             )
#         export_all_models(out_root=args.out_root, n=args.n)
#         return

#     selected_models = [m for m in [explicit_model, shortcut_model] if m is not None]

#     if len(selected_models) == 0:
#         raise SystemExit(
#             'Error: no model selected. Use --all or choose one model with '
#             '--model or a shortcut like --katz2003.'
#         )

#     if len(selected_models) > 1:
#         raise SystemExit('Error: please choose only one of --model or one shortcut flag.')

#     export_one_model_from_cli(selected_models[0], args)


# if __name__ == '__main__':
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate melting curves in pressure–temperature (P–T) and pressure–entropy (P–S)
space for several literature parametrizations of peridotite / silicate melting.

This version is designed to work with the legacy EOS lookup tables:

    - temperature_solid.dat
    - temperature_melt.dat

These tables provide temperature as a function of entropy and pressure,
T(S, P), on structured grids. The script therefore:

1. Builds solidus and liquidus curves in P–T space from literature fits.
2. Converts those curves into P–S space by inverting the EOS relation T(S, P).
3. Resamples the solidus and liquidus entropy curves onto a common pressure grid.
4. Saves both the P–T and P–S versions to disk.

Usage
-----
Export one model with a dedicated shortcut:

    python solidus_func.py --katz2003 --X-h2o 30
    python solidus_func.py --lin2024 --fO2 -4

Export one model by explicit internal name:

    python solidus_func.py --model wolf_bower_2018

Export all supported models:

    python solidus_func.py --all

Notes
-----
- Pressure is handled internally in GPa for the melting-curve parametrizations,
  unless a given published fit is explicitly defined in Pa and converted.
- The EOS tables are assumed to have the SPIDER-style format with scaling factors
  in the header.
- The exported entropy files use the same scaled SPIDER-like header so they can
  be re-used by downstream tools expecting that format.

References included here
------------------------
- Andrault et al. (2011), DOI: 10.1016/j.epsl.2011.02.006
- Monteux et al. (2016), DOI: 10.1016/j.epsl.2016.05.010
- Lin et al. (2024), DOI: 10.1038/s41561-024-01495-1
- Hirschmann (2000), DOI: 10.1029/2000GC000070
- Katz et al. (2003), DOI: 10.1029/2002GC000433
- Stixrude (2014), DOI: 10.1098/rsta.2013.0076
- Fei et al. (2021), DOI: 10.1038/s41467-021-21170-y
- Belonoshko et al. (2005), DOI: 10.1103/PhysRevLett.94.195701
- Fiquet et al. (2010), DOI: 10.1126/science.1192448
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import platformdirs
from scipy.interpolate import RegularGridInterpolator, interp1d

# =============================================================================
# DEFAULT OUTPUT LOCATION
# =============================================================================

FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', platformdirs.user_data_dir('fwl_data')))
MELTING_DIR = FWL_DATA_DIR / 'interior_lookup_tables' / 'Melting_curves'


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def make_pressure_grid(Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500) -> np.ndarray:
    r"""
    Create a uniformly sampled pressure grid in GPa.

    Parameters
    ----------
    Pmin : float, optional
        Minimum pressure in GPa.
    Pmax : float, optional
        Maximum pressure in GPa.
    n : int, optional
        Number of grid points.

    Returns
    -------
    np.ndarray
        One-dimensional pressure array in GPa.
    """
    return np.linspace(Pmin, Pmax, n)


def solidus_from_liquidus_stixrude(T_liq: np.ndarray) -> np.ndarray:
    r"""
    Estimate the solidus from a liquidus using the Stixrude ratio.
    """
    return T_liq / (1.0 - np.log(0.79))


def liquidus_from_solidus_stixrude(T_sol: np.ndarray) -> np.ndarray:
    r"""
    Estimate the liquidus from a solidus using the inverse Stixrude ratio.
    """
    return T_sol * (1.0 - np.log(0.79))


def _fmt_range(arr: np.ndarray, fmt: str = '.3f') -> str:
    """
    Format the finite minimum and maximum of an array as a string.
    """
    arr = np.asarray(arr, dtype=float)
    mask = np.isfinite(arr)

    if not np.any(mask):
        return '[nan, nan]'

    amin = np.min(arr[mask])
    amax = np.max(arr[mask])
    return f'[{amin:{fmt}}, {amax:{fmt}}]'


DISPLAY_NAMES = {
    'andrault_2011': 'Andrault et al. (2011)',
    'monteux_2016': 'Monteux et al. (2016)',
    'wolf_bower_2018': 'Wolf & Bower (2018)',
    'katz_2003': 'Katz et al. (2003)',
    'fei_2021': 'Fei et al. (2021)',
    'belonoshko_2005': 'Belonoshko et al. (2005)',
    'fiquet_2010': 'Fiquet et al. (2010)',
    'hirschmann_2000': 'Hirschmann (2000)',
    'stixrude_2014': 'Stixrude (2014)',
    'lin_2024': 'Lin et al. (2024)',
}


def print_model_summary(
    model_name: str,
    P_sol: np.ndarray,
    T_sol: np.ndarray,
    P_liq: np.ndarray,
    T_liq: np.ndarray,
    P_common: np.ndarray,
    S_sol_common: np.ndarray,
    S_liq_common: np.ndarray,
):
    """
    Print a compact summary of the exported P-T and P-S ranges for one model.
    """
    label = DISPLAY_NAMES.get(model_name, model_name)
    print(f'{label}:')
    print(
        f'  P-T solidus   : P_range = {_fmt_range(P_sol)} GPa, T_range = {_fmt_range(T_sol)} K'
    )
    print(
        f'  P-T liquidus  : P_range = {_fmt_range(P_liq)} GPa, T_range = {_fmt_range(T_liq)} K'
    )
    print(
        f'  P-S solidus   : P_range = {_fmt_range(P_common)} GPa, S_range = {_fmt_range(S_sol_common)} J kg^-1 K^-1'
    )
    print(
        f'  P-S liquidus  : P_range = {_fmt_range(P_common)} GPa, S_range = {_fmt_range(S_liq_common)} J kg^-1 K^-1'
    )
    print()


# =============================================================================
# LITERATURE MELTING CURVES
# =============================================================================


def andrault_2011(kind: str = 'solidus', Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500):
    r"""
    Melting curve from Andrault et al. (2011).
    """
    P = make_pressure_grid(Pmin, Pmax, n)
    P_pa = P * 1e9

    if kind == 'solidus':
        T0, a, c = 2045, 92e9, 1.3
    elif kind == 'liquidus':
        T0, a, c = 1940, 29e9, 1.9
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    T = T0 * ((P_pa / a) + 1.0) ** (1.0 / c)
    return P, T


def fei_2021(kind: str = 'liquidus', Pmin: float = 1.0, Pmax: float = 1000.0, n: int = 500):
    r"""
    Melting curve based on Fei et al. (2021).
    """
    P = make_pressure_grid(Pmin, Pmax, n)
    T_liq = 6000.0 * (P / 140.0) ** 0.26

    if kind == 'liquidus':
        T = T_liq
    elif kind == 'solidus':
        T = solidus_from_liquidus_stixrude(T_liq)
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return P, T


def belonoshko_2005(
    kind: str = 'liquidus', Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500
):
    r"""
    Melting curve based on Belonoshko et al. (2005).
    """
    P = make_pressure_grid(Pmin, Pmax, n)
    T_liq = 1831.0 * (1.0 + P / 4.6) ** 0.33

    if kind == 'liquidus':
        T = T_liq
    elif kind == 'solidus':
        T = solidus_from_liquidus_stixrude(T_liq)
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return P, T


def fiquet_2010(kind: str = 'liquidus', Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500):
    r"""
    Melting curve based on Fiquet et al. (2010).
    """
    P = make_pressure_grid(Pmin, Pmax, n)
    T_liq = np.zeros_like(P, dtype=float)

    low = P <= 20.0
    high = P > 20.0

    # Original high-pressure constant is given in Pa in the paper.
    # Here pressure is in GPa, so 4.054e6 Pa -> 0.004056 GPa.
    T_liq[low] = 1982.1 * ((P[low] / 6.594) + 1.0) ** (1.0 / 5.374)
    T_liq[high] = 78.74 * ((P[high] / 0.004056) + 1.0) ** (1.0 / 2.44)

    if kind == 'liquidus':
        T = T_liq
    elif kind == 'solidus':
        T = solidus_from_liquidus_stixrude(T_liq)
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return P, T


def monteux_2016(kind: str = 'solidus', Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500):
    r"""
    Melting curve from Monteux et al. (2016).
    """
    P = make_pressure_grid(Pmin, Pmax, n)
    P_pa = P * 1e9

    params = {
        'solidus': {
            'low': {'T0': 1661.2, 'a': 1.336e9, 'c': 7.437},
            'high': {'T0': 2081.8, 'a': 101.69e9, 'c': 1.226},
        },
        'liquidus': {
            'low': {'T0': 1982.1, 'a': 6.594e9, 'c': 5.374},
            'high': {'T0': 2006.8, 'a': 34.65e9, 'c': 1.844},
        },
    }

    if kind not in params:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    p = params[kind]
    T = np.zeros_like(P_pa, dtype=float)

    mask_low = P_pa <= 20e9
    mask_high = P_pa > 20e9

    T[mask_low] = p['low']['T0'] * ((P_pa[mask_low] / p['low']['a']) + 1.0) ** (
        1.0 / p['low']['c']
    )
    T[mask_high] = p['high']['T0'] * ((P_pa[mask_high] / p['high']['a']) + 1.0) ** (
        1.0 / p['high']['c']
    )

    return P, T


def hirschmann_2000(kind: str = 'solidus', Pmin: float = 0.0, Pmax: float = 10.0, n: int = 500):
    r"""
    Melting curve from Hirschmann (2000).
    """
    P = make_pressure_grid(Pmin, Pmax, n)

    coeffs = {
        'solidus': (1085.7, 132.9, -5.1),
        'liquidus': (1475.0, 80.0, -3.2),
    }

    if kind not in coeffs:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    A1, A2, A3 = coeffs[kind]
    T_c = A1 + A2 * P + A3 * P**2
    T = T_c + 273.15

    return P, T


def stixrude_2014(
    kind: str = 'liquidus', Pmin: float = 1.0, Pmax: float = 1000.0, n: int = 500
):
    r"""
    Melting curve based on Stixrude (2014).
    """
    P = make_pressure_grid(Pmin, Pmax, n)
    T_liq = 5400.0 * (P / 140.0) ** 0.480

    if kind == 'liquidus':
        T = T_liq
    elif kind == 'solidus':
        T = solidus_from_liquidus_stixrude(T_liq)
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return P, T


def wolf_bower_2018(
    kind: str = 'solidus', Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500
):
    r"""
    Piecewise melting curve based on Wolf & Bower (2018) style fits.
    """
    P = make_pressure_grid(Pmin, Pmax, n)

    params = {
        'solidus': (
            7.696777581585296,
            870.4767697319186,
            101.52655163737373,
            15.959022187236807,
            3.090844734784906,
            1417.4258954709148,
        ),
        'liquidus': (
            8.864665249317456,
            408.58442302949794,
            46.288444869816615,
            17.549174419770257,
            3.679647802112376,
            2019.967799687511,
        ),
    }

    if kind not in params:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    cp1, cp2, s1, s2, s3, intercept = params[kind]

    c1 = intercept
    c2 = c1 + (s1 - s2) * cp1
    c3 = c2 + (s2 - s3) * cp2

    T = np.zeros_like(P, dtype=float)

    m1 = P < cp1
    m2 = (P >= cp1) & (P < cp2)
    m3 = P >= cp2

    T[m1] = s1 * P[m1] + c1
    T[m2] = s2 * P[m2] + c2
    T[m3] = s3 * P[m3] + c3

    return P, T


def katz_2003(
    kind: str = 'solidus',
    X_h2o: float = 30.0,
    Pmin: float = 0.0,
    Pmax: float = 1000.0,
    n: int = 500,
):
    r"""
    Hydrous melting-curve correction following Katz et al. (2003).
    """
    gamma = 0.75
    K = 43.0

    if kind not in {'solidus', 'liquidus'}:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    P, T_anhydrous = wolf_bower_2018(kind=kind, Pmin=Pmin, Pmax=Pmax, n=n)
    delta_T = K * X_h2o**gamma
    T = T_anhydrous - delta_T

    return P, T


def lin_2024(
    kind: str = 'solidus',
    fO2: float = -4.0,
    Pmin: float = 0.0,
    Pmax: float = 1000.0,
    n: int = 500,
):
    r"""
    Oxygen-fugacity-dependent solidus following Lin et al. (2024).
    """
    P, T_anhydrous = wolf_bower_2018(kind='solidus', Pmin=Pmin, Pmax=Pmax, n=n)

    delta_T = (340.0 / 3.2) * (2.0 - fO2)
    T_sol = T_anhydrous + delta_T

    if kind == 'solidus':
        T = T_sol
    elif kind == 'liquidus':
        T = liquidus_from_solidus_stixrude(T_sol)
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return P, T


# =============================================================================
# PHYSICAL-INTERVAL FILTER
# =============================================================================


def truncate_to_physical_interval(func):
    r"""
    Wrap a melting-curve function so only the main interval with
    T_sol < T_liq is retained.
    """

    def wrapped(kind='solidus', Pmin=0.0, Pmax=1000.0, n=2000, **kwargs):
        P_sol, T_sol = func(kind='solidus', Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)
        P_liq, T_liq = func(kind='liquidus', Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)

        good = T_sol < T_liq
        idx = np.where(good)[0]

        if len(idx) == 0:
            raise ValueError(f'{func.__name__}: no physical interval where solidus < liquidus')

        splits = np.where(np.diff(idx) > 1)[0] + 1
        blocks = np.split(idx, splits)
        main_block = max(blocks, key=len)

        if kind == 'solidus':
            return P_sol[main_block], T_sol[main_block]
        if kind == 'liquidus':
            return P_liq[main_block], T_liq[main_block]
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return wrapped


andrault_2011_cut = truncate_to_physical_interval(andrault_2011)
monteux_2016_cut = truncate_to_physical_interval(monteux_2016)
wolf_bower_2018_cut = truncate_to_physical_interval(wolf_bower_2018)
katz_2003_cut = truncate_to_physical_interval(katz_2003)


# =============================================================================
# MODEL DISPATCHER
# =============================================================================

SUPPORTED_MODELS = [
    'andrault_2011',
    'monteux_2016',
    'wolf_bower_2018',
    'katz_2003',
    'fei_2021',
    'belonoshko_2005',
    'fiquet_2010',
    'hirschmann_2000',
    'stixrude_2014',
    'lin_2024',
]


def get_melting_curves(
    model_name: str, Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 2000, **kwargs
):
    r"""
    Return solidus and liquidus curves for a given model.
    """
    models = {
        'andrault_2011': andrault_2011_cut,
        'monteux_2016': monteux_2016_cut,
        'wolf_bower_2018': wolf_bower_2018_cut,
        'katz_2003': katz_2003_cut,
        'fei_2021': fei_2021,
        'belonoshko_2005': belonoshko_2005,
        'fiquet_2010': fiquet_2010,
        'hirschmann_2000': hirschmann_2000,
        'stixrude_2014': stixrude_2014,
        'lin_2024': lin_2024,
    }

    if model_name not in models:
        raise ValueError(f'Unknown model: {model_name}')

    func = models[model_name]
    P_sol, T_sol = func(kind='solidus', Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)
    P_liq, T_liq = func(kind='liquidus', Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)

    return P_sol, T_sol, P_liq, T_liq


# =============================================================================
# OUTPUT HELPERS
# =============================================================================


def save_PT_table(path: Path, P_gpa: np.ndarray, T_k: np.ndarray):
    r"""
    Save a pressure-temperature table to disk.
    """
    data = np.column_stack([P_gpa, T_k])
    np.savetxt(path, data, fmt='%.18e %.18e', header='pressure temperature', comments='#')


# =============================================================================
# EOS LOOKUP TABLE SETTINGS
# =============================================================================

nP = 2020
nS_solid = 125
nS_liquid = 95
skip_header = 5

SCALE_P_EOS = 1e9
SCALE_T_EOS = 1.0
SCALE_S_SOLID_EOS = 4.82426684604467e6
SCALE_S_LIQUID_EOS = 4.805046659407042e6

SCALE_P_OUT = 1_000_000_000.0
SCALE_S_OUT = 4_824_266.84604467

def make_entropy_header(n_rows: int) -> str:
    """
    Build the SPIDER-style header for a pressure-entropy table.

    Parameters
    ----------
    n_rows : int
        Number of data rows in the file.

    Returns
    -------
    str
        Multiline header string.
    """
    return '\n'.join(
        [
            f'# 5 {n_rows}',
            '# Pressure, Entropy',
            '# column * scaling factor should be SI units',
            '# scaling factors (constant) for each column given on line below',
            '# 1000000000.0 4824266.84604467',
        ]
    )

def get_default_spider_dir() -> Path:
    """
    Return the default SPIDER directory relative to this module.
    """
    script_dir = Path(__file__).resolve().parent
    return (script_dir / '../../../SPIDER').resolve()

def validate_entropy_export_arrays(
    P_common: np.ndarray,
    S_sol_common: np.ndarray,
    S_liq_common: np.ndarray,
    model_name: str,
):
    """
    Validate the common P-S arrays before writing them to disk.

    Parameters
    ----------
    P_common : np.ndarray
        Common pressure grid in GPa.
    S_sol_common : np.ndarray
        Solidus entropy values on the common pressure grid.
    S_liq_common : np.ndarray
        Liquidus entropy values on the common pressure grid.
    model_name : str
        Name of the melting model, used in error messages.

    Raises
    ------
    ValueError
        If the entropy export arrays are empty, mismatched, or non-finite.
    """
    if len(P_common) == 0 or len(S_sol_common) == 0 or len(S_liq_common) == 0:
        raise ValueError(
            f'{model_name}: could not build a valid common P-S grid. '
            'EOS inversion may have failed or the solidus/liquidus entropy '
            'ranges may not overlap.'
        )

    if not (len(P_common) == len(S_sol_common) == len(S_liq_common)):
        raise ValueError(
            f'{model_name}: inconsistent P-S array lengths: '
            f'len(P_common)={len(P_common)}, '
            f'len(S_sol_common)={len(S_sol_common)}, '
            f'len(S_liq_common)={len(S_liq_common)}'
        )

    if not (
        np.all(np.isfinite(P_common))
        and np.all(np.isfinite(S_sol_common))
        and np.all(np.isfinite(S_liq_common))
    ):
        raise ValueError(
            f'{model_name}: non-finite values found in common P-S export arrays.'
        )

def resolve_eos_paths(spider_dir: Path | str | None = None) -> tuple[Path, Path]:
    """
    Resolve and validate the default solid and liquid EOS lookup table paths.

    Parameters
    ----------
    spider_dir : Path | str | None, optional
        Root directory of the SPIDER repository. If None, a path relative to
        this module is used.

    Returns
    -------
    tuple[Path, Path]
        Paths to the solid and liquid EOS tables.

    Raises
    ------
    FileNotFoundError
        If one or both EOS files are missing.
    """
    if spider_dir is None:
        spider_dir = get_default_spider_dir()

    spider_dir = Path(spider_dir).resolve()

    eos_solid_path = (
        spider_dir / 'lookup_data' / '1TPa-dK09-elec-free' / 'temperature_solid.dat'
    )
    eos_liquid_path = (
        spider_dir / 'lookup_data' / '1TPa-dK09-elec-free' / 'temperature_melt.dat'
    )

    if not eos_solid_path.exists():
        raise FileNotFoundError(f'Missing EOS file: {eos_solid_path}')

    if not eos_liquid_path.exists():
        raise FileNotFoundError(f'Missing EOS file: {eos_liquid_path}')

    return eos_solid_path, eos_liquid_path


def load_eos_T_of_SP(eos_path: Path, nS: int, scale_S_axis: float):
    r"""
    Load an EOS lookup table and build an interpolator for T(S, P).
    """
    raw = np.genfromtxt(eos_path, skip_header=skip_header)
    resh = raw.reshape((nS, nP, 3))

    P_axis_GPa = resh[0, :, 0] * SCALE_P_EOS / 1e9
    S_axis = resh[:, 0, 1] * scale_S_axis
    T_grid = resh[:, :, 2] * SCALE_T_EOS

    T_interp = RegularGridInterpolator(
        (S_axis, P_axis_GPa),
        T_grid,
        bounds_error=False,
        fill_value=np.nan,
    )
    return S_axis, P_axis_GPa, T_interp


def load_default_eos_interpolators(
    spider_dir: Path | str | None = None,
) -> tuple[np.ndarray, RegularGridInterpolator, np.ndarray, RegularGridInterpolator]:
    """
    Load the default solid and liquid EOS interpolators.

    Parameters
    ----------
    spider_dir : Path | str | None, optional
        Root directory of the SPIDER repository. If None, a path relative to
        this module is used.

    Returns
    -------
    tuple
        A tuple containing:
        - solid entropy axis
        - solid T(S, P) interpolator
        - liquid entropy axis
        - liquid T(S, P) interpolator
    """
    eos_solid_path, eos_liquid_path = resolve_eos_paths(spider_dir=spider_dir)

    S_axis_solid, _, T_of_SP_solid = load_eos_T_of_SP(
        eos_solid_path, nS_solid, SCALE_S_SOLID_EOS
    )
    S_axis_liquid, _, T_of_SP_liquid = load_eos_T_of_SP(
        eos_liquid_path, nS_liquid, SCALE_S_LIQUID_EOS
    )

    return S_axis_solid, T_of_SP_solid, S_axis_liquid, T_of_SP_liquid


def invert_to_entropy_along_profile(
    P_gpa: np.ndarray, T_k: np.ndarray, S_axis: np.ndarray, T_of_SP
):
    r"""
    Convert a P-T curve into a P-S curve by inverting T(S, P).
    """
    S_out = np.full_like(T_k, np.nan, dtype=float)

    for i, (P_i_gpa, T_i) in enumerate(zip(P_gpa, T_k)):
        P_col = np.full_like(S_axis, P_i_gpa)
        T_vals = T_of_SP(np.column_stack([S_axis, P_col]))

        valid = np.isfinite(T_vals)
        if np.count_nonzero(valid) < 2:
            continue

        Tv = T_vals[valid]
        Sv = S_axis[valid]

        order = np.argsort(Tv)
        T_sorted = Tv[order]
        S_sorted = Sv[order]

        T_unique, idx_unique = np.unique(T_sorted, return_index=True)
        S_unique = S_sorted[idx_unique]

        if len(T_unique) < 2:
            continue

        if T_i < T_unique[0] or T_i > T_unique[-1]:
            continue

        try:
            f = interp1d(T_unique, S_unique, kind='linear', assume_sorted=True)
            S_out[i] = float(f(T_i))
        except Exception:
            try:
                f = interp1d(T_unique, S_unique, kind='nearest', assume_sorted=True)
                S_out[i] = float(f(T_i))
            except Exception:
                continue

    return S_out


def build_common_entropy_grid(
    P_sol: np.ndarray,
    S_sol: np.ndarray,
    P_liq: np.ndarray,
    S_liq: np.ndarray,
    n_common: int | None = None,
):
    r"""
    Resample solidus and liquidus entropy curves onto a shared pressure grid.
    """
    mask_sol = np.isfinite(S_sol)
    mask_liq = np.isfinite(S_liq)

    P_sol_v = np.asarray(P_sol[mask_sol], dtype=float)
    S_sol_v = np.asarray(S_sol[mask_sol], dtype=float)
    P_liq_v = np.asarray(P_liq[mask_liq], dtype=float)
    S_liq_v = np.asarray(S_liq[mask_liq], dtype=float)

    if len(P_sol_v) < 2 or len(P_liq_v) < 2:
        return np.array([]), np.array([]), np.array([])

    Pmin_common = max(np.min(P_sol_v), np.min(P_liq_v))
    Pmax_common = min(np.max(P_sol_v), np.max(P_liq_v))

    if (
        not np.isfinite(Pmin_common)
        or not np.isfinite(Pmax_common)
        or Pmax_common <= Pmin_common
    ):
        return np.array([]), np.array([]), np.array([])

    if n_common is None:
        n_common = min(len(P_sol_v), len(P_liq_v))

    order_sol = np.argsort(P_sol_v)
    order_liq = np.argsort(P_liq_v)

    P_sol_s = P_sol_v[order_sol]
    S_sol_s = S_sol_v[order_sol]
    P_liq_s = P_liq_v[order_liq]
    S_liq_s = S_liq_v[order_liq]

    P_sol_u, idx_sol = np.unique(P_sol_s, return_index=True)
    S_sol_u = S_sol_s[idx_sol]

    P_liq_u, idx_liq = np.unique(P_liq_s, return_index=True)
    S_liq_u = S_liq_s[idx_liq]

    if len(P_sol_u) < 2 or len(P_liq_u) < 2:
        return np.array([]), np.array([]), np.array([])

    P_common = np.linspace(Pmin_common, Pmax_common, n_common)

    f_sol = interp1d(
        P_sol_u,
        S_sol_u,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    f_liq = interp1d(
        P_liq_u,
        S_liq_u,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )

    S_sol_common = f_sol(P_common)
    S_liq_common = f_liq(P_common)

    mask = np.isfinite(S_sol_common) & np.isfinite(S_liq_common)
    return P_common[mask], S_sol_common[mask], S_liq_common[mask]


def save_entropy_table_with_header(path: Path, P_gpa: np.ndarray, S_jpk: np.ndarray):
    r"""
    Save a pressure-entropy table in SPIDER-style scaled format.
    """
    P_pa = P_gpa * 1e9
    data = np.column_stack([P_pa / SCALE_P_OUT, S_jpk / SCALE_S_OUT])
    header = make_entropy_header(len(P_gpa))
    np.savetxt(path, data, fmt='%.18e %.18e', header=header, comments='')

# =============================================================================
# MAIN EXPORTER
# =============================================================================


def export_model_curves(
    model_name: str,
    out_root: Path | str = MELTING_DIR,
    Pmin: float = 0.0,
    Pmax: float = 1000.0,
    n: int = 2000,
    spider_dir: Path | str | None = None,
    **kwargs,
):
    r"""
    Export one melting model in both P-T and P-S space.
    """
    out_dir = Path(out_root) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    P_sol, T_sol, P_liq, T_liq = get_melting_curves(
        model_name, Pmin=Pmin, Pmax=Pmax, n=n, **kwargs
    )

    save_PT_table(out_dir / 'solidus_P-T.dat', P_sol, T_sol)
    save_PT_table(out_dir / 'liquidus_P-T.dat', P_liq, T_liq)

    S_axis_solid, T_of_SP_solid, S_axis_liquid, T_of_SP_liquid = (
        load_default_eos_interpolators(spider_dir=spider_dir)
    )

    S_sol = invert_to_entropy_along_profile(P_sol, T_sol, S_axis_solid, T_of_SP_solid)
    S_liq = invert_to_entropy_along_profile(P_liq, T_liq, S_axis_liquid, T_of_SP_liquid)

    P_common, S_sol_common, S_liq_common = build_common_entropy_grid(
        P_sol, S_sol, P_liq, S_liq, n_common=n
    )

    validate_entropy_export_arrays(
        P_common,
        S_sol_common,
        S_liq_common,
        model_name=model_name,
    )

    save_entropy_table_with_header(
        out_dir / 'solidus_P-S.dat',
        P_common,
        S_sol_common,
    )

    save_entropy_table_with_header(
        out_dir / 'liquidus_P-S.dat',
        P_common,
        S_liq_common,
    )

    print_model_summary(
        model_name,
        P_sol,
        T_sol,
        P_liq,
        T_liq,
        P_common,
        S_sol_common,
        S_liq_common,
    )

    print(f'  Saved to      : {out_dir.resolve()}')
    print()

    return {
        'P_sol': P_sol,
        'T_sol': T_sol,
        'P_liq': P_liq,
        'T_liq': T_liq,
        'S_sol': S_sol,
        'S_liq': S_liq,
        'P_entropy_common': P_common,
        'S_sol_common': S_sol_common,
        'S_liq_common': S_liq_common,
    }


# =============================================================================
# BATCH EXPORTER
# =============================================================================


def export_all_models(
    out_root: Path | str = MELTING_DIR,
    n: int = 2000,
    spider_dir: Path | str | None = None,
):
    r"""
    Export all supported melting parametrizations.
    """
    for model in SUPPORTED_MODELS:
        if model == 'katz_2003':
            _ = export_model_curves(
                model, out_root=out_root, n=n, X_h2o=30.0, spider_dir=spider_dir
            )
        elif model == 'lin_2024':
            _ = export_model_curves(
                model, out_root=out_root, n=n, fO2=-4.0, spider_dir=spider_dir
            )
        elif model == 'hirschmann_2000':
            _ = export_model_curves(
                model, out_root=out_root, n=n, Pmax=10.0, spider_dir=spider_dir
            )
        elif model == 'fei_2021':
            _ = export_model_curves(
                model, out_root=out_root, n=n, Pmin=1.0, spider_dir=spider_dir
            )
        elif model == 'stixrude_2014':
            _ = export_model_curves(
                model, out_root=out_root, n=n, Pmin=1.0, spider_dir=spider_dir
            )
        else:
            _ = export_model_curves(model, out_root=out_root, n=n, spider_dir=spider_dir)


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Export solidus and liquidus melting curves in P-T and P-S space '
            'for one or more literature parametrizations.'
        ),
        epilog=(
            'Examples:\n'
            '  python solidus_func.py --all\n'
            '  python solidus_func.py --katz2003 --X-h2o 30\n'
            '  python solidus_func.py --lin2024 --fO2 -4\n'
            '  python solidus_func.py --model wolf_bower_2018\n'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Export all supported models.',
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=SUPPORTED_MODELS,
        help='Export a single model by internal name.',
    )

    parser.add_argument(
        '--katz2003',
        action='store_true',
        help='Export Katz et al. (2003). Requires --X-h2o.',
    )
    parser.add_argument(
        '--lin2024',
        action='store_true',
        help='Export Lin et al. (2024). Requires --fO2.',
    )
    parser.add_argument(
        '--wolfbower2018',
        action='store_true',
        help='Export Wolf & Bower (2018).',
    )
    parser.add_argument(
        '--andrault2011',
        action='store_true',
        help='Export Andrault et al. (2011).',
    )
    parser.add_argument(
        '--monteux2016',
        action='store_true',
        help='Export Monteux et al. (2016).',
    )
    parser.add_argument(
        '--fei2021',
        action='store_true',
        help='Export Fei et al. (2021).',
    )
    parser.add_argument(
        '--belonoshko2005',
        action='store_true',
        help='Export Belonoshko et al. (2005).',
    )
    parser.add_argument(
        '--fiquet2010',
        action='store_true',
        help='Export Fiquet et al. (2010).',
    )
    parser.add_argument(
        '--hirschmann2000',
        action='store_true',
        help='Export Hirschmann (2000).',
    )
    parser.add_argument(
        '--stixrude2014',
        action='store_true',
        help='Export Stixrude (2014).',
    )

    parser.add_argument(
        '--out-root',
        type=str,
        default=str(MELTING_DIR),
        help='Root directory where output folders will be created.',
    )

    parser.add_argument(
        '--spider-dir',
        type=str,
        default=None,
        help='Path to the SPIDER root directory containing lookup_data/.',
    )

    parser.add_argument(
        '--Pmin',
        type=float,
        default=0.0,
        help='Minimum pressure in GPa.',
    )
    parser.add_argument(
        '--Pmax',
        type=float,
        default=1000.0,
        help='Maximum pressure in GPa.',
    )
    parser.add_argument(
        '-n',
        type=int,
        default=2000,
        help='Number of pressure samples.',
    )

    parser.add_argument(
        '--X-h2o',
        dest='X_h2o',
        type=float,
        default=None,
        help='Water content parameter for Katz (2003). Required for --katz2003.',
    )
    parser.add_argument(
        '--fO2',
        type=float,
        default=None,
        help='Oxygen fugacity offset for Lin (2024). Required for --lin2024.',
    )

    return parser.parse_args()


def resolve_requested_model(args) -> str | None:
    """
    Resolve which single-model shortcut flag was requested.
    """
    shortcut_map = {
        'katz2003': 'katz_2003',
        'lin2024': 'lin_2024',
        'wolfbower2018': 'wolf_bower_2018',
        'andrault2011': 'andrault_2011',
        'monteux2016': 'monteux_2016',
        'fei2021': 'fei_2021',
        'belonoshko2005': 'belonoshko_2005',
        'fiquet2010': 'fiquet_2010',
        'hirschmann2000': 'hirschmann_2000',
        'stixrude2014': 'stixrude_2014',
    }

    chosen = [model for flag, model in shortcut_map.items() if getattr(args, flag)]

    if len(chosen) > 1:
        raise SystemExit('Error: please select only one model shortcut flag at a time.')

    if len(chosen) == 1:
        return chosen[0]

    return None


def export_one_model_from_cli(model_name: str, args):
    """
    Export a single model, applying model-specific defaults and enforcing
    required parameters.
    """
    kwargs = {}
    Pmin = args.Pmin
    Pmax = args.Pmax
    n = args.n

    if model_name == 'katz_2003':
        if args.X_h2o is None:
            raise SystemExit(
                'Error: --X-h2o is required when using Katz (2003).\n'
                'Example: python solidus_func.py --katz2003 --X-h2o 30'
            )
        kwargs['X_h2o'] = args.X_h2o

    elif model_name == 'lin_2024':
        if args.fO2 is None:
            raise SystemExit(
                'Error: --fO2 is required when using Lin (2024).\n'
                'Example: python solidus_func.py --lin2024 --fO2 -4'
            )
        kwargs['fO2'] = args.fO2

    elif model_name == 'hirschmann_2000':
        if args.Pmax == 1000.0:
            Pmax = 10.0

    elif model_name == 'fei_2021':
        if args.Pmin == 0.0:
            Pmin = 1.0

    elif model_name == 'stixrude_2014':
        if args.Pmin == 0.0:
            Pmin = 1.0

    _ = export_model_curves(
        model_name,
        out_root=args.out_root,
        Pmin=Pmin,
        Pmax=Pmax,
        n=n,
        spider_dir=args.spider_dir,
        **kwargs,
    )


def main():
    args = parse_args()

    shortcut_model = resolve_requested_model(args)
    explicit_model = args.model

    if args.all:
        if explicit_model is not None or shortcut_model is not None:
            raise SystemExit(
                'Error: please use either --all or a single model selection, not both.'
            )
        try:
            export_all_models(out_root=args.out_root, n=args.n, spider_dir=args.spider_dir)
        except FileNotFoundError as exc:
            raise SystemExit(f'Error: {exc}') from exc
        return

    selected_models = [m for m in [explicit_model, shortcut_model] if m is not None]

    if len(selected_models) == 0:
        raise SystemExit(
            'Error: no model selected. Use --all or choose one model with '
            '--model or a shortcut like --katz2003.'
        )

    if len(selected_models) > 1:
        raise SystemExit('Error: please choose only one of --model or one shortcut flag.')

    try:
        export_one_model_from_cli(selected_models[0], args)
    except FileNotFoundError as exc:
        raise SystemExit(f'Error: {exc}') from exc


if __name__ == '__main__':
    main()
