from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d


# =========================================================
# Common pressure grid helper
# =========================================================
def make_pressure_grid(Pmin=0.0, Pmax=1000.0, n=500):
    """
    Create a pressure grid in GPa.
    """
    return np.linspace(Pmin, Pmax, n)


# =========================================================
# Generic helpers
# =========================================================
def solidus_from_liquidus_stixrude(T_liq):
    """
    Approximate solidus from liquidus using the Stixrude ratio.
    """
    return T_liq / (1.0 - np.log(0.79))


def liquidus_from_solidus_stixrude(T_sol):
    """
    Approximate liquidus from solidus using the inverse Stixrude ratio.
    """
    return T_sol * (1.0 - np.log(0.79))


# =========================================================
# Andrault et al. (2011)
# Formula expects P in Pa in the coefficients below
# =========================================================
def andrault_2011(kind="solidus", Pmin=0.0, Pmax=1000.0, n=500):
    P = make_pressure_grid(Pmin, Pmax, n)   # GPa
    P_pa = P * 1e9                          # Pa

    if kind == "solidus":
        T0, a, c = 2045, 92e9, 1.3
    elif kind == "liquidus":
        T0, a, c = 1940, 29e9, 1.9
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    T = T0 * ((P_pa / a) + 1.0) ** (1.0 / c)
    return P, T


# =========================================================
# Fei et al. (2021)
# Expects P in GPa
# =========================================================
def fei_2021(kind="liquidus", Pmin=1.0, Pmax=1000.0, n=500):
    P = make_pressure_grid(Pmin, Pmax, n)
    T_liq = 6000.0 * (P / 140.0) ** 0.26

    if kind == "liquidus":
        T = T_liq
    elif kind == "solidus":
        T = solidus_from_liquidus_stixrude(T_liq)
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return P, T


# =========================================================
# Belonoshko et al. (2005)
# Expects P in GPa
# =========================================================
def belonoshko_2005(kind="liquidus", Pmin=0.0, Pmax=1000.0, n=500):
    P = make_pressure_grid(Pmin, Pmax, n)
    T_liq = 1831.0 * (1.0 + P / 4.6) ** 0.33

    if kind == "liquidus":
        T = T_liq
    elif kind == "solidus":
        T = solidus_from_liquidus_stixrude(T_liq)
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return P, T


# =========================================================
# Fiquet et al. (2010)
# Expects P in GPa
# =========================================================
def fiquet_2010(kind="liquidus", Pmin=0.0, Pmax=1000.0, n=500):
    P = make_pressure_grid(Pmin, Pmax, n)
    T_liq = np.zeros_like(P, dtype=float)

    low = P <= 20.0
    high = P > 20.0

    T_liq[low] = 1982.1 * ((P[low] / 6.594) + 1.0) ** (1.0 / 5.374)
    T_liq[high] = 78.74 * ((P[high] / 0.004056) + 1.0) ** (1.0 / 2.44)

    if kind == "liquidus":
        T = T_liq
    elif kind == "solidus":
        T = solidus_from_liquidus_stixrude(T_liq)
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return P, T


# =========================================================
# Monteux et al. (2016)
# Original coefficients are in Pa
# =========================================================
def monteux_2016(kind="solidus", Pmin=0.0, Pmax=1000.0, n=500):
    P = make_pressure_grid(Pmin, Pmax, n)   # GPa
    P_pa = P * 1e9                          # Pa

    params = {
        "solidus": {
            "low":  {"T0": 1661.2, "a": 1.336e9,  "c": 7.437},
            "high": {"T0": 2081.8, "a": 101.69e9, "c": 1.226},
        },
        "liquidus": {
            "low":  {"T0": 1982.1, "a": 6.594e9,  "c": 5.374},
            "high": {"T0": 2006.8, "a": 34.65e9,  "c": 1.844},
        }
    }

    if kind not in params:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    p = params[kind]
    T = np.zeros_like(P_pa, dtype=float)

    mask_low = P_pa <= 20e9
    mask_high = P_pa > 20e9

    T[mask_low] = p["low"]["T0"] * ((P_pa[mask_low] / p["low"]["a"]) + 1.0) ** (1.0 / p["low"]["c"])
    T[mask_high] = p["high"]["T0"] * ((P_pa[mask_high] / p["high"]["a"]) + 1.0) ** (1.0 / p["high"]["c"])

    return P, T


# =========================================================
# Hirschmann (2000)
# Valid only at relatively low pressure
# =========================================================
def hirschmann_2000(kind="solidus", Pmin=0.0, Pmax=10.0, n=500):
    P = make_pressure_grid(Pmin, Pmax, n)

    coeffs = {
        "solidus":  (1085.7, 132.9, -5.1),
        "liquidus": (1475.0, 80.0, -3.2),
    }

    if kind not in coeffs:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    A1, A2, A3 = coeffs[kind]
    T_c = A1 + A2 * P + A3 * P**2
    T = T_c + 273.15

    return P, T


# =========================================================
# Stixrude (2014)
# Expects P in GPa
# =========================================================
def stixrude_2014(kind="liquidus", Pmin=1.0, Pmax=1000.0, n=500):
    P = make_pressure_grid(Pmin, Pmax, n)
    T_liq = 5400.0 * (P / 140.0) ** 0.480

    if kind == "liquidus":
        T = T_liq
    elif kind == "solidus":
        T = solidus_from_liquidus_stixrude(T_liq)
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return P, T


# =========================================================
# Wolf & Bower (2018) / piecewise fits
# Expects P in GPa
# =========================================================
def wolf_bower_2018(kind="solidus", Pmin=0.0, Pmax=1000.0, n=500):
    P = make_pressure_grid(Pmin, Pmax, n)

    params = {
        "solidus": (
            7.696777581585296,
            870.4767697319186,
            101.52655163737373,
            15.959022187236807,
            3.090844734784906,
            1417.4258954709148
        ),
        "liquidus": (
            8.864665249317456,
            408.58442302949794,
            46.288444869816615,
            17.549174419770257,
            3.679647802112376,
            2019.967799687511
        )
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


# =========================================================
# Katz (2003)
# Applies the same hydrous depression to both curves
# =========================================================
def katz_2003(kind="solidus", X_h2o=30.0, Pmin=0.0, Pmax=1000.0, n=500):
    gamma = 0.75
    K = 43.0

    if kind not in {"solidus", "liquidus"}:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    P, T_anhydrous = wolf_bower_2018(kind=kind, Pmin=Pmin, Pmax=Pmax, n=n)
    delta_T = K * X_h2o ** gamma
    T = T_anhydrous - delta_T

    return P, T


# =========================================================
# Lin et al. (2024)
# =========================================================
def lin_2024(kind="solidus", fO2=-4.0, Pmin=0.0, Pmax=1000.0, n=500):
    P, T_anhydrous = wolf_bower_2018(kind="solidus", Pmin=Pmin, Pmax=Pmax, n=n)

    delta_T = (340.0 / 3.2) * (2.0 - fO2)
    T_sol = T_anhydrous + delta_T

    if kind == "solidus":
        T = T_sol
    elif kind == "liquidus":
        T = liquidus_from_solidus_stixrude(T_sol)
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return P, T


# =========================================================
# Keep only the main physical interval where solidus < liquidus
# Works for Andrault and for high-P crossovers
# =========================================================
def truncate_to_physical_interval(func):
    def wrapped(kind="solidus", Pmin=0.0, Pmax=1000.0, n=2000, **kwargs):
        P_sol, T_sol = func(kind="solidus", Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)
        P_liq, T_liq = func(kind="liquidus", Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)

        good = T_sol < T_liq
        idx = np.where(good)[0]

        if len(idx) == 0:
            raise ValueError(f"{func.__name__}: no physical interval where solidus < liquidus")

        splits = np.where(np.diff(idx) > 1)[0] + 1
        blocks = np.split(idx, splits)
        main_block = max(blocks, key=len)

        if kind == "solidus":
            return P_sol[main_block], T_sol[main_block]
        elif kind == "liquidus":
            return P_liq[main_block], T_liq[main_block]
        else:
            raise ValueError("kind must be 'solidus' or 'liquidus'")

    return wrapped


# Wrapped physical versions
andrault_2011_cut = truncate_to_physical_interval(andrault_2011)
monteux_2016_cut = truncate_to_physical_interval(monteux_2016)
wolf_bower_2018_cut = truncate_to_physical_interval(wolf_bower_2018)
katz_2003_cut = truncate_to_physical_interval(katz_2003)


# =========================================================
# Dispatcher
# =========================================================
def get_melting_curves(model_name, Pmin=0.0, Pmax=1000.0, n=2000, **kwargs):
    """
    Return physical solidus and liquidus curves for a given model name.

    Returns
    -------
    P_sol, T_sol, P_liq, T_liq
    """
    models = {
        "andrault_2011": andrault_2011_cut,
        "monteux_2016": monteux_2016_cut,
        "wolf_bower_2018": wolf_bower_2018_cut,
        "katz_2003": katz_2003_cut,
        "fei_2021": fei_2021,
        "belonoshko_2005": belonoshko_2005,
        "fiquet_2010": fiquet_2010,
        "hirschmann_2000": hirschmann_2000,
        "stixrude_2014": stixrude_2014,
        "lin_2024": lin_2024,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    func = models[model_name]
    P_sol, T_sol = func(kind="solidus", Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)
    P_liq, T_liq = func(kind="liquidus", Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)

    return P_sol, T_sol, P_liq, T_liq


# =========================================================
# Save helpers
# =========================================================
def save_PT_table(path: Path, P_gpa, T_k):
    """
    Save a P-T profile with header:
    #pressure temperature
    """
    data = np.column_stack([P_gpa, T_k])
    np.savetxt(path, data, fmt="%.18e %.18e", header="pressure temperature", comments="#")


# ================== EOS settings ==================
eos_solid_path = Path("temperature_solid.dat")
eos_liquid_path = Path("temperature_melt.dat")

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

COMMON_HEADER = "\n".join([
    "# 5 400",
    "# Pressure, Entropy, Quantity",
    "# column * scaling factor should be SI units",
    "# scaling factors (constant) for each column given on line below",
    "# 1000000000.0 4824266.84604467",
])


def load_eos_T_of_SP(eos_path: Path, nS: int, scale_S_axis: float):
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


def invert_to_entropy_along_profile(P_gpa, T_k, S_axis, T_of_SP):
    """
    Invert T(S,P) -> S along a P-T profile.
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

        if T_i < T_sorted[0] or T_i > T_sorted[-1]:
            continue

        try:
            f = interp1d(T_sorted, S_sorted, kind="linear", assume_sorted=True)
            S_out[i] = float(f(T_i))
        except Exception:
            f = interp1d(T_sorted, S_sorted, kind="nearest", assume_sorted=True)
            S_out[i] = float(f(T_i))

    return S_out


def save_entropy_table_with_header(path: Path, P_gpa, S_jpk):
    """
    Save entropy table in the scaled 2-column format.
    """
    P_pa = P_gpa * 1e9
    data = np.column_stack([P_pa / SCALE_P_OUT, S_jpk / SCALE_S_OUT])
    np.savetxt(path, data, fmt="%.18e %.18e", header=COMMON_HEADER, comments="")


# Load EOS interpolators once
S_axis_solid, P_axis_solid, T_of_SP_solid = load_eos_T_of_SP(
    eos_solid_path, nS_solid, SCALE_S_SOLID_EOS
)

S_axis_liquid, P_axis_liquid, T_of_SP_liquid = load_eos_T_of_SP(
    eos_liquid_path, nS_liquid, SCALE_S_LIQUID_EOS
)


# =========================================================
# Main exporter: save P-T and P-S
# =========================================================
def export_model_curves(model_name, out_root="outputs_entropy_curves",
                        Pmin=0.0, Pmax=1000.0, n=2000, **kwargs):
    """
    Generate melting curves for one model, save:
      - solidus.dat
      - liquidus.dat
      - solidus_entropy.dat
      - liquidus_entropy.dat

    Returns
    -------
    dict with arrays
    """
    out_dir = Path(out_root) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    P_sol, T_sol, P_liq, T_liq = get_melting_curves(
        model_name, Pmin=Pmin, Pmax=Pmax, n=n, **kwargs
    )

    # Save P-T
    save_PT_table(out_dir / "solidus.dat", P_sol, T_sol)
    save_PT_table(out_dir / "liquidus.dat", P_liq, T_liq)

    # Convert to entropy
    S_sol = invert_to_entropy_along_profile(
        P_sol, T_sol, S_axis_solid, T_of_SP_solid
    )
    S_liq = invert_to_entropy_along_profile(
        P_liq, T_liq, S_axis_liquid, T_of_SP_liquid
    )

    # Remove NaNs before saving P-S files
    mask_sol = np.isfinite(S_sol)
    mask_liq = np.isfinite(S_liq)

    save_entropy_table_with_header(
        out_dir / "solidus_entropy.dat",
        P_sol[mask_sol],
        S_sol[mask_sol],
    )

    save_entropy_table_with_header(
        out_dir / "liquidus_entropy.dat",
        P_liq[mask_liq],
        S_liq[mask_liq],
    )

    return {
        "P_sol": P_sol,
        "T_sol": T_sol,
        "P_liq": P_liq,
        "T_liq": T_liq,
        "S_sol": S_sol,
        "S_liq": S_liq,
    }


# =========================================================
# Convenience: export all models
# =========================================================
def export_all_models(out_root="outputs_entropy_curves"):
    models = [
        "andrault_2011",
        "monteux_2016",
        "wolf_bower_2018",
        "katz_2003",
        "fei_2021",
        "belonoshko_2005",
        "fiquet_2010",
        "hirschmann_2000",
        "stixrude_2014",
        "lin_2024",
    ]

    for model in models:
        print(f"[INFO] Exporting {model}")

        if model == "katz_2003":
            export_model_curves(model, out_root=out_root, X_h2o=30.0)
        elif model == "lin_2024":
            export_model_curves(model, out_root=out_root, fO2=-4.0)
        elif model == "hirschmann_2000":
            export_model_curves(model, out_root=out_root, Pmax=10.0)
        elif model == "fei_2021":
            export_model_curves(model, out_root=out_root, Pmin=1.0)
        elif model == "stixrude_2014":
            export_model_curves(model, out_root=out_root, Pmin=1.0)
        else:
            export_model_curves(model, out_root=out_root)

    print("[INFO] Done.")


# =========================================================
# Optional quick plot
# =========================================================
def show_melting_curves(model_name, Pmin=0.0, Pmax=1000.0, n=2000, **kwargs):
    P_sol, T_sol, P_liq, T_liq = get_melting_curves(
        model_name, Pmin=Pmin, Pmax=Pmax, n=n, **kwargs
    )

    plt.figure(figsize=(6, 6))
    plt.plot(T_sol, P_sol, label="Solidus")
    plt.plot(T_liq, P_liq, label="Liquidus")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (GPa)")
    plt.gca().invert_yaxis()
    plt.title(model_name)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return P_sol, T_sol, P_liq, T_liq

def show_entropy_curves(model_name, Pmin=0.0, Pmax=1000.0, n=2000, **kwargs):
    """
    Plot melting curves in P–S space.
    """

    P_sol, T_sol, P_liq, T_liq = get_melting_curves(
        model_name, Pmin=Pmin, Pmax=Pmax, n=n, **kwargs
    )

    # convert to entropy
    S_sol = invert_to_entropy_along_profile(
        P_sol, T_sol, S_axis_solid, T_of_SP_solid
    )

    S_liq = invert_to_entropy_along_profile(
        P_liq, T_liq, S_axis_liquid, T_of_SP_liquid
    )

    # remove NaNs before plotting
    mask_sol = np.isfinite(S_sol)
    mask_liq = np.isfinite(S_liq)

    plt.figure(figsize=(6,6))

    plt.plot(S_sol[mask_sol], P_sol[mask_sol], label="Solidus")
    plt.plot(S_liq[mask_liq], P_liq[mask_liq], label="Liquidus")

    plt.xlabel("Entropy (J kg$^{-1}$ K$^{-1}$)")
    plt.ylabel("Pressure (GPa)")
    plt.gca().invert_yaxis()

    plt.title(f"{model_name} (P–S)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return (
        P_sol[mask_sol], S_sol[mask_sol],
        P_liq[mask_liq], S_liq[mask_liq],
    )

# =========================================================
# Example usage
# =========================================================
# One model:
# export_model_curves("andrault_2011")
# export_model_curves("katz_2003", X_h2o=30.0)
# export_model_curves("lin_2024", fO2=-4.0)

# All models:
export_all_models()

# Quick plot:
# show_melting_curves("wolf_bower_2018")
# show_melting_curves("andrault_2011")
# show_melting_curves("katz_2003", X_h2o=30.0)
# show_melting_curves("lin_2024", fO2=-4.0)
# show_melting_curves("fei_2021")
# show_melting_curves("belonoshko_2005")
# show_melting_curves("fiquet_2010")
# show_melting_curves("hirschmann_2000", Pmax=10.0)
# show_melting_curves("stixrude_2014", Pmin=1.0)
show_entropy_curves("wolf_bower_2018")
show_entropy_curves("andrault_2011")
show_entropy_curves("katz_2003", X_h2o=30.0)
show_entropy_curves("lin_2024", fO2=-4.0)
show_entropy_curves("fei_2021")
show_entropy_curves("belonoshko_2005")
show_entropy_curves("fiquet_2010")
show_entropy_curves("hirschmann_2000", Pmax=10.0)
show_entropy_curves("stixrude_2014", Pmin=1.0)
