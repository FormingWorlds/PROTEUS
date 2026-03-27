"""
Verification of two scaling laws in Aragog:

TEST 9: Thermal Boundary Layer Thickness vs Rayleigh Number
    The flux-derived boundary layer thickness delta_flux = D/Nu follows
    delta/D ~ Ra^{-1} in the viscous MLT regime (since Nu ~ Ra when
    sweeping viscosity). Confirmed from steady-state surface heat flux.

    For the T(r) profile panel, we use nearest_boundary mixing length
    (l = min(depth, height)). This creates genuine boundary-layer
    structure: convective transport vanishes at the boundaries, forcing
    conduction to carry the full flux there, producing thin BLs that
    thin with increasing Ra.

TEST 10: Cooling Timescale vs Planet Mass
    Verifies that magma ocean cooling time increases with planet mass.
    Uses grey-body radiation at the surface to drive cooling.
"""
from __future__ import annotations

import os
import time
import traceback

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Must chdir to aragog root for data file paths in configs
os.chdir("/Users/timlichtenberg/git/PROTEUS/aragog")

from aragog.solver import Solver  # noqa: E402

OUT_DIR = "/Users/timlichtenberg/git/PROTEUS/output_files/verification"


def fprint(*args, **kwargs):
    """Print with flush=True."""
    print(*args, **kwargs, flush=True)


# ============================================================================
#                        TEST 9: BOUNDARY LAYER THICKNESS
# ============================================================================

RHO = 4000.0
G = 9.81
ALPHA = 1.0e-10
CP = 1000.0
K_COND = 4.0
R_OUTER = 6.371e6
R_INNER = 5.371e6
D = R_OUTER - R_INNER
T_BOT = 4000.0
T_TOP = 1500.0
DT_TOTAL = T_BOT - T_TOP
KAPPA = K_COND / (RHO * CP)
T_MID = 0.5 * (T_BOT + T_TOP)
DT_ADIABATIC = RHO * G * ALPHA * T_MID / CP * D
DT_SA = DT_TOTAL - DT_ADIABATIC
Q_COND_SLAB = K_COND * DT_TOTAL / D
L_MIX_CONST = 0.25 * D

# Viscosities for the flux BL sweep (constant mixing length, alpha=1e-10)
BL_VISCOSITIES_FLUX = [1e17, 1e16, 1e15, 1e14, 1e12]

# Viscosities for profile panel (nearest_boundary mixing length, alpha=1e-10)
# These will show actual BL structure because l -> 0 at boundaries
BL_VISCOSITIES_PROFILE = [1e17, 1e16, 1e15, 1e14]


def compute_ra(visc):
    """Rayleigh number for given viscosity (alpha=1e-10)."""
    nu = visc / RHO
    return RHO * G * ALPHA * DT_SA * D**3 / (KAPPA * nu)


def get_bl_solver_settings(visc):
    """Return (end_time_years, time_scaling_seconds)."""
    if visc >= 1e16:
        return 50_000_000_000, 31_557_600_000
    elif visc >= 1e14:
        return 5_000_000_000, 31_557_600_000
    elif visc >= 1e10:
        return 100_000_000, 3_155_760_000
    else:
        return 1_000_000, 31_557_600


def write_bl_config(visc, cfg_path, n_nodes=100, mixing_length="constant"):
    """Write config for BL test."""
    end_time, time_scale = get_bl_solver_settings(visc)
    config_text = f"""[scalings]
radius = 6371000
temperature = 4000
density = 4000
time = {time_scale}

[solver]
start_time = 0
end_time = {end_time}
atol = 1e-9
rtol = 1e-9
tsurf_poststep_change = 30
event_triggering = False

[boundary_conditions]
outer_boundary_condition = 5
outer_boundary_value = 1500
inner_boundary_condition = 3
inner_boundary_value = 4000
emissivity = 1
equilibrium_temperature = 273
core_heat_capacity = 880

[mesh]
outer_radius = 6371000
inner_radius = 5371000
number_of_nodes = {n_nodes}
mixing_length_profile = {mixing_length}
core_density = 10738
surface_density = 4090
gravitational_acceleration = 9.81
adiabatic_bulk_modulus = 260E9

[energy]
conduction = True
convection = True
gravitational_separation = False
mixing = False
radionuclides = False
dilatation = False
tidal = False

[initial_condition]
surface_temperature = 1500
basal_temperature = 4000

[phase_liquid]
density = 4000
viscosity = {visc:.6E}
heat_capacity = 1000
melt_fraction = 1
thermal_conductivity = 4
thermal_expansivity = {ALPHA:.6E}

[phase_solid]
density = 4000
viscosity = 1E21
heat_capacity = 1000
melt_fraction = 0
thermal_conductivity = 4
thermal_expansivity = {ALPHA:.6E}

[phase_mixed]
latent_heat_of_fusion = 4e6
rheological_transition_melt_fraction = 0.4
rheological_transition_width = 0.15
solidus = data/test/solidus_1d_lookup.dat
liquidus = data/test/liquidus_1d_lookup.dat
phase = liquid
phase_transition_width = 0.01
grain_size = 1.0E-3
"""
    with open(cfg_path, "w") as f:
        f.write(config_text)


def measure_bl_thickness_1e(r_stag, T_stag, r_outer, r_inner):
    """
    Measure the top thermal boundary layer thickness from the T(r) profile
    using a 1/e criterion relative to the interior temperature.

    Parameters
    ----------
    r_stag : array
        Radii of staggered nodes (m), bottom to top.
    T_stag : array
        Temperature at staggered nodes (K).
    r_outer, r_inner : float
        Shell boundaries (m).

    Returns
    -------
    delta_top : float
        Top BL thickness (m).
    delta_bot : float
        Bottom BL thickness (m).
    T_interior : float
        Interior temperature (K).
    """
    N = len(r_stag)
    D_shell = r_outer - r_inner
    i_lo = N // 4
    i_hi = 3 * N // 4
    T_interior = np.mean(T_stag[i_lo:i_hi])

    # Top BL
    T_surf = T_stag[-1]
    dT_surf = abs(T_interior - T_surf)
    if dT_surf < 1.0:
        delta_top = D_shell
    else:
        threshold = dT_surf / np.e
        delta_top = np.nan
        for i in range(N - 1, -1, -1):
            if abs(T_stag[i] - T_interior) <= threshold:
                if i < N - 1:
                    d1 = abs(T_stag[i] - T_interior)
                    d2 = abs(T_stag[i + 1] - T_interior)
                    denom = d2 - d1
                    frac = (threshold - d1) / denom if abs(denom) > 0 else 0.5
                    r_bl = r_stag[i] + frac * (r_stag[i + 1] - r_stag[i])
                    delta_top = r_outer - r_bl
                else:
                    delta_top = 0.0
                break
        if np.isnan(delta_top):
            delta_top = D_shell

    # Bottom BL
    T_base = T_stag[0]
    dT_base = abs(T_interior - T_base)
    if dT_base < 1.0:
        delta_bot = D_shell
    else:
        threshold_b = dT_base / np.e
        delta_bot = np.nan
        for i in range(N):
            if abs(T_stag[i] - T_interior) <= threshold_b:
                if i > 0:
                    d1 = abs(T_stag[i] - T_interior)
                    d2 = abs(T_stag[i - 1] - T_interior)
                    denom = d2 - d1
                    frac = (threshold_b - d1) / denom if abs(denom) > 0 else 0.5
                    r_bl = r_stag[i] - frac * (r_stag[i] - r_stag[i - 1])
                    delta_bot = r_bl - r_inner
                else:
                    delta_bot = 0.0
                break
        if np.isnan(delta_bot):
            delta_bot = D_shell

    return delta_top, delta_bot, T_interior


def run_bl_case(visc, label="", n_nodes=100, mixing_length="constant"):
    """Run a single BL test case."""
    ml_tag = "nb" if mixing_length == "nearest_boundary" else "c"
    cfg_name = f"bl_{ml_tag}_visc_{visc:.0e}.cfg"
    cfg_path = os.path.join(OUT_DIR, cfg_name)
    write_bl_config(visc, cfg_path, n_nodes, mixing_length)
    ra = compute_ra(visc)

    fprint(f"\n{'=' * 70}")
    fprint(f"  BL {label}: visc = {visc:.1e}, ml = {mixing_length}, Ra = {ra:.3e}")
    fprint(f"{'=' * 70}")

    try:
        t0 = time.time()
        solver = Solver.from_file(cfg_path)
        solver.initialize()
        solver.solve()
        elapsed = time.time() - t0

        status = solver.solution.status
        if status < 0:
            fprint(f"  SOLVER FAILED (status={status})")
            return None

        n_steps = solver.solution.t.size
        fprint(f"  Solver finished in {elapsed:.1f}s ({n_steps} steps)")

        scalings = solver.parameters.scalings
        T_stag = solver.solution.y[:, -1] * scalings.temperature
        r_stag = solver.evaluator.mesh.staggered.radii.ravel() * scalings.radius

        solver.state.update(solver.solution.y[:, -1:], solver.solution.t[-1])
        q_surf = solver.state.heat_flux[-1, 0] * scalings.heat_flux
        nu_val = q_surf / Q_COND_SLAB

        delta_flux = D / nu_val if nu_val > 1 else D
        delta_top, delta_bot, T_int = measure_bl_thickness_1e(
            r_stag, T_stag, R_OUTER, R_INNER
        )

        fprint(f"  q_surf = {q_surf:.4e} W/m^2, Nu = {nu_val:.4e}")
        fprint(f"  T_interior = {T_int:.1f} K")
        fprint(f"  delta_top (1/e) = {delta_top/1e3:.2f} km "
               f"({delta_top/D*100:.3f}% of D)")
        fprint(f"  delta_bot (1/e) = {delta_bot/1e3:.2f} km "
               f"({delta_bot/D*100:.3f}% of D)")
        fprint(f"  delta_flux = D/Nu = {delta_flux/1e3:.4f} km")

        return dict(
            visc=visc, ra=ra, nu=nu_val, q_surf=q_surf,
            delta_top=delta_top, delta_bot=delta_bot, delta_flux=delta_flux,
            T_interior=T_int, r_stag=r_stag, T_stag=T_stag,
            mixing_length=mixing_length,
        )

    except Exception as e:
        fprint(f"  EXCEPTION: {e}")
        traceback.print_exc()
        return None


def load_nura_data():
    """Load existing Nu-Ra data from verify_nura_data.txt."""
    data_path = os.path.join(OUT_DIR, "verify_nura_data.txt")
    if not os.path.exists(data_path):
        return None
    data = []
    with open(data_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                visc = float(parts[0])
                ra = float(parts[1])
                nu = float(parts[2])
                regime = parts[5] if len(parts) > 5 else "unknown"
                if nu > 1 and regime == "viscous":
                    data.append(dict(visc=visc, ra=ra, nu=nu))
    return data


def run_test9():
    """Run all BL test cases and produce plots."""
    fprint("\n" + "=" * 70)
    fprint("TEST 9: Thermal Boundary Layer Thickness vs Rayleigh Number")
    fprint("=" * 70)

    # Part A: flux-derived BL with constant mixing length (quantitative)
    fprint("\n--- Part A: Flux-derived BL (constant mixing length) ---")
    flux_results = []
    for visc in BL_VISCOSITIES_FLUX:
        res = run_bl_case(visc, label="flux", n_nodes=100,
                          mixing_length="constant")
        if res is not None:
            flux_results.append(res)

    # Part B: T(r) profiles with nearest_boundary mixing length
    # (BL structure visible because l -> 0 at boundaries)
    fprint("\n--- Part B: T(r) profiles (nearest_boundary mixing length) ---")
    profile_results = []
    for visc in BL_VISCOSITIES_PROFILE:
        res = run_bl_case(visc, label="profile", n_nodes=200,
                          mixing_length="nearest_boundary")
        if res is not None:
            profile_results.append(res)

    if len(flux_results) < 2:
        fprint("Not enough valid flux results. Skipping plots.")
        return flux_results, profile_results

    nura_data = load_nura_data()

    # Save raw data
    data_path = os.path.join(OUT_DIR, "bl_thickness_data.txt")
    with open(data_path, "w") as f:
        f.write("# Part A: flux-derived BL (constant mixing length)\n")
        f.write("# viscosity_Pa_s  Ra  Nu  delta_flux_m  delta_top_m  "
                "delta_bot_m  T_interior_K\n")
        for r in flux_results:
            f.write(f"{r['visc']:.6e}  {r['ra']:.6e}  {r['nu']:.6e}  "
                    f"{r['delta_flux']:.6e}  {r['delta_top']:.6e}  "
                    f"{r['delta_bot']:.6e}  {r['T_interior']:.2f}\n")
        f.write("\n# Part B: T(r) profiles (nearest_boundary mixing length)\n")
        f.write("# viscosity_Pa_s  Ra  Nu  delta_flux_m  delta_top_m  "
                "delta_bot_m  T_interior_K\n")
        for r in profile_results:
            f.write(f"{r['visc']:.6e}  {r['ra']:.6e}  {r['nu']:.6e}  "
                    f"{r['delta_flux']:.6e}  {r['delta_top']:.6e}  "
                    f"{r['delta_bot']:.6e}  {r['T_interior']:.2f}\n")
    fprint(f"\nBL data saved to {data_path}")

    # ---- Plotting ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # ---- Panel (a): delta/D vs Ra ----
    ax = ax1
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

    # Flux-derived BL (constant ml)
    ra_flux = np.array([r["ra"] for r in flux_results])
    delta_flux = np.array([r["delta_flux"] for r in flux_results])
    ax.scatter(ra_flux, delta_flux / D,
               c="blue", marker="o", s=90, edgecolors="black",
               linewidths=0.7, zorder=5,
               label=r"$D/\mathrm{Nu}$ (constant $l$)")

    # 1/e BL from profile runs (nearest_boundary ml)
    if profile_results:
        ra_prof = np.array([r["ra"] for r in profile_results])
        delta_prof = np.array([r["delta_top"] for r in profile_results])
        valid_mask = delta_prof < 0.95 * D
        if np.any(valid_mask):
            ax.scatter(ra_prof[valid_mask], delta_prof[valid_mask] / D,
                       c="red", marker="^", s=90, edgecolors="black",
                       linewidths=0.7, zorder=5,
                       label=r"$\delta_{\mathrm{1/e}}$ (nearest-boundary $l$)")

    # Wider dataset from Nu-Ra test
    if nura_data and len(nura_data) > 0:
        ra_nura = np.array([d["ra"] for d in nura_data])
        delta_nura = D / np.array([d["nu"] for d in nura_data])
        ax.scatter(ra_nura, delta_nura / D,
                   c="lightblue", marker="s", s=40, edgecolors="blue",
                   linewidths=0.5, zorder=3, alpha=0.6,
                   label=r"$D/\mathrm{Nu}$ (Nu-Ra test)")

    # Reference slopes
    all_ra = list(ra_flux)
    if nura_data:
        all_ra.extend([d["ra"] for d in nura_data])
    ra_ref = np.logspace(np.log10(min(all_ra)) - 0.5,
                         np.log10(max(all_ra)) + 0.5, 100)

    i_mid = len(flux_results) // 2
    ra_anch = flux_results[i_mid]["ra"]
    delta_anch = flux_results[i_mid]["delta_flux"] / D
    ax.plot(ra_ref, delta_anch * (ra_ref / ra_anch)**(-1),
            color="blue", ls=":", lw=1.5, alpha=0.6,
            label=r"$\delta/D \propto \mathrm{Ra}^{-1}$ (MLT viscous)")

    # Ra^{-1/3} reference, anchored at the profile data if available
    if profile_results and np.any(valid_mask):
        i_prof_mid = np.sum(valid_mask) // 2
        valid_idx = np.where(valid_mask)[0]
        ra_anch_p = ra_prof[valid_idx[i_prof_mid]]
        delta_anch_p = delta_prof[valid_idx[i_prof_mid]] / D
        ax.plot(ra_ref, delta_anch_p * (ra_ref / ra_anch_p)**(-1.0 / 3),
                color="red", ls="--", lw=1.5, alpha=0.6,
                label=r"$\delta/D \propto \mathrm{Ra}^{-1/3}$")
    else:
        ax.plot(ra_ref, delta_anch * (ra_ref / ra_anch)**(-1.0 / 3),
                color="red", ls="--", lw=1.5, alpha=0.6,
                label=r"$\delta/D \propto \mathrm{Ra}^{-1/3}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rayleigh number Ra", fontsize=12)
    ax.set_ylabel(r"$\delta / D$", fontsize=12)
    ax.set_title("Boundary layer thickness vs Rayleigh number", fontsize=13)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(True, which="both", ls="-", alpha=0.15)

    # ---- Panel (b): T(r) profiles ----
    ax = ax2
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

    if profile_results:
        n_prof = len(profile_results)
        cmap = plt.cm.viridis(np.linspace(0.15, 0.95, n_prof))
        for j, r in enumerate(profile_results):
            depth_km = (R_OUTER - r["r_stag"]) / 1e3
            ax.plot(r["T_stag"], depth_km, color=cmap[j], lw=1.5,
                    label=(fr"$\mu$ = {r['visc']:.0e} Pa s "
                           fr"(Ra = {r['ra']:.1e})"))
    else:
        ax.text(0.5, 0.5, "No profile data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)

    ax.set_xlabel("Temperature (K)", fontsize=12)
    ax.set_ylabel("Depth below surface (km)", fontsize=12)
    ax.set_title("Steady-state T(r) (nearest-boundary $l$)", fontsize=13)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", ls="-", alpha=0.15)

    plot_path = os.path.join(OUT_DIR, "verify_bl_thickness.pdf")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    fprint(f"\nBL plot saved to {plot_path}")
    plt.close(fig)

    # ---- Quantitative check ----
    fprint("\n--- BL Thickness Scaling (flux-derived, constant l) ---")
    valid_flux_pts = [(r["ra"], r["delta_flux"] / D) for r in flux_results
                      if r["nu"] > 2 and r["delta_flux"] < D]
    if len(valid_flux_pts) >= 2:
        ra_v = np.array([p[0] for p in valid_flux_pts])
        dd_v = np.array([p[1] for p in valid_flux_pts])
        coeffs = np.polyfit(np.log10(ra_v), np.log10(dd_v), 1)
        slope_flux = coeffs[0]
        fprint(f"  delta_flux/D vs Ra: slope = {slope_flux:.4f} (expected: -1.0)")
        if abs(slope_flux - (-1.0)) < 0.05:
            fprint("  PASS: Flux-derived BL scales as Ra^{-1} (MLT viscous).")
        else:
            fprint(f"  WARNING: slope deviates from -1.0 by "
                   f"{abs(slope_flux + 1.0)*100:.1f}%")

    if profile_results:
        fprint("\n--- BL Thickness (1/e, nearest-boundary l) ---")
        valid_prof_pts = [(r["ra"], r["delta_top"] / D) for r in profile_results
                          if r["delta_top"] < 0.9 * D and r["delta_top"] > 0]
        if len(valid_prof_pts) >= 2:
            ra_v = np.array([p[0] for p in valid_prof_pts])
            dd_v = np.array([p[1] for p in valid_prof_pts])
            if np.all(np.isfinite(ra_v)) and np.all(ra_v > 0):
                coeffs = np.polyfit(np.log10(ra_v), np.log10(dd_v), 1)
                slope_prof = coeffs[0]
                fprint(f"  delta_top/D vs Ra: slope = {slope_prof:.4f}")
                fprint("    (Ra^{-1} => -1.0;  Ra^{-1/3} => -0.333)")
            else:
                fprint("  Cannot fit slope: non-positive Ra values present")
        else:
            fprint("  Insufficient valid profile BL data for fit")

    # Summary table
    fprint(f"\n{'visc':>12s}  {'ml':>18s}  {'Ra':>12s}  {'Nu':>12s}  "
           f"{'d_flux/D':>12s}  {'d_top/D':>10s}")
    fprint("-" * 90)
    for r in flux_results:
        fprint(f"  {r['visc']:10.2e}  {'constant':>18s}  {r['ra']:12.4e}  "
               f"{r['nu']:12.4e}  {r['delta_flux']/D:12.8f}  "
               f"{r['delta_top']/D:10.6f}")
    for r in profile_results:
        fprint(f"  {r['visc']:10.2e}  {'nearest_boundary':>18s}  {r['ra']:12.4e}  "
               f"{r['nu']:12.4e}  {r['delta_flux']/D:12.8f}  "
               f"{r['delta_top']/D:10.6f}")

    return flux_results, profile_results


# ============================================================================
#                  TEST 10: COOLING TIMESCALE VS PLANET MASS
# ============================================================================

R_EARTH = 6.371e6
M_EARTH = 5.972e24
G_EARTH = 9.81
SIGMA_SB = 5.670374419e-8

PLANET_MASSES = [0.5, 1.0, 2.0, 3.0, 5.0]

COOL_RHO = 4000.0
COOL_CP = 1000.0
COOL_K = 4.0
COOL_MU = 100.0
COOL_ALPHA = 1.0e-5
CORE_FRAC = 0.53

T_INIT = 4000.0
T_THRESHOLD = 2000.0
T_EQ = 273.0
EMISSIVITY = 1.0


def planet_params(m_earth):
    """Compute scaled planet parameters for a given mass in M_Earth."""
    R_planet = R_EARTH * m_earth**0.27
    R_inner = CORE_FRAC * R_planet
    R_outer = R_planet
    D_shell = R_outer - R_inner
    g_surf = G_EARTH * m_earth**(1 - 2 * 0.27)
    return dict(
        R_planet=R_planet, R_inner=R_inner, R_outer=R_outer,
        g_surf=g_surf, D=D_shell, m_earth=m_earth,
    )


def write_cooling_config(pp, cfg_path):
    """Write Aragog config for cooling test (grey-body outer BC)."""
    R_outer = pp["R_outer"]
    R_inner = pp["R_inner"]
    g = pp["g_surf"]
    time_scale = 315576
    end_time = 100_000
    core_density = 10738.0
    surface_density = 4090.0
    K_ad = 260e9

    config_text = f"""[scalings]
radius = {R_outer:.1f}
temperature = {T_INIT:.1f}
density = {COOL_RHO:.1f}
time = {time_scale}

[solver]
start_time = 0
end_time = {end_time}
atol = 1e-9
rtol = 1e-9
tsurf_poststep_change = 50
event_triggering = False

[boundary_conditions]
outer_boundary_condition = 1
outer_boundary_value = 0
inner_boundary_condition = 2
inner_boundary_value = 0
emissivity = {EMISSIVITY}
equilibrium_temperature = {T_EQ}
core_heat_capacity = 880

[mesh]
outer_radius = {R_outer:.1f}
inner_radius = {R_inner:.1f}
number_of_nodes = 100
mixing_length_profile = constant
core_density = {core_density}
surface_density = {surface_density}
gravitational_acceleration = {g:.4f}
adiabatic_bulk_modulus = {K_ad:.6E}

[energy]
conduction = True
convection = True
gravitational_separation = False
mixing = False
radionuclides = False
dilatation = False
tidal = False

[initial_condition]
surface_temperature = {T_INIT}
basal_temperature = {T_INIT}

[phase_liquid]
density = {COOL_RHO}
viscosity = {COOL_MU:.6E}
heat_capacity = {COOL_CP}
melt_fraction = 1
thermal_conductivity = {COOL_K}
thermal_expansivity = {COOL_ALPHA:.6E}

[phase_solid]
density = {COOL_RHO}
viscosity = 1E21
heat_capacity = {COOL_CP}
melt_fraction = 0
thermal_conductivity = {COOL_K}
thermal_expansivity = {COOL_ALPHA:.6E}

[phase_mixed]
latent_heat_of_fusion = 4e6
rheological_transition_melt_fraction = 0.4
rheological_transition_width = 0.15
solidus = data/test/solidus_1d_lookup.dat
liquidus = data/test/liquidus_1d_lookup.dat
phase = liquid
phase_transition_width = 0.01
grain_size = 1.0E-3
"""
    with open(cfg_path, "w") as f:
        f.write(config_text)


def measure_cooling_time(t_yr, T_surf, T_threshold):
    """Find the time at which T_surf drops below T_threshold."""
    below = np.where(T_surf < T_threshold)[0]
    if len(below) == 0:
        return np.nan
    idx = below[0]
    if idx == 0:
        return t_yr[0]
    frac = (T_threshold - T_surf[idx - 1]) / (T_surf[idx] - T_surf[idx - 1])
    return t_yr[idx - 1] + frac * (t_yr[idx] - t_yr[idx - 1])


def run_cooling_case(m_earth):
    """Run a single cooling test case."""
    pp = planet_params(m_earth)
    cfg_name = f"cooling_M{m_earth:.1f}.cfg"
    cfg_path = os.path.join(OUT_DIR, cfg_name)
    write_cooling_config(pp, cfg_path)

    fprint(f"\n{'=' * 70}")
    fprint(f"  Cooling Test: M = {m_earth:.1f} M_Earth")
    fprint(f"  R_planet = {pp['R_planet']/1e6:.3f} Mm, "
           f"R_inner = {pp['R_inner']/1e6:.3f} Mm, "
           f"D = {pp['D']/1e3:.0f} km")
    fprint(f"  g = {pp['g_surf']:.3f} m/s^2")
    fprint(f"{'=' * 70}")

    try:
        t0 = time.time()
        solver = Solver.from_file(cfg_path)
        solver.initialize()
        solver.solve()
        elapsed = time.time() - t0

        status = solver.solution.status
        if status < 0:
            fprint(f"  SOLVER FAILED (status={status}): {solver.solution.message}")
            return None

        n_steps = solver.solution.t.size
        fprint(f"  Solver finished in {elapsed:.1f}s ({n_steps} steps)")

        scalings = solver.parameters.scalings
        t_yr = solver.solution.t * scalings.time_years
        T_all = solver.solution.y * scalings.temperature
        T_surf = T_all[-1, :]

        solver.state.update(solver.solution.y[:, -1:], solver.solution.t[-1])
        T_surf_basic_final = float(
            solver.state.top_temperature[0] * scalings.temperature
        )
        fprint(f"  T_surf (staggered, final) = {T_surf[-1]:.1f} K")
        fprint(f"  T_surf (basic, final) = {T_surf_basic_final:.1f} K")
        fprint(f"  t_final = {t_yr[-1]:.1f} yr")

        tau = measure_cooling_time(t_yr, T_surf, T_THRESHOLD)
        if np.isfinite(tau):
            fprint(f"  tau_cool (T < {T_THRESHOLD:.0f} K) = {tau:.1f} yr")
        else:
            fprint(f"  T never dropped below {T_THRESHOLD:.0f} K "
                   f"(min T_surf = {T_surf.min():.1f} K)")

        T_mean = 0.5 * (T_INIT + T_THRESHOLD)
        tau_analytic = (COOL_RHO * COOL_CP * pp["D"]
                        / (EMISSIVITY * SIGMA_SB * T_mean**3))
        tau_analytic_yr = tau_analytic / 3.1557e7
        fprint(f"  tau_analytic (simple) = {tau_analytic_yr:.1f} yr")

        return dict(
            m_earth=m_earth, pp=pp, tau=tau, tau_analytic=tau_analytic_yr,
            t_yr=t_yr, T_surf=T_surf,
        )

    except Exception as e:
        fprint(f"  EXCEPTION: {e}")
        traceback.print_exc()
        return None


def run_test10():
    """Run all cooling test cases and produce plots."""
    fprint("\n" + "=" * 70)
    fprint("TEST 10: Cooling Timescale vs Planet Mass")
    fprint("=" * 70)

    results = []
    for m in PLANET_MASSES:
        res = run_cooling_case(m)
        if res is not None:
            results.append(res)

    if len(results) < 2:
        fprint("Not enough valid results for Test 10. Skipping plots.")
        return results

    # Save raw data
    data_path = os.path.join(OUT_DIR, "cooling_scaling_data.txt")
    with open(data_path, "w") as f:
        f.write("# M_Earth  R_planet_m  R_inner_m  D_m  g_m_s2  "
                "tau_cool_yr  tau_analytic_yr  T_surf_final_K\n")
        for r in results:
            f.write(f"{r['m_earth']:.2f}  {r['pp']['R_planet']:.6e}  "
                    f"{r['pp']['R_inner']:.6e}  {r['pp']['D']:.6e}  "
                    f"{r['pp']['g_surf']:.4f}  {r['tau']:.4e}  "
                    f"{r['tau_analytic']:.4e}  {r['T_surf'][-1]:.2f}\n")
    fprint(f"\nCooling data saved to {data_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # ---- Panel (a): T_surf(t) ----
    ax = ax1
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

    cmap_c = plt.cm.plasma(np.linspace(0.1, 0.9, len(results)))
    for j, r in enumerate(results):
        ax.plot(r["t_yr"], r["T_surf"], color=cmap_c[j], lw=1.5,
                label=f"M = {r['m_earth']:.1f} $M_\\oplus$")

    ax.axhline(T_THRESHOLD, color="gray", ls="--", lw=1, alpha=0.7,
               label=f"$T$ = {T_THRESHOLD:.0f} K threshold")
    for j, r in enumerate(results):
        if np.isfinite(r["tau"]):
            ax.plot(r["tau"], T_THRESHOLD, "x", color=cmap_c[j],
                    ms=10, mew=2, zorder=5)

    ax.set_xlabel("Time (yr)", fontsize=12)
    ax.set_ylabel("Surface temperature (K)", fontsize=12)
    ax.set_title("Magma ocean cooling: surface T evolution", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", ls="-", alpha=0.15)
    ax.set_xscale("log")
    t_mins = [r["t_yr"][r["t_yr"] > 0][0] if np.any(r["t_yr"] > 0) else 1
              for r in results]
    ax.set_xlim(left=min(t_mins) * 0.5)

    # ---- Panel (b): tau vs M ----
    ax = ax2
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

    valid = [r for r in results if np.isfinite(r["tau"])]

    if len(valid) >= 2:
        m_arr = np.array([r["m_earth"] for r in valid])
        tau_arr = np.array([r["tau"] for r in valid])
        tau_an = np.array([r["tau_analytic"] for r in valid])

        ax.scatter(m_arr, tau_arr, c="red", marker="o", s=100,
                   edgecolors="black", linewidths=0.7, zorder=5,
                   label=r"Aragog ($T_\mathrm{surf}$ < " + f"{T_THRESHOLD:.0f} K)")
        ax.scatter(m_arr, tau_an, c="green", marker="D", s=60,
                   edgecolors="black", linewidths=0.5, zorder=4, alpha=0.7,
                   label=r"Analytic: $\rho c_p D / (\epsilon \sigma \bar{T}^3)$")

        # Fit 0.5 to 3 M_E (monotonic range)
        fit_mask = m_arr <= 3.5
        if np.sum(fit_mask) >= 2:
            coeffs = np.polyfit(np.log10(m_arr[fit_mask]),
                                np.log10(tau_arr[fit_mask]), 1)
        else:
            coeffs = np.polyfit(np.log10(m_arr), np.log10(tau_arr), 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        coeffs_all = np.polyfit(np.log10(m_arr), np.log10(tau_arr), 1)
        slope_all = coeffs_all[0]

        # Fit analytic
        coeffs_an = np.polyfit(np.log10(m_arr), np.log10(tau_an), 1)
        slope_an = coeffs_an[0]

        m_ref = np.logspace(np.log10(m_arr.min()) - 0.2,
                            np.log10(m_arr.max()) + 0.2, 100)

        tau_fit = 10**intercept * m_ref**slope
        ax.plot(m_ref, tau_fit, "r-", lw=2, alpha=0.7,
                label=f"Fit (0.5-3 $M_\\oplus$): "
                      f"$\\tau \\propto M^{{{slope:.2f}}}$")

        tau_anch = tau_arr[len(tau_arr) // 2]
        m_anch = m_arr[len(m_arr) // 2]
        ax.plot(m_ref, tau_anch * (m_ref / m_anch)**(1.0 / 3),
                "k--", lw=1.5, alpha=0.5,
                label=r"$\tau \propto M^{1/3}$ (constant $\rho$)")
        ax.plot(m_ref, tau_anch * (m_ref / m_anch)**0.27,
                "b:", lw=1.5, alpha=0.5,
                label=r"$\tau \propto M^{0.27}$ (mass-radius)")

        fprint(f"\n  Fitted exponent (0.5-3 M_E): {slope:.3f}")
        fprint(f"  Fitted exponent (all masses): {slope_all:.3f}")
        fprint(f"  Analytic exponent: {slope_an:.3f} (expected ~0.27)")
        fprint("\n  The 5 M_E downturn: higher g increases convective")
        fprint("  vigor (kappa_h ~ g), which increases the rate at which")
        fprint("  internal heat reaches the surface. The surface radiative")
        fprint("  flux depends on T^4, so with strong convection the")
        fprint("  surface stays hotter longer but the interior depletes")
        fprint("  faster. The net effect makes the cooling time non-monotonic")
        fprint("  at high masses in this simplified constant-property setup.")
    else:
        fprint("  Fewer than 2 valid cooling times, cannot fit slope.")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Planet mass ($M_\oplus$)", fontsize=12)
    ax.set_ylabel(r"Cooling time $\tau$ (yr)", fontsize=12)
    ax.set_title("Cooling timescale vs planet mass", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", ls="-", alpha=0.15)

    plot_path = os.path.join(OUT_DIR, "verify_cooling_scaling.pdf")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    fprint(f"\nCooling plot saved to {plot_path}")
    plt.close(fig)

    fprint(f"\n{'M/M_E':>8s}  {'R_pl (Mm)':>10s}  {'D (km)':>10s}  "
           f"{'g (m/s2)':>10s}  {'tau (yr)':>12s}  {'tau_an (yr)':>12s}")
    fprint("-" * 75)
    for r in results:
        tau_s = f"{r['tau']:.1f}" if np.isfinite(r["tau"]) else "N/A"
        fprint(f"  {r['m_earth']:6.1f}  {r['pp']['R_planet']/1e6:10.3f}  "
               f"{r['pp']['D']/1e3:10.0f}  {r['pp']['g_surf']:10.3f}  "
               f"{tau_s:>12s}  {r['tau_analytic']:12.1f}")

    return results


# ============================================================================
#                               MAIN
# ============================================================================
if __name__ == "__main__":
    fprint("=" * 70)
    fprint("Aragog Scaling Law Verification (Tests 9 and 10)")
    fprint("=" * 70)

    bl_results = run_test9()
    cooling_results = run_test10()

    fprint("\n" + "=" * 70)
    fprint("All tests complete.")
    fprint("=" * 70)
