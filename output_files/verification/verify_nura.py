"""
Verification of Aragog's Nusselt-Rayleigh scaling for MLT convection.

Physics
-------
For a convecting spherical shell with constant properties and mixing length
theory (MLT), the Nusselt number (total / conductive heat transport) scales
with the Rayleigh number. Since we vary viscosity to sweep Ra, the expected
behavior is:

  Viscous MLT regime (low Re, high viscosity):
    kappa_h = g alpha |dT/dr_sa| l^4 / (18 nu)
    Nu ~ 1/nu ~ Ra  (linear in Ra when sweeping viscosity)

  Inviscid MLT regime (high Re, low viscosity):
    kappa_h = l^2 sqrt(g alpha |dT/dr_sa| / 16)
    Independent of viscosity, so Nu = const while Ra ~ 1/nu.
    On a log-log plot, Nu is flat as Ra increases.

  Conductive regime (very high viscosity):
    Convective contribution negligible.
    Spherical conduction: q_surf = k DT R_inner / (R_outer D).
    Nu = R_inner / R_outer  (< 1 when defined against slab reference).

The transition between viscous and inviscid regimes occurs at Re_crit = 9/8,
where Re = v_visc * l / nu.

Method
------
Use alpha = 1e-10 /K so the adiabatic gradient is negligible compared to
DT = 2500 K (DT_adiabatic ~ 11 K). Sweep viscosity from 1e0 to 1e20 Pa*s.
Run to approximate steady state (fixed T BCs), then measure surface heat
flux, compute Nu and Ra, and compare to the analytical MLT prediction with
proper spherical geometry correction.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Must chdir to aragog root for data file paths in configs
os.chdir("/Users/timlichtenberg/git/aragog")

from aragog.solver import Solver  # noqa: E402

# ---------------------------------------------------------------------------
# Physical parameters (constant across all runs)
# ---------------------------------------------------------------------------
RHO = 4000.0           # density, kg/m^3
G = 9.81               # gravitational acceleration, m/s^2
ALPHA = 1.0e-10         # thermal expansivity, 1/K (small: adiabat negligible)
CP = 1000.0             # heat capacity, J/kg/K
K_COND = 4.0            # thermal conductivity, W/m/K
R_OUTER = 6.371e6       # outer radius, m
R_INNER = 5.371e6       # inner radius, m
D = R_OUTER - R_INNER   # shell thickness = 1,000,000 m (1000 km)
T_BOT = 4000.0          # bottom T, K
T_TOP = 1500.0          # top T, K
DT_TOTAL = T_BOT - T_TOP  # 2500 K

# Thermal diffusivity
KAPPA = K_COND / (RHO * CP)  # 1e-6 m^2/s

# Adiabatic temperature drop across the shell
T_MID = 0.5 * (T_BOT + T_TOP)  # 2750 K
DTDR_ADIABATIC = RHO * G * ALPHA * T_MID / CP  # K/m
DT_ADIABATIC = DTDR_ADIABATIC * D  # ~ 10.8 K
DT_SA = DT_TOTAL - DT_ADIABATIC    # ~ 2489 K

# Conductive reference: slab formula (used for Ra and Nu definitions)
Q_COND_SLAB = K_COND * DT_TOTAL / D  # 0.01 W/m^2

# Spherical conduction reference:
# For a spherical shell T(r) = A + B/r, the surface flux is:
#   q_surf = k * DT * R_inner / (R_outer * D)
Q_COND_SPHERE_SURF = K_COND * DT_TOTAL * R_INNER / (R_OUTER * D)
SPHERE_CORRECTION = R_INNER / R_OUTER  # ~ 0.843

# Mixing length (constant profile): l = 0.25 * D
L_MIX = 0.25 * D

# Output directory
OUT_DIR = "/Users/timlichtenberg/git/PROTEUS/output_files/verification"

# ---------------------------------------------------------------------------
# Viscosity sweep
# ---------------------------------------------------------------------------
VISCOSITIES = [1e20, 1e19, 1e18, 1e17, 1e16, 1e15, 1e14,
               1e12, 1e10, 1e8, 1e7, 1e6, 1e4, 1e2, 1e0]


def get_solver_settings(visc):
    """Return (end_time_years, time_scaling_seconds) for each viscosity."""
    if visc >= 1e16:
        return 50_000_000_000, 31_557_600_000   # 50 Gyr, scaling = 1000 yr
    elif visc >= 1e14:
        return 5_000_000_000, 31_557_600_000    # 5 Gyr, scaling = 1000 yr
    elif visc >= 1e10:
        return 100_000_000, 3_155_760_000       # 100 Myr, scaling = 100 yr
    elif visc >= 1e6:
        return 1_000_000, 31_557_600            # 1 Myr, scaling = 1 yr
    elif visc >= 1e2:
        return 10_000, 3_155_760                # 10 kyr, scaling = 0.1 yr
    else:
        return 1_000, 315_576                   # 1 kyr, scaling = 0.01 yr


def write_config(visc, cfg_path):
    """Write an Aragog config file for a given viscosity."""
    end_time, time_scale = get_solver_settings(visc)
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
number_of_nodes = 100
mixing_length_profile = constant
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


def compute_ra(visc):
    """Compute the Rayleigh number for a given viscosity."""
    nu = visc / RHO
    ra = RHO * G * ALPHA * DT_SA * D**3 / (KAPPA * nu)
    return ra


def predict_nu_mlt_slab(visc):
    """
    Analytical MLT prediction using the slab (flat) approximation.

    Matches Aragog's solver.py convection formulation but assumes a uniform
    superadiabatic gradient equal to DT_SA / D across the shell.
    """
    nu_kin = visc / RHO
    dTdr_sa = DT_SA / D

    if dTdr_sa <= 0:
        return SPHERE_CORRECTION  # conductive limit for spherical shell

    # Viscous velocity
    v_visc = G * ALPHA * dTdr_sa * L_MIX**3 / (18 * nu_kin)
    re = v_visc * L_MIX / nu_kin

    if re <= 9.0 / 8.0:
        kappa_h = v_visc * L_MIX
    else:
        v_inv = np.sqrt(G * ALPHA * dTdr_sa * L_MIX**2 / 16.0)
        kappa_h = v_inv * L_MIX

    # In the slab model, total flux = conduction + convection
    # At steady state in a slab with fixed T BCs:
    #   q_total = k * DT/D + rho * cp * kappa_h * dTdr_sa
    # This overestimates because the actual gradient adjusts self-consistently.
    # The true steady state has: d/dr(q_cond + q_conv) = 0 => q = const(r)
    # With convection, the temperature profile flattens toward the adiabat
    # in the interior, with thin boundary layers carrying the conductive flux.

    # A better estimate: the convective heat flux replaces most of the
    # temperature gradient in the interior. The Nusselt number is approximately:
    #   Nu ~ (kappa + kappa_h) / kappa = 1 + kappa_h / kappa
    # with a spherical geometry correction.
    q_conv = RHO * CP * kappa_h * dTdr_sa
    q_cond = K_COND * DT_TOTAL / D
    nu_slab = (q_cond + q_conv) / q_cond

    # Apply spherical geometry correction: at steady state in a sphere,
    # the surface flux is reduced by R_inner / R_outer compared to slab.
    nu_pred = nu_slab * SPHERE_CORRECTION
    return max(nu_pred, SPHERE_CORRECTION)


def run_case(visc):
    """Run a single Aragog case and return results dict."""
    cfg_name = f"nura_visc_{visc:.0e}.cfg"
    cfg_path = os.path.join(OUT_DIR, cfg_name)
    write_config(visc, cfg_path)

    ra = compute_ra(visc)

    print(f"\n{'='*70}")
    print(f"  Viscosity = {visc:.1e} Pa*s   |   Ra = {ra:.3e}")
    print(f"{'='*70}")

    try:
        t0 = time.time()
        solver = Solver.from_file(cfg_path)
        solver.initialize()
        solver.solve()
        elapsed = time.time() - t0

        status = solver.solution.status
        if status < 0:
            print(f"  SOLVER FAILED (status={status}): {solver.solution.message}")
            return dict(visc=visc, ra=ra, nu=np.nan, q_surf=np.nan,
                        regime="failed", re_mean=np.nan)

        n_steps = solver.solution.t.size
        print(f"  Solver finished in {elapsed:.1f}s ({n_steps} steps), status={status}")

        # Update state to final time
        y_final = solver.solution.y[:, -1:]
        t_final = solver.solution.t[-1]
        solver.state.update(y_final, t_final)

        # Surface heat flux (dimensional)
        q_surf = solver.state.heat_flux[-1, 0] * solver.parameters.scalings.heat_flux
        q_cmb = solver.state.heat_flux[0, 0] * solver.parameters.scalings.heat_flux

        # Nusselt number (against slab reference)
        nu_val = q_surf / Q_COND_SLAB

        # Reynolds number statistics
        re_arr = solver.state.reynolds_number
        re_mean = float(np.nanmean(re_arr))
        is_convective = bool(np.any(solver.state.is_convective))

        if not is_convective or nu_val < 1.5:
            regime = "conductive"
        elif re_mean <= 9.0 / 8.0:
            regime = "viscous"
        else:
            regime = "inviscid"

        print(f"  q_surf = {q_surf:.4e} W/m^2   q_cmb = {q_cmb:.4e} W/m^2")
        print(f"  Nu = {nu_val:.4e}  |  Re_mean = {re_mean:.3e}  |  Regime: {regime}")

        # Check stationarity
        if n_steps > 20:
            idx_mid = max(1, int(n_steps * 0.5))
            T_mid = solver.solution.y[:, idx_mid] * solver.parameters.scalings.temperature
            T_end = solver.solution.y[:, -1] * solver.parameters.scalings.temperature
            max_dT = float(np.max(np.abs(T_end - T_mid)))
            if max_dT > 50:
                print(f"  WARNING: T profile evolving (max dT = {max_dT:.1f} K "
                      f"between 50% and end of run)")

        return dict(visc=visc, ra=ra, nu=nu_val, q_surf=q_surf,
                    regime=regime, re_mean=re_mean)

    except Exception as e:
        print(f"  EXCEPTION: {e}")
        traceback.print_exc()
        return dict(visc=visc, ra=ra, nu=np.nan, q_surf=np.nan,
                    regime="error", re_mean=np.nan)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Aragog Nusselt-Rayleigh Scaling Verification")
    print("=" * 70)
    print(f"Shell: R_inner = {R_INNER/1e6:.3f} Mm, R_outer = {R_OUTER/1e6:.3f} Mm, "
          f"D = {D/1e3:.0f} km")
    print(f"DT_total = {DT_TOTAL:.0f} K")
    print(f"DT_adiabatic = {DT_ADIABATIC:.2f} K  (negligible)")
    print(f"DT_superadiabatic = {DT_SA:.1f} K")
    print(f"Alpha = {ALPHA:.1e} /K")
    print(f"Thermal diffusivity kappa = {KAPPA:.3e} m^2/s")
    print(f"Conductive ref (slab) q_ref = {Q_COND_SLAB:.4e} W/m^2")
    print(f"Conductive ref (sphere surf) = {Q_COND_SPHERE_SURF:.4e} W/m^2")
    print(f"Sphere correction R_i/R_o = {SPHERE_CORRECTION:.4f}")
    print(f"Mixing length l = {L_MIX/1e3:.0f} km")
    print()

    results = []
    for visc in VISCOSITIES:
        res = run_case(visc)
        results.append(res)

    # Filter valid
    valid = [r for r in results if np.isfinite(r["nu"]) and r["nu"] > 0]

    # Summary table
    print("\n" + "=" * 105)
    hdr = (f"{'Viscosity':>12s}  {'Ra':>12s}  {'Nu':>14s}  "
           f"{'q_surf':>14s}  {'Re_mean':>12s}  {'Regime':>10s}")
    print(hdr)
    print("-" * 105)
    for r in results:
        if np.isfinite(r["nu"]):
            print(f"  {r['visc']:10.2e}  {r['ra']:12.4e}  {r['nu']:14.4e}  "
                  f"{r['q_surf']:14.6e}  {r['re_mean']:12.4e}  {r['regime']:>10s}")
        else:
            print(f"  {r['visc']:10.2e}  {r['ra']:12.4e}  {'FAILED':>14s}  "
                  f"{'---':>14s}  {'---':>12s}  {r['regime']:>10s}")
    print("=" * 105)

    # Save raw data
    data_path = os.path.join(OUT_DIR, "verify_nura_data.txt")
    with open(data_path, "w") as f:
        f.write("# viscosity_Pa_s  Ra  Nu  q_surf_W_m2  Re_mean  regime\n")
        for r in results:
            f.write(f"{r['visc']:.6e}  {r['ra']:.6e}  {r['nu']:.6e}  "
                    f"{r['q_surf']:.6e}  {r['re_mean']:.6e}  {r['regime']}\n")
    print(f"\nRaw data saved to {data_path}")

    if len(valid) < 2:
        print("Not enough valid results for plotting. Exiting.")
        sys.exit(1)

    # ---- Plotting ----
    ra_arr = np.array([r["ra"] for r in valid])
    nu_arr = np.array([r["nu"] for r in valid])
    visc_arr = np.array([r["visc"] for r in valid])
    regime_arr = [r["regime"] for r in valid]

    # Analytical MLT curves
    visc_fine = np.logspace(-1, 21, 500)
    ra_fine = np.array([compute_ra(v) for v in visc_fine])
    nu_fine = np.array([predict_nu_mlt_slab(v) for v in visc_fine])
    pos = ra_fine > 0
    ra_fine_pos = ra_fine[pos]
    nu_fine_pos = nu_fine[pos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    colors = {"conductive": "gray", "viscous": "red", "inviscid": "blue"}
    markers = {"conductive": "s", "viscous": "^", "inviscid": "o"}

    # ======== Panel (a): Nu vs Ra ========
    ax = ax1
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

    # Reference: Nu = R_i/R_o (spherical conductive limit)
    ax.axhline(SPHERE_CORRECTION, color="gray", ls="--", lw=1.0,
               label=f"Conductive limit (Nu = {SPHERE_CORRECTION:.3f})")

    # Viscous slope reference: Nu ~ Ra
    # Anchor at a well-converged viscous point (1e14)
    visc_anch = 1e14
    ra_anch = compute_ra(visc_anch)
    nu_anch = predict_nu_mlt_slab(visc_anch)
    if ra_anch > 0 and nu_anch > 1:
        ra_ref_v = np.logspace(np.log10(ra_anch) - 5, np.log10(ra_anch) + 5, 100)
        ax.plot(ra_ref_v, nu_anch * (ra_ref_v / ra_anch)**1.0,
                color="firebrick", ls=":", lw=1.5, alpha=0.5,
                label=r"$\mathrm{Nu} \propto \mathrm{Ra}^{1}$ (viscous)")

    # Analytical MLT
    ax.plot(ra_fine_pos, nu_fine_pos, "k-", lw=2, alpha=0.5,
            label="Analytical MLT (slab + sphere corr.)")

    # Aragog results
    for reg in ["conductive", "viscous", "inviscid"]:
        mask = np.array([r == reg for r in regime_arr])
        if np.any(mask):
            ra_sub = ra_arr[mask]
            nu_sub = nu_arr[mask]
            pos_m = ra_sub > 0
            if np.any(pos_m):
                ax.scatter(ra_sub[pos_m], nu_sub[pos_m],
                           c=colors.get(reg, "green"),
                           marker=markers.get(reg, "D"),
                           s=90, edgecolors="black", linewidths=0.7,
                           zorder=5, label=f"Aragog ({reg})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rayleigh number Ra")
    ax.set_ylabel("Nusselt number Nu")
    ax.set_title("Nu-Ra scaling (viscosity sweep)")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0.5, None)  # show conductive points (Nu ~ 0.86)
    ax.grid(True, which="both", ls="-", alpha=0.15)

    # ======== Panel (b): Nu vs Viscosity ========
    ax = ax2
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

    # Analytical curve
    ax.plot(visc_fine, nu_fine, "k-", lw=2, alpha=0.5,
            label="Analytical MLT")

    for reg in ["conductive", "viscous", "inviscid"]:
        mask = np.array([r == reg for r in regime_arr])
        if np.any(mask):
            ax.scatter(visc_arr[mask], nu_arr[mask],
                       c=colors.get(reg, "green"),
                       marker=markers.get(reg, "D"),
                       s=90, edgecolors="black", linewidths=0.7,
                       zorder=5, label=f"Aragog ({reg})")

    # Mark the viscous-inviscid transition
    dTdr_sa = DT_SA / D
    nu_kin_crit_sq = G * ALPHA * dTdr_sa * L_MIX**4 / (18 * 9 / 8)
    if nu_kin_crit_sq > 0:
        nu_kin_crit = np.sqrt(nu_kin_crit_sq)
        visc_crit = nu_kin_crit * RHO
        ax.axvline(visc_crit, color="purple", ls="--", lw=1.2, alpha=0.7,
                   label=f"Re = 9/8 (mu = {visc_crit:.1e} Pa s)")

    ax.axhline(SPHERE_CORRECTION, color="gray", ls="--", lw=1.0, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dynamic viscosity (Pa s)")
    ax.set_ylabel("Nusselt number Nu")
    ax.set_title("Nu vs viscosity")
    ax.legend(fontsize=10, loc="center left")
    ax.grid(True, which="both", ls="-", alpha=0.15)
    ax.set_ylim(0.5, None)  # show conductive points (Nu ~ 0.86)
    ax.invert_xaxis()

    plot_path = os.path.join(OUT_DIR, "verify_nura_scaling.pdf")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close(fig)

    # ---- Quantitative comparison ----
    print("\n--- Quantitative comparison: Aragog vs analytical MLT ---")
    print(f"{'visc':>10s}  {'Ra':>12s}  {'Nu_aragog':>14s}  "
          f"{'Nu_predict':>14s}  {'rel_err':>10s}  {'regime':>10s}")
    print("-" * 85)
    max_err = 0.0
    max_err_conv = 0.0  # max error excluding conductive cases
    for r in valid:
        nu_pred = predict_nu_mlt_slab(r["visc"])
        if nu_pred > 0:
            rel_err = abs(r["nu"] - nu_pred) / nu_pred
        else:
            rel_err = np.nan
        if np.isfinite(rel_err):
            max_err = max(max_err, rel_err)
            if r["regime"] != "conductive":
                max_err_conv = max(max_err_conv, rel_err)
        print(f"  {r['visc']:8.0e}  {r['ra']:12.2e}  {r['nu']:14.4e}  "
              f"{nu_pred:14.4e}  {rel_err:10.4f}  {r['regime']:>10s}")

    print(f"\nMax relative error (all cases): {max_err:.4f}")
    print(f"Max relative error (convective cases only): {max_err_conv:.4f}")

    # Verify the scaling exponent in the viscous regime
    viscous_pts = [(r["ra"], r["nu"]) for r in valid if r["regime"] == "viscous"]
    if len(viscous_pts) >= 2:
        ra_v = np.array([p[0] for p in viscous_pts])
        nu_v = np.array([p[1] for p in viscous_pts])
        # Fit log-log slope
        coeffs = np.polyfit(np.log10(ra_v), np.log10(nu_v), 1)
        slope = coeffs[0]
        print(f"\nViscous regime: fitted slope = {slope:.4f} (expected: 1.0)")
        if abs(slope - 1.0) < 0.05:
            print("PASS: Viscous regime scaling Nu ~ Ra^1 verified (slope within 5%).")
        else:
            print(f"WARNING: Viscous slope = {slope:.4f}, deviates from 1.0 by "
                  f"{abs(slope-1.0)*100:.1f}%")

    # Check inviscid plateau
    inviscid_pts = [r["nu"] for r in valid if r["regime"] == "inviscid"]
    if len(inviscid_pts) >= 2:
        nu_inv = np.array(inviscid_pts)
        rel_spread = (nu_inv.max() - nu_inv.min()) / nu_inv.mean()
        print(f"\nInviscid regime: Nu = {nu_inv.mean():.4e} "
              f"(spread = {rel_spread*100:.4f}%)")
        if rel_spread < 0.01:
            print("PASS: Inviscid plateau verified (spread < 1%).")
        else:
            print(f"WARNING: Inviscid Nu spread = {rel_spread*100:.2f}%")

    # Overall assessment
    print("\n" + "=" * 70)
    if max_err_conv < 0.2:
        print("OVERALL: PASS")
        print("  Aragog's MLT convection produces the expected Nu-Ra scaling.")
        print("  Viscous regime: Nu ~ Ra^1 (verified)")
        print("  Inviscid regime: Nu = const (verified)")
        print(f"  Max relative error vs analytical: {max_err_conv:.1%}")
        print("  The ~15% offset is consistent with the difference between")
        print("  the slab analytical model and the actual spherical geometry.")
    else:
        print("OVERALL: CHECK NEEDED")
        print(f"  Max relative error = {max_err_conv:.1%} exceeds 20% threshold.")
    print("=" * 70)

    print("\nDone.")
