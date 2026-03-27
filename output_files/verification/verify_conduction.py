#!/usr/bin/env python3
"""
Verification: Steady-state spherical conduction in Aragog.

Compares Aragog's numerical solution for pure conduction in a spherical
shell against the analytical solution:

    T(r) = A/r + B

where:
    A = (T_a - T_b) * a * b / (b - a)
    B = (T_b * b - T_a * a) / (b - a)

for a shell with inner radius a, outer radius b, constant conductivity k,
fixed T_a at r=a and T_b at r=b, and no internal heating.

The heat flux (total power through any spherical surface) is:
    Q = 4 * pi * k * A   (constant, independent of r)

Produces:
    verify_conduction_profile.pdf     - T(r) overlay + residual
    verify_conduction_convergence.pdf - L2 error vs N (log-log)

All output goes to output_files/verification/.
"""
from __future__ import annotations

import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

# ---------------------------------------------------------------------------
# Physical and geometric parameters (must match the .cfg files)
# ---------------------------------------------------------------------------
R_INNER = 5371000.0       # m  (inner radius, a)
R_OUTER = 6371000.0       # m  (outer radius, b)
T_INNER = 4000.0          # K  (T at inner boundary)
T_OUTER = 1500.0          # K  (T at outer boundary)
K_COND  = 4.0             # W/m/K  (thermal conductivity)
RHO     = 4000.0          # kg/m^3
CP      = 1000.0          # J/kg/K

# Analytical coefficients
A_COEFF = (T_INNER - T_OUTER) * R_INNER * R_OUTER / (R_OUTER - R_INNER)
B_COEFF = (T_OUTER * R_OUTER - T_INNER * R_INNER) / (R_OUTER - R_INNER)
Q_ANALYTICAL = 4.0 * np.pi * K_COND * A_COEFF   # total power [W]


def analytical_T(r):
    """Analytical steady-state temperature at radius r."""
    return A_COEFF / r + B_COEFF


# ---------------------------------------------------------------------------
# Config file template
# ---------------------------------------------------------------------------
CFG_TEMPLATE = textwrap.dedent("""\
[scalings]
radius = 6371000
temperature = 4000
density = 4000
time = 31557600000

[solver]
start_time = 0
end_time = {end_time}
atol = 1e-12
rtol = 1e-12
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
number_of_nodes = {number_of_nodes}
mixing_length_profile = constant
core_density = 10738
surface_density = 4090
gravitational_acceleration = 9.81
adiabatic_bulk_modulus = 260E9

[energy]
conduction = True
convection = False
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
viscosity = 1E2
heat_capacity = 1000
melt_fraction = 1
thermal_conductivity = 4
thermal_expansivity = 1.0E-5

[phase_solid]
density = 4000
viscosity = 1E21
heat_capacity = 1000
melt_fraction = 0
thermal_conductivity = 4
thermal_expansivity = 1.0E-5

[phase_mixed]
latent_heat_of_fusion = 4e6
rheological_transition_melt_fraction = 0.4
rheological_transition_width = 0.15
solidus = data/test/solidus_1d_lookup.dat
liquidus = data/test/liquidus_1d_lookup.dat
phase = solid
phase_transition_width = 0.01
grain_size = 1.0E-3
""")


def write_config(cfg_path, number_of_nodes, end_time_years):
    """Write an Aragog .cfg file with the given resolution and end time."""
    cfg_text = CFG_TEMPLATE.format(
        number_of_nodes=number_of_nodes,
        end_time=int(end_time_years),
    )
    with open(cfg_path, "w") as f:
        f.write(cfg_text)


def run_aragog(cfg_path):
    """
    Run Aragog from the given config file and return
    (r_staggered [m], T_final [K], t_final [yr]).

    Parameters
    ----------
    cfg_path : str
        Absolute path to the .cfg file.

    Returns
    -------
    r : ndarray, shape (N-1,)
        Staggered radii in metres.
    T : ndarray, shape (N-1,)
        Final temperature profile in Kelvin.
    t_yr : float
        Final time in years.
    """
    # Aragog expects to find data/ relative to the aragog submodule root
    aragog_root = "/Users/timlichtenberg/git/aragog"
    orig_dir = os.getcwd()
    os.chdir(aragog_root)

    try:
        from aragog.solver import Solver

        solver = Solver.from_file(cfg_path)
        solver.initialize()
        solver.solve()

        # Extract results
        scalings = solver.parameters.scalings

        # Radii: staggered mesh, shape (N-1, 1) -> flatten to (N-1,)
        r_stag = solver.evaluator.mesh.staggered.radii.ravel() * scalings.radius

        # Temperature: solution.y shape (N-1, n_times), last column is final
        T_final = solver.solution.y[:, -1] * scalings.temperature

        # Time: solution.t is nondimensional, convert to years
        t_final_yr = solver.solution.t[-1] * scalings.time_years

        return r_stag, T_final, t_final_yr

    finally:
        os.chdir(orig_dir)


def compute_errors(r, T_numerical, T_exact):
    """
    Compute L2 and Linf errors between numerical and exact temperature.

    Uses volume-weighted L2 norm for proper spatial convergence measure.

    Parameters
    ----------
    r : ndarray
        Radii at staggered nodes (cell centers).
    T_numerical : ndarray
        Numerical temperature.
    T_exact : ndarray
        Analytical temperature.

    Returns
    -------
    L2 : float
        Volume-weighted RMS error.
    Linf : float
        Maximum absolute error.
    """
    diff = T_numerical - T_exact
    # Volume-weighted L2: weight each cell by r^2 (proportional to shell volume)
    weights = r**2
    L2 = np.sqrt(np.sum(weights * diff**2) / np.sum(weights))
    Linf = np.max(np.abs(diff))
    return L2, Linf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    out_dir = "/Users/timlichtenberg/git/PROTEUS/output_files/verification"
    os.makedirs(out_dir, exist_ok=True)

    # Thermal diffusion timescale: tau ~ rho*cp*D^2/k
    D = R_OUTER - R_INNER  # 1e6 m
    tau_diff = RHO * CP * D**2 / K_COND  # ~1e15 s ~ 31.7 Myr
    tau_diff_yr = tau_diff / 3.1557e7
    print(f"Thermal diffusion timescale: {tau_diff_yr:.2e} yr")

    # Use 3x the diffusion timescale to ensure steady state
    end_time_years = int(3 * tau_diff_yr)
    print(f"Integration end time: {end_time_years:.2e} yr")

    # ------------------------------------------------------------------
    # 1. Reference run at N=100 for the profile plot
    # ------------------------------------------------------------------
    N_ref = 100
    cfg_ref = os.path.join(out_dir, "conduction_steady.cfg")
    write_config(cfg_ref, N_ref, end_time_years)

    print(f"\nRunning reference case (N={N_ref})...")
    r_ref, T_ref, t_ref = run_aragog(cfg_ref)
    T_exact_ref = analytical_T(r_ref)
    L2_ref, Linf_ref = compute_errors(r_ref, T_ref, T_exact_ref)
    print(f"  Final time: {t_ref:.4e} yr")
    print(f"  L2 error:   {L2_ref:.4e} K")
    print(f"  Linf error: {Linf_ref:.4e} K")

    # ------------------------------------------------------------------
    # 2. Profile plot
    # ------------------------------------------------------------------
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 7), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.08}
    )

    r_km = r_ref / 1e3
    r_fine = np.linspace(R_INNER, R_OUTER, 500)
    r_fine_km = r_fine / 1e3

    # Top panel: T(r)
    ax_top.plot(r_fine_km, analytical_T(r_fine), "k-", lw=1.8,
                label="Analytical: $T = A/r + B$")
    ax_top.plot(r_km, T_ref, "o", color="#b2182b", ms=5, alpha=0.7,
                label=f"Aragog ($N={N_ref}$)")
    ax_top.set_ylabel("Temperature [K]")
    ax_top.legend(loc="upper right")
    ax_top.set_title("Steady-state conduction in a spherical shell")
    ax_top.text(0.02, 0.95, "(a)", transform=ax_top.transAxes,
                fontsize=14, fontweight="bold", va="top")
    ax_top.text(0.02, 0.05,
                f"$a$ = {R_INNER/1e3:.0f} km, $b$ = {R_OUTER/1e3:.0f} km\n"
                f"$T_a$ = {T_INNER:.0f} K, $T_b$ = {T_OUTER:.0f} K\n"
                f"$k$ = {K_COND} W/m/K",
                transform=ax_top.transAxes, fontsize=10,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Bottom panel: residual
    residual = T_ref - T_exact_ref
    ax_bot.plot(r_km, residual, ".-", color="#2166ac", ms=5, lw=0.8)
    ax_bot.axhline(0, color="#636363", lw=0.5, ls="--")
    ax_bot.set_xlabel("Radius [km]")
    ax_bot.set_ylabel("$T_{\\mathrm{num}} - T_{\\mathrm{exact}}$ [K]")
    ax_bot.text(0.02, 0.95, "(b)", transform=ax_bot.transAxes,
                fontsize=14, fontweight="bold", va="top")
    ax_bot.text(0.02, 0.90,
                f"$L_2$ = {L2_ref:.2e} K\n$L_\\infty$ = {Linf_ref:.2e} K",
                transform=ax_bot.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

    fig.savefig(os.path.join(out_dir, "verify_conduction_profile.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_dir}/verify_conduction_profile.pdf")

    # ------------------------------------------------------------------
    # 3. Convergence study
    # ------------------------------------------------------------------
    N_values = [25, 50, 100, 200, 400]
    L2_errors = []
    Linf_errors = []

    print("\nConvergence study:")
    print(f"{'N':>6s}  {'N_cells':>7s}  {'L2 [K]':>12s}  {'Linf [K]':>12s}")
    print("-" * 45)

    for N in N_values:
        cfg_path = os.path.join(out_dir, f"conduction_N{N}.cfg")
        write_config(cfg_path, N, end_time_years)

        print(f"  Running N={N}...", end="", flush=True)
        r, T, t = run_aragog(cfg_path)
        T_exact = analytical_T(r)
        L2, Linf = compute_errors(r, T, T_exact)
        L2_errors.append(L2)
        Linf_errors.append(Linf)
        print(f"  done (t_final={t:.2e} yr)")
        print(f"{N:>6d}  {N-1:>7d}  {L2:>12.4e}  {Linf:>12.4e}")

    # Compute convergence orders
    N_arr = np.array(N_values, dtype=float)
    L2_arr = np.array(L2_errors)
    Linf_arr = np.array(Linf_errors)

    print("\nConvergence orders (successive pairs):")
    print(f"{'N_i':>6s} -> {'N_j':>6s}  {'L2 order':>10s}  {'Linf order':>10s}")
    for i in range(len(N_values) - 1):
        # N is proportional to 1/h, so error ~ h^p means error ~ N^{-p}
        # p = -log(E_j/E_i) / log(N_j/N_i)
        order_L2 = -np.log(L2_arr[i+1] / L2_arr[i]) / np.log(N_arr[i+1] / N_arr[i])
        order_Linf = -np.log(Linf_arr[i+1] / Linf_arr[i]) / np.log(N_arr[i+1] / N_arr[i])
        print(f"{N_values[i]:>6d} -> {N_values[i+1]:>6d}  {order_L2:>10.2f}  {order_Linf:>10.2f}")

    # Overall order from log-log fit
    coeffs_L2 = np.polyfit(np.log(N_arr), np.log(L2_arr), 1)
    coeffs_Linf = np.polyfit(np.log(N_arr), np.log(Linf_arr), 1)
    print("\nOverall convergence order (log-log fit):")
    print(f"  L2:   {-coeffs_L2[0]:.2f}")
    print(f"  Linf: {-coeffs_Linf[0]:.2f}")

    # ------------------------------------------------------------------
    # 4. Convergence plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.loglog(N_arr, L2_arr, "o-", color="#2166ac", ms=8, lw=1.8, label="$L_2$ error")
    ax.loglog(N_arr, Linf_arr, "s--", color="#b2182b", ms=8, lw=1.8, label="$L_\\infty$ error")

    # Reference slope: 2nd order
    N_slope = np.array([N_arr[0], N_arr[-1]])
    ref_L2 = L2_arr[0] * (N_slope / N_slope[0])**(-2)
    ax.loglog(N_slope, ref_L2, ":", color="#636363", lw=1.5, alpha=0.6,
              label="$\\propto N^{-2}$ (2nd order)")

    ax.set_xlabel("Number of basic nodes $N$")
    ax.set_ylabel("Temperature error [K]")
    ax.set_title("Convergence: spherical conduction steady state")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax.text(0.98, 0.98,
            f"Fitted order (L2): {-coeffs_L2[0]:.2f}\n"
            f"Fitted order (Linf): {-coeffs_Linf[0]:.2f}",
            transform=ax.transAxes, fontsize=10,
            ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.savefig(os.path.join(out_dir, "verify_conduction_convergence.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_dir}/verify_conduction_convergence.pdf")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Analytical heat power Q = {Q_ANALYTICAL:.4e} W")
    print(f"Reference (N=100):  L2 = {L2_ref:.4e} K,  Linf = {Linf_ref:.4e} K")
    print(f"Convergence order:  L2 ~ N^{{{-coeffs_L2[0]:.2f}}},  "
          f"Linf ~ N^{{{-coeffs_Linf[0]:.2f}}}")
    print("Expected:           ~N^{-2} for centered finite-volume")
    print("=" * 60)

    # Save raw data for replotting
    data_path = os.path.join(out_dir, "convergence_data.npz")
    np.savez(data_path,
             N_values=N_arr, L2_errors=L2_arr, Linf_errors=Linf_arr,
             r_ref=r_ref, T_ref=T_ref, T_exact_ref=T_exact_ref,
             order_L2=-coeffs_L2[0], order_Linf=-coeffs_Linf[0])
    print(f"\nRaw data saved to: {data_path}")


if __name__ == "__main__":
    main()
