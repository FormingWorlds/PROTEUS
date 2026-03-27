#!/usr/bin/env python3
"""
Verification Tests 5 and 6 for Aragog:

TEST 5: Method of Manufactured Solutions (MMS) with Uniform Heating
    Spherical shell with constant k, rho, cp, uniform volumetric heating
    Q = rho*H, and fixed-T BCs. Analytical steady-state solution exists.
    Convergence study at N = 25, 50, 100, 200.

TEST 6: Grey-Body Radiative Steady State
    Shell with uniform tidal heating H, zero CMB flux, and grey-body outer
    BC. In steady state the surface temperature satisfies energy balance:
        epsilon * sigma * (T_surf^4 - T_eq^4) * A_surf = rho * H * V_shell

All output goes to output_files/verification/.
"""
from __future__ import annotations

import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as spc

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

# ===================================================================
# Shared utilities
# ===================================================================
ARAGOG_ROOT = "/Users/timlichtenberg/git/aragog"
OUT_DIR = "/Users/timlichtenberg/git/PROTEUS/output_files/verification"


def run_aragog(cfg_path, tidal_H_Wkg=None):
    """
    Run Aragog and return the solver object.

    Parameters
    ----------
    cfg_path : str
        Absolute path to the .cfg file.
    tidal_H_Wkg : float or None
        If not None, set tidal heating in W/kg before initialize().
        The value is divided by the power_per_mass scaling to convert to
        non-dimensional units (matching what scale_attributes would do).

    Returns
    -------
    solver : Solver
        The solved Aragog solver object.
    """
    orig_dir = os.getcwd()
    os.chdir(ARAGOG_ROOT)

    try:
        from aragog.solver import Solver

        solver = Solver.from_file(cfg_path)

        # Set tidal heating programmatically (already-scaled parameters)
        if tidal_H_Wkg is not None:
            ppm = solver.parameters.scalings.power_per_mass
            solver.parameters.energy.tidal = True
            solver.parameters.energy.tidal_array = np.array(
                [tidal_H_Wkg / ppm], dtype=float
            )

        solver.initialize()
        solver.solve()
        return solver

    finally:
        os.chdir(orig_dir)


def compute_errors(r, T_numerical, T_exact):
    """
    Volume-weighted L2 and Linf errors.

    Parameters
    ----------
    r : ndarray
        Radii at staggered nodes.
    T_numerical, T_exact : ndarray
        Temperature arrays.

    Returns
    -------
    L2, Linf : float
    """
    diff = T_numerical - T_exact
    weights = r**2
    L2 = np.sqrt(np.sum(weights * diff**2) / np.sum(weights))
    Linf = np.max(np.abs(diff))
    return L2, Linf


# ===================================================================
# TEST 5: MMS with Uniform Heating
# ===================================================================

# Physical parameters
R_INNER = 5371000.0   # m
R_OUTER = 6371000.0   # m
T_INNER = 4000.0       # K  (fixed T at inner boundary, BC type 3)
T_OUTER = 1500.0       # K  (fixed T at outer boundary, BC type 5)
K_COND = 4.0           # W/m/K
RHO = 4000.0           # kg/m^3
CP = 1000.0            # J/kg/K
H_MMS = 5e-8           # W/kg (tidal heating rate)
RHO_H = RHO * H_MMS   # W/m^3


def analytical_T_mms(r):
    """
    Analytical steady-state T(r) for a spherical shell with uniform
    volumetric heating Q = rho*H and fixed-T BCs at r=a and r=b.

    The governing equation in spherical coordinates is:
        d/dr(r^2 k dT/dr) = -rho*H r^2

    Integrating twice:
        T(r) = -(rho*H)/(6k) r^2 - A1/(k r) + A2

    where A1 and A2 are determined by T(a) = T_a, T(b) = T_b.
    """
    a, b = R_INNER, R_OUTER
    k = K_COND
    qH = RHO_H

    # From the two BC equations:
    # T_a = -(qH)/(6k) a^2 - A1/(k a) + A2
    # T_b = -(qH)/(6k) b^2 - A1/(k b) + A2
    # Subtracting:
    # T_a - T_b = -(qH)/(6k)(a^2 - b^2) - A1/k (1/a - 1/b)
    #           = (qH)/(6k)(b^2 - a^2) + A1/k (1/b - 1/a)
    # So: A1 = k * [T_a - T_b - (qH)/(6k)(b^2 - a^2)] / (1/b - 1/a)

    inv_diff = 1.0 / b - 1.0 / a  # negative
    A1 = k * (T_INNER - T_OUTER - (qH / (6 * k)) * (b**2 - a**2)) / inv_diff
    A2 = T_INNER + (qH / (6 * k)) * a**2 + A1 / (k * a)

    T = -(qH / (6 * k)) * r**2 - A1 / (k * r) + A2
    return T


# Config template for Test 5
CFG_MMS = textwrap.dedent("""\
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
tidal = True

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


def run_test5():
    """MMS with uniform heating: profile + convergence study."""
    print("=" * 60)
    print("TEST 5: MMS with Uniform Heating")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Diffusion timescale
    D = R_OUTER - R_INNER
    tau_diff_s = RHO * CP * D**2 / K_COND
    tau_diff_yr = tau_diff_s / spc.Julian_year
    end_time_yr = int(3 * tau_diff_yr)
    print(f"Diffusion timescale: {tau_diff_yr:.2e} yr")
    print(f"Integration time: {end_time_yr:.2e} yr")

    # Verify analytical solution at boundaries
    T_check_a = analytical_T_mms(R_INNER)
    T_check_b = analytical_T_mms(R_OUTER)
    print(f"\nAnalytical check: T(a)={T_check_a:.4f} K (expect {T_INNER})")
    print(f"Analytical check: T(b)={T_check_b:.4f} K (expect {T_OUTER})")

    # ---------------------------------------------------------------
    # Reference run at N=100
    # ---------------------------------------------------------------
    N_ref = 100
    cfg_path = os.path.join(OUT_DIR, "mms_N100.cfg")
    with open(cfg_path, "w") as f:
        f.write(CFG_MMS.format(number_of_nodes=N_ref, end_time=end_time_yr))

    print(f"\nRunning reference case N={N_ref}...")
    solver = run_aragog(cfg_path, tidal_H_Wkg=H_MMS)
    sc = solver.parameters.scalings
    r_stag = solver.evaluator.mesh.staggered.radii.ravel() * sc.radius
    T_num = solver.solution.y[:, -1] * sc.temperature
    T_exact = analytical_T_mms(r_stag)
    L2_ref, Linf_ref = compute_errors(r_stag, T_num, T_exact)
    t_final = solver.solution.t[-1] * sc.time_years
    print(f"  Final time: {t_final:.4e} yr")
    print(f"  L2 error:   {L2_ref:.4e} K")
    print(f"  Linf error: {Linf_ref:.4e} K")

    # ---------------------------------------------------------------
    # Profile plot
    # ---------------------------------------------------------------
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 7), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.08}
    )

    r_km = r_stag / 1e3
    r_fine = np.linspace(R_INNER, R_OUTER, 500)
    r_fine_km = r_fine / 1e3

    ax_top.plot(r_fine_km, analytical_T_mms(r_fine), "k-", lw=1.8,
                label="Analytical (conduction + heating)")
    ax_top.plot(r_km, T_num, "o", color="#b2182b", ms=5, alpha=0.7,
                label=f"Aragog ($N={N_ref}$)")
    ax_top.set_ylabel("Temperature [K]")
    ax_top.legend(loc="upper right")
    ax_top.set_title("Steady-state conduction with uniform heating")
    ax_top.text(0.02, 0.95, "(a)", transform=ax_top.transAxes,
                fontsize=14, fontweight="bold", va="top")
    ax_top.text(
        0.02, 0.05,
        f"$a$ = {R_INNER/1e3:.0f} km, $b$ = {R_OUTER/1e3:.0f} km\n"
        f"$T_a$ = {T_INNER:.0f} K, $T_b$ = {T_OUTER:.0f} K\n"
        f"$k$ = {K_COND} W/m/K, H = {H_MMS:.0e} W/kg",
        transform=ax_top.transAxes, fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    residual = T_num - T_exact
    ax_bot.plot(r_km, residual, ".-", color="#2166ac", ms=5, lw=0.8)
    ax_bot.axhline(0, color="#636363", lw=0.5, ls="--")
    ax_bot.set_xlabel("Radius [km]")
    ax_bot.set_ylabel("$T_{\\mathrm{num}} - T_{\\mathrm{exact}}$ [K]")
    ax_bot.text(0.02, 0.95, "(b)", transform=ax_bot.transAxes,
                fontsize=14, fontweight="bold", va="top")
    ax_bot.text(
        0.02, 0.90,
        f"$L_2$ = {L2_ref:.2e} K\n$L_\\infty$ = {Linf_ref:.2e} K",
        transform=ax_bot.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    fig.savefig(os.path.join(OUT_DIR, "verify_mms_heating.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR}/verify_mms_heating.pdf")

    # ---------------------------------------------------------------
    # Convergence study
    # ---------------------------------------------------------------
    N_values = [25, 50, 100, 200]
    L2_errors = []
    Linf_errors = []

    print("\nConvergence study:")
    print(f"{'N':>6s}  {'L2 [K]':>12s}  {'Linf [K]':>12s}")
    print("-" * 35)

    for N in N_values:
        cfg_path = os.path.join(OUT_DIR, f"mms_N{N}.cfg")
        with open(cfg_path, "w") as f:
            f.write(CFG_MMS.format(number_of_nodes=N, end_time=end_time_yr))

        print(f"  Running N={N}...", end="", flush=True)
        slv = run_aragog(cfg_path, tidal_H_Wkg=H_MMS)
        sc2 = slv.parameters.scalings
        r = slv.evaluator.mesh.staggered.radii.ravel() * sc2.radius
        T = slv.solution.y[:, -1] * sc2.temperature
        Te = analytical_T_mms(r)
        L2, Linf = compute_errors(r, T, Te)
        L2_errors.append(L2)
        Linf_errors.append(Linf)
        print("  done")
        print(f"{N:>6d}  {L2:>12.4e}  {Linf:>12.4e}")

    N_arr = np.array(N_values, dtype=float)
    L2_arr = np.array(L2_errors)
    Linf_arr = np.array(Linf_errors)

    # Pairwise convergence orders
    print("\nConvergence orders (successive pairs):")
    for i in range(len(N_values) - 1):
        o_L2 = -np.log(L2_arr[i + 1] / L2_arr[i]) / np.log(
            N_arr[i + 1] / N_arr[i]
        )
        o_Li = -np.log(Linf_arr[i + 1] / Linf_arr[i]) / np.log(
            N_arr[i + 1] / N_arr[i]
        )
        print(f"  {N_values[i]:>4d} -> {N_values[i+1]:>4d}: "
              f"L2 order = {o_L2:.2f}, Linf order = {o_Li:.2f}")

    coeffs_L2 = np.polyfit(np.log(N_arr), np.log(L2_arr), 1)
    coeffs_Linf = np.polyfit(np.log(N_arr), np.log(Linf_arr), 1)
    order_L2 = -coeffs_L2[0]
    order_Linf = -coeffs_Linf[0]
    print(f"\nOverall fitted order:  L2 ~ N^{{{-coeffs_L2[0]:.2f}}}, "
          f"Linf ~ N^{{{-coeffs_Linf[0]:.2f}}}")

    # Convergence plot (added to mms figure as second page)
    fig2, ax2 = plt.subplots(figsize=(7, 5.5))
    ax2.loglog(N_arr, L2_arr, "o-", color="#2166ac", ms=8, lw=1.8, label="$L_2$ error")
    ax2.loglog(N_arr, Linf_arr, "s--", color="#b2182b", ms=8, lw=1.8, label="$L_\\infty$ error")
    N_slope = np.array([N_arr[0], N_arr[-1]])
    ref_L2 = L2_arr[0] * (N_slope / N_slope[0]) ** (-2)
    ax2.loglog(N_slope, ref_L2, ":", color="#636363", lw=1.5, alpha=0.6,
               label="$\\propto N^{-2}$ (2nd order)")
    ax2.set_xlabel("Number of basic nodes $N$")
    ax2.set_ylabel("Temperature error [K]")
    ax2.set_title("Convergence: MMS with uniform heating")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    ax2.text(
        0.98, 0.98,
        f"Fitted order (L2): {order_L2:.2f}\n"
        f"Fitted order (Linf): {order_Linf:.2f}",
        transform=ax2.transAxes, fontsize=10,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    fig2.savefig(os.path.join(OUT_DIR, "verify_mms_convergence.pdf"),
                 bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {OUT_DIR}/verify_mms_convergence.pdf")

    # Save convergence data
    data_path = os.path.join(OUT_DIR, "mms_convergence_data.txt")
    with open(data_path, "w") as f:
        f.write("# MMS Convergence Data\n")
        f.write(f"# H = {H_MMS} W/kg, k = {K_COND} W/m/K\n")
        f.write(f"# Overall fitted L2 order = {order_L2:.4f}\n")
        f.write(f"# Overall fitted Linf order = {order_Linf:.4f}\n")
        f.write(f"# {'N':>6s}  {'L2_error_K':>14s}  {'Linf_error_K':>14s}\n")
        for i, N in enumerate(N_values):
            f.write(f"  {N:>6d}  {L2_arr[i]:>14.6e}  {Linf_arr[i]:>14.6e}\n")
    print(f"Saved: {data_path}")

    # Pass/fail
    pass5 = order_L2 > 1.8
    print(f"\n*** TEST 5 {'PASS' if pass5 else 'FAIL'}: "
          f"L2 convergence order = {order_L2:.2f} "
          f"(expected >= 1.8 for 2nd-order scheme) ***")

    return pass5, order_L2, order_Linf


# ===================================================================
# TEST 6: Grey-Body Radiative Steady State
# ===================================================================

# Physical parameters for Test 6
R_INNER_6 = 5371000.0   # m
R_OUTER_6 = 6371000.0   # m
RHO_6 = 4000.0
CP_6 = 1000.0
K_COND_6 = 4.0
ALPHA_6 = 1e-5
EMISSIVITY = 1.0
T_EQ = 273.0            # K
H_TIDAL = 1e-3          # W/kg
SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant, W/m^2/K^4

# Shell geometry
V_SHELL = (4.0 / 3.0) * np.pi * (R_OUTER_6**3 - R_INNER_6**3)
A_SURF = 4.0 * np.pi * R_OUTER_6**2

# Analytical steady-state surface temperature
T_SURF_ANALYTICAL = (
    T_EQ**4 + RHO_6 * H_TIDAL * V_SHELL / (A_SURF * EMISSIVITY * SIGMA)
) ** 0.25

# Config template for Test 6: grey-body outer BC, zero flux inner BC
CFG_GREY = textwrap.dedent("""\
[scalings]
radius = 6371000
temperature = 4000
density = 4000
time = 3155760

[solver]
start_time = 0
end_time = {end_time}
atol = 1e-8
rtol = 1e-8
tsurf_poststep_change = 200
event_triggering = False

[boundary_conditions]
outer_boundary_condition = 1
outer_boundary_value = 0
inner_boundary_condition = 2
inner_boundary_value = 0
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
convection = True
gravitational_separation = False
mixing = False
radionuclides = False
dilatation = False
tidal = True

[initial_condition]
surface_temperature = {T_init}
basal_temperature = {T_init}

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
phase = liquid
phase_transition_width = 0.01
grain_size = 1.0E-3
""")


def run_test6():
    """Grey-body radiative steady state verification."""
    print("\n" + "=" * 60)
    print("TEST 6: Grey-Body Radiative Steady State")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Shell volume:  V = {V_SHELL:.4e} m^3")
    print(f"Surface area:  A = {A_SURF:.4e} m^2")
    print(f"Tidal heating: H = {H_TIDAL:.0e} W/kg")
    print(f"Total heating power: P = rho*H*V = {RHO_6*H_TIDAL*V_SHELL:.4e} W")
    print(f"Analytical T_surf (steady state): {T_SURF_ANALYTICAL:.2f} K")

    # Verification: at T_surf_analytical, does power balance hold?
    P_out = EMISSIVITY * SIGMA * (T_SURF_ANALYTICAL**4 - T_EQ**4) * A_SURF
    P_in = RHO_6 * H_TIDAL * V_SHELL
    print(f"Check: P_in = {P_in:.4e} W, P_out = {P_out:.4e} W, "
          f"ratio = {P_out/P_in:.6f}")

    # ---------------------------------------------------------------
    # (a) Heated case: H = 1e-3 W/kg, start from T_init near expected
    # ---------------------------------------------------------------
    N_ref = 50
    T_init = 500.0  # K, near expected steady state
    end_time_yr = 50000  # years, long enough for convective equilibration

    cfg_path_heat = os.path.join(OUT_DIR, "greybody_heated.cfg")
    with open(cfg_path_heat, "w") as f:
        f.write(CFG_GREY.format(
            number_of_nodes=N_ref,
            end_time=end_time_yr,
            T_init=T_init,
        ))

    print(f"\nRunning heated case (H={H_TIDAL:.0e} W/kg, "
          f"T_init={T_init} K, N={N_ref})...")
    solver_heat = run_aragog(cfg_path_heat, tidal_H_Wkg=H_TIDAL)
    sc = solver_heat.parameters.scalings

    # Extract time series of surface temperature
    n_times = solver_heat.solution.y.shape[1]
    T_surf_series = np.zeros(n_times)
    P_out_series = np.zeros(n_times)
    t_yr = solver_heat.solution.t * sc.time_years

    for i in range(n_times):
        solver_heat.state.update(
            solver_heat.solution.y[:, i:i + 1],
            solver_heat.solution.t[i],
        )
        solver_heat.evaluator.boundary_conditions.apply_flux_boundary_conditions(
            solver_heat.state
        )
        T_surf_series[i] = solver_heat.state.top_temperature.ravel()[0] * sc.temperature
        q_surf = solver_heat.state.heat_flux[-1, 0] * sc.heat_flux
        P_out_series[i] = q_surf * A_SURF

    T_surf_final = T_surf_series[-1]
    P_in_total = RHO_6 * H_TIDAL * V_SHELL

    # Final T(r) profile
    r_stag = solver_heat.evaluator.mesh.staggered.radii.ravel() * sc.radius
    T_final_profile = solver_heat.solution.y[:, -1] * sc.temperature

    T_err = abs(T_surf_final - T_SURF_ANALYTICAL) / T_SURF_ANALYTICAL * 100
    print(f"  Final T_surf = {T_surf_final:.2f} K "
          f"(analytical: {T_SURF_ANALYTICAL:.2f} K)")
    print(f"  Relative error: {T_err:.2f}%")
    print(f"  Final P_out = {P_out_series[-1]:.4e} W "
          f"(expected: {P_in_total:.4e} W)")

    # ---------------------------------------------------------------
    # (b) Cooling case: H = 0, start hot, should cool to T_eq
    # ---------------------------------------------------------------
    T_init_cool = 4000.0
    end_time_cool = 500000  # years

    CFG_COOL = CFG_GREY.replace("tidal = True", "tidal = False")
    cfg_path_cool = os.path.join(OUT_DIR, "greybody_cooling.cfg")
    with open(cfg_path_cool, "w") as f:
        f.write(CFG_COOL.format(
            number_of_nodes=N_ref,
            end_time=end_time_cool,
            T_init=T_init_cool,
        ))

    print(f"\nRunning cooling case (H=0, T_init={T_init_cool} K)...")
    solver_cool = run_aragog(cfg_path_cool, tidal_H_Wkg=None)
    sc_c = solver_cool.parameters.scalings

    n_times_c = solver_cool.solution.y.shape[1]
    T_surf_cool = np.zeros(n_times_c)
    t_yr_cool = solver_cool.solution.t * sc_c.time_years

    for i in range(n_times_c):
        solver_cool.state.update(
            solver_cool.solution.y[:, i:i + 1],
            solver_cool.solution.t[i],
        )
        solver_cool.evaluator.boundary_conditions.apply_flux_boundary_conditions(
            solver_cool.state
        )
        T_surf_cool[i] = solver_cool.state.top_temperature.ravel()[0] * sc_c.temperature

    print(f"  Final T_surf (cooling): {T_surf_cool[-1]:.2f} K "
          f"(should approach T_eq = {T_EQ} K)")
    print(f"  Final time: {t_yr_cool[-1]:.2e} yr")

    # Energy conservation check for cooling: total energy lost = integral of
    # surface flux. Compute both from the solution.
    V_cells_c = solver_cool.evaluator.mesh.basic.volume.ravel() * sc_c.radius**3

    T_init_profile = solver_cool.solution.y[:, 0] * sc_c.temperature
    T_final_cool = solver_cool.solution.y[:, -1] * sc_c.temperature

    E_init = np.sum(RHO_6 * CP_6 * T_init_profile * V_cells_c)
    E_final = np.sum(RHO_6 * CP_6 * T_final_cool * V_cells_c)
    dE_internal = E_init - E_final  # energy lost from interior

    print("  Energy check (cooling):")
    print(f"    E_init = {E_init:.4e} J, E_final = {E_final:.4e} J")
    print(f"    dE = {dE_internal:.4e} J")

    # ---------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("Grey-body radiative steady state verification", fontsize=15)

    # Panel (a): T_surf(t) for heated case
    ax = axes[0]
    ax.plot(t_yr, T_surf_series, "-", color="#2166ac", lw=1.5, label=r"Aragog $T_\mathrm{surf}(t)$")
    ax.axhline(T_SURF_ANALYTICAL, color="#b2182b", ls="--", lw=1.8,
               label=f"Analytical: {T_SURF_ANALYTICAL:.1f} K")
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("Surface temperature [K]")
    ax.legend(fontsize=10)
    ax.set_title("(a) $T_\\mathrm{surf}$ to steady state")
    ax.text(0.02, 0.02,
            f"$H$ = {H_TIDAL:.0e} W/kg\n"
            f"Final $T_\\mathrm{{surf}}$ = {T_surf_final:.1f} K\n"
            f"Error = {T_err:.2f}%",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel (b): T(r) at steady state
    ax = axes[1]
    r_km = r_stag / 1e3
    ax.plot(r_km, T_final_profile, "-", color="#2166ac", lw=1.5, label="Final $T(r)$")
    ax.axhline(T_SURF_ANALYTICAL, color="#b2182b", ls="--", lw=1, alpha=0.7,
               label=r"Analytical $T_\mathrm{surf}$")
    ax.set_xlabel("Radius [km]")
    ax.set_ylabel("Temperature [K]")
    ax.legend(fontsize=10)
    ax.set_title("(b) Steady-state $T(r)$ profile")

    # Panel (c): Power balance
    ax = axes[2]
    ax.plot(t_yr, P_out_series, "-", color="#2166ac", lw=1.5,
            label=r"$\epsilon\sigma T_s^4 A_\mathrm{surf}$")
    ax.axhline(P_in_total, color="#b2182b", ls="--", lw=1.8,
               label=f"$\\rho H V$ = {P_in_total:.2e} W")
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("Power [W]")
    ax.legend(fontsize=10)
    ax.set_title("(c) Heating vs radiation")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "verify_greybody_steady.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR}/verify_greybody_steady.pdf")

    # Pass/fail
    pass6_heat = T_err < 5.0  # 5% tolerance
    pass6_cool = T_surf_cool[-1] < T_EQ + 50  # should be near T_eq
    pass6 = pass6_heat and pass6_cool

    print(f"\n*** TEST 6a (heated steady state) "
          f"{'PASS' if pass6_heat else 'FAIL'}: "
          f"T_surf error = {T_err:.2f}% (threshold: 5%) ***")
    print(f"*** TEST 6b (cooling to T_eq) "
          f"{'PASS' if pass6_cool else 'FAIL'}: "
          f"Final T_surf = {T_surf_cool[-1]:.1f} K "
          f"(threshold: T_eq + 50 = {T_EQ+50} K) ***")

    return pass6, T_err


# ===================================================================
# Main
# ===================================================================
def main():
    print("Aragog Verification: Tests 5 and 6")
    print("=" * 60)
    print()

    pass5, order5_L2, order5_Linf = run_test5()
    pass6, err6 = run_test6()

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"TEST 5 (MMS + heating): {'PASS' if pass5 else 'FAIL'} "
          f"(L2 order = {order5_L2:.2f})")
    print(f"TEST 6 (Grey-body):     {'PASS' if pass6 else 'FAIL'} "
          f"(T_surf error = {err6:.2f}%)")
    all_pass = pass5 and pass6
    print(f"\nAll tests passed: {all_pass}")
    print("=" * 60)

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
