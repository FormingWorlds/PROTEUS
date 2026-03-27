#!/usr/bin/env python3
"""Verification: Temporal convergence (BDF tolerance sweep) and mesh-type independence.

Test 7: Temporal Convergence
    Run the transient conduction test (perturbed IC decaying to steady state)
    at fixed spatial resolution (N=101 basic nodes) but varying BDF tolerances:
    rtol = atol = 1e-4, 1e-6, 1e-8, 1e-10, 1e-12.
    Extract the decay timescale tau by fitting max|dT(t)| to an exponential,
    and compare to the analytical tau = D^2 / (pi^2 * kappa) = 3.2107 Gyr.

Test 8: Mesh-Type Independence
    Run the steady-state conduction test at N = 25, 50, 100, 200 with both
    uniform radial spacing (mass_coordinates = False) and mass-coordinate
    spacing (mass_coordinates = True). Both should converge to the same
    analytical solution T(r) = A/r + B at second order.

Output
------
Script: output_files/verification/verify_convergence.py
Plot:   output_files/verification/verify_convergence.pdf (2-panel figure)
Data:   output_files/verification/temporal_convergence_data.txt
        output_files/verification/mesh_convergence_data.txt
"""
from __future__ import annotations

import os
import time as timer

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# ============================================================================
# Physical parameters (all SI)
# ============================================================================
A_INNER = 5.371e6       # inner radius [m]
B_OUTER = 6.371e6       # outer radius [m]
D_SHELL = B_OUTER - A_INNER  # 1e6 m
T_INNER = 4000.0        # inner BC temperature [K]
T_OUTER = 1500.0        # outer BC temperature [K]
RHO = 4000.0            # density [kg/m^3]
K_COND = 4.0            # thermal conductivity [W/m/K]
CP = 1000.0             # heat capacity [J/kg/K]
KAPPA = K_COND / (RHO * CP)  # thermal diffusivity = 1e-6 m^2/s
DELTA = 200.0           # perturbation amplitude [K]

# Analytical eigenvalue decay timescale for fundamental mode
LAMBDA_1 = np.pi / D_SHELL
TAU_ANALYTICAL = 1.0 / (KAPPA * LAMBDA_1**2)   # seconds
SEC_PER_YR = 31557600.0  # Julian year
TAU_ANALYTICAL_YR = TAU_ANALYTICAL / SEC_PER_YR  # ~3.2107e9 yr

# Scalings (same as abe_solid.cfg)
SCALING_RADIUS = 6371000.0
SCALING_TEMPERATURE = 4000.0
SCALING_DENSITY = 4000.0
SCALING_TIME = 31557600000.0  # 1000 yr in seconds

# Integration end time
END_TIME_YR = 5.0e9  # 5 Gyr

ARAGOG_DIR = "/Users/timlichtenberg/git/PROTEUS/aragog"
OUTPUT_DIR = "/Users/timlichtenberg/git/PROTEUS/output_files/verification"

# Analytical steady-state coefficients
A_COEFF = (T_INNER - T_OUTER) * A_INNER * B_OUTER / (B_OUTER - A_INNER)
B_COEFF = (T_OUTER * B_OUTER - T_INNER * A_INNER) / (B_OUTER - A_INNER)


def steady_state(r):
    """Analytical steady-state temperature: T(r) = A/r + B."""
    return A_COEFF / r + B_COEFF


def perturbation(r, delta=DELTA):
    """Fundamental eigenmode perturbation (n=1).

    P(r) = delta * sin(pi*(r-a)/(b-a)) / r * r_ref
    """
    r_ref = (A_INNER + B_OUTER) / 2.0
    return delta * np.sin(np.pi * (r - A_INNER) / D_SHELL) / r * r_ref


def analytical_decay(t_sec, amplitude, tau):
    """Exponential decay: A * exp(-t/tau)."""
    return amplitude * np.exp(-t_sec / tau)


# ============================================================================
# Config writers
# ============================================================================

def write_transient_cfg(cfg_path, ic_file_path, n_nodes, end_time_yr, rtol, atol):
    """Write an Aragog config for the transient conduction test with specified tolerances."""
    cfg_text = f"""\
# Transient conduction temporal convergence test
# rtol = {rtol}, atol = {atol}

[scalings]
radius = {SCALING_RADIUS}
temperature = {SCALING_TEMPERATURE}
density = {SCALING_DENSITY}
time = {SCALING_TIME}

[solver]
start_time = 0
end_time = {end_time_yr:.1f}
atol = {atol:.1e}
rtol = {rtol:.1e}
tsurf_poststep_change = 1000
event_triggering = False

[boundary_conditions]
outer_boundary_condition = 5
outer_boundary_value = {T_OUTER}
inner_boundary_condition = 3
inner_boundary_value = {T_INNER}
emissivity = 1
equilibrium_temperature = 273
core_heat_capacity = 880

[mesh]
outer_radius = {B_OUTER}
inner_radius = {A_INNER}
number_of_nodes = {n_nodes}
mixing_length_profile = nearest_boundary
core_density = 10738.332568062382
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
initial_condition = 2
surface_temperature = {T_OUTER}
basal_temperature = {T_INNER}
init_file = {ic_file_path}

[phase_liquid]
density = {RHO}
viscosity = 1E2
heat_capacity = {CP}
melt_fraction = 1
thermal_conductivity = {K_COND}
thermal_expansivity = 1.0E-5

[phase_solid]
density = {RHO}
viscosity = 1E21
heat_capacity = {CP}
melt_fraction = 0
thermal_conductivity = {K_COND}
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

[radionuclide_K40]
name = K40
t0_years = 4.55E9
abundance = 1.1668E-4
concentration = 310
heat_production = 2.8761E-5
half_life_years = 1248E6

[radionuclide_Th232]
name = Th232
t0_years = 4.55E9
abundance = 1.0
concentration = 0.124
heat_production = 2.6368E-5
half_life_years = 14000E6

[radionuclide_U235]
name = U235
t0_years = 4.55E9
abundance = 0.0072045
concentration = 0.031
heat_production = 5.68402E-4
half_life_years = 704E6

[radionuclide_U238]
name = U238
t0_years = 4.55E9
abundance = 0.9927955
concentration = 0.031
heat_production = 9.4946E-5
half_life_years = 4468E6
"""
    with open(cfg_path, "w") as f:
        f.write(cfg_text)


def write_steady_cfg(cfg_path, n_nodes, end_time_yr, mass_coordinates=False):
    """Write an Aragog config for the steady-state conduction test."""
    mass_coord_str = "True" if mass_coordinates else "False"
    cfg_text = f"""\
# Steady-state conduction mesh independence test
# mass_coordinates = {mass_coord_str}

[scalings]
radius = {SCALING_RADIUS}
temperature = {SCALING_TEMPERATURE}
density = {SCALING_DENSITY}
time = {SCALING_TIME}

[solver]
start_time = 0
end_time = {int(end_time_yr)}
atol = 1e-12
rtol = 1e-12
tsurf_poststep_change = 30
event_triggering = False

[boundary_conditions]
outer_boundary_condition = 5
outer_boundary_value = {T_OUTER}
inner_boundary_condition = 3
inner_boundary_value = {T_INNER}
emissivity = 1
equilibrium_temperature = 273
core_heat_capacity = 880

[mesh]
outer_radius = {B_OUTER}
inner_radius = {A_INNER}
number_of_nodes = {n_nodes}
mixing_length_profile = constant
core_density = 10738.332568062382
surface_density = 4090
gravitational_acceleration = 9.81
adiabatic_bulk_modulus = 260E9
mass_coordinates = {mass_coord_str}

[energy]
conduction = True
convection = False
gravitational_separation = False
mixing = False
radionuclides = False
dilatation = False
tidal = False

[initial_condition]
surface_temperature = {T_OUTER}
basal_temperature = {T_INNER}

[phase_liquid]
density = {RHO}
viscosity = 1E2
heat_capacity = {CP}
melt_fraction = 1
thermal_conductivity = {K_COND}
thermal_expansivity = 1.0E-5

[phase_solid]
density = {RHO}
viscosity = 1E21
heat_capacity = {CP}
melt_fraction = 0
thermal_conductivity = {K_COND}
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

[radionuclide_K40]
name = K40
t0_years = 4.55E9
abundance = 1.1668E-4
concentration = 310
heat_production = 2.8761E-5
half_life_years = 1248E6

[radionuclide_Th232]
name = Th232
t0_years = 4.55E9
abundance = 1.0
concentration = 0.124
heat_production = 2.6368E-5
half_life_years = 14000E6

[radionuclide_U235]
name = U235
t0_years = 4.55E9
abundance = 0.0072045
concentration = 0.031
heat_production = 5.68402E-4
half_life_years = 704E6

[radionuclide_U238]
name = U238
t0_years = 4.55E9
abundance = 0.9927955
concentration = 0.031
heat_production = 9.4946E-5
half_life_years = 4468E6
"""
    with open(cfg_path, "w") as f:
        f.write(cfg_text)


# ============================================================================
# IC generation
# ============================================================================

def generate_ic_file(ic_path, n_nodes):
    """Generate initial condition file with perturbed temperature profile.

    Parameters
    ----------
    ic_path : str
        Output file path.
    n_nodes : int
        Number of basic nodes (staggered = n_nodes - 1).

    Returns
    -------
    r_stag : ndarray
        Staggered radii in metres.
    T_init : ndarray
        Initial temperature at staggered nodes in K.
    """
    r_basic = np.linspace(A_INNER, B_OUTER, n_nodes)
    r_stag = 0.5 * (r_basic[:-1] + r_basic[1:])

    T_ss = steady_state(r_stag)
    T_init = T_ss + perturbation(r_stag)

    np.savetxt(ic_path, T_init)
    return r_stag, T_init


# ============================================================================
# Runners
# ============================================================================

def run_transient(n_nodes, end_time_yr, rtol, atol):
    """Run transient conduction test with specified tolerances.

    Returns
    -------
    r_stag : ndarray, shape (N_stag,)
        Staggered radii [m].
    times_yr : ndarray, shape (N_times,)
        Output times [yr].
    T_stag : ndarray, shape (N_stag, N_times)
        Temperature at staggered nodes [K].
    """
    tag = f"N{n_nodes}_rtol{rtol:.0e}"
    ic_path = os.path.join(OUTPUT_DIR, f"ic_temporal_{tag}.dat")
    cfg_path = os.path.join(OUTPUT_DIR, f"cfg_temporal_{tag}.cfg")

    generate_ic_file(ic_path, n_nodes)
    write_transient_cfg(cfg_path, ic_path, n_nodes, end_time_yr, rtol, atol)

    saved_cwd = os.getcwd()
    os.chdir(ARAGOG_DIR)

    try:
        from aragog.solver import Solver

        solver = Solver.from_file(cfg_path)
        solver.initialize()

        print(f"  Running transient: N={n_nodes}, rtol={rtol:.0e} ...", end="", flush=True)
        t0 = timer.time()
        solver.solve()
        elapsed = timer.time() - t0
        print(f" done ({elapsed:.1f} s, nfev={solver.solution.nfev}, "
              f"N_out={solver.solution.t.size})")

        if solver.solution.status != 0:
            print(f"  WARNING: solver status={solver.solution.status}: "
                  f"{solver.solution.message}")

        scalings = solver.parameters.scalings
        T_stag = solver.solution.y * scalings.temperature
        if T_stag.ndim == 3:
            T_stag = T_stag[:, 0, :]
        times_yr = solver.solution.t * scalings.time_years

        r_stag_mesh = solver.evaluator.mesh.staggered.radii * scalings.radius
        r_stag_mesh = r_stag_mesh.ravel()

    finally:
        os.chdir(saved_cwd)

    return r_stag_mesh, times_yr, T_stag


def run_steady(n_nodes, end_time_yr, mass_coordinates=False):
    """Run steady-state conduction test.

    Returns
    -------
    r_stag : ndarray, shape (N_stag,)
        Staggered radii [m].
    T_final : ndarray, shape (N_stag,)
        Final temperature [K].
    """
    tag = f"N{n_nodes}_mass{mass_coordinates}"
    cfg_path = os.path.join(OUTPUT_DIR, f"cfg_steady_{tag}.cfg")

    write_steady_cfg(cfg_path, n_nodes, end_time_yr, mass_coordinates)

    saved_cwd = os.getcwd()
    os.chdir(ARAGOG_DIR)

    try:
        from aragog.solver import Solver

        solver = Solver.from_file(cfg_path)
        solver.initialize()

        label = "mass-coord" if mass_coordinates else "uniform"
        print(f"  Running steady: N={n_nodes}, {label} ...", end="", flush=True)
        t0 = timer.time()
        solver.solve()
        elapsed = timer.time() - t0
        print(f" done ({elapsed:.1f} s)")

        if solver.solution.status != 0:
            print(f"  WARNING: solver status={solver.solution.status}: "
                  f"{solver.solution.message}")

        scalings = solver.parameters.scalings
        T_final = solver.solution.y[:, -1] * scalings.temperature

        r_stag_mesh = solver.evaluator.mesh.staggered.radii * scalings.radius
        r_stag_mesh = r_stag_mesh.ravel()

    finally:
        os.chdir(saved_cwd)

    return r_stag_mesh, T_final


# ============================================================================
# Analysis: temporal convergence
# ============================================================================

def fit_tau(r_stag, times_yr, T_stag):
    """Fit the decay timescale from the transient solution.

    Returns
    -------
    tau_fit_yr : float
        Fitted decay timescale [yr].
    A_fit : float
        Fitted amplitude [K].
    max_dT : ndarray
        max|T - T_ss| at each output time.
    """
    T_ss = steady_state(r_stag)
    dT = T_stag - T_ss[:, np.newaxis]
    max_dT = np.max(np.abs(dT), axis=0)

    times_sec = times_yr * SEC_PER_YR

    # Use points where signal is still significant (> 5% of initial)
    mask = max_dT > 0.05 * max_dT[0]
    if np.sum(mask) < 3:
        mask = np.ones(len(max_dT), dtype=bool)

    try:
        popt, _ = curve_fit(
            analytical_decay,
            times_sec[mask],
            max_dT[mask],
            p0=[max_dT[0], TAU_ANALYTICAL],
            bounds=([0, 0], [np.inf, np.inf]),
        )
        A_fit, tau_fit_sec = popt
    except RuntimeError:
        # Fallback: linear fit in log space
        log_dT = np.log(max_dT[mask])
        coeffs = np.polyfit(times_sec[mask], log_dT, 1)
        tau_fit_sec = -1.0 / coeffs[0]
        A_fit = np.exp(coeffs[1])

    tau_fit_yr = tau_fit_sec / SEC_PER_YR
    return tau_fit_yr, A_fit, max_dT


# ============================================================================
# Analysis: mesh convergence
# ============================================================================

def compute_L2_error(r, T_numerical, T_exact):
    """Volume-weighted L2 error."""
    diff = T_numerical - T_exact
    weights = r**2
    return np.sqrt(np.sum(weights * diff**2) / np.sum(weights))


# ============================================================================
# Test 7: Temporal convergence
# ============================================================================

def test_temporal_convergence():
    """Run Test 7: BDF tolerance sweep."""
    print("=" * 70)
    print("TEST 7: Temporal Convergence (BDF tolerance sweep)")
    print("=" * 70)
    print()
    print(f"Shell: a = {A_INNER/1e6:.3f} Mm, b = {B_OUTER/1e6:.3f} Mm")
    print(f"kappa = {KAPPA:.2e} m^2/s")
    print(f"Analytical tau = {TAU_ANALYTICAL_YR:.6e} yr = {TAU_ANALYTICAL_YR/1e9:.4f} Gyr")
    print("N = 101 basic nodes (fixed)")
    print()

    tolerances = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    n_nodes = 101
    results = []

    for tol in tolerances:
        r_stag, times_yr, T_stag = run_transient(n_nodes, END_TIME_YR, rtol=tol, atol=tol)
        tau_fit_yr, A_fit, max_dT = fit_tau(r_stag, times_yr, T_stag)
        rel_err = abs(tau_fit_yr - TAU_ANALYTICAL_YR) / TAU_ANALYTICAL_YR

        print(f"  rtol={tol:.0e}: tau_fit = {tau_fit_yr/1e9:.6f} Gyr, "
              f"rel_err = {rel_err:.3e}")

        results.append({
            "tol": tol,
            "tau_fit_yr": tau_fit_yr,
            "rel_err": rel_err,
            "A_fit": A_fit,
            "n_output": len(times_yr),
        })

    print()
    return results


# ============================================================================
# Test 8: Mesh-type independence
# ============================================================================

def test_mesh_independence():
    """Run Test 8: uniform vs mass-coordinate spacing convergence."""
    print("=" * 70)
    print("TEST 8: Mesh-Type Independence")
    print("=" * 70)
    print()

    # Thermal diffusion timescale for steady-state end time
    tau_diff = RHO * CP * D_SHELL**2 / K_COND
    tau_diff_yr = tau_diff / SEC_PER_YR
    end_time_yr = int(3 * tau_diff_yr)
    print(f"Integration time: {end_time_yr:.2e} yr (3 x diffusion timescale)")
    print()

    node_counts = [25, 50, 100, 200]
    results_uniform = []
    results_mass = []
    mass_coord_failed = False

    # Uniform spacing
    print("--- Uniform radial spacing ---")
    for N in node_counts:
        r_stag, T_final = run_steady(N, end_time_yr, mass_coordinates=False)
        T_exact = steady_state(r_stag)
        L2 = compute_L2_error(r_stag, T_final, T_exact)
        results_uniform.append({"N": N, "L2": L2})
        print(f"  N={N:4d}: L2 = {L2:.4e} K")
    print()

    # Mass-coordinate spacing
    print("--- Mass-coordinate spacing ---")
    for N in node_counts:
        try:
            r_stag, T_final = run_steady(N, end_time_yr, mass_coordinates=True)
            T_exact = steady_state(r_stag)
            L2 = compute_L2_error(r_stag, T_final, T_exact)
            results_mass.append({"N": N, "L2": L2})
            print(f"  N={N:4d}: L2 = {L2:.4e} K")
        except Exception as e:
            print(f"  N={N:4d}: FAILED ({type(e).__name__}: {e})")
            mass_coord_failed = True
            break
    print()

    # Convergence orders
    if len(results_uniform) >= 2:
        Ns = np.array([r["N"] for r in results_uniform], dtype=float)
        L2s = np.array([r["L2"] for r in results_uniform])
        coeffs = np.polyfit(np.log(Ns), np.log(L2s), 1)
        print(f"Uniform: overall convergence order = {-coeffs[0]:.2f}")

    if not mass_coord_failed and len(results_mass) >= 2:
        Ns = np.array([r["N"] for r in results_mass], dtype=float)
        L2s = np.array([r["L2"] for r in results_mass])
        coeffs = np.polyfit(np.log(Ns), np.log(L2s), 1)
        print(f"Mass-coord: overall convergence order = {-coeffs[0]:.2f}")

    print()
    return results_uniform, results_mass, mass_coord_failed


# ============================================================================
# Plotting
# ============================================================================

def plot_results(temporal_results, uniform_results, mass_results, mass_coord_failed):
    """Create the 2-panel figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel (a): Temporal convergence ---
    tols = np.array([r["tol"] for r in temporal_results])
    rel_errs = np.array([r["rel_err"] for r in temporal_results])

    ax1.loglog(tols, rel_errs, "ko-", markersize=8, linewidth=2, label="Fitted tau error")

    # Reference 1:1 line
    tol_range = np.array([tols.min() / 3, tols.max() * 3])
    ax1.loglog(tol_range, tol_range, "r--", linewidth=1.5, alpha=0.7,
               label="1:1 (error = tolerance)")

    # Indicate spatial discretization floor
    # At the tightest tolerances, spatial error dominates.
    # Draw a horizontal line at the minimum measured error.
    min_err = rel_errs.min()
    ax1.axhline(min_err, color="blue", linestyle=":", linewidth=1, alpha=0.6,
                label=f"Spatial floor ({min_err:.1e})")

    ax1.set_xlabel("Solver tolerance (rtol = atol)")
    ax1.set_ylabel(r"$|\tau_{\mathrm{fit}} - \tau_{\mathrm{exact}}| / \tau_{\mathrm{exact}}$")
    ax1.set_title("(a) Temporal convergence: BDF tolerance sweep")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.invert_xaxis()  # tighter tolerance to the right

    ax1.text(0.97, 0.03,
             "N = 101 nodes\n"
             r"$\tau_{\mathrm{exact}}$" + f" = {TAU_ANALYTICAL_YR/1e9:.4f} Gyr",
             transform=ax1.transAxes, fontsize=9,
             ha="right", va="bottom",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    # --- Panel (b): Mesh-type independence ---
    Ns_uni = np.array([r["N"] for r in uniform_results], dtype=float)
    L2_uni = np.array([r["L2"] for r in uniform_results])

    ax2.loglog(Ns_uni, L2_uni, "bo-", markersize=8, linewidth=2, label="Uniform spacing")

    if not mass_coord_failed and len(mass_results) > 0:
        Ns_mass = np.array([r["N"] for r in mass_results], dtype=float)
        L2_mass = np.array([r["L2"] for r in mass_results])
        ax2.loglog(Ns_mass, L2_mass, "rs--", markersize=8, linewidth=2, label="Mass-coord spacing")

    # Reference N^{-2} slope
    N_range = np.array([Ns_uni[0], Ns_uni[-1]])
    ref_L2 = L2_uni[0] * (N_range / N_range[0])**(-2)
    ax2.loglog(N_range, ref_L2, "k:", linewidth=1.5, alpha=0.6,
               label=r"$\propto N^{-2}$ (2nd order)")

    ax2.set_xlabel("Number of basic nodes $N$")
    ax2.set_ylabel("$L_2$ temperature error [K]")
    ax2.set_title("(b) Mesh-type independence: steady-state conduction")
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)

    # Annotate convergence order
    if len(uniform_results) >= 2:
        coeffs_uni = np.polyfit(np.log(Ns_uni), np.log(L2_uni), 1)
        order_text = f"Uniform: order = {-coeffs_uni[0]:.2f}"
        if not mass_coord_failed and len(mass_results) >= 2:
            Ns_m = np.array([r["N"] for r in mass_results], dtype=float)
            L2_m = np.array([r["L2"] for r in mass_results])
            coeffs_mass = np.polyfit(np.log(Ns_m), np.log(L2_m), 1)
            order_text += f"\nMass-coord: order = {-coeffs_mass[0]:.2f}"
        ax2.text(0.97, 0.97, order_text,
                 transform=ax2.transAxes, fontsize=9,
                 ha="right", va="top",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    if mass_coord_failed:
        ax2.text(0.5, 0.5,
                 "mass_coordinates=True\nfailed (see console)",
                 transform=ax2.transAxes, fontsize=11,
                 ha="center", va="center", color="red", alpha=0.6)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "verify_convergence.pdf")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


# ============================================================================
# Data saving
# ============================================================================

def save_temporal_data(results):
    """Save temporal convergence data."""
    path = os.path.join(OUTPUT_DIR, "temporal_convergence_data.txt")
    with open(path, "w") as f:
        f.write("# Test 7: Temporal convergence (BDF tolerance sweep)\n")
        f.write(f"# N = 101 basic nodes, end_time = {END_TIME_YR:.1e} yr\n")
        f.write(f"# tau_analytical = {TAU_ANALYTICAL_YR:.8e} yr\n")
        f.write("# rtol_atol  tau_fit_yr  relative_error  N_output\n")
        for r in results:
            f.write(f"{r['tol']:.1e}  {r['tau_fit_yr']:.10e}  "
                    f"{r['rel_err']:.10e}  {r['n_output']}\n")
    print(f"Saved: {path}")


def save_mesh_data(uniform_results, mass_results, mass_coord_failed):
    """Save mesh convergence data."""
    path = os.path.join(OUTPUT_DIR, "mesh_convergence_data.txt")
    with open(path, "w") as f:
        f.write("# Test 8: Mesh-type independence (steady-state conduction)\n")
        f.write("# mesh_type  N_basic_nodes  L2_error_K\n")
        for r in uniform_results:
            f.write(f"uniform     {r['N']:4d}  {r['L2']:.10e}\n")
        if not mass_coord_failed:
            for r in mass_results:
                f.write(f"mass_coord  {r['N']:4d}  {r['L2']:.10e}\n")
        else:
            f.write("# mass_coordinates=True failed, no data available\n")
    print(f"Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Test 7
    temporal_results = test_temporal_convergence()
    save_temporal_data(temporal_results)

    # Test 8
    uniform_results, mass_results, mass_coord_failed = test_mesh_independence()
    save_mesh_data(uniform_results, mass_results, mass_coord_failed)

    # Combined plot
    plot_results(temporal_results, uniform_results, mass_results, mass_coord_failed)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Test 7: Temporal convergence")
    for r in temporal_results:
        print(f"  rtol={r['tol']:.0e}: tau_err = {r['rel_err']:.3e}")
    print(f"  Analytical tau = {TAU_ANALYTICAL_YR/1e9:.4f} Gyr")
    print()
    print("Test 8: Mesh-type independence")
    for r in uniform_results:
        print(f"  Uniform  N={r['N']:4d}: L2 = {r['L2']:.4e} K")
    if not mass_coord_failed:
        for r in mass_results:
            print(f"  MassCoord N={r['N']:4d}: L2 = {r['L2']:.4e} K")
    else:
        print("  mass_coordinates=True FAILED")
    print()


if __name__ == "__main__":
    main()
