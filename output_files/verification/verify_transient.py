"""Verification: Transient conductive cooling eigenvalue decay in a spherical shell.

Tests Aragog's time integration against the analytical eigenvalue decay for
transient conduction in a spherical shell with fixed-T BCs.

Physics
-------
For a spherical shell [a, b] with constant thermal properties (k, rho, cp),
no internal heating, and fixed T at both boundaries, the solution decomposes into:

    T(r,t) = T_ss(r) + sum_n  c_n * sin(n*pi*(r-a)/(b-a)) / r * exp(-lambda_n^2 * kappa * t)

where T_ss(r) = A/r + B is the steady state, kappa = k/(rho*cp) is the
thermal diffusivity, and lambda_n = n*pi/(b-a) are the eigenvalues.

The fundamental mode (n=1) decay timescale is:

    tau = (b-a)^2 / (pi^2 * kappa)

This script:
1. Generates a perturbed IC = T_ss + delta * sin(pi*(r-a)/(b-a)) / r * r_ref
2. Runs Aragog with conduction only and fixed-T BCs
3. Extracts the deviation from steady state at each output time
4. Fits the decay and compares to the analytical timescale
5. Optionally runs a resolution convergence study

All output goes to output_files/verification/ (gitignored).
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
D_SHELL = B_OUTER - A_INNER  # shell thickness = 1e6 m
T_INNER = 4000.0        # inner BC temperature [K]
T_OUTER = 1500.0        # outer BC temperature [K]
RHO = 4000.0            # density [kg/m^3]  (solid phase)
K_COND = 4.0            # thermal conductivity [W/m/K]
CP = 1000.0             # heat capacity [J/kg/K]
KAPPA = K_COND / (RHO * CP)  # thermal diffusivity = 1e-6 m^2/s
DELTA = 200.0           # perturbation amplitude [K]

# Analytical eigenvalue decay timescale for fundamental mode
LAMBDA_1 = np.pi / D_SHELL
TAU_ANALYTICAL = 1.0 / (KAPPA * LAMBDA_1**2)   # seconds
TAU_ANALYTICAL_YR = TAU_ANALYTICAL / 31557600.0  # years (Julian year)

# Scalings (same as abe_solid.cfg)
SCALING_RADIUS = 6371000.0
SCALING_TEMPERATURE = 4000.0
SCALING_DENSITY = 4000.0
SCALING_TIME = 31557600000.0  # 1000 yr in seconds

# Integration end time: ~1.5 e-folding times
END_TIME_YR = 5.0e9  # 5 Gyr in years

ARAGOG_DIR = "/Users/timlichtenberg/git/PROTEUS/aragog"
OUTPUT_DIR = "/Users/timlichtenberg/git/PROTEUS/output_files/verification"


def steady_state(r):
    """Steady-state temperature for fixed-T BCs on a spherical shell.

    T_ss(r) = A_coeff / r + B_coeff
    """
    A_coeff = (T_INNER - T_OUTER) * A_INNER * B_OUTER / (B_OUTER - A_INNER)
    B_coeff = (T_OUTER * B_OUTER - T_INNER * A_INNER) / (B_OUTER - A_INNER)
    return A_coeff / r + B_coeff


def perturbation(r, delta=DELTA):
    """Fundamental eigenmode perturbation (n=1).

    P(r) = delta * sin(pi*(r-a)/(b-a)) / r * r_ref
    where r_ref = (a+b)/2 for normalization.
    """
    r_ref = (A_INNER + B_OUTER) / 2.0
    return delta * np.sin(np.pi * (r - A_INNER) / D_SHELL) / r * r_ref


def analytical_decay(t_sec, amplitude, tau):
    """Exponential decay: A * exp(-t/tau)."""
    return amplitude * np.exp(-t_sec / tau)


def write_cfg(cfg_path, ic_file_path, n_nodes, end_time_yr):
    """Write an Aragog configuration file for this test.

    Parameters
    ----------
    cfg_path : str
        Output config file path.
    ic_file_path : str
        Absolute path to the initial condition temperature file.
    n_nodes : int
        Number of basic mesh nodes (staggered = n_nodes - 1).
    end_time_yr : float
        End time in years.
    """
    cfg_text = f"""\
# Transient conduction verification test
# All units are SI unless otherwise specified.

[scalings]
radius = {SCALING_RADIUS}
temperature = {SCALING_TEMPERATURE}
density = {SCALING_DENSITY}
time = {SCALING_TIME}

[solver]
start_time = 0
end_time = {end_time_yr:.1f}
atol = 1e-10
rtol = 1e-10
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


def run_aragog(n_nodes, end_time_yr):
    """Run Aragog and return solution arrays.

    Parameters
    ----------
    n_nodes : int
        Number of basic mesh nodes.
    end_time_yr : float
        End time in years.

    Returns
    -------
    r_stag : ndarray, shape (N_stag,)
        Staggered radii in metres.
    times_yr : ndarray, shape (N_times,)
        Output times in years.
    T_stag : ndarray, shape (N_stag, N_times)
        Temperature at staggered nodes in K.
    """
    # Create temporary files for config and IC
    ic_path = os.path.join(OUTPUT_DIR, f"ic_transient_N{n_nodes}.dat")
    cfg_path = os.path.join(OUTPUT_DIR, f"cfg_transient_N{n_nodes}.cfg")

    r_stag, T_init = generate_ic_file(ic_path, n_nodes)
    write_cfg(cfg_path, ic_path, n_nodes, end_time_yr)

    # Must chdir to aragog directory for data file paths (solidus, liquidus)
    saved_cwd = os.getcwd()
    os.chdir(ARAGOG_DIR)

    try:
        from aragog import Solver

        solver = Solver.from_file(cfg_path)
        solver.initialize()

        print(f"  Running Aragog with N={n_nodes}, end_time={end_time_yr:.2e} yr ...")
        t0 = timer.time()
        solver.solve()
        elapsed = timer.time() - t0
        print(f"  Completed in {elapsed:.1f} s, "
              f"status={solver.solution.status}, "
              f"nfev={solver.solution.nfev}, "
              f"N_output={solver.solution.t.size}")

        if solver.solution.status != 0:
            print(f"  WARNING: solver status = {solver.solution.status}: "
                  f"{solver.solution.message}")

        # Extract results
        scalings = solver.parameters.scalings
        T_stag = solver.solution.y * scalings.temperature  # (N_stag, N_times) or (N_stag,1,N_times)?

        # Handle potential extra dimensions
        if T_stag.ndim == 3:
            T_stag = T_stag[:, 0, :]
        times_yr = solver.solution.t * scalings.time_years

        # Get actual staggered radii from mesh (in metres)
        r_stag_mesh = solver.evaluator.mesh.staggered.radii * scalings.radius
        r_stag_mesh = r_stag_mesh.ravel()

    finally:
        os.chdir(saved_cwd)

    return r_stag_mesh, times_yr, T_stag


def extract_decay(r_stag, times_yr, T_stag):
    """Extract max deviation from steady state at each time.

    Parameters
    ----------
    r_stag : ndarray, shape (N,)
    times_yr : ndarray, shape (M,)
    T_stag : ndarray, shape (N, M)

    Returns
    -------
    times_yr : ndarray
    max_dT : ndarray
        Max absolute deviation from steady state at each time.
    dT_profiles : ndarray, shape (N, M)
        Full deviation profiles.
    """
    T_ss = steady_state(r_stag)

    # Deviation from steady state
    # T_stag may be (N, M) or (N, 1, M)
    if T_stag.ndim == 2:
        dT = T_stag - T_ss[:, np.newaxis]
    else:
        raise ValueError(f"Unexpected T_stag shape: {T_stag.shape}")

    max_dT = np.max(np.abs(dT), axis=0)

    return times_yr, max_dT, dT


def fit_decay(times_yr, max_dT):
    """Fit exponential decay to max|dT| vs time.

    Parameters
    ----------
    times_yr : ndarray
    max_dT : ndarray

    Returns
    -------
    tau_fit_yr : float
        Fitted decay timescale in years.
    A_fit : float
        Fitted amplitude.
    """
    # Convert to seconds for fitting
    sec_per_yr = 31557600.0  # Julian year
    times_sec = times_yr * sec_per_yr

    # Use points where signal is still significant (> 5% of initial)
    mask = max_dT > 0.05 * max_dT[0]
    if np.sum(mask) < 3:
        # Fall back to all points
        mask = np.ones(len(max_dT), dtype=bool)

    try:
        popt, pcov = curve_fit(
            analytical_decay,
            times_sec[mask],
            max_dT[mask],
            p0=[max_dT[0], TAU_ANALYTICAL],
            bounds=([0, 0], [np.inf, np.inf]),
        )
        A_fit, tau_fit_sec = popt
        tau_fit_yr = tau_fit_sec / sec_per_yr
    except RuntimeError:
        print("  WARNING: curve_fit failed, using linear fit in log space")
        # Fallback: linear fit in log space
        log_dT = np.log(max_dT[mask])
        t_sec = times_sec[mask]
        coeffs = np.polyfit(t_sec, log_dT, 1)
        tau_fit_sec = -1.0 / coeffs[0]
        A_fit = np.exp(coeffs[1])
        tau_fit_yr = tau_fit_sec / sec_per_yr

    return tau_fit_yr, A_fit


# ============================================================================
# Plotting
# ============================================================================

def plot_decay(r_stag, times_yr, T_stag, max_dT, dT_profiles, tau_fit_yr, A_fit, n_nodes):
    """Plot 1: Decay curve and spatial profile shapes.

    Parameters
    ----------
    r_stag : ndarray
    times_yr : ndarray
    T_stag : ndarray, shape (N, M)
    max_dT : ndarray
    dT_profiles : ndarray, shape (N, M)
    tau_fit_yr : float
    A_fit : float
    n_nodes : int
    """
    sec_per_yr = 31557600.0
    tau_fit_sec = tau_fit_yr * sec_per_yr

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # --- Top panel: max|dT| decay ---
    ax = axes[0]
    times_sec = times_yr * sec_per_yr
    ax.semilogy(times_yr / 1e9, max_dT, "k.", markersize=3, label="Aragog")

    # Analytical curve
    t_fine = np.linspace(0, times_sec[-1], 500)
    dT_analytical = A_fit * np.exp(-t_fine / TAU_ANALYTICAL)
    ax.semilogy(
        t_fine / (sec_per_yr * 1e9),
        dT_analytical,
        "r-",
        linewidth=1.5,
        label=f"Analytical (tau = {TAU_ANALYTICAL_YR/1e9:.3f} Gyr)",
    )

    # Fitted curve
    dT_fitted = A_fit * np.exp(-t_fine / tau_fit_sec)
    ax.semilogy(
        t_fine / (sec_per_yr * 1e9),
        dT_fitted,
        "b--",
        linewidth=1.5,
        label=f"Fitted (tau = {tau_fit_yr/1e9:.3f} Gyr)",
    )

    ax.set_xlabel("Time [Gyr]")
    ax.set_ylabel("max |T - T_ss| [K]")
    ax.set_title(f"Transient conduction decay (N = {n_nodes} basic nodes)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate relative error
    rel_err = abs(tau_fit_yr - TAU_ANALYTICAL_YR) / TAU_ANALYTICAL_YR
    ax.text(
        0.02,
        0.02,
        f"Relative error in tau: {rel_err:.2e}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    # --- Bottom panel: normalized spatial profiles at selected times ---
    ax = axes[1]
    r_km = (r_stag - A_INNER) / 1e3  # depth from inner boundary in km

    # Select ~6 times spread across the run, ensuring distinct labels
    n_times = len(times_yr)
    if n_times > 6:
        # Pick indices at roughly equal fractions of total time
        target_fracs = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
        t_max = times_yr[-1]
        indices = []
        for frac in target_fracs:
            t_target = frac * t_max
            idx = np.argmin(np.abs(times_yr - t_target))
            if idx not in indices:
                indices.append(idx)
        indices = np.array(indices)
    else:
        indices = np.arange(n_times)

    # Analytical fundamental mode shape
    r_ref = (A_INNER + B_OUTER) / 2.0
    mode_shape = np.sin(np.pi * (r_stag - A_INNER) / D_SHELL) / r_stag * r_ref

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(indices)))
    for j, idx in enumerate(indices):
        profile = dT_profiles[:, idx]
        if max_dT[idx] > 1e-3:  # only plot if signal is still visible
            normalized = profile / np.max(np.abs(profile))
            ax.plot(
                r_km,
                normalized,
                color=colors[j],
                linewidth=1.5,
                label=f"t = {times_yr[idx]/1e9:.2f} Gyr",
            )

    # Analytical shape (normalized)
    ax.plot(
        r_km,
        mode_shape / np.max(np.abs(mode_shape)),
        "k--",
        linewidth=2,
        label="Analytical n=1 mode",
    )

    ax.set_xlabel("Distance from inner boundary [km]")
    ax.set_ylabel("Normalized dT")
    ax.set_title("Spatial profile shape (normalized)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "verify_transient_decay.pdf")
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")


def plot_convergence(results):
    """Plot 2: Convergence of decay timescale with resolution.

    Parameters
    ----------
    results : list of (n_nodes, tau_fit_yr, rel_err)
    """
    if len(results) < 2:
        print("  Skipping convergence plot (need >= 2 resolutions)")
        return

    ns = np.array([r[0] for r in results])
    errs = np.array([r[2] for r in results])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ns, errs, "ko-", markersize=8, linewidth=2, label="Measured")

    # Overlay second-order reference line (BDF is second order)
    if len(ns) >= 2:
        # h ~ 1/N, so error ~ N^{-2} for second-order spatial discretization
        ref_errs = errs[0] * (ns[0] / ns) ** 2
        ax.loglog(ns, ref_errs, "r--", linewidth=1.5, label=r"$\propto N^{-2}$ (reference)")

    ax.set_xlabel("Number of basic nodes N")
    ax.set_ylabel("Relative error in decay timescale")
    ax.set_title("Convergence: transient conduction eigenvalue")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "verify_transient_convergence.pdf")
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Aragog Verification: Transient Conductive Cooling Eigenvalue Decay")
    print("=" * 70)
    print()
    print(f"Shell: a = {A_INNER/1e6:.3f} Mm, b = {B_OUTER/1e6:.3f} Mm, "
          f"D = {D_SHELL/1e6:.3f} Mm")
    print(f"Properties: k = {K_COND} W/m/K, rho = {RHO} kg/m^3, "
          f"cp = {CP} J/kg/K")
    print(f"Thermal diffusivity: kappa = {KAPPA:.2e} m^2/s")
    print(f"Analytical tau (fundamental mode): {TAU_ANALYTICAL:.4e} s "
          f"= {TAU_ANALYTICAL_YR:.4e} yr "
          f"= {TAU_ANALYTICAL_YR/1e9:.4f} Gyr")
    print(f"Perturbation amplitude: delta = {DELTA} K")
    print(f"Integration time: {END_TIME_YR:.2e} yr "
          f"= {END_TIME_YR/TAU_ANALYTICAL_YR:.2f} tau")
    print()

    # Resolution study: 50, 100, 200 basic nodes
    node_counts = [51, 101, 201]
    convergence_results = []

    for n_nodes in node_counts:
        print(f"--- N = {n_nodes} basic nodes ({n_nodes - 1} staggered) ---")

        r_stag, times_yr, T_stag = run_aragog(n_nodes, END_TIME_YR)

        # Handle shape
        print(f"  T_stag shape: {T_stag.shape}, times shape: {times_yr.shape}")
        print(f"  r_stag shape: {r_stag.shape}")

        times_yr_out, max_dT, dT_profiles = extract_decay(r_stag, times_yr, T_stag)

        print(f"  max|dT| at t=0: {max_dT[0]:.2f} K")
        print(f"  max|dT| at t=end: {max_dT[-1]:.4f} K")

        tau_fit_yr, A_fit = fit_decay(times_yr_out, max_dT)
        rel_err = abs(tau_fit_yr - TAU_ANALYTICAL_YR) / TAU_ANALYTICAL_YR

        print(f"  Fitted tau:      {tau_fit_yr:.4e} yr = {tau_fit_yr/1e9:.4f} Gyr")
        print(f"  Analytical tau:  {TAU_ANALYTICAL_YR:.4e} yr = {TAU_ANALYTICAL_YR/1e9:.4f} Gyr")
        print(f"  Relative error:  {rel_err:.4e}")
        print()

        convergence_results.append((n_nodes, tau_fit_yr, rel_err))

        # Plot for the highest resolution
        if n_nodes == node_counts[-1]:
            plot_decay(
                r_stag, times_yr_out, T_stag, max_dT, dT_profiles,
                tau_fit_yr, A_fit, n_nodes
            )

    # Convergence plot
    plot_convergence(convergence_results)

    # Save numerical data
    data_path = os.path.join(OUTPUT_DIR, "transient_convergence_data.txt")
    with open(data_path, "w") as f:
        f.write("# N_basic_nodes  tau_fit_yr  tau_analytical_yr  relative_error\n")
        for n_nodes, tau_fit, rel_err in convergence_results:
            f.write(f"{n_nodes:6d}  {tau_fit:.8e}  {TAU_ANALYTICAL_YR:.8e}  {rel_err:.8e}\n")
    print(f"Saved convergence data: {data_path}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for n_nodes, tau_fit, rel_err in convergence_results:
        status = "PASS" if rel_err < 0.01 else ("MARGINAL" if rel_err < 0.05 else "FAIL")
        print(f"  N={n_nodes:4d}: tau_fit={tau_fit/1e9:.4f} Gyr, "
              f"rel_err={rel_err:.2e}  [{status}]")
    print(f"  Analytical:  tau={TAU_ANALYTICAL_YR/1e9:.4f} Gyr")
    print()


if __name__ == "__main__":
    main()
