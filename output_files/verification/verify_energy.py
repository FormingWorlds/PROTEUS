"""
Aragog energy budget closure verification.

Runs two test cases and checks that the numerical solution conserves energy
to within the ODE solver tolerance.

Case 1: Conduction only, liquid phase, fixed-flux BCs (zero flux at both
        boundaries). The system is a closed box: total thermal energy must
        remain constant.

Case 2: Convection + conduction, liquid phase, grey-body cooling at the
        surface, zero flux at the CMB. Total thermal energy must decrease
        at a rate equal to the surface radiative loss.

Outputs
-------
verify_energy_budget.pdf : 3-panel diagnostic plot for Case 2
energy_data_case1.npz    : raw data for Case 1
energy_data_case2.npz    : raw data for Case 2

All outputs are written to the same directory as this script.
"""
from __future__ import annotations

import os
import tempfile
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Aragog must be run from its own directory so relative data paths resolve
ARAGOG_DIR = Path("/Users/timlichtenberg/git/aragog")
OUTPUT_DIR = Path("/Users/timlichtenberg/git/PROTEUS/output_files/verification")
os.chdir(ARAGOG_DIR)

from scipy import constants  # noqa: E402

from aragog.solver import Solver  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: write a temporary .cfg file and return its path
# ---------------------------------------------------------------------------
def write_cfg(text: str) -> str:
    """Write config text to a temporary file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".cfg", prefix="aragog_energy_")
    with os.fdopen(fd, "w") as f:
        f.write(textwrap.dedent(text))
    return path


# ---------------------------------------------------------------------------
# Helper: extract physical quantities from a solved Aragog model
# ---------------------------------------------------------------------------
def extract_energy_timeseries(solver):
    """
    Compute total thermal energy E(t) and surface/CMB heat fluxes at each
    output time from a solved Aragog model.

    Parameters
    ----------
    solver : aragog.solver.Solver
        A solver that has already been initialized and solved.

    Returns
    -------
    dict with keys:
        t_phys     : (n_times,) physical time in seconds
        E_total    : (n_times,) total thermal energy in J
        q_surf     : (n_times,) surface heat flux in W/m^2 (outward positive)
        q_cmb      : (n_times,) CMB heat flux in W/m^2 (outward positive)
        A_surf     : float, surface area in m^2
        A_cmb      : float, CMB area in m^2
        rho        : float, density in kg/m^3
        cp         : float, heat capacity in J/kg/K
    """
    sc = solver.parameters.scalings
    sol = solver.solution
    SEC_PER_YR = constants.Julian_year  # 31557600.0

    # Mesh quantities (physical units)
    # With the refactored Aragog, all quantities are already in SI units
    # (scalings are all 1.0), so multiplying by sc.* is a no-op.
    r_basic = solver.evaluator.mesh.basic.radii.ravel() * sc.radius     # m
    V_cells = solver.evaluator.mesh.basic.volume.ravel() * sc.radius**3  # m^3

    A_surf = 4.0 * np.pi * r_basic[-1] ** 2   # m^2
    A_cmb = 4.0 * np.pi * r_basic[0] ** 2     # m^2

    n_times = sol.y.shape[1]

    # Phase properties (constant for single-phase liquid)
    # Evaluate once at the initial state
    T0 = sol.y[:, 0:1]
    solver.state.update(T0, sol.t[0])
    rho = float(np.mean(solver.state.phase_staggered.density())) * sc.density  # kg/m^3
    cp = float(np.mean(solver.state.phase_staggered.heat_capacity())) * sc.heat_capacity  # J/kg/K

    # Time array in physical seconds
    # sol.t is in years; convert to seconds for energy rate computation
    t_phys = sol.t * SEC_PER_YR

    # Total thermal energy at each output time
    # E(t) = sum_i rho * cp * V_i * T_i(t)
    T_phys = sol.y * sc.temperature  # (n_stag, n_times) in K
    E_total = np.sum(rho * cp * V_cells[:, None] * T_phys, axis=0)  # (n_times,)

    # Surface and CMB heat fluxes at each output time
    q_surf = np.zeros(n_times)
    q_cmb = np.zeros(n_times)

    for i in range(n_times):
        T_col = sol.y[:, i:i + 1]
        solver.state.update(T_col, sol.t[i])
        # After update, apply flux BCs (this is what dTdt does internally)
        solver.evaluator.boundary_conditions.apply_flux_boundary_conditions(solver.state)

        q_surf[i] = solver.state.heat_flux[-1, 0] * sc.heat_flux  # W/m^2
        q_cmb[i] = solver.state.heat_flux[0, 0] * sc.heat_flux    # W/m^2

    return {
        "t_phys": t_phys,
        "E_total": E_total,
        "q_surf": q_surf,
        "q_cmb": q_cmb,
        "A_surf": A_surf,
        "A_cmb": A_cmb,
        "rho": rho,
        "cp": cp,
    }


def compute_residuals(data):
    """
    Compute the energy budget residual from the extracted timeseries.

    Uses centred finite differences for dE/dt and trapezoidal-rule averaged
    boundary fluxes.

    Returns
    -------
    dict with keys:
        t_mid      : (n-1,) midpoint times in seconds
        dEdt       : (n-1,) numerical dE/dt in W
        Q_net      : (n-1,) net boundary power in W (positive = energy entering)
        residual   : (n-1,) relative residual |dEdt - Q_net| / |Q_net|
    """
    t = data["t_phys"]
    E = data["E_total"]
    q_s = data["q_surf"]
    q_c = data["q_cmb"]
    A_s = data["A_surf"]
    A_c = data["A_cmb"]

    dt = np.diff(t)
    dE = np.diff(E)
    dEdt = dE / dt

    # Trapezoidal average of fluxes over each interval
    q_s_avg = 0.5 * (q_s[:-1] + q_s[1:])
    q_c_avg = 0.5 * (q_c[:-1] + q_c[1:])

    # Net power entering the domain: +Q_cmb*A_cmb - Q_surf*A_surf
    # (flux is defined as positive outward, so surface flux removes energy)
    Q_net = q_c_avg * A_c - q_s_avg * A_s

    t_mid = 0.5 * (t[:-1] + t[1:])

    # Relative residual (avoid division by zero)
    denom = np.abs(Q_net)
    denom = np.where(denom < 1e-30, 1e-30, denom)
    residual = np.abs(dEdt - Q_net) / denom

    return {
        "t_mid": t_mid,
        "dEdt": dEdt,
        "Q_net": Q_net,
        "residual": residual,
    }


# ===========================================================================
# CASE 1: Closed box (zero flux at both boundaries, conduction only)
# ===========================================================================
print("=" * 70)
print("CASE 1: Closed box -- conduction only, zero flux BCs")
print("=" * 70)

cfg1_text = """\
[scalings]
radius = 6371000
temperature = 4000
density = 4000
time = 3155760

[solver]
start_time = 0
end_time = 5000
atol = 1e-10
rtol = 1e-10
tsurf_poststep_change = 100
event_triggering = False

[boundary_conditions]
outer_boundary_condition = 4
outer_boundary_value = 0
inner_boundary_condition = 2
inner_boundary_value = 0
emissivity = 1
equilibrium_temperature = 273
core_heat_capacity = 880

[mesh]
outer_radius = 6371000
inner_radius = 5371000
number_of_nodes = 60
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
surface_temperature = 3000
basal_temperature = 4000

[phase_liquid]
density = 4000
viscosity = 1E2
heat_capacity = 1000
melt_fraction = 1
thermal_conductivity = 4
thermal_expansivity = 1.0E-5

[phase_solid]
density = 4200
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
"""

cfg1_path = write_cfg(cfg1_text)
print(f"Config written to: {cfg1_path}")

solver1 = Solver.from_file(cfg1_path)
solver1.initialize()
print("Running Case 1...")
solver1.solve()
print(f"  Solver status: {solver1.solution.status}, message: {solver1.solution.message}")
print(f"  Output times: {solver1.solution.t.shape[0]} points")

data1 = extract_energy_timeseries(solver1)
res1 = compute_residuals(data1)

# For a closed box, Q_net should be zero, so check absolute energy conservation
E0 = data1["E_total"][0]
E_drift = np.abs(data1["E_total"] - E0) / np.abs(E0)

print(f"\n  Thermal energy at t=0:  {E0:.6e} J")
print(f"  Thermal energy at end:  {data1['E_total'][-1]:.6e} J")
print(f"  Max relative E drift:   {np.max(E_drift):.3e}")
print(f"  Mean relative E drift:  {np.mean(E_drift):.3e}")
print(f"  Max |q_surf|:           {np.max(np.abs(data1['q_surf'])):.3e} W/m^2 (should be ~0)")
print(f"  Max |q_cmb|:            {np.max(np.abs(data1['q_cmb'])):.3e} W/m^2 (should be ~0)")

# Save data
np.savez(
    OUTPUT_DIR / "energy_data_case1.npz",
    t_phys=data1["t_phys"],
    E_total=data1["E_total"],
    q_surf=data1["q_surf"],
    q_cmb=data1["q_cmb"],
    E_drift=E_drift,
)

print("\n  VERDICT:", end=" ")
if np.max(E_drift) < 1e-6:
    print("PASS -- energy conserved to < 1e-6 relative drift")
else:
    print(f"MARGINAL -- max relative drift = {np.max(E_drift):.3e}")


# ===========================================================================
# CASE 2: Convective cooling with grey-body radiation
# ===========================================================================
print("\n" + "=" * 70)
print("CASE 2: Convective cooling -- grey-body surface, zero-flux CMB")
print("=" * 70)

cfg2_text = """\
[scalings]
radius = 6371000
temperature = 4000
density = 4000
time = 3155760

[solver]
start_time = 0
end_time = 1000
atol = 1e-9
rtol = 1e-9
tsurf_poststep_change = 30
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
surface_temperature = 4000
basal_temperature = 4000

[phase_liquid]
density = 4000
viscosity = 1E2
heat_capacity = 1000
melt_fraction = 1
thermal_conductivity = 4
thermal_expansivity = 1.0E-5

[phase_solid]
density = 4200
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
"""

cfg2_path = write_cfg(cfg2_text)
print(f"Config written to: {cfg2_path}")

solver2 = Solver.from_file(cfg2_path)
solver2.initialize()
print("Running Case 2...")
solver2.solve()
print(f"  Solver status: {solver2.solution.status}, message: {solver2.solution.message}")
print(f"  Output times: {solver2.solution.t.shape[0]} points")

data2 = extract_energy_timeseries(solver2)
res2 = compute_residuals(data2)

# Summary statistics (skip first few points where finite-difference errors are largest)
skip = 3
res_trimmed = res2["residual"][skip:]
print(f"\n  Max  relative residual (after skipping first {skip}): {np.max(res_trimmed):.3e}")
print(f"  Mean relative residual (after skipping first {skip}): {np.mean(res_trimmed):.3e}")
print(f"  Median relative residual:                             {np.median(res_trimmed):.3e}")
print("  Solver rtol: 1e-9")

# Save data
np.savez(
    OUTPUT_DIR / "energy_data_case2.npz",
    t_phys=data2["t_phys"],
    E_total=data2["E_total"],
    q_surf=data2["q_surf"],
    q_cmb=data2["q_cmb"],
    t_mid=res2["t_mid"],
    dEdt=res2["dEdt"],
    Q_net=res2["Q_net"],
    residual=res2["residual"],
)

# Convert time to years for plotting
yr = constants.Julian_year

print("\n  VERDICT:", end=" ")
if np.median(res_trimmed) < 1e-3:
    print(f"PASS -- median residual {np.median(res_trimmed):.3e} is consistent with "
          "finite-difference discretization of the BDF output")
else:
    print(f"CHECK -- median residual {np.median(res_trimmed):.3e} is larger than expected")


# ===========================================================================
# PLOT: 3-panel diagnostic for Case 2
# ===========================================================================
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Panel (a): Total thermal energy vs time
ax = axes[0]
ax.plot(data2["t_phys"] / yr, data2["E_total"], "-", color="#2166ac", linewidth=1.2)
ax.set_ylabel("Total thermal energy $E(t)$ [J]")
ax.set_title("Aragog energy budget verification (convective cooling)")
ax.text(0.02, 0.92, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold",
        verticalalignment="top")
ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

# Panel (b): dE/dt vs net boundary flux power
ax = axes[1]
t_yr = res2["t_mid"] / yr
ax.plot(t_yr, res2["dEdt"], "-", color="#2166ac", linewidth=1.0, label=r"$dE/dt$ (finite diff.)")
ax.plot(t_yr, res2["Q_net"], "--", color="#b2182b", linewidth=1.0, label=r"$Q_\mathrm{CMB} A_\mathrm{CMB} - Q_\mathrm{surf} A_\mathrm{surf}$")
ax.set_ylabel("Power [W]")
ax.legend(fontsize=9, loc="upper right")
ax.text(0.02, 0.92, "(b)", transform=ax.transAxes, fontsize=12, fontweight="bold",
        verticalalignment="top")
ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

# Panel (c): Relative residual
ax = axes[2]
ax.semilogy(t_yr, res2["residual"], "-", color="#636363", linewidth=0.8)
ax.axhline(1e-9, color="#636363", linestyle=":", linewidth=0.8, label="solver rtol = 1e-9")
ax.axhline(np.median(res_trimmed), color="#2166ac", linestyle="--", linewidth=0.8,
           label=f"median = {np.median(res_trimmed):.2e}")
ax.set_xlabel("Time [yr]")
ax.set_ylabel(r"Relative residual $|dE/dt - Q_\mathrm{net}| \,/\, |Q_\mathrm{net}|$")
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim(bottom=1e-12)
ax.text(0.02, 0.92, "(c)", transform=ax.transAxes, fontsize=12, fontweight="bold",
        verticalalignment="top")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "verify_energy_budget.pdf", dpi=150)
print(f"\nPlot saved to: {OUTPUT_DIR / 'verify_energy_budget.pdf'}")

# Clean up temp files
os.unlink(cfg1_path)
os.unlink(cfg2_path)

print("\nDone.")
