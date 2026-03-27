"""
Verification test: Aragog phase-change energy conservation.

Tests:
  A) Cumulative energy budget: the total enthalpy lost by the mantle
     (computed from temperature changes and phase-appropriate rho*cp)
     must equal the time-integrated surface radiative power loss.
     This tests the BDF integrator's accuracy through phase transitions,
     not just the RHS consistency.
  B) Latent heat slows cooling: a composite-phase run cools slower through
     the solidus-liquidus range than a single-phase liquid run with
     identical initial conditions and boundary conditions.

The key subtlety: in the composite phase, the effective heat capacity cp_eff
includes latent heat (cp_eff = L / delta_T_fusion in the mixed zone). So the
"thermal energy" E = sum(rho*cp*T*V) is NOT a conserved quantity when cp(T)
changes. Instead, we must compute enthalpy changes incrementally:

   delta_H = sum_i [ rho_i(T_n) * cp_i(T_n) * (T_{n+1,i} - T_{n,i}) * V_i ]

and compare with the time-integrated surface power:

   delta_W = integral P_surf dt

Output: verify_phasechange_energy.pdf (4 panels)
"""
from __future__ import annotations

import configparser
import os

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

# Must chdir to aragog root so relative data paths resolve
os.chdir("/Users/timlichtenberg/git/aragog")

from aragog.solver import Solver  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OUT_DIR = "/Users/timlichtenberg/git/PROTEUS/output_files/verification"


def write_cfg(path: str, phase: str = "composite", end_time: float = 5000.0,
              n_nodes: int = 60, rtol: float = 1e-9, atol: float = 1e-9,
              inner_radius: float = 6271000.0, outer_radius: float = 6371000.0,
              surface_temp: float = 2500.0, basal_temp: float = 2500.0) -> str:
    """Write an Aragog .cfg file and return the path."""
    cfg = configparser.ConfigParser()

    cfg["scalings"] = {
        "radius": "6371000",
        "temperature": "4000",
        "density": "4000",
        "time": "3155760",  # 0.1 yr in seconds
    }

    cfg["solver"] = {
        "start_time": "0",
        "end_time": str(end_time),
        "atol": str(atol),
        "rtol": str(rtol),
        "tsurf_poststep_change": "500",  # large to avoid early termination
        "event_triggering": "False",
    }

    cfg["boundary_conditions"] = {
        "outer_boundary_condition": "1",  # grey-body
        "outer_boundary_value": "0",
        "inner_boundary_condition": "2",  # prescribed flux
        "inner_boundary_value": "0",      # zero CMB flux
        "emissivity": "1",
        "equilibrium_temperature": "273",
        "core_heat_capacity": "880",
    }

    cfg["mesh"] = {
        "outer_radius": str(outer_radius),
        "inner_radius": str(inner_radius),
        "number_of_nodes": str(n_nodes),
        "mixing_length_profile": "constant",
        "core_density": "10738.332568062382",
        "surface_density": "4090",
        "gravitational_acceleration": "9.81",
        "adiabatic_bulk_modulus": "260E9",
    }

    cfg["energy"] = {
        "conduction": "True",
        "convection": "True",
        "gravitational_separation": "False",
        "mixing": "False",
        "radionuclides": "False",
        "dilatation": "False",
        "tidal": "False",
    }

    cfg["initial_condition"] = {
        "surface_temperature": str(surface_temp),
        "basal_temperature": str(basal_temp),
    }

    cfg["phase_liquid"] = {
        "density": "4000",
        "viscosity": "1E2",
        "heat_capacity": "1000",
        "melt_fraction": "1",
        "thermal_conductivity": "4",
        "thermal_expansivity": "1.0E-5",
    }

    cfg["phase_solid"] = {
        "density": "4200",
        "viscosity": "1E21",
        "heat_capacity": "1000",
        "melt_fraction": "0",
        "thermal_conductivity": "4",
        "thermal_expansivity": "1.0E-5",
    }

    cfg["phase_mixed"] = {
        "latent_heat_of_fusion": "4e6",
        "rheological_transition_melt_fraction": "0.4",
        "rheological_transition_width": "0.15",
        "solidus": "data/test/solidus_1d_lookup.dat",
        "liquidus": "data/test/liquidus_1d_lookup.dat",
        "phase": phase,
        "phase_transition_width": "0.1",
        "grain_size": "1.0E-3",
    }

    with open(path, "w") as f:
        cfg.write(f)
    return path


def run_solver(cfg_path: str) -> Solver:
    """Instantiate, initialize, and solve."""
    solver = Solver.from_file(cfg_path)
    solver.initialize()
    solver.solve()
    return solver


def extract_timeseries(solver):
    """
    Extract dimensional time series from a solved Aragog model.

    Computes three independent energy measures:

    1. Cumulative enthalpy change (incremental):
       delta_H(n) = sum over steps k=0..n-1 of
         sum_i [rho_i(k) * cp_i(k) * (T_i(k+1) - T_i(k)) * V_i]
       This properly accounts for cp changing at phase transitions.

    2. Cumulative surface energy loss:
       delta_W(n) = integral_0^t_n P_surf dt
       (trapezoidal rule on the output timesteps)

    3. Instantaneous P_surf for each timestep (for plotting).

    Returns
    -------
    dict with keys:
        t_yr            : time in years (n_times,)
        T_surf          : surface temperature in K (n_times,)
        phi_mean        : volume-averaged melt fraction (n_times,)
        P_surf          : surface power loss in W (n_times,)
        cum_dH          : cumulative enthalpy change in J (n_times,)
        cum_Wsurf       : cumulative surface energy loss in J (n_times,)
    """
    sc = solver.parameters.scalings
    sol = solver.solution
    n_stag = sol.y.shape[0]
    n_times = sol.y.shape[1]

    SEC_PER_YR = 31557600.0  # Julian year
    t_yr = sol.t * sc.time_years
    t_s = sol.t * SEC_PER_YR  # time in seconds (sol.t is in years)

    # Mesh quantities (dimensional)
    r_basic = solver.evaluator.mesh.basic.radii.ravel() * sc.radius
    V_cells = solver.evaluator.mesh.basic.volume.ravel() * sc.radius**3  # m^3
    A_surf = 4.0 * np.pi * r_basic[-1]**2  # m^2

    T_surf = np.zeros(n_times)
    phi_mean = np.zeros(n_times)
    P_surf = np.zeros(n_times)
    cum_dH = np.zeros(n_times)
    cum_Wsurf = np.zeros(n_times)

    # Store previous step's state for incremental enthalpy computation
    prev_T_K = None
    prev_rho_cp = None  # rho * cp in dimensional units at each cell

    for i in range(n_times):
        T_col = sol.y[:, i:i+1]
        t_val = sol.t[i]

        # Update state (this sets phase properties at current temperature)
        solver.state.update(T_col, t_val)
        solver.evaluator.boundary_conditions.apply_flux_boundary_conditions(
            solver.state)

        # Dimensional temperature at staggered nodes
        T_K = T_col.ravel() * sc.temperature  # (n_stag,)

        # Phase-appropriate rho * cp at staggered nodes (dimensional)
        rho_nd = solver.state.phase_staggered.density()
        cp_nd = solver.state.phase_staggered.heat_capacity()
        if np.ndim(rho_nd) == 0:
            rho = float(rho_nd) * sc.density * np.ones(n_stag)
        else:
            rho = np.asarray(rho_nd).ravel() * sc.density
        if np.ndim(cp_nd) == 0:
            cp = float(cp_nd) * sc.heat_capacity * np.ones(n_stag)
        else:
            cp = np.asarray(cp_nd).ravel() * sc.heat_capacity
        rho_cp = rho * cp

        # Surface temperature
        T_basic = solver.evaluator.mesh.quantity_at_basic_nodes(T_col)
        T_surf[i] = float(T_basic[-1, 0]) * sc.temperature

        # Surface heat flux
        q_surf = float(solver.state.heat_flux[-1, 0]) * sc.heat_flux
        P_surf[i] = q_surf * A_surf

        # Melt fraction
        phi = solver.state.phase_staggered.melt_fraction()
        if np.ndim(phi) == 0:
            phi_mean[i] = float(phi)
        else:
            phi_arr = np.asarray(phi).ravel()
            phi_mean[i] = np.sum(phi_arr * V_cells) / np.sum(V_cells)

        # Cumulative enthalpy change (incremental, trapezoidal rule)
        if i == 0:
            cum_dH[i] = 0.0
        else:
            # Use average of rho*cp at start and end of interval
            # (trapezoidal rule) to match the order of the surface
            # power integration
            dT = T_K - prev_T_K
            rho_cp_mid = 0.5 * (prev_rho_cp + rho_cp)
            dH_step = np.sum(rho_cp_mid * dT * V_cells)
            cum_dH[i] = cum_dH[i - 1] + dH_step

        # Cumulative surface energy loss (trapezoidal rule)
        if i == 0:
            cum_Wsurf[i] = 0.0
        else:
            dt_s = t_s[i] - t_s[i - 1]
            cum_Wsurf[i] = cum_Wsurf[i - 1] + 0.5 * (
                P_surf[i] + P_surf[i - 1]) * dt_s

        prev_T_K = T_K.copy()
        prev_rho_cp = rho_cp.copy()

    return dict(
        t_yr=t_yr, T_surf=T_surf, phi_mean=phi_mean,
        P_surf=P_surf, cum_dH=cum_dH, cum_Wsurf=cum_Wsurf,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Run 1: composite phase (with phase change)
    # ------------------------------------------------------------------
    cfg_composite = os.path.join(OUT_DIR, "phasechange_composite.cfg")
    write_cfg(cfg_composite, phase="composite", end_time=5000.0, n_nodes=60)
    print("Running composite-phase model...")
    solver_c = run_solver(cfg_composite)
    print(f"  solver status = {solver_c.solution.status}, "
          f"n_timesteps = {solver_c.solution.t.size}")
    ts_c = extract_timeseries(solver_c)

    # ------------------------------------------------------------------
    # Run 2: liquid phase (no phase change)
    # ------------------------------------------------------------------
    cfg_liquid = os.path.join(OUT_DIR, "phasechange_liquid.cfg")
    write_cfg(cfg_liquid, phase="liquid", end_time=5000.0, n_nodes=60)
    print("Running liquid-phase model...")
    solver_l = run_solver(cfg_liquid)
    print(f"  solver status = {solver_l.solution.status}, "
          f"n_timesteps = {solver_l.solution.t.size}")
    ts_l = extract_timeseries(solver_l)

    # ------------------------------------------------------------------
    # Energy budget analysis
    # ------------------------------------------------------------------
    # cum_dH should equal -cum_Wsurf (enthalpy lost = energy radiated)
    # cum_dH is negative (cooling), cum_Wsurf is positive (energy leaving)
    energy_residual = ts_c["cum_dH"] + ts_c["cum_Wsurf"]
    # Relative to total energy lost
    total_lost = np.abs(ts_c["cum_Wsurf"])
    rel_residual = np.abs(energy_residual) / np.maximum(total_lost, 1e-30)

    # Skip very early points where both are near zero
    mask_sig = total_lost > 1e20  # at least 1e20 J lost
    if np.any(mask_sig):
        rel_res_sig = rel_residual[mask_sig]
        max_rel = np.max(rel_res_sig)
        med_rel = np.median(rel_res_sig)
        mean_rel = np.mean(rel_res_sig)
    else:
        max_rel = med_rel = mean_rel = np.nan

    print("\n=== Cumulative energy budget (composite phase) ===")
    print(f"  Total enthalpy change: {ts_c['cum_dH'][-1]:.6e} J")
    print(f"  Total surface loss:    {ts_c['cum_Wsurf'][-1]:.6e} J")
    print(f"  Residual (dH + W):     {energy_residual[-1]:.6e} J")
    print(f"  Relative residual (final): {rel_residual[-1]:.3e}")
    print(f"  max |relative residual|:    {max_rel:.3e}")
    print(f"  median |relative residual|: {med_rel:.3e}")
    print(f"  mean |relative residual|:   {mean_rel:.3e}")
    print(f"  (from {np.sum(mask_sig)}/{len(rel_residual)} timesteps "
          f"where cumulative loss > 1e20 J)")

    # Solidus and liquidus at surface pressure (low P ~ 0)
    T_sol_surf = 1381.0  # K (from data file at P=0)
    T_liq_surf = 1800.0  # K (from data file at P=0)

    # Cooling timescale comparison
    mask_c = ts_c["T_surf"] > T_sol_surf
    mask_l = ts_l["T_surf"] > T_sol_surf
    t_solid_c = ts_c["t_yr"][mask_c][-1] if np.any(mask_c) else ts_c["t_yr"][-1]
    t_solid_l = ts_l["t_yr"][mask_l][-1] if np.any(mask_l) else ts_l["t_yr"][-1]
    print("\n=== Cooling timescale comparison ===")
    print(f"  Composite: T_surf reaches solidus ({T_sol_surf:.0f} K) "
          f"at t ~ {t_solid_c:.1f} yr")
    print(f"  Liquid:    T_surf reaches solidus ({T_sol_surf:.0f} K) "
          f"at t ~ {t_solid_l:.1f} yr")
    if t_solid_c > t_solid_l:
        print(f"  -> Composite cools {t_solid_c / t_solid_l:.2f}x slower "
              f"(latent heat effect confirmed)")
    else:
        print("  WARNING: composite did not cool slower than liquid")

    mask_c_liq = ts_c["T_surf"] > T_liq_surf
    t_enter = ts_c["t_yr"][mask_c_liq][-1] if np.any(mask_c_liq) else 0.0
    print(f"  Composite enters mushy zone (T < {T_liq_surf:.0f} K) "
          f"at t ~ {t_enter:.1f} yr")
    print(f"  Time in mushy zone: {t_solid_c - t_enter:.1f} yr")

    print(f"\n  Final T_surf (composite): {ts_c['T_surf'][-1]:.1f} K "
          f"at t = {ts_c['t_yr'][-1]:.1f} yr")
    print(f"  Final T_surf (liquid):    {ts_l['T_surf'][-1]:.1f} K "
          f"at t = {ts_l['t_yr'][-1]:.1f} yr")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Aragog Phase-Change Energy Conservation Verification", fontsize=13)

    # ---- (a) T_surf vs time ----
    ax = axes[0, 0]
    ax.plot(ts_c["t_yr"], ts_c["T_surf"], "C0-", lw=1.5,
            label="Composite (with phase change)")
    ax.plot(ts_l["t_yr"], ts_l["T_surf"], "C1--", lw=1.5,
            label="Liquid (no phase change)")
    ax.axhline(T_liq_surf, color="C3", ls=":", lw=0.8,
               label=f"Liquidus ({T_liq_surf:.0f} K)")
    ax.axhline(T_sol_surf, color="C2", ls=":", lw=0.8,
               label=f"Solidus ({T_sol_surf:.0f} K)")
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("Surface temperature [K]")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("(a) Surface temperature")
    ax.set_xlim(0, 200)
    ax.set_ylim(200, 2600)

    # ---- (b) Mean melt fraction vs time (composite only) ----
    ax = axes[0, 1]
    ax.plot(ts_c["t_yr"], ts_c["phi_mean"], "C0-", lw=1.5)
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("Volume-averaged melt fraction")
    ax.set_title("(b) Melt fraction (composite)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 200)

    # ---- (c) Cumulative energy budget ----
    ax = axes[1, 0]
    ax.plot(ts_c["t_yr"], ts_c["cum_dH"], "C0-", lw=1.5,
            label=r"$\Delta H$ (enthalpy change)")
    ax.plot(ts_c["t_yr"], -ts_c["cum_Wsurf"], "C1--", lw=1.5,
            label=r"$-\int P_{\mathrm{surf}}\,dt$")
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("Energy [J]")
    ax.legend(fontsize=9)
    ax.set_title("(c) Cumulative energy budget")
    ax.set_xlim(left=0)

    # ---- (d) Relative residual of cumulative budget ----
    ax = axes[1, 1]
    # Only plot where total_lost > 0
    mask_plot = total_lost > 0
    ax.semilogy(ts_c["t_yr"][mask_plot], rel_residual[mask_plot],
                "C0-", lw=1.0)
    ax.axhline(1e-2, color="C3", ls="--", lw=0.8, label="1% threshold")
    ax.axhline(1e-6, color="C4", ls=":", lw=0.8, label="1 ppm threshold")
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel(
        r"$|\Delta H + \int P_{\mathrm{surf}}\,dt|"
        r"\;/\;\int P_{\mathrm{surf}}\,dt$")
    ax.set_title("(d) Relative cumulative energy residual")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(1e-8, 1)

    plt.tight_layout()
    pdf_path = os.path.join(OUT_DIR, "verify_phasechange_energy.pdf")
    fig.savefig(pdf_path)
    print(f"\nPlot saved to: {pdf_path}")
    plt.close(fig)

    # Save raw data
    np.savez(
        os.path.join(OUT_DIR, "verify_phasechange_data.npz"),
        t_yr_composite=ts_c["t_yr"],
        T_surf_composite=ts_c["T_surf"],
        phi_mean_composite=ts_c["phi_mean"],
        P_surf_composite=ts_c["P_surf"],
        cum_dH_composite=ts_c["cum_dH"],
        cum_Wsurf_composite=ts_c["cum_Wsurf"],
        t_yr_liquid=ts_l["t_yr"],
        T_surf_liquid=ts_l["T_surf"],
        phi_mean_liquid=ts_l["phi_mean"],
        P_surf_liquid=ts_l["P_surf"],
        cum_dH_liquid=ts_l["cum_dH"],
        cum_Wsurf_liquid=ts_l["cum_Wsurf"],
    )
    print(f"Raw data saved to: "
          f"{os.path.join(OUT_DIR, 'verify_phasechange_data.npz')}")


if __name__ == "__main__":
    main()
