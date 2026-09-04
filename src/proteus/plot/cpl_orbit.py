from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from proteus.orbit.common import _solve_e_stationary
from proteus.orbit.wrapper import read_tides_data
from proteus.utils.constants import (
    AU,
    M_earth,
    R_earth,
    const_G,
    secs_per_day,
    secs_per_hour,
    secs_per_year,
)
from proteus.utils.plot import sample_output

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger('fwl.' + __name__)


def plot_orbit(
    hf_all: pd.DataFrame, output_dir: str, plot_format: str = 'pdf', t0: float = 100.0
):
    time = np.array(hf_all['Time'])
    if np.amax(time) <= t0:
        log.debug('Insufficient data to make plot_orbit')
        return

    log.info('Plot orbit')

    # Plotting parameters
    lw = 2.0
    figscale = 1.2
    yext = 1.05

    # 3 Rows (Semi-major axis, Eccentricity, Timescales)
    # 2 Columns (Planet on left, Satellite on right)
    fig, axs = plt.subplots(3, 2, figsize=(11 * figscale, 9 * figscale), sharex=True)

    # ----------------- COLUMN 0: PLANET -----------------
    # Panel 0,0: Planet Semi-major Axis
    y_a_pl = hf_all['semimajorax'] / AU
    axs[0, 0].plot(time, y_a_pl, lw=lw, color='black')
    axs[0, 0].set_ylabel('Semi-major Axis [AU]')
    axs[0, 0].set_ylim(np.amin(y_a_pl) / yext, np.amax(y_a_pl) * yext)
    axs[0, 0].set_title('Planet Orbiting Star')
    axs[0, 0].grid(alpha=0.2)

    # Panel 1,0: Planet Eccentricity
    y_e_pl = hf_all['eccentricity']
    axs[1, 0].plot(time, y_e_pl, lw=lw, color='tab:blue')
    axs[1, 0].set_ylabel('Eccentricity')
    ymin_e_pl = np.amin(y_e_pl) / yext
    ymax_e_pl = max(np.amax(y_e_pl) * yext, ymin_e_pl + 0.01)
    axs[1, 0].set_ylim(ymin_e_pl, ymax_e_pl)
    axs[1, 0].grid(alpha=0.2)

    # Panel 2,0: Planet Rotational & Orbital Periods (Time comparison)
    p_orb_pl = hf_all['orbital_period'] / secs_per_day
    p_spin_pl = hf_all['axial_period'] / secs_per_hour

    # Left Y-axis: Orbital Period
    ax_left = axs[2, 0]
    l1 = ax_left.plot(time, p_orb_pl, lw=lw, label='Orbital Period', color='tab:orange')
    ax_left.set_ylabel('Orbital Period [days]', color='tab:orange')
    ax_left.tick_params(axis='y', labelcolor='tab:orange')
    ax_left.set_yscale('log')
    ax_left.grid(alpha=0.2, which="both")

    # Right Y-axis: Spin Period
    ax_right = ax_left.twinx()
    l2 = ax_right.plot(time, p_spin_pl, lw=lw, label='Axial Spin Period', color='tab:red')
    ax_right.set_ylabel('Axial Spin Period [hours]', color='tab:red')
    ax_right.tick_params(axis='y', labelcolor='tab:red')
    ax_right.set_yscale('log')

    # Combined Legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc='best')

    # ----------------- COLUMN 1: SATELLITE -----------------
    # Check if satellite columns exist to avoid KeyErrors
    has_sat = 'semimajorax_sat' in hf_all.columns

    if has_sat:
        # Panel 0,1: Satellite Semi-major Axis
        # Using AU to keep consistent scale, or feel free to use e.g. 1e6 meters or Earth-Radii
        y_a_sat = hf_all['semimajorax_sat'] / R_earth
        axs[0, 1].plot(time, y_a_sat, lw=lw, color='black')
        axs[0, 1].set_ylabel('Semi-major Axis [R_earth]')
        axs[0, 1].set_ylim(np.amin(y_a_sat) / yext, np.amax(y_a_sat) * yext)
        axs[0, 1].set_title('Satellite Orbiting Planet')
        axs[0, 1].grid(alpha=0.2)

        # Panel 1,1: Satellite Eccentricity
        y_e_sat = hf_all['eccentricity_sat']
        axs[1, 1].plot(time, y_e_sat, lw=lw, color='tab:blue')
        axs[1, 1].set_ylabel('Eccentricity')
        ymin_e_sat = np.amin(y_e_sat) / yext
        ymax_e_sat = max(np.amax(y_e_sat) * yext, ymin_e_sat + 0.01)
        axs[1, 1].set_ylim(ymin_e_sat, ymax_e_sat)
        axs[1, 1].grid(alpha=0.2)

        # Panel 2,1: Satellite Periods & Optional Precession
        p_orb_sat = hf_all['orbital_period_sat'] / secs_per_hour
        p_spin_sat = hf_all['axial_period_sat'] / secs_per_hour

        axs[2, 1].plot(time, p_orb_sat, lw=lw, label='Orbital Period', color='tab:orange')
        axs[2, 1].plot(time, p_spin_sat, lw=lw, label='Axial Spin Period', color='tab:red')
        axs[2, 1].set_ylabel('Periods [hours]')
        axs[2, 1].set_yscale('log')
        axs[2, 1].legend(loc='best')
        axs[2, 1].grid(alpha=0.2, which="both")
    else:
        # Gracefully leave satellite panels blank/notate if not simulated
        for row in range(3):
            axs[row, 1].text(0.5, 0.5, 'No Satellite Data', transform=axs[row, 1].transAxes,
                             ha='center', va='center', color='grey')

    # ----------------- SHARED X-AXIS CONFIG -----------------
    for ax in axs.flat:
        ax.set_xscale('log')
        ax.set_xlim(left=t0, right=np.amax(time))

    axs[2, 0].set_xlabel('Time [yr]')
    axs[2, 1].set_xlabel('Time [yr]')

    fig.tight_layout()

    # Save the figure
    fpath = os.path.join(output_dir, 'plots', 'plot_orbit.%s' % plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')

    plt.close(fig)
    plt.ioff()


def plot_orbit_system(hf_all: pd.DataFrame, output_dir: str, plot_format: str = 'pdf', t0=1e3):
    if np.amax(hf_all['Time']) <= t0 + 1:
        log.debug('Insufficient data to make plot_system')
        return

    log.info('Plot orbit_system')

    # Plotting parameters
    lw_pla = 1.2
    lw_sat = 0.8
    figscale = 1.4
    fig, ax = plt.subplots(1, 1, figsize=(4 * figscale, 4 * figscale))

    # plot star
    ax.scatter(0, 0, color='orange', s=60, zorder=4, label='Star', marker='*')

    # Colors
    times = np.array(hf_all['Time'][:])
    norm = mpl.colors.LogNorm(vmin=t0, vmax=times[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlow, norm=norm)
    sm.set_array([])

    # plot planet at time
    t = np.linspace(0, np.pi * 2, 80)

    def _plot_planet(i):
        hf_row = hf_all.iloc[i]
        col = sm.to_rgba(hf_row['Time'])

        # planet orbit parameters
        a = hf_row['semimajorax'] / AU
        e = hf_row['eccentricity']
        b = a * np.sqrt(1 - e * e)

        # location of focus
        f = a * e

        # plot ellipse of planet orbit
        x = a * np.cos(t) - f
        y = b * np.sin(t)
        ax.plot(x, y, color=col, alpha=0.8, zorder=5, lw=lw_pla)

        # plot satellite orbit around planet
        asat = hf_row['semimajorax_sat'] / AU
        x0 = np.amin(x)
        xx = asat * np.cos(t) + x0
        yy = asat * np.sin(t)
        ax.plot(xx, yy, lw=lw_sat, color=col, alpha=0.4, zorder=5)

        return max(rmax, np.amax(np.abs(x)))

    # make orbits
    rmax = 0.01
    for i in range(len(hf_all)):
        rmax = max(_plot_planet(i), rmax)

    # roche radius of star
    roche = hf_all.iloc[-1]['roche_limit'] / AU
    ax.plot(roche * np.cos(t), roche * np.sin(t), ls='dashed', c='tab:red', label='Roche limit')

    # Plot colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=-0.2)
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Time [yr]')

    # dummy labels
    ax.plot([], [], label='Planet orbit', c='purple', lw=lw_pla)
    ax.plot([], [], label='Moon orbit', c='purple', lw=lw_sat)

    # decorate
    rmax *= 1.2
    lims = (-rmax, rmax)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xticklabels([])
    ax.set_ylabel('Distance [AU]')
    ax.grid(zorder=0, alpha=0.3)
    ax.legend(loc='upper right')

    plt.close()
    plt.ioff()

    fig.tight_layout()

    fpath = os.path.join(output_dir, 'plots', 'plot_orbit_system.%s' % plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


def plot_evection(
    hf_all: pd.DataFrame, output_dir: str, plot_format: str = 'pdf', t0: float = 100.0,
    xscale: str = 'linear', t_max: float = 1e5,
    fine_t=None, fine_phi=None, filter_toggle_t=None,
):
    """Plot the evection diagnostics."""
    time = np.array(hf_all['Time'])
    if np.amax(time) <= t0:
        log.debug('Insufficient data to make plot_evection')
        return

    log.info('Plot evection')

    lw = 2.0
    figscale = 1.2
    yext = 1.05

    fig, axs = plt.subplots(4, 1, figsize=(11 * figscale, 9 * figscale), sharex=True)

    Omega_earth = np.sqrt(const_G * M_earth / R_earth**3)
    Omega_sun = 2 * np.pi / secs_per_year
    J_star = 0.315
    Lambda = np.sqrt(1.5 * J_star * Omega_earth / Omega_sun)
    Omega_ratio = Omega_sun / Omega_earth

    a_prime = (hf_all['semimajorax_sat'] / R_earth).to_numpy()
    e_arr = hf_all['eccentricity_sat'].to_numpy()
    s_prime = (2 * np.pi / hf_all['axial_period'].to_numpy()) / Omega_earth

    with np.errstate(invalid='ignore'):
        a_res = (Lambda * s_prime / (1.0 - e_arr**2)) ** (4.0 / 7.0)

    e_s = np.array([
        _solve_e_stationary(a_prime[i], s_prime[i], Lambda, Omega_ratio)
        for i in range(len(a_prime))
    ])

    # Panel (a): a' and a'_res
    y_a_sat = hf_all['semimajorax_sat'] / R_earth
    axs[0].plot(time, a_res, lw=lw, ls='--', color='#a8c6e8', label="a'_res (evection)", zorder=2)
    axs[0].plot(time, y_a_sat, lw=lw, color='black', label="a' (satellite)", zorder=3)
    axs[0].set_ylabel('Semi-major Axis [R_Earth]')
    axs[0].set_ylim(np.amin(y_a_sat) / yext, np.amax(y_a_sat) * yext)
    axs[0].set_title('Satellite Orbiting Planet')
    axs[0].legend(loc='best', fontsize=9, framealpha=0.9)
    axs[0].grid(alpha=0.2)

    # Panel (b): e_s and e
    y_e_sat = hf_all['eccentricity_sat']
    axs[1].plot(time, y_e_sat, lw=lw, color='tab:blue', label='e', zorder=2)
    axs[1].plot(time, e_s, lw=lw, ls='--', color='#ff8c00', label='e_s (stable stationary)', zorder=3)
    axs[1].set_ylabel('Eccentricity')
    ymin_e_sat = np.amin(y_e_sat) / yext
    ymax_e_sat = max(np.amax(y_e_sat) * yext, ymin_e_sat + 0.01)
    axs[1].set_ylim(ymin_e_sat, ymax_e_sat)
    axs[1].legend(loc='best', fontsize=9, framealpha=0.9)
    axs[1].grid(alpha=0.2)

    # Panel (c): Resonance/evection angle.
    if fine_t is not None and fine_phi is not None:
        y_evec = np.mod(np.asarray(fine_phi), 2 * np.pi)
        t_evec = np.asarray(fine_t)
    else:
        y_evec = hf_all['evection_angle'].to_numpy().copy()
        if np.ptp(y_evec) < 2 * np.pi - 1e-9:
            pass  # pure libration: bounded already, no wrap needed
        else:
            y_evec = np.mod(y_evec, 2 * np.pi)
        t_evec = time
        log.debug('plot_evection: no fine phi trace supplied -- panel (c) uses '
                   'the coarse, potentially aliased evection_angle column')

    axs[2].plot(t_evec, y_evec, lw=0.8 if fine_t is not None else lw, color='tab:green')
    axs[2].set_ylabel('Evection Angle [rad]')
    axs[2].set_yticks([0, np.pi, 2 * np.pi])
    axs[2].set_yticklabels(['0', r'$\pi$', r'$2\pi$'])
    axs[2].set_ylim(-0.1, 2 * np.pi + 0.1)
    axs[2].grid(alpha=0.2)
    if filter_toggle_t is not None:
        axs[2].axvline(filter_toggle_t, color='crimson', ls=':', lw=1.2, alpha=0.7,
                        label=f'filter activates (t~{filter_toggle_t:.0f} yr)')
        axs[2].legend(loc='upper right', fontsize=9, framealpha=0.9)
    if fine_t is None:
        pass

    # Panel (d): total AM and normalized planet spin
    norm_SR = Omega_earth
    norm_MoI = 0.335 * M_earth * R_earth**2
    norm_AM = norm_MoI * norm_SR

    y_AM = hf_all['plan_sat_am'] / norm_AM
    axs[3].plot(time, y_AM, lw=lw, label='Normalized Angular Momentum', color='tab:orange', zorder=3)

    Omega_p_arr = 2 * np.pi / hf_all['axial_period'].to_numpy()
    s_p_prime = Omega_p_arr / norm_SR
    axs[3].plot(time, s_p_prime, lw=lw, ls='-.', label="Planet Spin, s_p' (normalized)", color='tab:red', zorder=2)

    axs[3].set_ylabel('Normalized AM / Spin')
    axs[3].set_ylim(0.0, 1.0)
    axs[3].legend(loc='best', fontsize=9, framealpha=0.9)
    axs[3].grid(alpha=0.2)

    for ax in (axs[0], axs[1], axs[3]):
        if filter_toggle_t is not None:
            ax.axvline(filter_toggle_t, color='crimson', ls=':', lw=1.0, alpha=0.5)

    if xscale == 'log':
        for ax in axs.flat:
            ax.set_xscale('log')
            ax.set_xlim(left=t0, right=t_max)
        axs[3].set_xlabel('Time [log10(yr)]')
    else:
        import matplotlib.ticker as mticker
        for ax in axs.flat:
            ax.set_xscale('linear')
            ax.set_xlim(left=0.0, right=t_max)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(1e4))
        axs[3].set_xlabel('Time [yr]')

    fig.tight_layout()

    # Save figure
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    fpath = os.path.join(output_dir, 'plots', 'plot_evection.%s' % plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')

    plt.close(fig)
    plt.ioff()


def plot_Lovenumber(output_dir: str, times: list | np.ndarray, data: list, plot_format: str = 'pdf'):
    if times is None or len(times) == 0:
        log.debug('No times provided for plot_Lovenumber')
        return

    if np.amax(times) < 2:
        log.debug('Insufficient data to make plot_interior')
        return

    log.info('Plot Lovenumber')

    # Structure data by unique mode across all time steps
    # Key: (n, m, k), Value: dict of array lists
    modes = {}

    for i, time in enumerate(times):
        ds = data[i]

        n_arr = ds["n"][:]
        m_arr = ds["m"][:]
        k_arr = ds["k"][:]
        sigma_arr = ds["sigma_range"][:]
        raw_imag = ds["knms_total"]
        knms_total = raw_imag[0, :] + 1j * raw_imag[1, :]

        # Group data per mode index
        for j in range(len(n_arr)):
            mode_key = (int(n_arr[j]), int(m_arr[j]), int(k_arr[j]))
            if mode_key not in modes:
                modes[mode_key] = {
                    "time": [],
                    "sigma": [],
                    "real_log": [],
                    "imag_log": []
                }

            real_val = np.log10(np.abs(knms_total[j].real)) if knms_total[j].real != 0 else -np.inf
            imag_val = np.log10(np.abs(knms_total[j].imag)) if knms_total[j].imag != 0 else -np.inf

            modes[mode_key]["time"].append(time)
            modes[mode_key]["sigma"].append(np.abs(sigma_arr[j]))
            modes[mode_key]["real_log"].append(real_val)
            modes[mode_key]["imag_log"].append(imag_val)

    # Determine global colorbar bounds across all mode points
    all_real_log = [val for mode in modes.values() for val in mode["real_log"] if np.isfinite(val)]
    all_imag_log = [val for mode in modes.values() for val in mode["imag_log"] if np.isfinite(val)]

    if not all_real_log or not all_imag_log:
        log.warning("No valid non-zero Love numbers to plot.")
        return

    vmin_real, vmax_real = np.min(all_real_log), np.max(all_real_log)
    vmin_imag, vmax_imag = np.min(all_imag_log), np.max(all_imag_log)

    # Setup Figure
    scale = 1.0
    fig, axs = plt.subplots(1, 2, figsize=(14 * scale, 6 * scale), sharey=True)

    cmap_real = plt.get_cmap('plasma')
    cmap_imag = plt.get_cmap('viridis')

    # Plot connecting lines and mode markers
    for mode_key, mode_data in modes.items():
        # Sort trajectories chronologically by time
        sort_idx = np.argsort(mode_data["time"])
        t_sorted = np.array(mode_data["time"])[sort_idx]
        x_vals = np.log10(t_sorted)
        y_vals = np.array(mode_data["sigma"])[sort_idx]

        real_vals = np.array(mode_data["real_log"])[sort_idx]
        imag_vals = np.array(mode_data["imag_log"])[sort_idx]

        # Draw connecting trajectory lines across time
        axs[0].plot(x_vals, y_vals, color='gray', linestyle='-', linewidth=0.8, alpha=0.4, zorder=1)
        axs[1].plot(x_vals, y_vals, color='gray', linestyle='-', linewidth=0.8, alpha=0.4, zorder=1)

        # Overlay scatter points colored by magnitude
        sc_real = axs[0].scatter(
            x_vals, y_vals, c=real_vals, cmap=cmap_real,
            vmin=vmin_real, vmax=vmax_real, edgecolors='none', s=20, alpha=0.8, zorder=2
        )

        sc_imag = axs[1].scatter(
            x_vals, y_vals, c=imag_vals, cmap=cmap_imag,
            vmin=vmin_imag, vmax=vmax_imag, edgecolors='none', s=20, alpha=0.8, zorder=2
        )

    # Formatting & Colorbars
    for ax in axs:
        ax.set_yscale('log')
        ax.set_xlabel(r'$\log_{10}(\text{Time [yr]})$')
        ax.grid(True, which="both", ls="--", alpha=0.5)

    axs[0].set_ylabel(r'Forcing Frequency $|\sigma|$ (Log Scale)')
    axs[0].set_title(r'Real Part: $\log_{10}(|\text{Re}(k_{nm})|)$')
    axs[1].set_title(r'Imaginary Part: $\log_{10}(|\text{Im}(k_{nm})|)$')

    fig.colorbar(sc_real, ax=axs[0], orientation='vertical', shrink=0.8, label=r'$\log_{10}(|\text{Re}(k_{nm})|)$')
    fig.colorbar(sc_imag, ax=axs[1], orientation='vertical', shrink=0.8, label=r'$\log_{10}(|\text{Im}(k_{nm})|)$')

    fig.tight_layout()

    # Save figure
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    fpath = os.path.join(output_dir, 'plots', f'plot_Lovenumber.{plot_format}')
    fig.savefig(fpath, dpi=200, bbox_inches='tight')

    plt.close(fig)
    plt.ioff()



def plot_evection_A(
    hf_all: pd.DataFrame, output_dir: str, plot_format: str = 'pdf', t0: float = 100.0
):
    """Plot the relative strength of satellite- vs planet-raised tides.

    Generalizes the CTL/CPL 'A' parameter (e.g. Zahnle et al. 2015, eq. 9,
    A = k2m*dtm / (k2p*dt) * (Mp/m)^2 * (Rm/Rp)^5) to a dynamical-tide model
    with a full (n,m,k) Love-number spectrum. Rather than a single frequency-
    independent k2*dt per body, A_a and A_e are computed directly from the
    per-body contributions to da/dt and de/dt (see sma_dot_planet/sat and
    ecc_dot_planet/sat in ps1d): the ratio of the satellite-raised ('lunar')
    to planet-raised ('Earth') tidal contribution to each quantity.

    A > 1 means the satellite-raised tide dominates that quantity's evolution;
    A < 1 means the planet-raised tide dominates. Unlike the CTL 'A', these
    are generally time-dependent and process-dependent -- A_a need not equal
    A_e, since da/dt and de/dt sample different combinations of tidal modes.

    Parameters
    ----------
        hf_all : pd.DataFrame
            Helpfile history. Must contain 'Time', 'sma_dot_planet',
            'sma_dot_sat', 'ecc_dot_planet', 'ecc_dot_sat'.
        output_dir : str
            Directory in which to save the plot (a 'plots' subfolder is used).
        plot_format : str
            Image format for the saved figure.
        t0 : float
            Lower time bound [yr] for the plot / minimum data requirement.
    """

    time = np.array(hf_all['Time'])
    if np.amax(time) <= t0:
        log.debug('Insufficient data to make plot_evection_A')
        return

    required = ('sma_dot_planet', 'sma_dot_sat', 'ecc_dot_planet', 'ecc_dot_sat')
    missing = [c for c in required if c not in hf_all.columns]
    if missing:
        log.warning(f'plot_evection_A: missing columns {missing}, skipping plot')
        return

    log.info('Plot tidal dominance ratios (A_a, A_e)')

    # Ratio of |satellite-raised| to |planet-raised| tidal contributions.
    # np.errstate silences the expected divide-by-zero at synchronisation
    # crossings, where the planet-raised term passes through zero; those
    # points are masked to NaN below so the line shows a gap rather than a
    # spurious spike to +/- infinity.
    with np.errstate(divide='ignore', invalid='ignore'):
        A_a = np.abs(hf_all['sma_dot_sat'].to_numpy()) / np.abs(hf_all['sma_dot_planet'].to_numpy())
        A_e = np.abs(hf_all['ecc_dot_sat'].to_numpy()) / np.abs(hf_all['ecc_dot_planet'].to_numpy())
    A_a = np.where(np.isfinite(A_a) & (A_a > 0), A_a, np.nan)
    A_e = np.where(np.isfinite(A_e) & (A_e > 0), A_e, np.nan)

    lw = 2.0
    figscale = 1.2

    col_planet = 'tab:blue'   # validated diverging pair (CVD delta-E ~21-34)
    col_sat    = 'tab:red'
    col_mid    = 'dimgray'

    fig, axs = plt.subplots(2, 1, figsize=(11 * figscale, 6.0 * figscale), sharex=True)

    panels = (
        (axs[0], A_a, r'$A_a = |\dot{a}_{\rm sat}| \, / \, |\dot{a}_{\rm planet}|$'),
        (axs[1], A_e, r'$A_e = |\dot{e}_{\rm sat}| \, / \, |\dot{e}_{\rm planet}|$'),
    )

    for ax, A, ylabel in panels:
        ax.plot(time, A, lw=lw, color='black', zorder=3)
        ax.axhline(1.0, lw=1.2, ls='--', color=col_mid, zorder=2)

        # Diverging fill against the A=1 crossover
        ax.fill_between(time, A, 1.0, where=(A >= 1.0), color=col_sat,
                         alpha=0.15, interpolate=True, zorder=1)
        ax.fill_between(time, A, 1.0, where=(A < 1.0), color=col_planet,
                         alpha=0.15, interpolate=True, zorder=1)

        ax.set_yscale('log')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

    axs[0].set_title('Relative strength of satellite- vs planet-raised tides')
    axs[1].set_xlabel('Time [log10(yr)]')

    handles = [
        Line2D([0], [0], color='black', lw=lw, label='A'),
        Line2D([0], [0], color=col_mid, lw=1.2, ls='--', label='A = 1 (equal strength)'),
        Patch(color=col_sat, alpha=0.3, label='satellite tide dominates'),
        Patch(color=col_planet, alpha=0.3, label='planet tide dominates'),
    ]
    axs[0].legend(handles=handles, loc='upper right', fontsize=9, framealpha=0.9)

    for ax in axs.flat:
        ax.set_xscale('log')
        ax.set_xlim(left=t0, right=1e5)

    fig.tight_layout()

    fpath = os.path.join(output_dir, 'plots', 'plot_evection_A.%s' % plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')

    plt.close(fig)
    plt.ioff()


def plot_orbit_entry(handler: Proteus):
    # read helpfile
    hf_all = pd.read_csv(
        os.path.join(handler.directories['output'], 'runtime_helpfile.csv'), sep=r'\s+'
    )

    # plots for orbit
    # make plot
    plot_orbit(
        hf_all,
        handler.directories['output'],
        plot_format=handler.config.params.out.plot_fmt,
    )
    plot_orbit_system(
        hf_all,
        handler.directories['output'],
        plot_format=handler.config.params.out.plot_fmt,
    )

    if handler.config.orbit.planet_satellite_model == 'ps1d_evec':
        # get data from fine output for evection angle
        fine_t, fine_phi = None, None
        fine_path = os.path.join(handler.directories['output/data'], 'fine_evection_data.csv')

        if os.path.exists(fine_path):
            try:
                fine_t, fine_phi = np.loadtxt(fine_path, skiprows=1, delimiter=',').T
            except Exception as e:
                log.warning(f"Failed to load fine evection data: {e}")

        plot_evection(
            hf_all,
            handler.directories['output'],
            plot_format=handler.config.params.out.plot_fmt,
            t0=1e1,
            xscale='linear',
            t_max=1e5,
            fine_t=fine_t,
            fine_phi=fine_phi,
        )

    # plots for tides
    # if obliqua plot the Lovenumber spectrum evolution
    if handler.config.orbit.module == 'obliqua':
        extension = '_obliqua.nc'

        plot_times, _ = sample_output(handler, extension=extension, tmin=1e3)
        log.info('Snapshots: %s', plot_times)

        data = read_tides_data(handler.directories['output'], 'obliqua', plot_times)

        plot_Lovenumber(
            output_dir=handler.directories['output'],
            times=plot_times,
            data=data,
            plot_format=handler.config.params.out.plot_fmt
        )


if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv

    handler = get_handler_from_argv()
    plot_orbit_entry(handler)
