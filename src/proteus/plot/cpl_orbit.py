from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from proteus.orbit.wrapper import read_tides_data
from proteus.utils.constants import AU, secs_per_day
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
    p_spin_pl = hf_all['axial_period'] / secs_per_day

    axs[2, 0].plot(time, p_orb_pl, lw=lw, label='Orbital Period', color='tab:orange')
    axs[2, 0].plot(time, p_spin_pl, lw=lw, label='Axial Spin Period', color='tab:red')
    axs[2, 0].set_ylabel('Periods [days]')
    axs[2, 0].set_yscale('log') # Log scale handles wide differences in spin/orbit well
    axs[2, 0].legend(loc='best')
    axs[2, 0].grid(alpha=0.2, which="both")


    # ----------------- COLUMN 1: SATELLITE -----------------
    # Check if satellite columns exist to avoid KeyErrors
    has_sat = 'semimajorax_sat' in hf_all.columns

    if has_sat:
        # Panel 0,1: Satellite Semi-major Axis
        # Using AU to keep consistent scale, or feel free to use e.g. 1e6 meters or Earth-Radii
        y_a_sat = hf_all['semimajorax_sat'] / AU
        axs[0, 1].plot(time, y_a_sat, lw=lw, color='black')
        axs[0, 1].set_ylabel('Semi-major Axis [AU]')
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
        p_orb_sat = hf_all['orbital_period_sat'] / secs_per_day
        p_spin_sat = hf_all['axial_period_sat'] / secs_per_day

        axs[2, 1].plot(time, p_orb_sat, lw=lw, label='Orbital Period', color='tab:orange')
        axs[2, 1].plot(time, p_spin_sat, lw=lw, label='Axial Spin Period', color='tab:red')

        # If the apsidal precession angle is present, let's plot its rate/timescale
        if 'aps_prec_angle' in hf_all.columns:
            # Approximate precession timescale (2pi / average rate of change)
            dt = np.diff(time)
            dtheta = np.diff(np.unwrap(hf_all['aps_prec_angle']))
            # Avoid divide-by-zero for static steps
            valid = (dt > 0) & (dtheta != 0)
            if np.any(valid):
                # Calculate local precession period in days (assuming time in years, convert to days)
                prec_period_days = np.abs((2 * np.pi) / (dtheta[valid] / (dt[valid] * 365.25)))
                axs[2, 1].plot(time[1:][valid], prec_period_days, lw=lw, ls='--', label='Apsidal Prec. Period', color='tab:purple')

        axs[2, 1].set_ylabel('Periods [days]')
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


def plot_Lovenumber(output_dir: str, times: list | np.ndarray, data: list, plot_format: str = 'pdf'):
    if np.amax(times) < 2:
        log.debug('Insufficient data to make plot_interior')
        return

    log.info('Plot Lovenumber')

    # Init figure with 2 panels (Real and Imaginary)
    scale = 1.0
    fig, axs = plt.subplots(1, 2, figsize=(14 * scale, 6 * scale), sharey=True)

    # Store data across all times to establish consistent global colorbar limits
    all_real_log = []
    all_imag_log = []

    # Pre-parse loop to collect values for robust vmin/vmax limits
    for i in range(len(times)):
        ds = data[i]
        raw_imag = ds.variables["knms_total"]
        knms_total = raw_imag[0, :] + 1j * raw_imag[1, :]

        # Take absolute value to handle negative values safely before log10
        all_real_log.extend(np.log10(np.abs(knms_total.real)))
        all_imag_log.extend(np.log10(np.abs(knms_total.imag)))

    vmin_real, vmax_real = np.min(all_real_log), np.max(all_real_log)
    vmin_imag, vmax_imag = np.min(all_imag_log), np.max(all_imag_log)

    # Loop over all times and plot on the same two panels
    for i, time in enumerate(times):
        ds = data[i]

        # Read dimensions
        sigma_range = ds.variables["sigma_range"][:]

        # Extract complex Love numbers
        raw_imag   = ds.variables["knms_total"]
        knms_total = raw_imag[0, :] + 1j * raw_imag[1, :]

        # 1. Coordinate transformations
        # Horizontal: log10(time in years)
        x_vals = np.full_like(sigma_range, np.log10(time))

        # Vertical: forcing frequency. Ensure positive values for log10 y-axis
        y_vals = np.abs(sigma_range)

        # 2. Color metric transformations (convert to absolute + log10)
        real_color_vals = np.log10(np.abs(knms_total.real))
        imag_color_vals = np.log10(np.abs(knms_total.imag))

        # Panel 0: Real Part
        sc_real = axs[0].scatter(
            x_vals,
            y_vals,
            c=real_color_vals,
            cmap='plasma',
            vmin=vmin_real,
            vmax=vmax_real,
            edgecolors='none',
            alpha=0.7
        )

        # Panel 1: Imaginary Part
        sc_imag = axs[1].scatter(
            x_vals,
            y_vals,
            c=imag_color_vals,
            cmap='viridis',
            vmin=vmin_imag,
            vmax=vmax_imag,
            edgecolors='none',
            alpha=0.7
        )

    # Convert y-axis to logarithmic scaling for both panels
    for ax in axs:
        ax.set_yscale('log')
        ax.set_xlabel(r'$\log_{10}(\text{Time [yr]})$')
        ax.grid(True, which="both", ls="--", alpha=0.5)

    axs[0].set_ylabel(r'Forcing Frequency $|\sigma|$ (Log Scale)')
    axs[0].set_title('Real Part: ' + r'$\log_{10}(|k_{nm}|)$')
    axs[1].set_title('Imaginary Part: ' + r'$\log_{10}(|\text{Im}(k_{nm})|)$')

    # Add dedicated colorbars next to each subplot
    fig.colorbar(sc_real, ax=axs[0], orientation='vertical', shrink=0.8, label=r'$\log_{10}(|\text{Re}(k_{nm})|)$')
    fig.colorbar(sc_imag, ax=axs[1], orientation='vertical', shrink=0.8, label=r'$\log_{10}(|\text{Im}(k_{nm})|)$')

    fig.tight_layout()

    # Save the figure
    fpath = os.path.join(output_dir, 'plots', 'plot_Lovenumber.%s' % plot_format)
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
