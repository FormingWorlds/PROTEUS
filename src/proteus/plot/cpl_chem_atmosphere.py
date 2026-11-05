from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

from proteus.atmos_chem.common import chem_csv_name, read_result
from proteus.atmos_clim.common import list_atmosphere_times, read_ncdf_profile
from proteus.utils.constants import gas_list, vap_list
from proteus.utils.helper import find_nearest, list_snapshot_times, snapshot_path_for_time
from proteus.utils.plot import get_colour, latexify

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger('fwl.' + __name__)

GASES_STANDARD = (
    'C2H4',
    'CO',
    'H2O',
    'H2SO4',
    'N2',
    'O2',
    'O3',
    'OH',
    'H',
    'SO',
    'CH4',
    'CO2',
    'H2',
    'H2S',
    'HCN',
    'NH2',
    'NH3',
    'OCS',
    'S2',
    'S6',
    'S8',
    'SO2',
    'N2O',
    'NO',
    'NO2',
    'HNO3',
    'PH3',
    'C2H2',
    'NO3',
    'N2O5',
    'HONO',
    'HO2NO2',
    'H2O2',
    'C2H6',
    'CH3',
    'H2CO',
    'HO2',
    'C',
    'N',
    'O',
    'S',
    'SO',
    'CS2',
    'SiO',
    'SiO2',
    'TiO',
    'FeO',
    'MgO',
    'Na',
)


REFRACTORY_GASES = tuple(vap_list)


def _aerosol_gas(name: str, gases: list[str]) -> str | None:
    """Match an aerosol name to a gas name of the same atmosphere file.

    AGNI writes an aerosol as the name of its gas in lower case ('feo' for 'FeO').

    Parameters
    ----------
    name : str
        Aerosol name of the atmosphere file.
    gases : list of str
        Gas names of the same file.

    Returns
    -------
    str or None
        The one gas whose name equals ``name`` apart from case; None when no gas
        or more than one gas matches ('co' with both 'CO' and 'Co').
    """
    match = [g for g in gases if g.lower() == name.lower()]
    return match[0] if len(match) == 1 else None


def plot_chem_atmosphere(
    output_dir: str,
    chem_module: str,
    plot_format='pdf',
    plot_gases: list = None,
    plot_offchem: bool = True,
    xmin: float = 1e-10,
    time: float | None = None,
):
    """
    Plot the chemical composition of the atmosphere.

    Arguments
    ---------
    output_dir : str
        Path to the output directory for the simulation.
    chem_module : str
        Name of the chemistry module that was used.
    plot_format : str, optional
        Format for the output plot file.
    plot_gases : list, optional
        List of gases to plot (otherwise uses default list)
    plot_offchem : bool, optional
        Whether to plot offline chemistry results.
    xmin : float, optional
        Minimum VMR for x-axis of the plot.
    time : float, optional
        Time [yr] to plot. The chemistry snapshot nearest it is drawn, with the
        atmosphere snapshot nearest that. None draws the final chemistry
        snapshot, or the final atmosphere snapshot when there is no chemistry.
    """
    log.info('Plot atmosphere chemical composition')

    # Default species.
    #     Ensure that members of gas_list are first

    if not plot_gases:
        plot_gases = list(gas_list) + list(GASES_STANDARD)

    # Remove duplicates, preserving order
    plot_gases = list(dict.fromkeys(plot_gases))

    # Chemistry snapshot: the final one, or the one nearest the requested time
    atm_offchem = None
    target = time
    use_offchem = plot_offchem and (chem_module is not None) and (chem_module != 'none')
    if use_offchem:
        offchem_dir = os.path.join(output_dir, 'offchem')
        chem_times = list_snapshot_times(offchem_dir, '.csv', prefix=f'{chem_module}_')
        if chem_times:
            target = chem_times[-1] if time is None else find_nearest(chem_times, time)[0]
            atm_offchem = read_result(
                output_dir, chem_module, filename=chem_csv_name(chem_module, target)
            )
        else:
            # Output without per-snapshot files holds only the fixed name
            atm_offchem = read_result(output_dir, chem_module)
    has_offchem = atm_offchem is not None
    if has_offchem:
        # The chemistry grid is its own, so its profiles are drawn at its pressures
        p_offchem = atm_offchem['p'].to_numpy() * 1e-5  # Pa to bar
        atm_offchem.drop(columns=['tmp', 'p', 'z', 'Kzz'], inplace=True)

    # Atmosphere snapshot nearest the chemistry snapshot, else the final one
    atm_times = list_atmosphere_times(output_dir)
    if len(atm_times) == 0:
        log.warning('No atmosphere NetCDF files found in output folder')
        return
    year = atm_times[-1] if target is None else float(find_nearest(atm_times, target)[0])
    nc_fpath = snapshot_path_for_time(os.path.join(output_dir, 'data'), year, '_atm.nc')
    atm_profile = read_ncdf_profile(
        nc_fpath, extra_keys=['pl', 'tmpl', 'x_gas', 'cloud_mmr', 'aer_mmr', 'aerosols']
    )

    parr = atm_profile['pl'] * 1e-5  # convert to bar
    tarr = atm_profile['tmpl']  # temperature profile

    # init plot with two panels
    scale = 1.15
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5 * scale, 6 * scale))

    # ====================
    # Panel 1: Gas Species
    # ====================
    # plot species profiles
    lw = 0.9
    al = 0.8
    vmr_surf = []
    for i, gas in enumerate(plot_gases):
        col = get_colour(gas)
        lbl = latexify(gas)
        vmr = 0.0
        _lw = lw
        if gas in gas_list:
            _lw *= 1.25

        # plot from netCDF (dashed lines). The prepend of ``xarr[0]``
        # adds a TOA point for visual continuity when the source VMR
        # array is shorter than parr by one (the layer-vs-interface
        # convention some atmosphere backends use). When the VMR and
        # parr already match, the prepend overshoots by one and
        # matplotlib raises a shape mismatch; defend by truncating
        # both to the common length before plotting.
        key = gas + '_vmr'
        if key in atm_profile.keys():
            xarr = list(atm_profile[key])
            xarr = [xarr[0]] + xarr
            if np.amax(xarr) >= xmin:
                vmr = float(xarr[-1])
                n = min(len(xarr), len(parr))
                ax1.plot(xarr[:n], parr[:n], ls='dashed', color=col, lw=_lw, alpha=al)

        # plot from offline chemistry, if available (solid lines)
        if has_offchem and (gas in atm_offchem.keys()):
            xarr = list(atm_offchem[gas].values)
            if np.amax(xarr) >= xmin:
                vmr = float(xarr[-1])  # prefer vmr from offline chemistry
                ax1.plot(xarr, p_offchem, ls='solid', color=col, lw=_lw, alpha=al)

        # create legend entry and store surface vmr
        if vmr > 0.0:
            ax1.plot([], [], ls='solid', color=col, lw=_lw, alpha=al, label=lbl)
            vmr_surf.append(vmr)

    # Add temperature profile to Panel 1 (secondary x-axis at top)
    ax1_temp = ax1.twiny()
    ax1_temp.plot(tarr, parr, 'k', lw=1.5, alpha=0.6, label='Temperature')
    ax1_temp.set_xlabel('Temperature [K]')
    ax1_temp.set_xlim(left=0)

    # Decorate Panel 1
    ax1.set_xscale('symlog', linthresh=xmin)
    ax1.set_xlim(left=0, right=1.1)
    ax1.set_xlabel('Gas volume mixing ratio')

    ax1.set_ylabel('Pressure [bar]')
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=np.amax(parr), top=np.amin(parr))
    ax1.yaxis.set_major_locator(LogLocator(numticks=1000))

    # Legend (handles sorted by surface vmr)
    handles, labels = ax1.get_legend_handles_labels()
    order = np.argsort(vmr_surf)[::-1]
    ax1.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc='center left',
        bbox_to_anchor=(1.00, 0.5),
        ncols=max(1, len(order) // 18 + 1),
        fontsize=8,
        borderpad=0.0,
        labelspacing=0.3,
        columnspacing=1.0,
        handlelength=0.9,
        handletextpad=0.3,
        frameon=False,
    )

    # =============================
    # Panel 2: Aerosols and Clouds
    # =============================
    # Cloud profiles
    if 'cloud_mmr' in atm_profile.keys():
        cloud_mmr = atm_profile['cloud_mmr']
        # Prepend matches the gas VMR case above; truncate to the
        # common length so the plot does not fail when the source
        # mmr array is already the same length as parr.
        cloud_arr = [cloud_mmr[0]] + list(cloud_mmr)
        n = min(len(cloud_arr), len(parr))
        ax2.plot(
            cloud_arr[:n],
            parr[:n],
            ls='solid',
            color=get_colour('cloud'),
            lw=1.5,
            alpha=0.7,
            label='cloud',
        )

    # Aerosol profiles
    # Check for individual aerosol species
    num_aerosols = 0
    if 'aerosols' in atm_profile.keys():
        gases = [k[:-4] for k in atm_profile if k.endswith('_vmr')]
        for aer_name in atm_profile['aerosols']:
            num_aerosols += 1
            gas = _aerosol_gas(aer_name, gases)
            aer_mmr = atm_profile[f'{aer_name}_mmr']
            aer_arr = [aer_mmr[0]] + list(aer_mmr)
            n = min(len(aer_arr), len(parr))
            ax2.plot(
                aer_arr[:n],
                parr[:n],
                ls='solid',
                lw=1.5,
                alpha=0.7,
                color=get_colour(gas or aer_name),
                label=latexify(gas) if gas else aer_name,
            )

    # Add temperature profile to Panel 2 (secondary x-axis at top)
    ax2_temp = ax2.twiny()
    ax2_temp.plot(tarr, parr, 'k', lw=1.5, alpha=0.6)
    ax2_temp.set_xticks([])  # hide x-tick labels on secondary axis
    ax2_temp.set_xlim(left=0)

    # Decorate Panel 2
    ax2.set_xscale('symlog', linthresh=xmin)
    ax2.set_xlim(left=0, right=1.1)
    ax2.set_xlabel('Aerosol mass mixing ratio')

    ax2.set_ylabel('Pressure [bar]')
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=np.amax(parr), top=np.amin(parr))
    ax2.yaxis.set_major_locator(LogLocator(numticks=1000))

    ax2.legend(
        loc='center left',
        bbox_to_anchor=(1.00, 0.5),
        ncols=max(1, num_aerosols // 18 + 1),
        fontsize=8,
        borderpad=0.0,
        labelspacing=0.3,
        columnspacing=1.0,
        handlelength=0.9,
        handletextpad=0.3,
        frameon=False,
    )

    # Add time annotation
    ax1.text(
        0.02,
        0.98,
        f't = {year:.2e} yr',
        transform=ax1.transAxes,
        ha='left',
        va='top',
        fontsize=11,
        weight='bold',
        zorder=999,
    )

    # Save file
    fpath = os.path.join(output_dir, 'plots', 'plot_chem_atmosphere.%s' % plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


def plot_chem_atmosphere_entry(handler: Proteus):
    plot_chem_atmosphere(
        output_dir=handler.directories['output'],
        chem_module=handler.config.atmos_chem.module,
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv

    handler = get_handler_from_argv()
    plot_chem_atmosphere_entry(handler)
