from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from proteus.utils.constants import gas_list
from proteus.utils.plot import get_colour, latexify

if TYPE_CHECKING:
    from proteus import Proteus
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def plot_global(hf_all: pd.DataFrame, output_dir: str, config: Config,
                logt: bool=True, tmin: float=1e3):

    if np.amax(hf_all["Time"]) < 3:
        log.debug("Insufficient data to make plot_global")
        return
    hf = hf_all.loc[hf_all["Time"]>0]

    log.info("Plot global")

    # Plotting parameters
    lw=2.0
    al=0.95
    fig_ratio=(3,2)
    fig_scale=4.0
    leg_kwargs = {
        "frameon":1,
        "fancybox":True,
        "framealpha":0.9,
        "labelspacing":0.2,
        "columnspacing":1.1,
        "handletextpad":0.7
    }

    F_asf = np.array(hf["F_ins"]) * config.orbit.s0_factor * (1.0 - config.atmos_clim.albedo_pl) * np.cos(config.orbit.zenith_angle * np.pi/180.0)

    #    Volatile parameters (keys=vols, vals=quantites_over_time)
    vol_present = {} # Is present ever? (true/false)
    vol_vmr     = {} # Volume mixing ratio
    vol_bars    = {} # Surface partial pressure [bar]
    vol_mol_atm = {} # Moles in atmosphere
    vol_mol_int = {} # Moles in interior
    vol_mol_tot = {} # Moles in total
    vol_intpart = {} # Partitioning into int

    for vol in gas_list:
        # Check vmr for presence
        this_vmr = np.array(hf[vol+"_vmr"])
        vol_present[vol] = True

        if np.amax(this_vmr) < 1.0e-10:
            vol_present[vol] = False
            continue
        vol_vmr[vol] = this_vmr

        # Do other variables
        vol_bars[vol]    = np.array(hf[vol+"_bar"])
        vol_mol_atm[vol] = np.array(hf[vol+"_mol_atm"])
        vol_mol_tot[vol] = np.array(hf[vol+"_mol_total"])
        vol_mol_int[vol] = vol_mol_tot[vol] - vol_mol_atm[vol]

        # Volatile partitioning into the interior
        # Requires special treatment for when moles=0, which occurs when atmosphere escapes.
        mask = np.argwhere(vol_mol_tot[vol] < 1e-10) # mask of values where moles=0
        vol_mol_int[vol][mask] = 0
        vol_mol_tot[vol][mask] = 1
        vol_intpart[vol] = vol_mol_int[vol]/vol_mol_tot[vol]

    # Init plot
    fig,axs = plt.subplots(3,2, figsize=(fig_ratio[0]*fig_scale, fig_ratio[1]*fig_scale), sharex=True)
    ax_tl = axs[0][0]
    ax_cl = axs[1][0]
    ax_bl = axs[2][0]
    ax_tr = axs[0][1]
    ax_cr = axs[1][1]
    ax_br = axs[2][1]
    axs = (ax_tl, ax_cl, ax_bl, ax_tr, ax_cr, ax_br) # flatten

    # Set y-labels
    ax_tl.set_ylabel('Upward flux [W m$^{-2}$]')
    ax_cl.set_ylabel(r'$T_\mathrm{s}$ [K]')
    ax_bl.set_ylabel('Planet fraction')
    ax_tr.set_ylabel(r'$p_{\mathrm{i}}$ [bar]')
    ax_cr.set_ylabel(r'$\chi^{\mathrm{atm}}_{\mathrm{i}}$ [%]')
    ax_br.set_ylabel(r'$m^{\mathrm{int}}_{\mathrm{i}}/m^{\mathrm{tot}}_{\mathrm{i}}$ [%]')

    # Set x-labels
    for ax in (ax_bl, ax_br):
        ax.set_xlabel("Time [yr]")

    # Set titles
    ax_titles = [
        "(a) Components of heat flux",
        "(c) Surface temperature",
        "(e) Mantle evolution",
        "(b) Surface gas partial pressure",
        "(d) Surface gas mole fraction",
        "(f) Volatile partitioning into interior"
    ]
    for i,ax in enumerate(axs):
        ax.text(0.011, 0.015, ax_titles[i],
                transform=ax.transAxes,
                horizontalalignment='left',
                verticalalignment='bottom',
                fontsize=11,
                zorder=20,
                bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.1, boxstyle='round')
                )

    # Move right subplots y-axes to right side
    for ax in (ax_tr, ax_cr, ax_br):
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    # Percentage plots
    for ax in (ax_cr, ax_br):
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    # Log axes
    if logt:
        for ax in axs:
            ax.set_xscale("log")
    ax_tl.set_yscale("symlog", linthresh=0.1)
    ax_tr.set_yscale("log")

    # Set xlim
    xmin = max(tmin, 1.0)
    xmax = np.amax(hf["Time"])
    if xmin > xmax/2:
        xmin = 1.0
    if logt:
        xmax = max(1.0e6,xmax)
        xlim = (xmin,10 ** np.ceil(np.log10(xmax*1.1)))
    else:
        xlim = (xmin, xmax)
    for ax in axs:
        ax.set_xlim(xlim[0],  max(xlim[1], xlim[0]+1))

    # PLOT ax_tl
    ax_tl.plot( hf["Time"], hf["F_radio"], color=get_colour("radio"), lw=lw,     alpha=al,  label="Radio")
    ax_tl.plot( hf["Time"], hf["F_tidal"], color=get_colour("tidal"), lw=lw,     alpha=al,  label="Tidal")
    ax_tl.plot( hf["Time"], hf["F_int"],   color=get_colour("int"),   lw=lw*1.5, alpha=al,  label="Net (int.)", ls='dashed')
    ax_tl.plot( hf["Time"], hf["F_atm"],   color=get_colour("atm"),   lw=lw*1.5, alpha=al,  label="Net (atm.)" )
    ax_tl.plot( hf["Time"], hf["F_olr"],   color=get_colour("OLR"),   lw=lw*0.8, alpha=al,  label="OLR" )
    ax_tl.plot( hf["Time"], F_asf,         color=get_colour("ASF"),   lw=lw,     alpha=al,  label="ASF" ,ls='dashed' )
    ax_tl.legend(loc='center left', **leg_kwargs)
    ymin, ymax = 0.0, 100.0
    for k in ("F_int","F_atm","F_olr","F_tidal","F_radio", F_asf):
        if type(k) is str:
            arr = np.array(hf[k])
        else:
            arr = np.array(k)
        ymin = min(ymin, np.amin(arr))
        ymax = max(ymax, np.amax(arr))
    ax_tl.set_ylim(bottom=ymin/1.5, top=ymax*1.5)

    # PLOT ax_cl
    min_temp = np.amin(hf["T_surf"])
    max_temp = np.amax(hf["T_magma"])
    ax_cl.plot(hf["Time"], hf["T_magma"], ls="dashed", lw=lw, alpha=al, color=get_colour("int"))
    ax_cl.plot(hf["Time"], hf["T_surf"],  ls="-",      lw=lw, alpha=al, color=get_colour("atm"))
    ax_cl.set_ylim(min(1000.0,min_temp-25) , max(3500.0,max_temp+25))

    # PLOT ax_bl
    ax_bl.axhline( y=config.struct.corefrac,     ls='dashed', lw=lw*1.5, alpha=al, color=get_colour("core"), label=r'C-M boundary' )
    ax_bl.plot( hf["Time"], 1.0-hf["RF_depth"],   color=get_colour("int"), ls="solid",    lw=lw, alpha=al, label=r'Rheol. front')
    ax_bl.plot( hf["Time"],     hf["Phi_global"], color=get_colour("atm"), linestyle=':', lw=lw, alpha=al, label=r'Melt fraction')

    ax_bl.legend(loc='center left', **leg_kwargs)
    ax_bl.set_ylim(0.0,1.01)

    # PLOT ax_tr
    ax_tr.plot( hf["Time"], hf["P_surf"], color='black', linestyle='dashed', lw=lw*1.5, label=r'Total')
    bar_min, bar_max = 0.1, 10.0
    bar_max = max(bar_max, np.amax(hf["P_surf"]))
    for vol in gas_list:
        if not vol_present[vol]:
            continue
        ax_tr.plot( hf["Time"], vol_bars[vol], color=get_colour(vol), lw=lw, alpha=al, label=latexify(vol))
        bar_min = min(bar_min, np.amin(vol_bars[vol]))
    ax_tr.set_ylim(max(1.0e-7,min(bar_min, 1.0e-1)), bar_max * 2.0)
    ax_tr.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5) )

    # PLOT ax_cr
    for vol in gas_list:
        if not vol_present[vol]:
            continue
        ax_cr.plot( hf["Time"], vol_vmr[vol]*100.0, color=get_colour(vol), lw=lw, alpha=al, label=latexify(vol))
    ax_cr.set_ylim(0, 101)

    # PLOT ax_br
    for vol in gas_list:
        if not vol_present[vol]:
            continue
        ax_br.plot( hf["Time"], vol_intpart[vol]*100.0, color=get_colour(vol), lw=lw, alpha=al, label=latexify(vol))
    ax_br.set_ylim(0,101)
    ax_br.legend(loc='center left', ncol=2, **leg_kwargs).set_zorder(20)

    # Save plot
    fig.subplots_adjust(wspace=0.05,hspace=0.1)
    plt_name = "plot_global"
    if logt:
        plt_name += "_log"
    else:
        plt_name += "_lin"

    fname = os.path.join(output_dir,"plots","%s.%s"%(plt_name,config.params.out.plot_fmt))
    fig.savefig(fname,bbox_inches='tight', dpi=200)


def plot_global_entry(handler: Proteus):
    # read helpfile
    hf_all = pd.read_csv(os.path.join(handler.directories['output'], "runtime_helpfile.csv"), sep=r"\s+")

    # make plot
    for logt in [True,False]:
        plot_global(
            hf_all=hf_all,
            output_dir=handler.directories['output'],
            config=handler.config,
            logt=logt,
            tmin=1e3,
        )

if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_global_entry(handler)
