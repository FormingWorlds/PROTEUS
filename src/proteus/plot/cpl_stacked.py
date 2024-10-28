from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib.ticker import MultipleLocator

from proteus.atmos_clim.common import read_atmosphere_data
from proteus.interior.wrapper import read_interior_data
from proteus.utils.plot import get_colour, latex_float, sample_output

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_stacked(output_dir: str, times: list, int_data:list, atm_data:list,
                        module:str, plot_format: str="pdf"):

    if np.amax(times) < 2:
        log.debug("Insufficient data to make plot_stacked")
        return
    if module not in ["spider","aragog"]:
        log.warning(f"Cannot make stacked plot for interior module '{module}'")
        return

    log.info("Plot stacked")

    scale = 1.0
    fig,(axt,axb) = plt.subplots(2,1, figsize=(5*scale,10*scale), sharex=True)

    norm = mpl.colors.LogNorm(vmin=max(1,times[0]), vmax=times[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm)
    sm.set_array([])

    # limits
    y_depth, y_height = 100, 100
    lw = 1.5

    # loop over times
    for i,time in enumerate(times):

        # Get atmosphere data for this time
        prof = atm_data[i]
        atm_z = prof["z"]/1e3
        atm_t = prof["t"]

        # Get temperuture-depth interior data for this time
        ds = int_data[i]
        if module == "aragog":
            temperature_interior = ds["temp_b"][:]
            xx_radius = ds["radius_b"][:]
            xx_depth = xx_radius[-1] - xx_radius
        elif module == "spider":
            temperature_interior = ds.get_dict_values(['data','temp_b'])
            xx_radius = ds.get_dict_values(['data','radius_b']) * 1e-3
            xx_depth = xx_radius[0] - xx_radius

        # Determine mixed phase region
        if module == "aragog":
            yy = np.array(ds["phi_b"])
            MASK_SO = yy < 0.05
            MASK_MI = (0.05 <= yy) & ( yy <= 0.95)
            MASK_ME = yy > 0.95
        elif module == "spider":
            MASK_MI = ds.get_mixed_phase_boolean_array('basic')
            MASK_ME = ds.get_melt_phase_boolean_array( 'basic')
            MASK_SO = ds.get_solid_phase_boolean_array('basic')

        # overlap lines by 1 node
        MASK_MI[2:-2] = MASK_MI[2:-2] | MASK_MI[1:-3] | MASK_MI[3:-1]

        # Plot decorators
        label = latex_float(time)+" yr"
        color = sm.to_rgba(time)

        # Plot atmosphere
        axt.plot( atm_t, atm_z, color=color, label=label, lw=lw)

        # Plot interior
        axb.plot( temperature_interior[MASK_SO], xx_depth[MASK_SO], linestyle='solid',  color=color, lw=lw )
        axb.plot( temperature_interior[MASK_MI], xx_depth[MASK_MI], linestyle='dashed', color=color, lw=lw )
        axb.plot( temperature_interior[MASK_ME], xx_depth[MASK_ME], linestyle='dotted', color=color, lw=lw )

        # update limits
        y_height = max(y_height, np.amax(atm_z))
        y_depth  = max(y_depth,  np.amax(xx_depth))


    ytick_spacing = 100.0 # km

    # Decorate top plot
    axt.set(ylabel="Atmosphere height [km]")
    axt.set_ylim(bottom=0.0, top=y_height)
    axt.legend()
    axt.yaxis.set_major_locator(MultipleLocator(ytick_spacing))
    axt.set_facecolor(get_colour("atm_bkg"))

    # Decorate bottom plot
    axb.set(ylabel="Interior depth [km]", xlabel="Temperature [K]")
    axb.invert_yaxis()
    axb.set_ylim(top=0.0, bottom=y_depth)
    axb.yaxis.set_minor_locator(MultipleLocator(ytick_spacing))
    axb.set_facecolor(get_colour("int_bkg"))
    axb.xaxis.set_major_locator(MultipleLocator(1000))
    axb.xaxis.set_minor_locator(MultipleLocator(250))

    axb.plot([1000,1001],[-100,-200], ls='solid',  c='k', lw=lw, label="Solid")
    axb.plot([1000,1001],[-100,-200], ls='dashed', c='k', lw=lw, label="Mush")
    axb.plot([1000,1001],[-100,-200], ls='dotted', c='k', lw=lw, label="Melt")
    axb.legend(fancybox=True, framealpha=0.9, loc='lower left')

    fig.subplots_adjust(hspace=0)

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plot_stacked.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


def plot_stacked_entry(handler: Proteus):
    plot_times,_ = sample_output(handler, extension="_atm.nc", tmin=1e3)
    print("Snapshots:", plot_times)

    int_module = handler.config.interior.module
    output_dir = handler.directories['output']

    int_data = read_interior_data(output_dir, int_module, plot_times)
    atm_data = read_atmosphere_data(output_dir, plot_times)

    plot_stacked(
        output_dir=output_dir, times=plot_times, int_data=int_data, atm_data=atm_data,
        module=int_module, plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_stacked_entry(handler)
