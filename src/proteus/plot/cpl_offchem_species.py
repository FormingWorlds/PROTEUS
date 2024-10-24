from __future__ import annotations

import glob
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from proteus.utils.plot import latexify
from proteus.utils.plot_offchem import offchem_read_year

if TYPE_CHECKING:
    from proteus import Proteus

mpl.use("Agg")


def plot_offchem_species(
        output_dir: str,
        specie: str,
        tmin: float=-1.0,
        tmax: float=-1.0,
        plot_init_mx: bool=False
    ):
    """Plot evolution of a single component according to offline VULCAN output

    Reads-in the data from output_dir for a single species. Can also
    include JANUS/SPIDER mixing ratios as dashed lines, which were used to
    initialise each VULCAN run.

    Parameters
    ----------
    output_dir : str
        Output directory that was specified in the PROTEUS cfg file
    specie : str
        Which species to plot? (e.g. H2O)
    tmin : float
        Initial year to include (-1 for start of data)
    tmax : float
        Final year to include (-1 for end of data)
    plot_init_mx : bool
        Include initial mixing ratios for each VULCAN run in plot?
    """

    mpl.use("Agg")

    ls = glob.glob(output_dir+"/offchem/*/output.vul")
    years = [int(f.split("/")[-2]) for f in ls]
    years_data = [offchem_read_year(output_dir,y,read_const=plot_init_mx) for y in years]

    if len(years) == 0:
        raise Exception('No VULCAN output files found')

    lw = 1.3
    alpha = 0.7

    fig, (ax0,ax1) = plt.subplots(1,2,sharey=True,figsize=(8,4))

    ax0.set_ylabel("Pressure [bar]")
    ax0.invert_yaxis()
    ax0.set_yscale("log")
    ax0.set_xlabel("Temperature [K]")
    ax0.set_title("Evolution of $T(p)$")

    ax1.set_title("Evolution of "+latexify(specie))
    ax1.set_xlabel("Mole fraction")
    ax1.set_xscale("log")

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    if (tmin > 0):
        cb_vmin = tmin
    else:
        cb_vmin = max(years_data[1]["year"],1.0)


    if (tmax > 0):
        cb_vmax = tmax
    else:
        cb_vmax = max(years_data[-1]["year"],2.0)


    norm = mpl.colors.LogNorm(vmin=cb_vmin, vmax=cb_vmax)
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')  #, ticks=np.linspace(0,2,N), boundaries=np.arange(-0.05,2.1,.1))
    cbar.set_label("Time [yr]")

    min_mix = 5e-1

    for i, yd in enumerate(years_data):

        if (tmin > -1) and (yd["year"] < tmin):
            continue

        if (tmax > -1) and (yd["year"] > tmax):
            continue

        if (tmin == -1) and (i == 0):
            color = 'blue'
        else:
            color = sm.to_rgba(yd["year"])

        p = yd["pressure"]

        # Temperature profile
        ax0.plot(yd["temperature"],p,color=color,lw=lw,alpha=alpha,label="%1.2e"%yd["year"])

        # Mixing ratios
        key = str("mx_"+specie)
        ax1.plot(yd[key],p,label=specie,color=color,lw=lw,alpha=alpha)

        min_mix = min(min_mix,np.amin(yd[key]))

        if plot_init_mx:
            key = str("mv_"+specie)
            if key in yd.keys():
                ax1.scatter(yd[key],p[0], s=40,color='grey')
                ax1.scatter(yd[key],p[0], s=20,color=color)

    ax1.set_xlim([min_mix,1])

    fig.tight_layout()
    fig.savefig(output_dir+"plot_offchem_species_{specie}.pdf")
    plt.close('all')


def plot_offchem_species_entry(handler: Proteus):
    plot_janus_result = True

    # Species to make plots for
    species = ["H2", "H2O", "H", "OH", "O3", "O2",
               "CO2", "CO", "CH4", "C2H2",
               "HCN", "NH3", "NH2", "N2", "NO"]

    # Call plotting function
    for specie in species:
        print(f"Species = {specie}")
        plot_offchem_species(
            output_dir=handler.directories["output"],
            specie=specie,
            tmin=-1,
            plot_init_mx=plot_janus_result,
        )


if __name__ == '__main__':
    print("Plotting offline chemistry (species vs pressure)...")

    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_offchem_species_entry(handler)

    print("Done!")
