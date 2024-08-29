from __future__ import annotations

import glob
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from proteus.utils.plot import dict_colors, vol_latex
from proteus.utils.plot_offchem import offchem_read_year

if TYPE_CHECKING:
    from proteus import Proteus

mpl.use("Agg")


def plot_offchem_year(
        output_dir: str,
        year_dict: dict,
        species: list,
        plot_init_mx: bool=False,
    ):
    """Plot a set of species in the atmosphere, for a single year only

    Reads-in the data from output_dir for a single year. Can also
    include JANUS/SPIDER mixing ratios as dashed lines, which were used to
    initialise each VULCAN run.

    Parameters
    ----------
    output_dir : str
        Output directory that was specified in the PROTEUS cfg file
    species : list
        Which species to plot? (e.g. H2O) If none, then plot all.
    tmin : float
        Initial year to include [yr]
    tmax : float
        Final yeat to include [yr]
    plot_init_mx : bool
        Include initial mixing ratios for each VULCAN run in plot?
    """

    if species is None:
        species = []
        for k in year_dict.keys():
            if "_" not in k:
                continue
            species.append(k.split("_")[1])

    lw = 2

    mpl.use("Agg")

    fig, (ax0,ax1) = plt.subplots(1,2,sharey=True,figsize=(8,4))

    # Temperature profile
    ax0.set_title("Time = %g kyr" % (year_dict["year"]/1.e3))
    ax0.set_ylabel("Pressure [bar]")
    ax0.invert_yaxis()
    ax0.set_yscale("log")
    ax0.set_xlabel("Temperature [K]")

    ax0.plot(year_dict["temperature"],year_dict["pressure"],color='black',lw=lw)

    # Mixing ratios
    ax1.set_xlabel("Mole fraction")
    ax1.set_xscale("log")


    min_mix = 5e-1
    for i,s in enumerate(species):

        if s in dict_colors.keys():
            color = dict_colors[s]
            ls = 'solid'
        else:
            color = None
            ls = 'dotted'
            print("Warning: could not find a defined colour for species '%s' " % s)

        pretty = s
        if s in vol_latex.keys():
            pretty = vol_latex[s]

        # VULCAN result
        key = str("mx_"+s)
        if key in year_dict.keys():
            line = ax1.plot(year_dict[key],year_dict["pressure"],label=pretty,ls=ls,lw=lw,color=color)[0]
            color = line.get_color()
            min_mix = min(min_mix,np.amin(year_dict[key]))

        # JANUS result
        if plot_init_mx:
            key = str("mv_"+s)
            if key in year_dict.keys():
                ax1.scatter([year_dict[key]],[np.amax(year_dict["pressure"])],color=color)

    ax1.set_xlim([min_mix,1])
    ax1.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(output_dir+"plot_offchem_year_%d.pdf"%year_dict["year"])
    plt.close(fig)


def plot_offchem_year_entry(handler: Proteus, **kwargs):
    plot_janus = True

    # Species to make plots for
    species = ["H2", "H2O", "H", "OH", "CO2", "CO", "CH4","HCN", "NH3", "N2", "NO"]

    # Load data
    ls = glob.glob(handler.directories["output"]+"offchem/*/output.vul")
    years = [int(f.split("/")[-2]) for f in ls]
    years_data = [offchem_read_year(handler.directories["output"],y,read_const=plot_janus) for y in years]

    if len(years) == 0:
        raise Exception('No VULCAN output files found')

    # Call plotting function
    for yd in years_data:
        print("Year = %d" % yd["year"])
        plot_offchem_year(
            output_dir=handler.directories["output"],
            year_dict=yd,
            species=species,
            plot_init_mx=plot_janus,
        )


if __name__ == '__main__':
    print("Plotting offline chemistry for each year (mixing ratios vs pressure)...")

    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_offchem_year_entry(handler)

    print("Done!")
