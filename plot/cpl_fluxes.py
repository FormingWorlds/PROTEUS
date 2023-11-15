#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_ext import *
from utils.plot import *

# Plotting fluxes
def plot_fluxes_atmosphere(output_dir, atm):

    print("Plotting fluxes at each level")

    mpl.use('Agg')

    fig,ax = plt.subplots()

    ax.axvline(0,color='black',lw=0.8)

    pl = atm.pl * 1.e-5

    ax.plot(atm.flux_up_total,pl,color='red',label='UP',lw=1)
    ax.plot(atm.SW_flux_up   ,pl,color='red',label='UP SW',linestyle='dotted',lw=2)
    ax.plot(atm.LW_flux_up   ,pl,color='red',label='UP LW',linestyle='dashed',lw=1)

    ax.plot(-1.0*atm.flux_down_total,pl,color='green',label='DN',lw=2)
    ax.plot(-1.0*atm.SW_flux_down   ,pl,color='green',label='DN SW',linestyle='dotted',lw=3)
    ax.plot(-1.0*atm.LW_flux_down   ,pl,color='green',label='DN LW',linestyle='dashed',lw=2)

    ax.plot(atm.net_flux ,pl,color='black',label='NET')

    ax.set_xscale("symlog")
    ax.set_xlabel("Upward-directed flux [W m-2]")
    ax.set_ylabel("Pressure [Bar]")
    ax.set_yscale("log")
    ax.set_ylim([pl[-1],pl[0]])

    ax.legend(loc='upper left')

    plt.close()
    plt.ioff()
    fig.savefig(output_dir+"/plot_fluxes_atmosphere.pdf")

# Plotting fluxes
def plot_fluxes_global(output_dir, COUPLER_options):

    print("Plotting fluxes")

    # Get values
    df = pd.read_csv(output_dir+"/runtime_helpfile.csv", sep="\t")

    df_atm = df.loc[df['Input']=='Atmosphere'].drop_duplicates(subset=['Time'], keep='last')
    F_net = df_atm["F_atm"]
    F_asf = df_atm["F_ins"] * COUPLER_options["asf_scalefactor"] * np.cos(COUPLER_options["zenith_angle"] * np.pi/180.0)
    F_ins = df_atm["F_ins"] 
    F_olr = df_atm["F_olr"]
    t_atm = df_atm["Time"]

    df_int = df.loc[df['Input']=='Interior'].drop_duplicates(subset=['Time'], keep='last')
    F_int = df_int["F_int"]
    t_int = df_int["Time"]

    # Create plot
    mpl.use('Agg')
    fig,ax = plt.subplots(figsize=(5,4))
    lw = 2.0

    # F=0 line
    # ax.axhline(y=0.0, color='black',lw=0.8)
    
    # Steam runaway line
    ax.axhline(y=285.0, color='black', lw=lw, linestyle='dashed', label="Runaway")

    # Plot fluxes
    ax.plot(t_int, F_int, lw=lw, color=dict_colors["qred"],   label="Interior")
    ax.plot(t_atm, F_net, lw=lw, color=dict_colors["qgray"],  label="Atmosphere")
    ax.plot(t_atm, F_ins, lw=lw, color=dict_colors["qturq"],  label="Instellation")
    ax.plot(t_atm, F_olr, lw=lw, color=dict_colors["NO_3"],   label="OLR")
    ax.plot(t_atm, F_asf, lw=lw, color=dict_colors["S_2"],    label="ASF")

    # Configure plot
    ax.set_yscale("symlog")
    ax.set_ylabel("Unsigned flux [W m-2]")
    ax.set_xscale("log")
    ax.set_xlabel("Time [yr]")
    ax.legend(loc='lower left')
    ax.grid(color='black', alpha=0.05)

    plt.close()
    plt.ioff()
    fig.savefig(output_dir+"/plot_fluxes.pdf", bbox_inches='tight')

if __name__ == '__main__':

    if len(sys.argv) == 2:
        cfg = sys.argv[1]
    else:
        cfg = 'init_coupler.cfg' 

    # Read in COUPLER input file
    from utils.coupler import ReadInitFile, SetDirectories
    COUPLER_options, time_dict = ReadInitFile( cfg )

    # Set directories dictionary
    dirs = SetDirectories(COUPLER_options)

    plot_fluxes_global(dirs["output"], COUPLER_options)
    
