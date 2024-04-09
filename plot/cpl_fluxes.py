#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_ext import *
from utils.plot import *

log = logging.getLogger("PROTEUS")

# Plotting fluxes
def plot_fluxes_atmosphere(output_dir, nc_fpath):

    log.info("Plotting atmosphere fluxes")

    # Read netCDF
    ds = nc.Dataset(nc_fpath)
    atm_pl      = np.array(ds.variables['pl'     ][:])
    atm_fl_D_LW = np.array(ds.variables['fl_D_LW'][:])
    atm_fl_U_LW = np.array(ds.variables['fl_U_LW'][:])
    atm_fl_D_SW = np.array(ds.variables['fl_D_SW'][:])
    atm_fl_U_SW = np.array(ds.variables['fl_U_SW'][:])
    atm_fl_D    = np.array(ds.variables['fl_D'   ][:])
    atm_fl_U    = np.array(ds.variables['fl_U'   ][:])
    atm_fl_N    = np.array(ds.variables['fl_N'   ][:])
    ds.close()

    fig,ax = plt.subplots()

    ax.axvline(0,color='black',lw=0.8)

    pl = atm_pl * 1.e-5

    ax.plot(atm_fl_U      ,pl,color='red',label='UP',lw=1)
    ax.plot(atm_fl_U_SW   ,pl,color='red',label='UP SW',linestyle='dotted',lw=2)
    ax.plot(atm_fl_U_LW   ,pl,color='red',label='UP LW',linestyle='dashed',lw=1)

    ax.plot(-1.0*atm_fl_D      ,pl,color='green',label='DN',lw=2)
    ax.plot(-1.0*atm_fl_D_SW   ,pl,color='green',label='DN SW',linestyle='dotted',lw=3)
    ax.plot(-1.0*atm_fl_D_LW   ,pl,color='green',label='DN LW',linestyle='dashed',lw=2)

    ax.plot(atm_fl_N ,pl,color='black',label='NET')

    ax.set_xscale("symlog")
    ax.set_xlabel("Upward-directed flux [W m-2]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_yscale("log")
    ax.set_ylim([pl[-1],pl[0]])

    ax.legend(loc='upper left')

    plt.close()
    plt.ioff()
    fig.savefig(output_dir+"/plot_fluxes_atmosphere.pdf", bbox_inches='tight')

# Plotting fluxes
def plot_fluxes_global(output_dir, COUPLER_options, t0=100.0):

    log.info("Plotting global fluxes")

    # Get values
    df = pd.read_csv(output_dir+"/runtime_helpfile.csv", sep=r"\s+")

    df_atm = df.loc[df['Input']=='Atmosphere'].drop_duplicates(subset=['Time'], keep='last')
    df_atm = df_atm.loc[df_atm['Time']>t0]
    F_net = np.array(df_atm["F_atm"])
    F_asf = np.array(df_atm["F_ins"]) * COUPLER_options["asf_scalefactor"] * (1.0 - COUPLER_options["albedo_pl"]) * np.cos(COUPLER_options["zenith_angle"] * np.pi/180.0)
    F_ins = np.array(df_atm["F_ins"]) 
    F_olr = np.array(df_atm["F_olr"])
    F_upw = np.array(df_atm["F_olr"]) + np.array(df_atm["F_sct"]) 
    t_atm = np.array(df_atm["Time"] )

    df_int = df.loc[df['Input']=='Interior'].drop_duplicates(subset=['Time'], keep='last')
    df_int = df_int.loc[df_int['Time']>t0]
    F_int = np.array(df_int["F_int"])
    t_int = np.array(df_int["Time"])

    if max(len(t_atm),len(t_int)) < 3:
        log.warning("Cannot make plot with less than 3 samples")
        return

    # Create plot
    mpl.use('Agg')
    scale = 1.1
    fig,ax = plt.subplots(1,1,figsize=(5*scale,4*scale))
    lw = 2.0
    al = 0.96

    # Steam runaway line
    ax.axhline(y=280.0, color='black', lw=lw, linestyle='dashed', label="S-N limit", zorder=1)

    # Plot fluxes
    ax.plot(t_int, F_int, lw=lw, alpha=al, zorder=2, color=dict_colors["int"],    label="Net (int.)")
    ax.plot(t_atm, F_net, lw=lw, alpha=al, zorder=2, color=dict_colors["atm"],    label="Net (atm.)")
    ax.plot(t_atm, F_olr, lw=lw, alpha=al, zorder=2, color=dict_colors["OLR"],    label="OLR")
    ax.plot(t_atm, F_upw, lw=lw, alpha=al, zorder=3, color=dict_colors["sct"],    label="OLR + Scat.")
    # ax.plot(t_atm, F_ins, lw=lw, alpha=al, zorder=2, color=dict_colors["ASF"],    label="Instellation")
    ax.plot(t_atm, F_asf, lw=lw, alpha=al, zorder=3, color=dict_colors["ASF"],    label="ASF", linestyle='dotted')

    # Configure plot
    # ax.set_yscale("symlog")
    ax.set_ylabel("Unsigned flux [W m$^{-2}$]")
    ax.set_xscale("log")
    ax.set_xlabel("Time [yr]")
    ax.set_xlim(t_atm[0], t_atm[-1])
    legend = ax.legend(loc='lower left')
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0.99))
    ax.grid(color='black', alpha=0.05)

    plt.close()
    plt.ioff()
    fig.savefig(output_dir+"/plot_fluxes_global.pdf", bbox_inches='tight')

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
    
