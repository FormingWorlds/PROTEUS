#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_ext import *
from utils.plot import *
from utils.spider import *

log = logging.getLogger(__name__)

def plot_global( output_dir , COUPLER_options, logt=True, tmin=1e1):

    log.info("Plot global")

    # Plotting parameters
    lw=2.0
    al=0.95
    fig_ratio=(3,2)
    fig_scale=4.0
    leg_kwargs = {
        "frameon":1, 
        "fancybox":True,
        "framealpha":0.9
    }

    # Read data
    #    Helpfiles...
    df = pd.read_csv(output_dir+"/runtime_helpfile.csv",sep=r'\s+')
    df_int = df.loc[df['Input']=='Interior'].drop_duplicates(subset=['Time'], keep='last')
    df_atm = df.loc[df['Input']=='Atmosphere'].drop_duplicates(subset=['Time'], keep='last')

    #    Volatile parameters (keys=vols, vals=quantites_over_time)
    vol_present = {} # Is present ever? (true/false)
    vol_vmr     = {} # Volume mixing ratio 
    vol_bars    = {} # Surface partial pressure [bar]
    vol_mol_atm = {} # Moles in atmosphere
    vol_mol_int = {} # Moles in interior
    vol_mol_tot = {} # Moles in total
    vol_atmpart = {} # Partitioning into atm
    vol_intpart = {} # Partitioning into int

    for vol in volatile_species:
        # Check vmr for presence
        this_vmr = np.array(df_atm[vol+"_mr"])
        vol_present[vol] = True 
        if np.amax(this_vmr) < 1.0e-30:
            vol_present[vol] = False 
            continue 
        vol_vmr[vol] = this_vmr 
        
        # Do other variables
        vol_bars[vol]    = np.array(df_atm[vol+"_atm_bar"])
        vol_mol_atm[vol] = np.array(df_atm[vol+"_mol_atm"])
        vol_mol_tot[vol] = np.array(df_atm[vol+"_mol_total"])
        vol_mol_int[vol] = vol_mol_tot[vol] - vol_mol_atm[vol]
        vol_atmpart[vol] = vol_mol_atm[vol]/vol_mol_tot[vol]
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
    ax_cr.set_ylabel(r'$\chi^{\mathrm{atm}}_{\mathrm{i}}$')
    ax_br.set_ylabel(r'$m^{\mathrm{int}}_{\mathrm{i}}/m^{\mathrm{tot}}_{\mathrm{i}}$')

    # Set x-labels
    for ax in (ax_bl, ax_br):
        ax.set_xlabel("Time [yr]")

    # Set titles 
    ax_titles = [
        "A) Net heat flux to space",
        "B) Surface temperature",
        "C) Mantle evolution",
        "D) Surface volatile partial pressure",
        "E) Surface volatile mole fraction",
        "F) Interior volatile partitioning"
    ]
    for i,ax in enumerate(axs):
        ax.text(0.015, 0.015, ax_titles[i],
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

    # Log axes
    if logt:
        for ax in axs:
            ax.set_xscale("log")
    ax_tl.set_yscale("symlog", linthresh=0.1)
    ax_tr.set_yscale("log")
    ax_cr.set_yscale("log")

    # Set xlim
    xmin = max(tmin, 1.0)
    if logt:
        xmax = max(1.0e6,np.amax(df_int["Time"]))
        xlim = (xmin,10 ** math.ceil(math.log10(xmax*1.1)))
    else:
        xmax = np.amax(df_int["Time"])
        xlim = (1.0, xmax)
    for ax in axs:
        ax.set_xlim(xlim[0],  max(xlim[1], xlim[0]+1))
    

    # ------------------------------------------------------------------------

    # PLOT ax_tl
    ax_tl.plot( df_atm["Time"], df_atm["F_int"], color=dict_colors["int"], lw=lw, alpha=al,  label="Int.",zorder=3, linestyle='dashed')
    ax_tl.plot( df_atm["Time"], df_atm["F_atm"], color=dict_colors["atm"], lw=lw, alpha=al,  label="Atm.",zorder=3)
    ax_tl.plot( df_atm["Time"], df_atm["F_olr"], color=dict_colors["OLR"], lw=lw, alpha=al,  label="OLR", zorder=2)
    ax_tl.legend(loc='center left', **leg_kwargs)
    ax_tl.set_ylim(0.0, np.amax(df_atm["F_olr"]*2.0))


    # PLOT ax_cl
    min_temp = np.amin(df_atm["T_surf"])
    max_temp = np.amax(df_int["T_surf"])
    ax_cl.plot(df_int["Time"], df_int["T_surf"], ls="dashed", lw=lw, alpha=al, color=dict_colors["int"])
    ax_cl.plot(df_atm["Time"], df_atm["T_surf"], ls="-",      lw=lw, alpha=al, color=dict_colors["atm"])
    ax_cl.set_ylim(min(1000.0,min_temp) , max(3500.0,max_temp))

    
    # PLOT ax_bl
    ax_bl.axhline( y=COUPLER_options["planet_coresize"], ls='dashed', lw=lw*1.5, alpha=al, color=dict_colors["qmagenta_dark"], label=r'C-M boundary' )
    ax_bl.plot( df_int["Time"], 1.0-df_int["RF_depth"],   color=dict_colors["int"], ls="solid",    lw=lw, alpha=al, label=r'Rheol. front')
    ax_bl.plot( df_int["Time"],     df_int["Phi_global"], color=dict_colors["atm"], linestyle=':', lw=lw, alpha=al, label=r'Melt fraction')
    ax_bl.legend(loc='center left', **leg_kwargs)
    ax_bl.set_ylim(0.0,1.0)


    # PLOT ax_tr
    ax_tr.plot( df_int["Time"], df_int["P_surf"], color='black', linestyle='dashed', lw=lw*1.5, label=r'Total')
    bar_min, bar_max = 0.1, 10.0
    bar_max = max(bar_max, np.amax(df_int["P_surf"]))
    for vol in volatile_species:
        if not vol_present[vol]:
            continue 
        ax_tr.plot( df_atm["Time"], vol_bars[vol], color=dict_colors[vol], lw=lw, alpha=al, label=vol_latex[vol], zorder=vol_zorder[vol])
        bar_min = min(bar_min, np.amin(vol_bars[vol]))
    ax_tr.set_ylim(max(1.0e-7,min(bar_min, 1.0e-1)), bar_max * 2.0)
    ax_tr.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5) )

    # PLOT ax_cr
    vmr_min = 1.0e-3
    for vol in volatile_species:
        if not vol_present[vol]:
            continue 
        ax_cr.plot( df_atm["Time"], vol_vmr[vol], color=dict_colors[vol], lw=lw, alpha=al, label=vol_latex[vol], zorder=vol_zorder[vol])
        vmr_min = min(vmr_min, np.amin(vol_vmr[vol]))
    ax_cr.set_ylim(vmr_min, 1.0)
    ax_cr.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5) )

    # PLOT ax_br
    for vol in volatile_species:
        if not vol_present[vol]:
            continue 
        ax_br.plot( df_atm["Time"], vol_intpart[vol], color=dict_colors[vol], lw=lw, alpha=al, label=vol_latex[vol], zorder=vol_zorder[vol])
    ax_br.set_ylim(0.0,1.0)
    ax_br.legend(loc='center left', ncol=2, **leg_kwargs).set_zorder(20)
    

    # ------------------------------------------------------------------------

    # Save plot
    fig.subplots_adjust(wspace=0.05,hspace=0.1)
    plt_name = "plot_global"
    if logt:
        plt_name += "_log"
    else:
        plt_name += "_lin"
    fig.savefig(output_dir+"/%s.pdf"%plt_name, bbox_inches='tight')

#====================================================================

if __name__ == "__main__":

    if len(sys.argv) == 2:
        cfg = sys.argv[1]
    else:
        cfg = 'init_coupler.cfg' 

    # Read in COUPLER input file
    log.info("Read cfg file")
    from utils.coupler import ReadInitFile, SetDirectories
    COUPLER_options, time_dict = ReadInitFile( cfg )

    # Set directories dictionary
    dirs = SetDirectories(COUPLER_options)

    for logt in [True,False]:
        plot_global(dirs['output'],COUPLER_options, logt=logt, tmin=1e1)
