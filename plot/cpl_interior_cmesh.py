#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_ext import *
from utils.plot import *
from utils.spider import *

log = logging.getLogger("PROTEUS")

#====================================================================

# Plotting function
def plot_interior_cmesh(output_dir, use_contour=True, cblevels=24, numticks=5, plot_format="pdf"):

    log.info("Plot interior colourmesh")

    # Gather data
    output_files = glob.glob(output_dir+"/data/*.json")
    output_times = [ int(str(f).split('/')[-1].split('.')[0]) for f in output_files]
    sort_mask = np.argsort(output_times)
    sorted_files = np.array(output_files)[sort_mask]
    sorted_times = np.array(output_times)[sort_mask]

    if len(sorted_times) < 3:
        log.warning("Too few samples to make interior_cmesh plot")
        return
    if len(sorted_times) < 1000:
        stride = int(1)
    else:
        stride = int(15)
    sorted_files = sorted_files[::stride]
    sorted_times = sorted_times[::stride]
    nfiles = len(sorted_times)

    # Initialise plot
    scale = 1.0
    fig,axs = plt.subplots(4,1, sharex=True, figsize=(5*scale,7*scale))
    ax1,ax2,ax3,ax4 = axs
    for ax in axs:
        ax.set_ylabel("Pressure, $P$ [GPa]")
    ax4.set_xlabel("Time, $t$ [yr]")

    # Loop through years to arrange data
    arr_yb  = np.array(MyJSON( output_dir+"/data/%d.json"%sorted_times[-1]).get_dict_values(['data','pressure_b'])) * 1.0E-9
    arr_ys  = np.array(MyJSON( output_dir+"/data/%d.json"%sorted_times[-1]).get_dict_values(['data','pressure_s'])) * 1.0E-9
    nlev_b   = len(arr_yb)
    nlev_s   = len(arr_ys)
    arr_z1 = np.zeros((nfiles,nlev_b),dtype=float)
    arr_z2 = np.zeros((nfiles,nlev_b),dtype=float)
    arr_z3 = np.zeros((nfiles,nlev_b),dtype=float)
    arr_z4 = np.zeros((nfiles,nlev_s),dtype=float)
    for i in range(nfiles):

        # Get data
        myjson_o = MyJSON( output_dir+"/data/%d.json"%sorted_times[i])

        y_tmp = myjson_o.get_dict_values(['data','temp_b'])
        y_phi = myjson_o.get_dict_values(['data','phi_b'])
        y_vis = myjson_o.get_dict_values(['data','visc_b'])
        y_ent = myjson_o.get_dict_values(['data','S_s'])

        for j in range(nlev_b):
            arr_z1[i,j] = y_tmp[j]
            arr_z2[i,j] = y_phi[j]
            arr_z3[i,j] = y_vis[j]
        for j in range(nlev_s):
            arr_z4[i,j] = y_ent[j]

    for a in (arr_z1,arr_z2,arr_z3,arr_z4,arr_yb,arr_ys):
        a = np.array(a,dtype=float)

    # Y-axis ticks
    yticks = np.linspace(arr_yb[0],arr_ys[-1],4)
    yticks = [round(v) for v in yticks]
    for ax in axs:
        ax.set_ylim([yticks[-1], yticks[0]])
        ax.set_yticks(yticks)

    # Plot panel label
    panel_labels = ['A)','B)','C)','D)']
    for i in range(4):
        axt = axs[i].text(0.03,0.94, panel_labels[i], transform=axs[i].transAxes, verticalalignment="top", horizontalalignment="left",
                    fontsize=12)
        axt.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))

    # Plot temperature
    cmap = cm.lajolla
    cax = make_axes_locatable(ax1).append_axes('right', size='5%', pad=0.05)
    norm = mpl.colors.Normalize(vmin=np.amin(arr_z1), vmax=np.amax(arr_z1))
    if use_contour:
        cf = ax1.contourf(sorted_times, arr_yb, arr_z1.T, cmap=cmap, norm=norm, levels=cblevels)
    else:
        cf = ax1.pcolormesh(sorted_times, arr_yb, arr_z1.T, cmap=cmap, norm=norm, rasterized=True)
    cb = fig.colorbar(cf, cax=cax, orientation='vertical')
    cb.set_label("Temperature [K]") 
    cb.set_ticks([float(round(v, -2)) for v in np.linspace(np.amin(arr_z1), np.amax(arr_z1), numticks)])


    # Plot melt fraction
    cmap = cm.grayC
    cax = make_axes_locatable(ax2).append_axes('right', size='5%', pad=0.05)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    if use_contour:
        cf = ax2.contourf(sorted_times, arr_yb, arr_z2.T, cmap=cmap, norm=norm, levels=np.linspace(0.0, 1.0, cblevels))
    else:
        cf = ax2.pcolormesh(sorted_times, arr_yb, arr_z2.T, cmap=cmap, norm=norm, rasterized=True)
    cb = fig.colorbar(cf, cax=cax, orientation='vertical')
    cb.set_label("Melt fraction")
    cb.set_ticks(list(np.linspace(0.0,1.0,numticks)))

    # Plot viscosity
    cmap = cm.imola_r
    cax = make_axes_locatable(ax3).append_axes('right', size='5%', pad=0.05)
    if (np.amax(arr_z3) > 100.0*np.amin(arr_z3)):
        norm = mpl.colors.LogNorm(vmin=np.amin(arr_z3), vmax=np.amax(arr_z3))
        # cbticks = [int(v) for v in np.linspace( np.log10(np.amin(arr_z3)), np.log10(np.amax(arr_z3)), numticks)]
        # cbticks = [10.0**float(v) for v in cbticks]
    else:
        norm = mpl.colors.Normalize(vmin=np.amin(arr_z3), vmax=np.amax(arr_z3))
        # cbticks = [float(int(v)) for v in np.linspace(np.amin(arr_z3), np.amax(arr_z3), numticks)]
    if use_contour:
        cf = ax3.contourf(sorted_times, arr_yb, arr_z3.T, cmap=cmap, norm=norm, levels=cblevels)
    else:
        cf = ax3.pcolormesh(sorted_times, arr_yb, arr_z3.T, cmap=cmap, norm=norm, rasterized=True)
    cb = fig.colorbar(cf, cax=cax, orientation='vertical')
    cb.set_label("Viscosity [Pa s]") 
    # if len(cb.get_ticks()) > numticks:
    #     cb.set_ticks(sorted(list(set(cbticks))))
    
    # Plot entropy
    cmap = cm.acton
    cax = make_axes_locatable(ax4).append_axes('right', size='5%', pad=0.05)
    norm = mpl.colors.Normalize(vmin=np.amin(arr_z4), vmax=np.amax(arr_z4))
    if use_contour:
        cf = ax4.contourf(sorted_times, arr_ys, arr_z4.T, cmap=cmap, norm=norm, levels=cblevels)
    else:
        cf = ax4.pcolormesh(sorted_times, arr_ys, arr_z4.T, cmap=cmap, norm=norm, rasterized=True)
    cb = fig.colorbar(cf, cax=cax, orientation='vertical')
    cb.set_label("Sp. entropy [J K$^{-1} $kg$^{-1}$]")
    cb.set_ticks([float(round(v, -1)) for v in np.linspace(np.amin(arr_z4), np.amax(arr_z4), numticks)])

    # Save plot
    ax4.set_xticks(np.linspace(sorted_times[0], sorted_times[-1], 5))
    fname = os.path.join(output_dir,"plot_interior_cmesh.%s"%plot_format)
    fig.subplots_adjust(top=0.98, bottom=0.07, right=0.85, left=0.13, hspace=0.11)
    fig.savefig(fname, dpi=200)

#====================================================================
def main():

    if len(sys.argv) == 2:
        cfg = sys.argv[1]
    else:
        cfg = 'init_coupler.cfg' 

    # Read in COUPLER input file
    log.info("Read cfg file")
    from utils.coupler import ReadInitFile, SetDirectories
    COUPLER_options = ReadInitFile( cfg )

    # Set directories dictionary
    dirs = SetDirectories(COUPLER_options)

    # Plot fixed set from above
    plot_interior_cmesh( output_dir=dirs["output"], plot_format=COUPLER_options["plot_format"])

#====================================================================

if __name__ == "__main__":
    main()
