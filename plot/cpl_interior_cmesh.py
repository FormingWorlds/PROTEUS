#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_ext import *
from utils.plot import *
from utils.spider import *

#====================================================================

# Plotting function
def plot_interior_cmesh(output_dir):

    print("Plot interior colourmesh")

    # Gather data
    output_files = glob.glob(output_dir+"/data/*.json")
    output_times = [ int(str(f).split('/')[-1].split('.')[0]) for f in output_files]
    sort_mask = np.argsort(output_times)
    sorted_files = np.array(output_files)[sort_mask]
    sorted_times = np.array(output_times)[sort_mask]

    if len(sorted_times) < 3:
        print("WARNING: Too few samples to make interior_cmesh plot")
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

    # Colour mapping
    cblevels = 40
    
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
    cmap = sci_colormaps['lajolla_r']
    cax = make_axes_locatable(ax1).append_axes('right', size='5%', pad=0.05)
    norm = mpl.colors.Normalize(vmin=np.amin(arr_z1), vmax=np.amax(arr_z1))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax1.contourf(sorted_times, arr_yb, arr_z1.T, cmap=cmap, norm=norm, levels=cblevels)
    fig.colorbar(sm, cax=cax, orientation='vertical').set_label("Temperature [K]") 

    # Plot melt fraction
    cmap = sci_colormaps['grayC_r']
    cax = make_axes_locatable(ax2).append_axes('right', size='5%', pad=0.05)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax2.contourf(sorted_times, arr_yb, arr_z2.T, cmap=cmap, norm=norm, levels=cblevels)
    fig.colorbar(sm, cax=cax, orientation='vertical').set_label("Melt fraction")

    # Plot viscosity
    cmap = sci_colormaps['imola_r']
    cax = make_axes_locatable(ax3).append_axes('right', size='5%', pad=0.05)
    if np.amax(arr_z3) > 100*np.amin(arr_z3):
        norm = mpl.colors.LogNorm(vmin=np.amin(arr_z3), vmax=np.amax(arr_z3))
    else:
        norm = mpl.colors.Normalize(vmin=np.amin(arr_z3), vmax=np.amax(arr_z3))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax3.contourf(sorted_times, arr_yb, arr_z3.T, cmap=cmap, norm=norm, levels=cblevels)
    fig.colorbar(sm, cax=cax, orientation='vertical').set_label("Viscosity [Pa s]") 
    
    # Plot entropy
    cmap = sci_colormaps['acton']
    cax = make_axes_locatable(ax4).append_axes('right', size='5%', pad=0.05)
    norm = mpl.colors.Normalize(vmin=np.amin(arr_z4), vmax=np.amax(arr_z4))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax4.contourf(sorted_times, arr_ys, arr_z4.T, cmap=cmap, norm=norm, levels=cblevels)
    fig.colorbar(sm, cax=cax, orientation='vertical').set_label("Sp. entropy [J K$^{-1} $kg$^{-1}$]")

    # Save plot
    fname = os.path.join(output_dir,"plot_interior_cmesh.pdf")
    fig.subplots_adjust(top=0.98, bottom=0.07, right=0.85, left=0.13, hspace=0.11)
    fig.savefig(fname, transparent=True)

#====================================================================
def main():

    # Optional command line arguments for running from the terminal
    parser = argparse.ArgumentParser(description='COUPLER plotting script')
    parser.add_argument('-odir', '--output_dir', type=str, help='Full path to output directory');
    args = parser.parse_args()

    # Define output directory for plots
    if args.output_dir:
        output_dir = args.output_dir
        print("Output directory:", output_dir)
    else:
        output_dir = os.getcwd()
        print("Output directory:", output_dir)

    # Plot fixed set from above
    plot_interior_cmesh( output_dir=output_dir)

#====================================================================

if __name__ == "__main__":
    main()
