#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_ext import *
from utils.plot import *

#====================================================================

# Helper function
def _read_nc(nc_fpath):
    ds = nc.Dataset(nc_fpath)

    p = np.array(ds.variables["p"][:])
    pl = np.array(ds.variables["pl"][:])

    t = np.array(ds.variables["tmp"][:])
    tl = np.array(ds.variables["tmpl"][:])

    z = np.array(ds.variables["z"][:])
    zl = np.array(ds.variables["zl"][:])

    nlev = len(p)
    arr_p = [pl[0]]
    arr_t = [tl[0]]
    arr_z = [zl[0]]
    for i in range(nlev):
        arr_p.append(p[i]); arr_p.append(pl[i+1])
        arr_t.append(t[i]); arr_t.append(tl[i+1])
        arr_z.append(z[i]); arr_z.append(zl[i+1])

    ds.close()

    return np.array(arr_p), np.array(arr_t), np.array(arr_z)


# Plotting function
def plot_atmosphere_cbar(output_dir, plot_both=False):

    print("Plot atmosphere temperatures colourbar")

    # Gather data files
    output_files = glob.glob(output_dir+"/data/*_atm.nc")
    output_times = [ int(str(f).split('/')[-1].split('_')[0]) for f in output_files]
    sort_mask = np.argsort(output_times)
    sorted_files = np.array(output_files)[sort_mask]
    sorted_times = np.array(output_times)[sort_mask] / 1e6

    if len(sorted_times) < 3:
        print("WARNING: Too few samples to make atmosphere_cbar plot")
        return

    # Parse NetCDF files
    stride = 1
    sorted_p = []
    sorted_t = []
    sorted_z = []
    for i in range(0,len(sorted_files),stride):
        p,t,z = _read_nc(sorted_files[i])
        sorted_p.append(p / 1.0e5)  # Convert Pa -> bar
        sorted_t.append(t ) 
        sorted_z.append(z / 1.0e3)  # Convert m -> km    
    nfiles = len(sorted_p)

    # Initialise plot
    scale = 1.1
    if plot_both:
        fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, figsize=(5*scale,8*scale))
        ax1.set_ylabel("Height [km]")
    else:
        fig,ax2 = plt.subplots(1,1,figsize=(5*scale,4*scale))
    ax2.set_ylabel("Pressure [bar]")
    ax2.set_xlabel("Temperature [K]")
    ax2.invert_yaxis()
    ax2.set_yscale("log")

    # Colour mapping
    norm = mpl.colors.Normalize(vmin=sorted_times[0], vmax=sorted_times[-1])
    sm = plt.cm.ScalarMappable(cmap=sci_colormaps['batlowK_r'], norm=norm) # 
    sm.set_array([])

    # Plot data
    for i in range(nfiles):
        a = 0.7
        c = sm.to_rgba(sorted_times[i])
        if plot_both:
            ax1.plot(sorted_t[i], sorted_z[i], color=c, alpha=a, zorder=3)
        ax2.plot(sorted_t[i], sorted_p[i], color=c, alpha=a, zorder=3)

    # Grid
    if plot_both:
        ax1.grid(alpha=0.2, zorder=2)
    ax2.grid(alpha=0.2, zorder=2)
    ax2.set_xlim(0,np.amax(sorted_t)+100)

    # Plot colourbar
    cax = inset_axes(ax2, width="50%", height="8%", loc='upper right', borderpad=1.0) 
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal') 
    cbar.ax.set_xticks([round(v,1) for v in np.linspace(sorted_times[0] , sorted_times[-1], 4)])
    cbar.set_label("Time [Myr]") 

    # Save plot
    fname = os.path.join(output_dir,"plot_atmosphere_cbar.pdf")
    fig.savefig(fname, transparent=True, bbox_inches='tight')

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
    plot_atmosphere_cbar( output_dir=output_dir)

#====================================================================

if __name__ == "__main__":
    main()
