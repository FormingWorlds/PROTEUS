#!/usr/bin/env python3

# Plot a cross-section of the GridOfflineChemistry output

# Import things
import matplotlib as mpl
mpl.use("Agg")
from utils.plot_offchem import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

cmap_name = 'batlowK_r'

def plot_offchem_grid_cross(grid_dir:str, x_var:str, y_var:str, z_var:str, cvar_dict:dict={}, 
                            contour:bool=True, labelcontrols:bool=False):
    """Plot a cross-section of the GridOfflineChemistry output.

    Parameters
    ----------
        grid_dir : str
            Path to grid output folder
        x_var : str
            Independent variable on the x-axis
        y_var : str
            Independent variable on the y-axis
        z_var : str
            Dependent variable on the z-axis (colour bar)

        cvar_dict : dict
            Control variable filter dictionary (see `offchem_slice_grid` for more info)
        contour : bool
            Plot using contours instead of scatter points.
        labelcontrols : bool
            Write the control variables on the plot

    """

    # Read grid
    years, opts, data  = offchem_read_grid(grid_dir)

    # Validate x_var and y_var choices
    for v in [x_var, y_var]:
        if v not in opts[0].keys():
            raise Exception("Invalid independent variable '%s'" % v)
        try:
            _ = float(opts[0][v])
        except ValueError:
            raise Exception("Independent variable is not a scalar quantity")
    
    # Slice grid 
    if len(cvar_dict.keys()) > 0:
        years, opts, data = offchem_slice_grid(years, opts, data, cvar_dict)
    gpoints = np.shape(opts)[0]

    if contour and (gpoints < 12):
        print("WARNING: grid size is small but contour plotting is enabled - expect ugly plots")

    # Flatten x,y,z data into 1D arrays...

    x_arr = []
    y_arr = []
    z_arr = []

    x_scale = 'linear'
    y_scale = 'linear'
    z_scale = 'linear'
    cb_vmin = None
    cb_vmax = None

    for gi in range(gpoints):

        # Get data at last time-sample for now (think about this later)
        
        y = years[gi][-1]
        d = data[gi][-1]

        x_arr.append(float(opts[gi][x_var]))
        y_arr.append(float(opts[gi][y_var]))

        # handle possible z_var values 
        z_this = None

        if z_var == 'p_toa':
            z_this = d['pressure'][-1]
            z_scale = 'log'
            z_label = 'Total pressure (TOA) [bar]'
        
        elif z_var == 'p_boa':
            z_this = d['pressure'][0]
            z_scale = 'log'
            z_label = 'Total pressure (BOA) [bar]'

        elif z_var == 'T_surf':
            z_this = d['temperature'][0]
            z_label = 'Surface temperature [K]'

        elif z_var[0] == 'x': # mole fraction cases

            prf = z_var[:6]
            sp = z_var[6:]
            k  = 'mx_' + sp

            if k in d.keys():
                if prf == 'x_toa_': # mixing ratio from VULCAN (TOA)
                        z_this = d[k][0]
                        z_scale = 'log'
                        z_label = '%s kinetic mole fraction (TOA)'

                elif prf == 'x_boa_': # mixing ratio from VULCAN (BOA)
                        z_this = d[k][-1]
                        z_scale = 'log'
                        z_label = '%s kinetic mole fraction (BOA)' % sp

                elif prf == 'x_avg_': # mixing ratio from VULCAN (column average)
                        z_this = np.average(d[k][:])
                        z_scale = 'log'
                        z_label = '%s kinetic mole fraction (column average)' % sp

                elif prf == 'x_run_': # runtime well-mixed mole fraction
                        z_this = d[k]
                        z_scale = 'log'
                        z_label = '%s MVE mole fraction' % sp
            
        elif z_var == 'lifetime': # magma ocean lifetime
            z_this = y
            z_label = 'Magma ocean liftime [yr]'

        else:
            raise Exception("Invalid dependent variable '%s'" % z_var)
        
        if z_this == None:
            raise Exception("Could not read dependent variable '%s' from grid" % z_var)

        z_arr.append(float(z_this))

    # Plot data
    fig,ax = plt.subplots(1,1)

    ax.set_ylabel(y_var)
    ax.set_yscale(y_scale)

    ax.set_xlabel(x_var)
    ax.set_xscale(x_scale)
    
    if cb_vmin == None:
        cb_vmin = np.amin(z_arr)
    
    if cb_vmax == None:
        cb_vmax = np.amax(z_arr)

    if z_scale == 'log':
        norm = mpl.colors.LogNorm(vmin=cb_vmin, vmax=cb_vmax)
    else:
        norm = mpl.colors.Normalize(vmin=cb_vmin, vmax=cb_vmax)

    cmap = sci_colormaps[cmap_name]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    cbar = fig.colorbar(sm, cax=cax, orientation='vertical') 
    cbar.ax.set_ylabel(z_label)

    if contour:
         ax.tricontourf(x_arr, y_arr, z_arr, cmap=cmap, norm=norm, levels=100)
    else:
        ax.scatter(x_arr, y_arr, c=z_arr, cmap=cmap, norm=norm)

    if labelcontrols:
        lbl = "Control variables:\n"
        for k in cvar_dict.keys():
             lbl += k + " = %g\n" % cvar_dict[k]
        ax.text(0.02, 0.98, lbl,
                horizontalalignment='left',
                verticalalignment='top',
                transform = ax.transAxes,
                color='white',
                backgroundcolor=(0.0, 0.0, 0.0, 0.3),
                fontsize='small')

    fig.tight_layout()
    fig.savefig(grid_dir+"/plot_offchem_grid_cross_[%s]_[%s]_[%s].pdf"%(x_var, y_var, z_var))
    plt.close('all') 


# If executed directly
if __name__ == '__main__':
    print("Plotting offchem grid (crossection)...")

    # Folder
    fol = "output/pspace_redox/"

    # Independent vars
    xv  = "fO2_shift_IW"
    yv  = "CH_ratio"

    # Dependent var
    zv  = "T_surf"

    # Filter
    cf = {"mean_distance":0.01154}

    # Plotting options
    contour = False
    lctrl   = True

    print("    x: %s" % xv)
    print("    y: %s" % yv)
    print("    z: %s" % zv)
    print("    f: %s" % cf)

    plot_offchem_grid_cross(os.path.abspath(fol), xv, yv, zv, cf, contour, lctrl)
    
    print("Done!")


