#!/usr/bin/env python

import spider_utils as su
import logging
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import argparse
from scipy.optimize import curve_fit
import os, sys

#====================================================================
def figure9():

    #---------------
    # set parameters
    #---------------
    # flag to enable data output
    EXPORT = True
    DYNAMIC = False # dynamic criterion, otherwise critical melt fraction
    # pressure cut-offs for curve fitting and plotting
    #PMIN = 1.0 # GPa # TODO: CURRENTLY NOT USED
    #PMAX = 135.0 # GPa # TODO: CURRENTLY NOT USED
    TPRIME1_PRESSURE = 15.0 # pressure at which rheological transition is assumed to be at the surface
    XMAXFRAC = 1.05 # plot maximum x value this factor beyond TPRIME1
    # currently not used RADFIT = 1 # fit using radiative cooling model (n=4), otherwise use linear fit
    markersize = 3.0
    alpha = 0.05
    color1 = (0,0,0.3)
    color2 = (0.4,0.4,0.8)
    color3 = (0.5,0.1,0.1)
    #---------------

    name = 'rheological_front_'
    if DYNAMIC:
        name += 'dynamic'
    else:
        name += 'phi'

    width = 4.7747
    height = 4.7747
    fig_o = su.FigureData( 2, 2, width, height, name )
    fig_o.fig.subplots_adjust(wspace=0.4,hspace=0.4)
    ax0 = fig_o.ax[0][0]
    ax1 = fig_o.ax[0][1]
    ax2 = fig_o.ax[1][0]
    ax3 = fig_o.ax[1][1]

    fig_o.time = su.get_all_output_times()
    time_yrs = fig_o.time
    time0 = fig_o.time[0]

    # pressure in GPa
    myjson_o = su.MyJSON( 'output/{}.json'.format(time0) )
    xx_pres_b = myjson_o.get_dict_values_internal( ('data','pressure_b') )
    xx_pres_b *= 1.0E-9 # to GPa

    data_l = []
    handle_l = []

    # by asking for all the data in one go, the code is much faster
    keys_t = ( ('rheological_front_phi','depth'),
               ('rheological_front_phi', 'pressure'),
               ('rheological_front_phi', 'temperature'),
               #('rheological_front_phi', 'below_middle', 'pressure' ),
               #('rheological_front_phi', 'below_middle', 'temperature' ),
               #('rheological_front_phi', 'above_middle', 'pressure' ),
               #('rheological_front_phi', 'above_middle', 'temperature' ),
               ('rheological_front_phi', 'below_mass_avg', 'pressure' ),
               ('rheological_front_phi', 'below_mass_avg', 'temperature' ),
               ('rheological_front_phi', 'above_mass_avg', 'pressure' ),
               ('rheological_front_phi', 'above_mass_avg', 'temperature' ),
               ('rheological_front_dynamic', 'depth'),
               ('rheological_front_dynamic', 'pressure'),
               ('rheological_front_dynamic', 'temperature'),
               #('rheological_front_dynamic', 'below_middle', 'pressure' ),
               #('rheological_front_dynamic', 'below_middle', 'temperature' ),
               #('rheological_front_dynamic', 'above_middle', 'pressure' ),
               #('rheological_front_dynamic', 'above_middle', 'temperature' ),
               ('rheological_front_dynamic', 'below_mass_avg', 'pressure' ),
               ('rheological_front_dynamic', 'below_mass_avg', 'temperature' ),
               ('rheological_front_dynamic', 'above_mass_avg', 'pressure' ),
               ('rheological_front_dynamic', 'above_mass_avg', 'temperature' ),
               ('atmosphere','temperature_surface'),
               ('data','Jtot_b') )
    data_a = su.get_dict_surface_values_for_times( keys_t, fig_o.time )

    temperature_surface = data_a[14,:]
    flux_surface = data_a[15,:]

    if DYNAMIC:
        # rheological front based on dynamic criterion
        rf_depth = data_a[7,:]
        rf_pressure = data_a[8,:] * 1.0E-9 # to GPa
        rf_temperature = data_a[9,:]
        mo_pressure = data_a[10,:] * 1.0E-9 # to GPa
        mo_temperature = data_a[11,:]
        so_pressure = data_a[12,:] * 1.0E-9 # to GPa
        so_temperature = data_a[13,:]
    else:
        # rheological front based on critical melt fraction
        rf_depth = data_a[0,:]
        rf_pressure = data_a[1,:] * 1.0E-9 # to GPa
        rf_temperature = data_a[2,:]
        mo_pressure = data_a[3,:] * 1.0E-9 # to GPa
        mo_temperature = data_a[4,:]
        so_pressure = data_a[5,:] * 1.0E-9 # to GPa
        so_temperature = data_a[6,:]

    # compute start time and end time of rheological front moving through the mantle
    (t0,n0,t1,n1) = get_t0_t1( fig_o.time, rf_pressure, xx_pres_b[-1], TPRIME1_PRESSURE )

    time_yrs_shift = fig_o.time[n0:n1] - t0 # shift time for plotting Pf and Tf
    rf_pressure_shift = rf_pressure[n0:n1]
    rf_temperature_shift = rf_temperature[n0:n1]
    mo_pressure_shift = mo_pressure[n0:n1]
    mo_temperature_shift = mo_temperature[n0:n1]
    so_pressure_shift = so_pressure[n0:n1]
    so_temperature_shift = so_temperature[n0:n1]

    # ---------------------------
    # ---- surface heat flux ----
    trans = transforms.blended_transform_factory(ax0.transData, ax0.transAxes)
    h1, = ax0.semilogy( fig_o.time, flux_surface, marker='o', markersize=markersize, color='0.8' )
    ax0.axvline( t0, ymin=0.1, ymax=0.8, color='0.25', linestyle=':')
    ax0.text( t0, 0.82, '$t^\prime_0$', ha='right', va='bottom', transform = trans )
    ax0.axvline( t1, ymin=0.1, ymax=0.8, color='0.25', linestyle=':')
    ax0.text( t1, 0.82, '$t^\prime_1$', ha='right', va='bottom', transform = trans )
    title = r'(a) $q_0(t)$, W/m$^2$'

    xmax = XMAXFRAC*t1
    ydata_plot = flux_surface[np.where(time_yrs<xmax)]
    yticks = get_yticks( ydata_plot, 1.0E4 )
    #yticks = get_yticks( ydata_plot, 1.0E5 )

    fig_o.set_myaxes( ax0, title=title, ylabel='$q_0$', xlabel='$t$ (yrs)' )
    ax0.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax0.yaxis.set_label_coords(-0.15,0.5)
    ax0.set_xlim( None, xmax )
    ax0.set_ylim( yticks[0], yticks[-1] )

    # --- end of surface heat flux ----
    # --------------------------------- 

    # -----------------------------
    # ---- surface temperature ----
    trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    ax1.plot( fig_o.time, temperature_surface, marker='o', markersize=markersize, color='0.8' )
    ax1.axvline( t0, ymin=0.1, ymax=0.8, color='0.25', linestyle=':')
    ax1.text( t0, 0.82, '$t^\prime_0$', ha='right', va='bottom', transform = trans )
    ax1.axvline( t1, ymin=0.1, ymax=0.8, color='0.25', linestyle=':')
    ax1.text( t1, 0.82, '$t^\prime_1$', ha='right', va='bottom', transform = trans )
    title = r'(b) $Ts(t)$, K'

    xmax = XMAXFRAC*t1
    ydata_plot = temperature_surface[np.where(time_yrs<xmax)]
    #yticks = [1300,1400,1500,1600,1700]
    yticks = get_yticks( ydata_plot, 400 ) # [1500,2000,2500,3000,3500,4000]

    fig_o.set_myaxes( ax1, title=title, ylabel='$Ts$', xlabel='$t$ (yrs)', yticks=yticks )
    #ax1.set_ylim( [1300,1700])
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #ax1.yaxis.set_label_coords(-0.2,0.575)
    ax1.yaxis.set_label_coords(-0.15,0.5)
    ax1.set_xlim( None, xmax )
    # ---- end surface temperature ----
    # ---------------------------------

    # ---------------------------
    # ---- rheological front ----
    #if EXPORT:
    #    np.savetxt( 'Pr.dat', np.column_stack((time_a,pres_a)))
    #    np.savetxt( 'Tr.dat', np.column_stack((time_a,temp_a)))
    ax2.plot( time_yrs_shift, rf_pressure_shift, marker='o', markersize=markersize, alpha=alpha, color=color1 )
    ax2.plot( time_yrs_shift, mo_pressure_shift, marker='o', markersize=markersize, alpha=alpha, color=color2 )
    ax2.plot( time_yrs_shift, so_pressure_shift, marker='o', markersize=markersize, alpha=alpha, color=color3 )
    title = r'(c) $P_f(t^\prime)$, GPa'
    yticks = [0,50,100,150]
    fig_o.set_myaxes( ax2, title=title, ylabel='$P_f$', xlabel='$t^\prime$ (yrs)', yticks=yticks )
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax2.yaxis.set_label_coords(-0.15,0.5)
    ax2.set_xlim( (0, t1-t0) )

    ax3.plot( time_yrs_shift, rf_temperature_shift, marker='o', markersize=markersize, alpha=alpha, color=color1 )
    ax3.plot( time_yrs_shift, mo_temperature_shift, marker='o', markersize=markersize, alpha=alpha, color=color2 )
    ax3.plot( time_yrs_shift, so_temperature_shift, marker='o', markersize=markersize, alpha=alpha, color=color3 )
    title = r'(d) $T_f(t^\prime)$, K'
    yticks = [1500,2500,3500,4500]
    fig_o.set_myaxes( ax3, title=title, ylabel='$T_f$', xlabel='$t^\prime$ (yrs)', yticks=yticks )
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax3.set_ylim( [1500,4600] )
    #ax3.yaxis.set_label_coords(-0.15,0.575)
    ax3.yaxis.set_label_coords(-0.15,0.5)
    ax3.set_xlim( (0,t1-t0) )

    ###################
    ##### fitting #####
    ###################

    # fitting can break if the motion of the rheological transition is complicated
    if 0:

        if DYNAMIC:
            deg = 2
        else:
            deg = 1

        # fit rheological front
        rf_pressure_fit = fit_data( time_yrs_shift, rf_pressure_shift, deg )
        ax2.plot( time_yrs_shift, rf_pressure_fit, color=color1 )
        rf_temperature_fit = fit_data( time_yrs_shift, rf_temperature_shift, deg )
        ax3.plot( time_yrs_shift, rf_temperature_fit, color=color1 )

        # fit magma ocean
        mo_pressure_fit = fit_data( time_yrs_shift, mo_pressure_shift, deg )
        ax2.plot( time_yrs_shift, mo_pressure_fit, color=color2 )
        mo_temperature_fit = fit_data( time_yrs_shift, mo_temperature_shift, deg )
        ax3.plot( time_yrs_shift, mo_temperature_fit, color=color2 )

        # fit solid
        so_pressure_fit = fit_data( time_yrs_shift, so_pressure_shift, deg )
        ax2.plot( time_yrs_shift, so_pressure_fit, color=color3 )
        so_temperature_fit = fit_data( time_yrs_shift, so_temperature_shift, deg )
        ax3.plot( time_yrs_shift, so_temperature_fit, color=color3 )

    fig_o.savefig(9)

#====================================================================
def fit_data( xx, yy, deg ):

    poly = np.polyfit( xx, yy, deg )
    poly_o = np.poly1d( poly )
    yy_fit = poly_o( xx )
    Rval = Rsquared( yy, yy_fit )
    print('----fit----')
    print( 'R^2=', Rval )
    print( 'poly=', poly )
    print( 'gradient=', poly[0] )
    print( 'intercept=', poly[1] )
    print( 'minimum for fit=', np.min(yy_fit) )
    print( 'maximum for fit=', np.max(yy_fit) )

    return yy_fit

#====================================================================
def get_t0_t1( time_l, pres_l, val_t0, val_t1 ):

    '''return absolute time and index when the rheological front
       leaves the CMB and arrives at the surface, as determined by
       the pressures val_t0 and val_t1, respectively'''

    FLAG_t0 = False
    FLAG_t1 = False

    for nn, time in enumerate(time_l):
        pres = pres_l[nn]
        if pres < val_t0 and not FLAG_t0:
            t0 = time
            n0 = nn
            FLAG_t0 = True
        if pres < val_t1 and not FLAG_t1:
            t1 = time
            n1 = nn
            FLAG_t1 = True

    print( 't0= ', t0 )
    print( 't1= ', t1 )

    return (t0,n0,t1,n1)

#====================================================================
def get_yticks( array_in, roundtonearest=500 ):

    # set the range of the y axis and determine the y labels based
    # on the data extent in the input array

    minimum = np.min( array_in )
    maximum = np.max( array_in )

    minimum = np.floor(minimum/roundtonearest) * roundtonearest
    maximum = np.ceil(maximum/roundtonearest) * roundtonearest

    labels = np.arange( minimum, maximum+1, roundtonearest )
    labels = labels.tolist()

    return labels

#====================================================================
def radiative_function_fit( x, *popt ):

    return (popt[0]*x+popt[1])**(-1.0/3.0) + popt[2]

#====================================================================
def Rsquared( ydata, fmodel ):

    # sum of squares of residuals
    ybar = np.mean( ydata )
    #print( 'ybar=', ybar )
    SSres = np.sum( np.square(ydata-fmodel) )
    #print( 'SSres=', SSres )
    SStot = np.sum( np.square(ydata-ybar) )
    #print( 'SStot=', SStot )
    Rsquared = 1.0 - SSres/SStot
    #print( 'Rsquared=', Rsquared )

    return Rsquared

#====================================================================
def main():

    figure9()
    plt.show()

#====================================================================

if __name__ == "__main__":

    main()
