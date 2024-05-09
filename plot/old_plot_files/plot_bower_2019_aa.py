#!/usr/bin/env python

import spider_utils as su
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import sys

logger = su.get_my_logger(__name__)

#====================================================================
def spiderplot_fig1( times ):

    # article class text width is 4.7747 inches
    # http://tex.stackexchange.com/questions/39383/determine-text-width

    logger.info( 'building spiderplot_fig1' )

    # must be 1.1 for legend to fit
    figw = (4.7747 / 2.0) * 1.1
    figh = (4.7747 / 2.0) * 1.1

    # record of times for plotting
    # these times are for a previous test case
    #times='0,150000,500000,575000000,4550000000'
    # with 0.25 contour
    # these times are for case3
    #times='0,250000,750000,573000000,4550000000'
    # these times are for case9
    #times='0,3850000,12050000,802000000,4550000000'
    # these times are for case1
    #times='0,94400,5450000,560000000,4550000000'
    # these times are for case1m
    times = '0,94400,300000,550000,3900000'
    # with 0.3 contour
    # these times are for case3
    #times='0,250000,650000,573000000,4550000000'
    # these times are for case3m5
    #times = '0,250000,650000,1050000,3050000'
    # these times are for case9
    #times='0,3850000,10950000,802000000,4550000000'
    # these times are for case1
    #times='0,94400,250000,560000000,4550000000'

    fig_o = su.FigureData( 1, 1, figw, figh, 'boweretal2019_fig1', times, units='Myr' )

    #ax0 = fig_o.ax[0]
    ax1 = fig_o.ax #1]
    #ax2 = fig_o.ax[1][0]
    #ax3 = fig_o.ax[1][1]

    time = fig_o.time[0] # first timestep since liquidus and solidus
                         # are time-independent

    myjson_o = su.MyJSON( 'output/{}.json'.format(time) )

    TIMEYRS = myjson_o.data_d['nstep']

    # hack to compute some average properties for Bower et al. (2018)
    #xx_liq, yy_liq = fig_o.get_xy_data( 'liquidus_rho', time )
    #xx_sol, yy_sol = fig_o.get_xy_data( 'solidus_rho', time )
    #diff = (yy_liq - yy_sol) / yy_sol * 100.0
    #print diff[40:]
    #print np.mean(diff[40:])
    #sys.exit(1)

    xx_pres = myjson_o.get_dict_values_internal(['data','pressure_b'])
    xx_pres *= 1.0E-9

    xx_pres_s = myjson_o.get_dict_values(['data','pressure_s'])
    xx_pres_s *= 1.0E-9

    # shade grey between liquidus and solidus
    yy_liq = myjson_o.get_dict_values_internal(['data','liquidus_b'])
    yy_sol = myjson_o.get_dict_values_internal(['data','solidus_b'])
    #ax0.fill_between( xx_pres, yy_liq, yy_sol, facecolor='grey', alpha=0.35, linewidth=0 )
    yy_liqt = myjson_o.get_dict_values_internal(['data','liquidus_temp_b'])
    yy_solt = myjson_o.get_dict_values_internal(['data','solidus_temp_b'])
    ax1.fill_between( xx_pres, yy_liqt, yy_solt, facecolor='grey', alpha=0.35, linewidth=0 )
    # hack to compute some average properties for Bower et al. (2018)
    #print xx_sol
    #print np.mean(yy_liq[20:]-yy_sol[20:])
    #sys.exit(1)

    # dotted lines of constant melt fraction
    #for xx in range( 0, 11, 2 ):
    #    yy_b = xx/10.0 * (yy_liq - yy_sol) + yy_sol
    #    if xx == 0:
            # solidus
    #        ax0.plot( xx_pres, yy_b, '-', linewidth=0.5, color='black' )
    #    elif xx == 3:
            # typically, the approximate location of the rheological transition
    #        ax0.plot( xx_pres, yy_b, '-', linewidth=1.0, color='white')
    #    elif xx == 10:
            # liquidus
    #        ax0.plot( xx_pres, yy_b, '-', linewidth=0.5, color='black' )
    #    else:
            # dashed constant melt fraction lines
    #        ax0.plot( xx_pres, yy_b, '--', linewidth=1.0, color='white' )

    handle_l = [] # handles for legend

    #labelsuff = ['(1)','(0.75)','(0.3)','(0.1)','(0.01)']
    labelsuff = ['(1.00)','(0.75)','(0.30)','(0.01)','(0.00)']

    for nn, time in enumerate( fig_o.time ):
        # read json
        myjson_o = su.MyJSON( 'output/{}.json'.format(time) )

        color = fig_o.get_color( nn )
        # use melt fraction to determine mixed region
        MIX = myjson_o.get_mixed_phase_boolean_array( 'basic_internal' )
        MIX_s = myjson_o.get_mixed_phase_boolean_array( 'staggered' )

        label = fig_o.get_legend_label( time )

        # entropy
        # plot staggered, since this is where entropy is defined
        # cell-wise and we can easily see the CMB boundary condition
        #yy = myjson_o.get_dict_values(['data','S_s'])
        #print( np.min(yy) )
        #yy = myjson_o.get_dict_values_internal('S_b')
        #ax0.plot( xx_pres_s, yy, '--', color=color )
        #handle, = ax0.plot( xx_pres_s*MIX_s, yy*MIX_s, '-', color=color, label=label )
        #handle_l.append( handle )
        # temperature
        #yy = myjson_o.get_dict_values_internal(['data','temp_b'])
        #ax1.plot( xx_pres, yy, '--', color=color )
        #ax1.plot( xx_pres*MIX, yy*MIX, '-', color=color )
        # temperature
        yy = myjson_o.get_dict_values(['data','temp_s'])
        ax1.plot( xx_pres_s, yy, '--', color=color )
        labelo = label + ' ' + labelsuff[nn]
        handle, = ax1.plot( xx_pres_s*MIX_s, yy*MIX_s, '-', color=color, label=labelo )
        handle_l.append( handle )

        # melt fraction
        #yy = myjson_o.get_dict_values_internal(['data','phi_b'])
        #ax2.plot( xx_pres, yy, '-', color=color )
        # viscosity
        #visc_const = 1 # this is used for the arcsinh scaling
        #visc_fmt = su.MyFuncFormatter( visc_const )
        #yy = myjson_o.get_dict_values_internal(['data','visc_b'], visc_fmt)
        #ax3.plot( xx_pres, yy, '-', color=color )

    xticks = [0,25,50,75,100,135]
    xmax = 138

    # titles and axes labels, legends, etc
    #units = myjson_o.get_dict_units(['data','S_b'])
    #title = '(a) Entropy, {}'.format(units)
    #yticks = [400,1200,2000,2800]
    # Bower et al. (2018)
    #yticks = [1600,2000,2400,2800,3200]
    # DJB used this next range for work with Bayreuth
    #yticks = [300,1000,1600,2000,2400,2800,3200]
    #fig_o.set_myaxes( ax0, title=title, ylabel='$S$', xticks=xticks, xmax=xmax, yticks=yticks )
    #ax0.yaxis.set_label_coords(-0.075,0.5)
    units = myjson_o.get_dict_units(['data','temp_b'])
    title = r'\textbf{Mantle temperature (Case 3v)}' #.format(units)
    # Bower et al. (2018)
    yticks= [250,1000,2000,3000,4000,5000]
    # DJB used this next range for work with Bayreuth
    #yticks= [300,1000,2000,3000,4000,5000]
    fig_o.set_myaxes( ax1, title=title, ylabel='$T$ (K)', xticks=xticks, xlabel='$P$ (GPa)', xmax=xmax, yticks=yticks )
    ax1.set_xlim( xticks[0], 138 )
    ax1.yaxis.set_label_coords(-0.1,0.45)
    fig_o.set_mylegend( ax1, handle_l, loc=4, ncol=2, TITLE='Time, Myr ($\phi_g$)' )
    #fig_o.set_mylegend( ax0, handle_l, loc=2, ncol=2 )
    #title = '(c) Melt fraction'
    #yticks = [0,0.2,0.4,0.6,0.8,1.0]
    #fig_o.set_myaxes( ax2, title=title, xlabel='$P$ (GPa)',
    #    ylabel=r'$\phi$', xticks=xticks, xmax=xmax, yticks=yticks )
    #ax2.yaxis.set_label_coords(-0.075,0.475)
    #ax2.set_ylim( [0, 1] )
    #units = myjson_o.get_dict_units(['data','visc_b'])
    #title = '(d) Viscosity, ' + units
    #yticks = [1.0E2, 1.0E6, 1.0E12, 1.0E18, 1.0E21]
    #fig_o.set_myaxes( ax3, title=title, xlabel='$P$ (GPa)',
    #    ylabel=r'$\eta$', xticks=xticks, xmax=xmax, yticks=yticks, fmt=visc_fmt )
    #ax3.yaxis.set_label_coords(-0.075,0.67)

    if 0:
        # add sol and liq text boxes to mark solidus and liquidus
        # these have to manually adjusted according to the solidus
        # and liquidus used for the model
        prop_d = {'boxstyle':'round', 'facecolor':'white', 'linewidth': 0}
        ax0.text(120, 2925, r'liq', fontsize=8, bbox=prop_d )
        ax0.text(120, 2075, r'sol', fontsize=8, bbox=prop_d )
        # DJB commented out for now, below labels 0.2 melt fraction
        # so the reader knows that dashed white lines denote melt
        # fraction contours
        ax0.text(110, 2275, r'$\phi=0.2$', fontsize=6 )

    fig_o.savefig(1)

#====================================================================
def spiderplot_fig2( times ):

    logger.info( 'building spiderplot_fig2' )

    fig_o = su.FigureData( 2, 2, 4.7747, 4.7747, 'spiderplot_fluxes', times )

    ax0 = fig_o.ax[0][0]
    ax1 = fig_o.ax[0][1]
    ax2 = fig_o.ax[1][0]
    ax3 = fig_o.ax[1][1]

    handle_l = []

    # for arcsinh scaling
    flux_const = 1.0E6
    flux_fmt = su.MyFuncFormatter( flux_const )

    for nn, time in enumerate( fig_o.time ):
        # In Bower et al. (2018), just every other line
        # is plotted for visual clarity (uncomment below)
        #if (nn-1) % 2:
        #    continue

        # read json
        myjson_o = su.MyJSON( 'output/{}.json'.format(time) )

        color = fig_o.get_color( nn )
        # use melt fraction to determine mixed region
        MIX = myjson_o.get_mixed_phase_boolean_array( 'basic_internal')

        label = fig_o.get_legend_label( time )

        # pressure for x-axis
        xx_pres = myjson_o.get_dict_values_internal(['data','pressure_b'])
        xx_pres *= 1.0E-9

        # Jconv_b
        yy = myjson_o.get_dict_values_internal(['data','Jconv_b'], flux_fmt)
        ax0.plot( xx_pres, yy, '--', color=color )
        handle, = ax0.plot( xx_pres*MIX, yy*MIX, '-', label=label,
            color=color )
        handle_l.append( handle )
        # Jmix_b
        yy = myjson_o.get_dict_values_internal(['data','Jmix_b'], flux_fmt)
        ax2.plot( xx_pres, yy, '--', color=color )
        ax2.plot( xx_pres*MIX, yy*MIX, '-', color=color )
        # Jgrav_b
        yy = myjson_o.get_dict_values_internal(['data','Jgrav_b'], flux_fmt)
        ax1.plot( xx_pres, yy, '--', color=color )
        ax1.plot( xx_pres*MIX, yy*MIX, '-', color=color )
        # Jtot_b
        yy = myjson_o.get_dict_values_internal(['data','Jtot_b'], flux_fmt)
        ax3.plot( xx_pres, yy, '--', color=color )
        ax3.plot( xx_pres*MIX, yy*MIX, '-', color=color )

    # titles and axes labels, legends, etc.
    xticks = [0,50,100,135]
    xmax = 138
    yticks = [-1E15,-1E12, -1E6, -1E0, 0, 1E0, 1E6, 1E12, 1E15]
    units = myjson_o.get_dict_units(['data','Jconv_b'])
    title = '(a) Convective flux, {}'.format(units)
    fig_o.set_myaxes( ax0, title=title, ylabel='$F_\mathrm{conv}$',
        yticks=yticks, xticks=xticks, xmax=xmax, fmt=flux_fmt )
    ax0.yaxis.set_label_coords(-0.16,0.54)
    fig_o.set_mylegend( ax0, handle_l, ncol=2 )
    units = myjson_o.get_dict_units(['data','Jmix_b'])
    title = '(c) Mixing flux, {}'.format(units)
    fig_o.set_myaxes( ax2, title=title, xlabel='$P$ (GPa)',
        ylabel='$F_\mathrm{mix}$', yticks=yticks, xticks=xticks, xmax=xmax, fmt=flux_fmt )
    ax2.yaxis.set_label_coords(-0.16,0.54)
    units = myjson_o.get_dict_units(['data','Jgrav_b'])
    title = '(b) Separation flux, {}'.format(units)
    fig_o.set_myaxes( ax1, title=title,
        ylabel='$F_\mathrm{grav}$', yticks=yticks, xticks=xticks, xmax=xmax, fmt=flux_fmt )
    ax1.yaxis.set_label_coords(-0.16,0.54)
    units = myjson_o.get_dict_units(['data','Jtot_b'])
    title = '(d) Total flux, {}'.format(units)
    fig_o.set_myaxes( ax3, title=title, xlabel='$P$ (GPa)',
        ylabel='$F_\mathrm{tot}$', yticks=yticks, xticks=xticks, xmax=xmax, fmt=flux_fmt )
    ax3.yaxis.set_label_coords(-0.16,0.54)

    fig_o.savefig(2)

#====================================================================
def spiderplot_fig3( times ):

    logger.info( 'building spiderplot_fig3' )

    # keep y the same by scaling (7.1621)
    fig_o = su.FigureData( 3, 2, 4.7747, 7.1621, 'spiderplot_matprop', times )

    ax0 = fig_o.ax[0][0]
    ax1 = fig_o.ax[0][1]
    ax2 = fig_o.ax[1][0]
    ax3 = fig_o.ax[1][1]
    ax4 = fig_o.ax[2][0]
    ax5 = fig_o.ax[2][1]

    handle_l = []

    eddy_const = 1.0
    eddy_fmt = su.MyFuncFormatter( eddy_const )
    alpha_const = 1.0E6
    alpha_fmt = su.MyFuncFormatter( alpha_const )
    dSdr_const = 1.0E15
    dSdr_fmt = su.MyFuncFormatter( dSdr_const )

    for nn, time in enumerate( fig_o.time ):
        # In Bower et al. (2018), just every other line
        # is plotted for visual clarity (uncomment below)
        #if (nn-1) % 2:
        #    continue

        # read json
        myjson_o = su.MyJSON( 'output/{}.json'.format(time) )

        color = fig_o.get_color( nn )
        # use melt fraction to determine mixed region
        MIX = myjson_o.get_mixed_phase_boolean_array( 'basic_internal' )

        label = fig_o.get_legend_label( time )

        # pressure for x-axis
        xx_pres = myjson_o.get_dict_values_internal(['data','pressure_b'])
        xx_pres *= 1.0E-9

        # eddy diffusivity
        yy = myjson_o.get_dict_values_internal(['data','kappah_b'], eddy_fmt)
        ax0.plot( xx_pres, yy, '--', color=color )
        handle, = ax0.plot( xx_pres*MIX, yy*MIX, '-', label=label,
            color=color )
        handle_l.append( handle )
        # density
        yy = myjson_o.get_dict_values_internal(['data','rho_b'])
        ax1.plot( xx_pres, yy, '--', color=color )
        ax1.plot( xx_pres*MIX, yy*MIX, '-', color=color )
        # thermal expansion coefficient
        yy = myjson_o.get_dict_values_internal(['data','alpha_b'], alpha_fmt)
        ax3.plot( xx_pres, yy, '--', color=color )
        ax3.plot( xx_pres*MIX, yy*MIX, '-', color=color )
        # heat capacity
        yy = myjson_o.get_dict_values_internal(['data','cp_b'])
        ax4.plot( xx_pres, yy, '--', color=color )
        ax4.plot( xx_pres*MIX, yy*MIX, '-', color=color )
        # dTdrs
        yy = myjson_o.get_dict_values_internal(['data','dTdrs_b'])
        ax5.plot( xx_pres, yy, '--', color=color )
        ax5.plot( xx_pres*MIX, yy*MIX, '-', color=color )
        # entropy gradient
        yy = myjson_o.get_dict_values_internal(['data','dSdr_b'], dSdr_fmt )
        yy *= -1.0
        ax2.plot( xx_pres, yy, '--', color=color )
        ax2.plot( xx_pres*MIX, yy*MIX, '-', color=color )

    xticks = [0,50,100,135]
    xmax = 138

    # titles and axes labels, legends, etc.
    units = myjson_o.get_dict_units(['data','kappah_b'])
    title = '(a) Eddy diffusivity, {}'.format(units)
    yticks = [1.0, 1.0E3, 1.0E6, 1.0E9]
    fig_o.set_myaxes( ax0, title=title, ylabel='$\kappa_h$',
        yticks=yticks, fmt=eddy_fmt, xticks=xticks, xmax=xmax )
    ax0.yaxis.set_label_coords(-0.1,0.475)

    units = myjson_o.get_dict_units(['data','rho_b'])
    title = '(b) Density, {}'.format(units)
    yticks = [2000,3000,4000,5000,6000]
    fig_o.set_myaxes( ax1, title=title, ylabel='$\\rho$',
        yticks=yticks, xticks=xticks, xmax=xmax )
    ax1.yaxis.set_label_coords(-0.1,0.6)
    fig_o.set_mylegend( ax1, handle_l, loc=4, ncol=2 )

    units = myjson_o.get_dict_units(['data','dSdr_b'])
    title = '(c) Entropy grad, {}'.format(units)
    #yticks= [-1.0E-3, -1.0E-6, -1.0E-9, -1.0E-12, -1.0E-15]
    yticks = [1E-15, 1E-12, 1E-9, 1E-6, 1E-3]
    fig_o.set_myaxes( ax2, title=title,
        yticks=yticks, xticks=xticks, xmax=xmax,
        ylabel='$-\\frac{\partial S}{\partial r}$', fmt=dSdr_fmt)
    ax2.yaxis.set_label_coords(-0.1,0.565)

    units = myjson_o.get_dict_units(['data','alpha_b'])
    title = '(d) Thermal expansion, {}'.format(units)
    yticks = [1.0E-5, 1.0E-4, 1.0E-3, 1.0E-2]
    fig_o.set_myaxes( ax3, title=title, ylabel='$\\alpha$',
        yticks=yticks, fmt=alpha_fmt, xticks=xticks, xmax=xmax )
    ax3.yaxis.set_label_coords(-0.1,0.475)

    units = myjson_o.get_dict_units(['data','cp_b'])
    title = '(e) Heat capacity, {}'.format(units)
    yticks = [0,5000,10000,15000]
    fig_o.set_myaxes( ax4, title=title, xlabel='$P$ (GPa)', ylabel='$c$',
        yticks=yticks, xticks=xticks, xmax=xmax )
    ax4.yaxis.set_label_coords(-0.1,0.475)

    units = myjson_o.get_dict_units(['data','dTdrs_b'])
    title = '(f) Adiabatic grad, {}'.format(units)
    yticks = [-3E-3, -2E-3, -1E-3, 0]
    fig_o.set_myaxes( ax5, title=title, xlabel='$P$ (GPa)',
        yticks=yticks, xticks=xticks, xmax=xmax,
        ylabel='$\\left(\\frac{\partial T}{\partial r}\\right)_S$')
    ax5.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    ax5.yaxis.set_label_coords(-0.15,0.43)

    fig_o.savefig(3)

#====================================================================
def main():

    # arguments (run with -h to summarize)
    parser = argparse.ArgumentParser(description='SPIDER plotting script')
    parser.add_argument('-t', '--times', type=str, help='Comma-separated (no spaces) list of times');
    parser.add_argument('-f1', '--fig1', help='Plot figure 1', action="store_true")
    parser.add_argument('-f2', '--fig2', help='Plot figure 2', action="store_true")
    parser.add_argument('-f3', '--fig3', help='Plot figure 3', action="store_true")
    args = parser.parse_args()

    # if nothing specified, choose a default set
    if not (args.fig1 or args.fig2 or args.fig3):
        args.fig1 = True;
        args.fig2 = True;
        args.fig3 = True;

    if not args.times:
        logger.critical( 'You must specify times in a comma-separated list (no spaces) using -t' )
        sys.exit(0)

    # reproduce staple figures in Bower et al. (2018)
    # i.e., figs 3,4,5,6,7,8
    if args.fig1 :
        spiderplot_fig1( times=args.times )
    if args.fig2 :
        spiderplot_fig2( times=args.times )
    if args.fig3 :
        spiderplot_fig3( times=args.times )

    plt.show()

#====================================================================

if __name__ == "__main__":

    main()
