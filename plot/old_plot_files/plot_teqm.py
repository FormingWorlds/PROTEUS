#!/usr/bin/env python

import numpy as np
import spider_utils as su
import argparse
import logging
import os
import matplotlib.pyplot as plt

#====================================================================
def plot_teqm():

    figw = 4.7747/2.0
    figh = 4.7747/2.0

    fig_o = su.FigureData( 1, 1, figw, figh, 'teqm', units='Myr' )

    ax0 = fig_o.ax

    # equilibrium temperatures to process
    teqm_l = [1000,1100,1300,1500,1600,1700,1750,1800,1850,1900,2100]

    time_l = su.get_all_output_times( odir='1000/output' )
    time_myr = time_l * 1.0E-6 # to Myr

    handles = []

    for tt, teqm in enumerate( teqm_l ):
        # read json
        indir = os.path.join( str(teqm), 'output' )
        rf_depth = su.get_dict_values_for_times( ['rheological_front_phi','depth'], time_l, indir=indir )
        rf_depth *= 1.0E-3 # to km
        h1, = ax0.semilogx( time_myr, rf_depth, label=str(teqm) )
        handles.append( h1 )

    fig_o.set_mylegend( ax0, handles, loc='upper right', TITLE='Teqm' )

    xlabel = 'Time (Myr)'
    ylabel = 'Magma ocean depth (km)'
    fig_o.set_myaxes( ax0, xlabel=xlabel, ylabel=ylabel, yrotation=90 )
    #ax0.set_xlim( 1.0E-1, 1 )
    ax0.invert_yaxis()

    fig_o.savefig(1)

#====================================================================

if __name__ == "__main__":

    # arguments (run with -h to summarize)
    #parser = argparse.ArgumentParser(description='SPIDER plotting script')
    #parser.add_argument('-t', '--times', type=str, help='Comma-separated (no spaces) list of times');
    #args = parser.parse_args()

    #if not args.times:
    #    logger.critical( 'You must specify times in a comma-separated list (no spaces) using -t' )
    #    sys.exit(0)

    plot_teqm()
    plt.show()
