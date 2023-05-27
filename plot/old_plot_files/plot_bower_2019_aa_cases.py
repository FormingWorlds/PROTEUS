#!/usr/bin/env python

import spider_utils as su
import matplotlib.pyplot as plt
import numpy as np
import os, json
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
from scipy.interpolate import interp1d

#====================================================================
def get_param_data_d():

    '''parameter dictionary with key attributes of the cases that will
       processed and stored in a single json'''

    data_d = {  'case9': {
                'dir':'case9_2558_116',
                'CO2_ppm': 2558,
                'H2O_ppm': 116,
                'mix': 9.0/10.0},
                'case9m': {
                'dir':'case9m_2558_116',
                'CO2_ppm': 2558,
                'H2O_ppm': 116,
                'mix': 9.0/10.0},
                 # case 8
                'case8': {
                'dir':'case8_1483_121',
                'CO2_ppm':1483,
                'H2O_ppm':121,
                'mix': 0.83},
                 # case 7
                'case7': {
                'dir':'case7_941_128',
                'CO2_ppm': 941,
                'H2O_ppm': 128,
                'mix': 0.75},
                'case7w': {
                'dir':'case7w_941_128',
                'CO2_ppm': 941,
                'H2O_ppm': 128,
                'mix': 0.75},
                'case7m': {
                'dir':'case7m_941_128',
                'CO2_ppm': 941,
                'H2O_ppm': 128,
                'mix': 0.75},
                'case7mw': {
                'dir':'case7mw_941_128',
                'CO2_ppm': 941,
                'H2O_ppm': 128,
                'mix': 0.75},
                 # case 6
                'case6': {
                'dir':'case6_666_136',
                'CO2_ppm':666,
                'H2O_ppm':136,
                'mix': 0.66},
                 # case 5
                'case5': {
                'dir':'case5_380_155',
                'CO2_ppm': 380,
                'H2O_ppm': 155,
                'mix': 1.0/2.0},
                'case5m': {
                'dir':'case5m_380_155',
                'CO2_ppm': 380,
                'H2O_ppm': 155,
                'mix': 1.0/2.0},
                 # case 4
                'case4': {
                'dir':'case4_221_181',
                'CO2_ppm': 221,
                'H2O_ppm': 181,
                'mix': 1.0/3.0},
                 # case 3
                'case3': {
                'dir':'case3_160_197',
                'CO2_ppm': 160,
                'H2O_ppm': 197,
                'mix': 1.0/4.0},
                'case3w': {
                'dir':'case3w_160_197',
                'CO2_ppm': 160,
                'H2O_ppm': 197,
                'mix': 1.0/4.0},
                'case3m': {
                'dir':'case3m_160_197',
                'CO2_ppm': 160,
                'H2O_ppm': 197,
                'mix': 1.0/4.0},
                'case3mw': {
                'dir':'case3mw_160_197',
                'CO2_ppm': 160,
                'H2O_ppm': 197,
                'mix': 1.0/4.0},
                'case3m5': {
                'dir':'case3m5_160_197',
                'CO2_ppm': 160,
                'H2O_ppm': 197,
                'mix': 1.0/4.0},
                 # case 2
                'case2': {
                'dir':'case2_79_227',
                'CO2_ppm': 79,
                'H2O_ppm': 227,
                'mix': 1.0/8.0},
                 # case 1
                'case1': {
                'dir':'case1_63_234',
                'CO2_ppm': 63,
                'H2O_ppm': 234,
                'mix': 1.0/10.0},
                'case1m': {
                'dir':'case1m_63_234',
                'CO2_ppm': 63,
                'H2O_ppm': 234,
                'mix': 1.0/10.0},
                 # case N
                'caseN': {
                'dir':'caseN_120_410',
                'CO2_ppm': 120,
                'H2O_ppm': 410,
                'mix': 0.0},
                'caseNw': {
                'dir':'caseNw_120_410',
                'CO2_ppm': 120,
                'H2O_ppm': 410,
                'mix': 0.0}
            }

    return data_d

#====================================================================
def get_model_data( indir ):

    '''return dictionary of data for a particular case/model'''

    width = 4.7747 / 2.0
    height = 4.7747 / 2.0
    fig_o = su.FigureData( 1, 1, width, height, 'atmosphere', units='Myr' )
    fig_o.fig.subplots_adjust(wspace=0.5,hspace=1.0)

    ax0 = fig_o.ax

    fig_o.time = su.get_all_output_times( indir )

    keys_t = ( ('atmosphere','mass_liquid'),
               ('atmosphere','mass_solid'),
               ('atmosphere','mass_mantle'),
               ('atmosphere','CO2','liquid_kg'),
               ('atmosphere','CO2','liquid_ppm' ),
               ('atmosphere','CO2','solid_kg'),
               ('atmosphere','CO2','initial_kg'),
               ('atmosphere','CO2','atmosphere_kg'),
               ('atmosphere','CO2','atmosphere_bar'),
               ('atmosphere','CO2','mixing_ratio'),
               ('atmosphere','H2O','liquid_kg'),
               ('atmosphere','H2O','liquid_ppm' ),
               ('atmosphere','H2O','solid_kg'),
               ('atmosphere','H2O','initial_kg'),
               ('atmosphere','H2O','atmosphere_kg'),
               ('atmosphere','H2O','atmosphere_bar'),
               ('atmosphere','H2O','mixing_ratio'),
               ('atmosphere','temperature_surface'),
               ('atmosphere','emissivity'),
               ('rheological_front_phi','phi_global') )

    data_a = su.get_dict_surface_values_for_times( keys_t, fig_o.time, indir=indir )

    return fig_o.time, data_a

#====================================================================
def save_row_to_dict( param_d, strcase, key, data_a, rownum ):

    param_d[strcase][key] = data_a[rownum,:].tolist()

#====================================================================
def dump_all_data():

    '''for all cases listed in case_l, get the data from the output
       files and dump everything into a single json'''

    param_d = get_param_data_d()

    case_l = [1,'1m',2,3,'3w','3m','3mw','3m5',4,5,'5m',6,7,'7w','7m','7mw',8,9,'9m','N','Nw']
    #case_l = ['N','Nw']
    #case_l = ['3m5']

    for case in case_l:
        print(case_l,case)
        strcase = 'case'+str(case)
        print( 'working on', strcase )
        indir = param_d[strcase]['dir'] + '/output'
        print( indir )
        time_a, data_a = get_model_data( indir )
        # add to dictionary

        save_row_to_dict( param_d, strcase, 'mass_liquid', data_a, 0 )
        save_row_to_dict( param_d, strcase, 'mass_solid', data_a, 1 )
        save_row_to_dict( param_d, strcase, 'mass_mantle', data_a, 2 )
        save_row_to_dict( param_d, strcase, 'CO2_liquid_kg', data_a, 3 )
        save_row_to_dict( param_d, strcase, 'CO2_liquid_ppm', data_a, 4 )
        save_row_to_dict( param_d, strcase, 'CO2_solid_kg', data_a, 5 )
        save_row_to_dict( param_d, strcase, 'CO2_initial_kg', data_a, 6 )
        save_row_to_dict( param_d, strcase, 'CO2_atmosphere_kg', data_a, 7 )
        save_row_to_dict( param_d, strcase, 'CO2_atmosphere_bar', data_a, 8 )
        save_row_to_dict( param_d, strcase, 'CO2_mixing_ratio', data_a, 9 )
        save_row_to_dict( param_d, strcase, 'H2O_liquid_kg', data_a, 10 )
        save_row_to_dict( param_d, strcase, 'H2O_liquid_ppm', data_a, 11 )
        save_row_to_dict( param_d, strcase, 'H2O_solid_kg', data_a, 12 )
        save_row_to_dict( param_d, strcase, 'H2O_initial_kg', data_a, 13 )
        save_row_to_dict( param_d, strcase, 'H2O_atmosphere_kg', data_a, 14 )
        save_row_to_dict( param_d, strcase, 'H2O_atmosphere_bar', data_a, 15 )
        save_row_to_dict( param_d, strcase, 'H2O_mixing_ratio', data_a, 16 )
        save_row_to_dict( param_d, strcase, 'temperature_surface', data_a, 17 )
        save_row_to_dict( param_d, strcase, 'emissivity', data_a, 18 )
        save_row_to_dict( param_d, strcase, 'phi_global', data_a, 19 )

        param_d[strcase]['time'] = time_a.tolist()

    with open('3m5data.json', 'w') as fp:
        json.dump(param_d, fp)

#====================================================================
def plot_interior_depletion():

    #case_l = [1,3,5,7,9]
    # for comparing models that are wrong
    #case_l = [3,'3w',7,'7w']
    #case_l = ['N','Nw']
    case_l = ['1m','3m5','5m','7m','9m']

    label_l = ['1c','3c','5c','7c','9c']

    width = 4.7747 / 2.0
    height = 4.7747 / 2.0

    with open('data.json', 'r') as fp:
        data_d = json.load(fp )

    # number of colors for plotting the cases
    times = '4550000000,0,0,0,0' # fake zeros to replicate number of cases
    fig_o = su.FigureData( 1, 1, width, height, 'interior_depletion', times, units='Myr' )
    fig_o.fig.subplots_adjust(wspace=0.3,hspace=0.4)
    ax0 = fig_o.ax

    handle_l = []

    for nn, case in enumerate(case_l):
        strcase = 'case'+str(case)
        print( 'working on ', strcase )
        phi = np.array(data_d[strcase]['phi_global'])
        time = np.array(data_d[strcase]['time'])
        time_myrs = time * 1.0E-6
        CO2_atmosphere_kg = np.array(data_d[strcase]['CO2_atmosphere_kg'])
        CO2_initial_kg = np.array(data_d[strcase]['CO2_initial_kg'] )
        CO2_interior = (CO2_initial_kg - CO2_atmosphere_kg)
        CO2_interior_t0 = CO2_interior[0]
        CO2_dep = (CO2_interior-CO2_interior_t0)/CO2_interior_t0 * 100.0
        H2O_atmosphere_kg = np.array(data_d[strcase]['H2O_atmosphere_kg'])
        H2O_initial_kg = np.array(data_d[strcase]['H2O_initial_kg'] )
        H2O_interior = (H2O_initial_kg - H2O_atmosphere_kg)
        H2O_interior_t0 = H2O_interior[0]
        H2O_dep = (H2O_interior-H2O_interior_t0)/H2O_interior_t0 * 100.0
        color = fig_o.get_color(nn)
        mix = data_d[strcase]['mix']
        #label = str(case) + ' (' + str(mix)[0:4] + ')'
        label = label_l[nn] + ' (' + str(mix)[0:4] + ')'
        color = fig_o.get_color(nn)
        ax0.plot( phi*100.0, -CO2_dep, color=color, label=label, linestyle='--' )
        h1, = ax0.plot( phi*100.0, -H2O_dep, color=color, label=label, linestyle='-' )
        handle_l.append( h1 )

    title = r'\textbf{Interior volatile depletion}'
    ylabel = r'Depletion (\%)'
    xlabel = r'$\phi_g$ (\%)'
    fig_o.set_myaxes( ax0, title=title, ylabel=ylabel, xlabel=xlabel, yrotation=90 )
    ax0.yaxis.set_label_coords(-0.125,0.5)
    ax0.xaxis.set_label_coords(0.5,-0.1)
    ax0.invert_xaxis()

    #title = r'\textbf{(b) H$_2$O interior depletion}'
    #ylabel = r'H$_2$O depletion (\%)'
    #xlabel = r'$\phi_g$'
    #fig_o.set_myaxes( ax1, title=title, ylabel=ylabel, xlabel=xlabel, yrotation=90 )
    #ax1.yaxis.set_label_coords(-0.15,0.5)
    #ax1.invert_xaxis()

    ax0.text( 4,78,'CO$_2$\n(dashed)', ha='center')
    ax0.text( 4,35,'H$_2$O\n(solid)', ha='center')

    TITLE = "Case ($r_{f,CO_2}$)"
    fig_o.set_mylegend( ax0, handle_l, loc='center left', ncol=1, TITLE=TITLE )
    #fig_o.set_mylegend( ax1, handle_l, loc='upper left', ncol=1, TITLE=TITLE )

    ax0.set_xlim(10,0)
    #ax1.set_xlim(0.1,0)

    fig_o.savefig(1)

#====================================================================
def plot_partial_pressure_versus_depletion():

    #case_l = (1,3,5,7,9)
    # for comparing models that are wrong
    #case_l = (3,'3w',7,'7w')
    case_l = ('7m','7mw','7')

    width = 4.7747 / 2.0 * 3.0
    height = 4.7747 / 2.0

    with open('data.json', 'r') as fp:
        data_d = json.load(fp )

    # number of colors for plotting the cases
    times = '4550000000,0,0,0,0' # fake zeros to replicate number of cases
    fig_o = su.FigureData( 1, 3, width, height, 'partialp_depletion', times, units='Myr' )
    fig_o.fig.subplots_adjust(wspace=0.3,hspace=0.4)

    ax0 = fig_o.ax[0]
    ax1 = fig_o.ax[1]
    ax2 = fig_o.ax[2]

    handle_l = []

    for nn, case in enumerate(case_l):
        strcase = 'case'+str(case)
        print( 'working on ', strcase )
        phi = np.array(data_d[strcase]['phi_global'])
        time = np.array(data_d[strcase]['time'])
        time_myrs = time * 1.0E-6
        CO2_atmosphere_bar = np.array(data_d[strcase]['CO2_atmosphere_bar'])
        CO2_atmosphere_kg = np.array(data_d[strcase]['CO2_atmosphere_kg'])
        CO2_initial_kg = np.array(data_d[strcase]['CO2_initial_kg'] )
        CO2_interior = (CO2_initial_kg - CO2_atmosphere_kg)
        CO2_interior_t0 = CO2_interior[0]
        CO2_dep = (CO2_interior-CO2_interior_t0)/CO2_interior_t0 * 100.0
        H2O_atmosphere_bar = np.array(data_d[strcase]['H2O_atmosphere_bar'])
        H2O_atmosphere_kg = np.array(data_d[strcase]['H2O_atmosphere_kg'])
        H2O_initial_kg = np.array(data_d[strcase]['H2O_initial_kg'] )
        H2O_interior = (H2O_initial_kg - H2O_atmosphere_kg)
        H2O_interior_t0 = H2O_interior[0]
        H2O_dep = (H2O_interior-H2O_interior_t0)/H2O_interior_t0 * 100.0
        color = fig_o.get_color(nn)
        mix = data_d[strcase]['mix']
        label = str(case) + ' (' + str(mix)[0:4] + ')'
        color = fig_o.get_color(nn)
        # figure a
        h1, = ax0.plot( -CO2_dep, CO2_atmosphere_bar, color=color, label=label, linestyle='-' )
        ax0.plot( -CO2_dep, H2O_atmosphere_bar, color=color, label=label, linestyle='--' )
        # figure b
        ax1.plot( -H2O_dep, CO2_atmosphere_bar, color=color, label=label, linestyle='-' )
        ax1.plot( -H2O_dep, H2O_atmosphere_bar, color=color, label=label, linestyle='--' )
        # figure c
        ax2.plot( phi*100, CO2_atmosphere_bar, color=color, label=label, linestyle='-' )
        ax2.plot( phi*100, H2O_atmosphere_bar, color=color, label=label, linestyle='--' )
        handle_l.append( h1 )

    title = r'\textbf{Partial pressure}'
    ylabel = r'$p$ (bar)'
    xlabel = r'CO$_2$ depletion (\%)'
    fig_o.set_myaxes( ax0, title=title, ylabel=ylabel, xlabel=xlabel, yrotation=90 )
    ax0.yaxis.set_label_coords(-0.2,0.5)
    ax0.xaxis.set_label_coords(0.5,-0.125)
    ax0.set_xlim(25,100)

    title = r'\textbf{Partial pressure}'
    ylabel = r'$p$ (bar)'
    xlabel = r'H$_2$O depletion (\%)'
    fig_o.set_myaxes( ax1, title=title, ylabel=ylabel, xlabel=xlabel, yrotation=90 )
    ax1.yaxis.set_label_coords(-0.2,0.5)
    ax1.xaxis.set_label_coords(0.5,-0.125)
    ax1.set_xlim(25,100)

    title = r'\textbf{Partial pressure}'
    ylabel = r'$p$ (bar)'
    xlabel = r'$\phi_g$ (\%)'
    fig_o.set_myaxes( ax2, title=title, ylabel=ylabel, xlabel=xlabel, yrotation=90 )
    ax2.yaxis.set_label_coords(-0.2,0.5)
    ax2.xaxis.set_label_coords(0.5,-0.125)
    ax2.invert_xaxis()
    ax2.set_xlim(10,0)


    #title = r'\textbf{(b) H$_2$O interior depletion}'
    #ylabel = r'H$_2$O depletion (\%)'
    #xlabel = r'$\phi_g$'
    #fig_o.set_myaxes( ax1, title=title, ylabel=ylabel, xlabel=xlabel, yrotation=90 )
    #ax1.yaxis.set_label_coords(-0.15,0.5)
    #ax1.invert_xaxis()

    #ax0.text( 0.04,78,'CO$_2$\n(dashed)', ha='center')
    #ax0.text( 0.04,35,'H$_2$O\n(solid)', ha='center')

    TITLE = "Case ($r_{f,CO_2}$)"
    fig_o.set_mylegend( ax0, handle_l, loc='center left', ncol=1, TITLE=TITLE )
    #fig_o.set_mylegend( ax1, handle_l, loc='upper left', ncol=1, TITLE=TITLE )

    fig_o.savefig(2)

#====================================================================
def plot_pressure_side_by_side():

    width = 4.7747 / 1.0
    height = 4.7747 / 2.0

    with open('data.json', 'r') as fp:
        data_d = json.load(fp)

    fig_o = su.FigureData( 1, 2, width, height, 'case1_case9_atmosphere', units='Myr' )
    fig_o.fig.subplots_adjust(wspace=0.3,hspace=0.3)

    ax0 = fig_o.ax[0]
    ax1 = fig_o.ax[1]

    title = r'\textbf{(a) Case 1c }' + r'($\bf{r_{f,CO_2}=0.1}$)'
    label_align_l = ('center','center','center','center')
    ylabeloff = 0.5
    plot_pressure_figure( title, fig_o, ax0, '1m', label_align_l, ylabeloff, data_d )

    title = r'\textbf{(b) Case 9c }' + r'($\bf{r_{f,CO_2}=0.9}$)'
    label_align_l = ('center','center','center','left')
    ylabeloff = 0.54
    plot_pressure_figure( title, fig_o, ax1, '9m', label_align_l, ylabeloff, data_d )

    fig_o.savefig(3)

#====================================================================
def plot_pressure_figure( title, fig_o, ax, case, label_align_l, ylabeloff, data_d ):

    strcase = 'case'+str(case)
    Case_time_myrs = np.array(data_d[strcase]['time']) * 1.0E-6
    Case_CO2_atmos_a = np.array(data_d[strcase]['CO2_atmosphere_bar'])
    Case_H2O_atmos_a = np.array(data_d[strcase]['H2O_atmosphere_bar'])
    Case_phi_global = np.array(data_d[strcase]['phi_global'])

    # to plot melt fraction contours on figure (a)
    # compute time at which desired melt fraction is reached
    phi_cont_l = [0.75,0.30,0.10,0.01]
    phi_cont_label = ['0.75','0.30','0.10','0.01']
    phi_time_l = [] # contains the times for each contour
    for cont in phi_cont_l:
        time_temp_l = su.find_xx_for_yy( Case_time_myrs, Case_phi_global, cont )
        index = su.get_first_non_zero_index( time_temp_l )
        if index is None:
            out = 0.0
        else:
            out = Case_time_myrs[index]
        phi_time_l.append( out )

    print( case, phi_time_l )

    #xticks = [1.0E-2, 1.0E-1, 1.0E0, 1.0E1, 1.0E2,1.0E3] #[1E-6,1E-4,1E-2,1E0,1E2,1E4,1E6]#,1]
    xlabel = 'Time (Myr)'

    if case=='1m':
        xlim = (1.0E-2, 1.0E1)# 4550)
    else:
        xlim = (1.0E0,1.0E2)

    red = (0.5,0.1,0.1)
    blue = (0.1,0.1,0.5)
    black = 'black'

    ylabel = '$p$\n(bar)'
    trans = transforms.blended_transform_factory( ax.transData, ax.transAxes)
    h1, = ax.semilogx( Case_time_myrs, Case_CO2_atmos_a, color=red, linestyle='-', label=r'CO$_2$')
    h2, = ax.semilogx( Case_time_myrs, Case_H2O_atmos_a, color=blue, linestyle='-', label=r'H$_2$O')
    #axb = ax.twinx()
    #h3, = axb.semilogx( Case_time_myrs, Case_phi_global, color=black, linestyle='--', label=r'Melt, $\phi_g$')
    for cc, cont in enumerate(phi_cont_l):
        ax.axvline( phi_time_l[cc], ymin=0.02, ymax=0.95, color='0.25', linestyle=':' )
        label = phi_cont_label[cc] #int(cont*100) # as percent
        #label = str(round(label,2))
        if cc == 0:
            label= r'$\phi_g=$ ' + str(label)
        ha = label_align_l[cc]
        #ax.text( phi_time_l[cc], 0.5, '{0:.2f}'.format(label), va='top', ha=ha, rotation=90, bbox=dict(facecolor='white', edgecolor='none', pad=2), transform=trans )
        ax.text( phi_time_l[cc], 0.6, label, va='top', ha=ha, rotation=90, bbox=dict(facecolor='white', edgecolor='none', pad=2), transform=trans )
    handle_l = [h1,h2]#,h3]
    fig_o.set_myaxes( ax, title=title, ylabel=ylabel, xlabel=xlabel)#, xticks=xticks )
    if case=='1m' or case=='9m':
        fig_o.set_mylegend( ax, handle_l, loc='upper left', ncol=1, facecolor='white', framealpha=1.0 )
        #fig_o.set_mylegend( ax, handle_l, loc=(0.02,0.1), ncol=1 )
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim( *xlim )
    ax.yaxis.set_label_coords(-0.1,ylabeloff)
    #axb.set_ylabel( r'$\phi_g$', rotation=0 )
    #axb.yaxis.set_label_coords(1.1,0.525)

#====================================================================
def plot_atmosphere_comparison():

    '''make four panel figure which compares various outputs associated
       with the atmosphere for the cases listed in case_l'''

    # models with conventional mixing length
    #case_l = [1,3,5,7,9]
    # models with constant mixing length
    case_l = ['1m','3m5','5m','7m','9m']
    label_l = ['1c','3c','5c','7c','9c']

    #case_l = [3,'3w',7,'7w']

    width = 4.7747
    height = 4.7747

    with open('data.json', 'r') as fp:
        data_d = json.load(fp )

    # number of colors for plotting the cases
    times = '4550000000,0,0,0,0' # fake zeros to replicate number of cases
    fig_o = su.FigureData( 2, 2, width, height, 'all_atmosphere', times, units='Myr' )
    fig_o.fig.subplots_adjust(wspace=0.3,hspace=0.4)
    ax0 = fig_o.ax[0][0]
    ax1 = fig_o.ax[0][1]
    ax2 = fig_o.ax[1][0]
    ax3 = fig_o.ax[1][1]

    h_l = []

    for nn, case in enumerate(case_l):
        strcase = 'case'+str(case)
        print( 'working on ', strcase )
        phi = np.array(data_d[strcase]['phi_global'])
        time = np.array(data_d[strcase]['time'])
        time_myrs = time * 1.0E-6
        CO2_mixing_ratio = np.array(data_d[strcase]['CO2_mixing_ratio'])
        #CO2_atmosphere_kg = np.array(data_d[strcase]['CO2_atmosphere_kg'])
        #CO2_initial_kg = np.array(data_d[strcase]['CO2_initial_kg'] )
        CO2_liquid_ppm = np.array(data_d[strcase]['CO2_liquid_ppm'] )
        #CO2_interior = (CO2_initial_kg - CO2_atmosphere_kg) / CO2_initial_kg
        H2O_mixing_ratio = np.array(data_d[strcase]['H2O_mixing_ratio'])
        H2O_liquid_ppm = np.array(data_d[strcase]['H2O_liquid_ppm'] )
        mix = data_d[strcase]['mix']
        #label = str(case) + ' (' + str(mix)[0:4] + ')'
        label = label_l[nn] + ' (' + str(mix)[0:4] + ')'
        color = fig_o.get_color(nn)
        MASK = phi > 0.01
        # must convert to float array otherwise np.nan masking does not work!
        MASK = MASK*1.0 # convert to float array
        MASK[MASK==0] = np.nan
        h1, = ax0.semilogx( time_myrs, phi, label=label, color=color )
        h2, = ax1.semilogx( time_myrs, CO2_mixing_ratio, label=label, color=color )
        h3, = ax2.loglog( time_myrs, MASK*CO2_liquid_ppm, label=label, color=color )
        h4, = ax3.loglog( time_myrs, MASK*H2O_liquid_ppm, label=label, color=color )
        h_l.append( h1 )

    # let's just say times are same for all models
    time_l = data_d['case'+str(case_l[0])]['time'][1:]

    time_myrs = np.array(time_l) * 1.0E-6

    fig_o.set_mylegend( ax1, h_l, loc='lower left', ncol=1, TITLE="Case ($r_{f,CO_2}$)" )

    xlabel = 'Time (Myr)'
    #xticks = [1.0E-2, 1.0E-1, 1.0E0, 1.0E1, 1.0E2,1.0E3]
    xlim = (1.0E-2, 1.0E2)#4550)

    title = r'\textbf{(a) Global melt fraction,} $\bf{\phi_g}$'
    ylabel = r'$\phi_g$'
    fig_o.set_myaxes( ax0, title=title, ylabel=ylabel, xlabel=xlabel )#, xticks=xticks )
    ax0.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax0.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax0.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax0.set_xlim( *xlim )
    ax0.yaxis.set_label_coords(-0.1,0.45)

    title = r'\textbf{(b) CO$_2$ volume mixing ratio}'
    ylabel = r'$r_{CO_2}$'
    fig_o.set_myaxes( ax1, title=title, ylabel=ylabel, xlabel=xlabel )#, xticks=xticks )
    ax1.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax1.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.set_xlim( *xlim )
    ax1.yaxis.set_label_coords(-0.15,0.45)
    ax1.set_ylim( 0, 1.05 )

    title = r'\textbf{(c) CO$_2$ in interior (ppm)}'
    ylabel = r'$X_{CO_2}$'
    fig_o.set_myaxes( ax2, title=title, ylabel=ylabel, yrotation=0, xlabel=xlabel)# xticks=xticks )
    ax2.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax2.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.set_xlim( *xlim )
    ax2.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax2.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.yaxis.set_label_coords(-0.15,0.55)
    ax2.set_xlim( *xlim )

    title = r'\textbf{(d) H$_2$O in interior (ppm)}'
    ylabel = r'$X_{H_2O}$'
    fig_o.set_myaxes( ax3, title=title, ylabel=ylabel, yrotation=0, xlabel=xlabel)#, xticks=xticks )
    ax3.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax3.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax3.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax3.set_xlim( *xlim )
    ax3.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax3.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax3.yaxis.set_label_coords(-0.15,0.55)
    ax3.set_ylim([1.0E2,1.0E4])

    fig_o.savefig(4)

#====================================================================
def plot_right_versus_wrong():

    '''make four panel figure which compares various outputs associated
       with the atmosphere for the cases listed in case_l'''

    # conventional mixing length
    #case_l = (3,'3w',7,'7w')

    # constant mixing length
    case_l = ['3m5','3mw','7m','7mw']
    #case_l = ('7m','7mw')
    #case_l = ('N','Nw')

    label_l = ['3c','3cw','7c','7cw']

    width = 4.7747 * 3.0 / 2.0
    height = 4.7747 / 2.0

    with open('data.json', 'r') as fp:
        data_d = json.load(fp )

    # number of colors for plotting the cases
    times = '4550000000,0,0,0,0' # fake zeros to replicate number of cases
    fig_o = su.FigureData( 1, 3, width, height, 'right_wrong', times, units='Myr' )
    fig_o.fig.subplots_adjust(wspace=0.3,hspace=0.4)
    ax0 = fig_o.ax[0]
    ax1 = fig_o.ax[1]
    ax2 = fig_o.ax[2]

    h_l = []

    # to exactly match colors of case 3 and 7 with the other figures
    red = fig_o.get_color(1)
    blue = fig_o.get_color(3)

    colors_l = [red,red,blue,blue]
    linespec_l = ['-','--','-','--']

    for nn, case in enumerate(case_l):
        strcase = 'case'+str(case)
        print( 'working on ', strcase )
        phi = np.array(data_d[strcase]['phi_global'])
        time = np.array(data_d[strcase]['time'])
        time_myrs = time * 1.0E-6

        phi = np.array(data_d[strcase]['phi_global'])
        time = np.array(data_d[strcase]['time'])
        CO2_atmosphere_kg = np.array(data_d[strcase]['CO2_atmosphere_kg'])
        CO2_initial_kg = np.array(data_d[strcase]['CO2_initial_kg'] )
        CO2_interior = (CO2_initial_kg - CO2_atmosphere_kg)
        CO2_interior_t0 = CO2_interior[0]
        CO2_dep = (CO2_interior-CO2_interior_t0)/CO2_interior_t0 * 100.0
        H2O_atmosphere_kg = np.array(data_d[strcase]['H2O_atmosphere_kg'])
        H2O_initial_kg = np.array(data_d[strcase]['H2O_initial_kg'] )
        H2O_interior = (H2O_initial_kg - H2O_atmosphere_kg)
        H2O_interior_t0 = H2O_interior[0]
        H2O_dep = (H2O_interior-H2O_interior_t0)/H2O_interior_t0 * 100.0
        CO2_mixing_ratio = np.array(data_d[strcase]['CO2_mixing_ratio'])
        CO2_liquid_ppm = np.array(data_d[strcase]['CO2_liquid_ppm'] )
        H2O_mixing_ratio = np.array(data_d[strcase]['H2O_mixing_ratio'])
        H2O_liquid_ppm = np.array(data_d[strcase]['H2O_liquid_ppm'] )
        mix = data_d[strcase]['mix']
        label = label_l[nn]
        if nn==0 or nn==2:
            right = H2O_dep
            rightphi = phi
            label += ' (' + str(mix)[0:4] + ')'
        else:
            label += r', \textbf{incorrect}'
            wrong = H2O_dep
            wrongphi = phi
            rw = right - wrong
            print( np.max(rw))
            print (phi[np.argmax( rw )])
            # check
            print( rw[np.argmax( rw )] )
        color = colors_l[nn]
        MASK = phi > 0.01
        # must convert to float array otherwise np.nan masking does not work!
        MASK = MASK*1.0 # convert to float array
        MASK[MASK==0] = np.nan
        linespec = linespec_l[nn]
        h0, = ax0.semilogx( time_myrs, CO2_mixing_ratio, label=label, color=color, linestyle=linespec )
        h1, = ax1.loglog( time_myrs, MASK*CO2_liquid_ppm, label=label, color=color, linestyle=linespec )
        h2, = ax2.plot( phi*100.0, -H2O_dep, color=color, label=label, linestyle=linespec )
        #h3, = ax2.plot( phi*100.0, -CO2_dep, color=color, label=label, linestyle=linespec )
        h_l.append( h0 )

    # let's just say times are same for all models
    time_l = data_d['case'+str(case_l[0])]['time'][1:]

    time_myrs = np.array(time_l) * 1.0E-6

    fig_o.set_mylegend( ax2, h_l, loc='upper left', ncol=1, TITLE="Case ($r_{f,CO_2}$)" )

    xlabel = 'Time (Myr)'
    #xticks = [1.0E-2, 1.0E-1, 1.0E0, 1.0E1, 1.0E2,1.0E3]
    #xlim = (1.0E-2, 4550)
    #xticks = (1.0E-6,1.0E-5,1.0E-4,1.0E-3,1.0E-2,1.0E-1,1.0E0)
    xlim = (1.0E-2,1.0E2)

    title = r'\textbf{(a) CO$_2$ volume mixing ratio}'
    ylabel = r'$r_{CO_2}$'
    fig_o.set_myaxes( ax0, title=title, ylabel=ylabel, xlabel=xlabel )#, xticks=xticks )
    ax0.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax0.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax0.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax0.set_xlim( *xlim )
    ax0.yaxis.set_label_coords(-0.15,0.45)
    ax0.set_ylim( 0, 1.05 )

    title = r'\textbf{(b) CO$_2$ in interior (ppm)}'
    ylabel = r'$X_{CO_2}$'
    fig_o.set_myaxes( ax1, title=title, ylabel=ylabel, yrotation=0, xlabel=xlabel)# xticks=xticks )
    ax1.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax1.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.set_xlim( *xlim )
    ax1.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    #ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.yaxis.set_label_coords(-0.15,0.55)
    ax1.set_xlim( *xlim )

    title = r'\textbf{(c) Interior H$_2$O depletion}'
    ylabel = r'Depletion (\%)'
    xlabel = r'$\phi_g$ (\%)'
    fig_o.set_myaxes( ax2, title=title, ylabel=ylabel, xlabel=xlabel, yrotation=90 )
    ax2.yaxis.set_label_coords(-0.15,0.5)
    ax2.xaxis.set_label_coords(0.5,-0.15)
    ax2.invert_xaxis()
    #ax2.set_ylim( 85,100)

    #title = r'\textbf{(b) H$_2$O interior depletion}'
    #ylabel = r'H$_2$O depletion (\%)'
    #xlabel = r'$\phi_g$'
    #fig_o.set_myaxes( ax1, title=title, ylabel=ylabel, xlabel=xlabel, yrotation=90 )
    #ax1.yaxis.set_label_coords(-0.15,0.5)
    #ax1.invert_xaxis()

    #ax2.text( 0.04,78,'CO$_2$\n(dashed)', ha='center')
    ax2.text( 5,35,'H$_2$O', ha='center')

    #TITLE = "Case ($r_{f,CO_2}$)"
    #fig_o.set_mylegend( ax2, handle_l, loc='center left', ncol=1, TITLE=TITLE )
    #fig_o.set_mylegend( ax1, handle_l, loc='upper left', ncol=1, TITLE=TITLE )

    ax2.set_xlim(15,0)
    #ax1.set_xlim(0.1,0)

    fig_o.savefig(5)

#====================================================================
def plot_mantle_temperature_at_end():

    case_l = (1,3,5,7,9)

    figw = 4.7747 / 2.0
    figh = 4.7747 / 2.0

    with open('data.json', 'r') as fp:
        data_d = json.load(fp )

    # fake zeros to ensure the list of times give the necessary
    # number of colors for plotting the cases
    times = '4550000000,0,0,0,0' # fake zeros to replicate number of cases
    fig_o = su.FigureData( 1, 1, figw, figh, 'mantle_temp_4550myr', times, units='Myr' )
    ax0 = fig_o.ax

    time = fig_o.time[0] # first timestep since liquidus and solidus
                         # are time-independent

    # some preliminaries
    param_d = get_param_data_d()
    strcase = 'case{}'.format( case_l[0] )
    case_dir = param_d[strcase]['dir']
    myjson_o = su.MyJSON( '{}/output/{}.json'.format(case_dir,time) )
    TIMEYRS = myjson_o.data_d['nstep']
    xx_pres = myjson_o.get_dict_values_internal(['data','pressure_b'])
    xx_pres *= 1.0E-9
    xx_pres_s = myjson_o.get_dict_values(['data','pressure_s'])
    xx_pres_s *= 1.0E-9

    # shade grey between liquidus and solidus
    yy_liq = myjson_o.get_dict_values_internal(['data','liquidus_b'])
    yy_sol = myjson_o.get_dict_values_internal(['data','solidus_b'])
    yy_liqt = myjson_o.get_dict_values_internal(['data','liquidus_temp_b'])
    yy_solt = myjson_o.get_dict_values_internal(['data','solidus_temp_b'])
    ax0.fill_between( xx_pres, yy_liqt, yy_solt, facecolor='grey', alpha=0.35, linewidth=0 )

    handle_l = [] # handles for legend

    for nn, case in enumerate( case_l ):
        strcase = 'case'+str(case)
        print( 'working on ', strcase )
        mix = param_d[strcase]['mix']
        label = str(case) + ' (' + str(mix)[0:4] + ')'
        case_dir = param_d[strcase]['dir']
        myjson_o = su.MyJSON( '{}/output/{}.json'.format(case_dir,time) )

        color = fig_o.get_color( nn )
        # use melt fraction to determine mixed region
        MIX = myjson_o.get_mixed_phase_boolean_array( 'basic_internal' )
        MIX_s = myjson_o.get_mixed_phase_boolean_array( 'staggered' )

        yy = myjson_o.get_dict_values(['data','temp_s'])
        ax0.plot( xx_pres_s, yy, '--', color=color )
        handle, = ax0.plot( xx_pres_s*MIX_s, yy*MIX_s, '-', color=color, label=label )
        handle_l.append( handle )


    xticks = [0,25,50,75,100,135]
    xmax = 138

    title = r'\textbf{Mantle temperature at 4.550 Gyr}'
    yticks= [250,1000,2000,3000,4000]
    fig_o.set_myaxes( ax0, title=title, ylabel='$T$ (K)', xticks=xticks, xlabel='$P$ (GPa)', xmax=xmax, yticks=yticks )
    ax0.set_xlim( xticks[0], 138 )
    ax0.yaxis.set_label_coords(-0.1,0.55)
    TITLE = "Case ($r_{f,CO_2}$)"
    fig_o.set_mylegend( ax0, handle_l, loc=4, ncol=2, TITLE=TITLE )

    fig_o.savefig(6)

#====================================================================
def plot_atmosphere( casenum ):

    '''plot atmosphere properties for a single case, which should be
       specified by casenum'''

    # change case number here for plotting
    #casenum = 1

    strcase = 'case'+str(casenum)

    width = 4.7747 * 3.0/2.0
    height = 4.7747 / 2.0
    fig_o = su.FigureData( 1, 3, width, height, strcase+'_atmosphere', units='Myr' )
    fig_o.fig.subplots_adjust(wspace=0.4,hspace=0.3)

    ax0 = fig_o.ax[0]
    ax1 = fig_o.ax[1]
    ax2 = fig_o.ax[2]

    with open('data.json', 'r') as fp:
        data_d = json.load(fp )

    case_d = data_d[strcase]

    mass_liquid_a = np.array(case_d['mass_liquid'])
    mass_solid_a = np.array(case_d['mass_solid'])
    mass_mantle_a = np.array(case_d['mass_mantle'])
    mass_mantle = mass_mantle_a[0] # time independent

    # compute total mass (kg) in each reservoir
    CO2_liquid_kg_a = np.array(case_d['CO2_liquid_kg'])
    CO2_solid_kg_a = np.array(case_d['CO2_solid_kg'])
    CO2_total_kg_a = np.array(case_d['CO2_initial_kg'])
    CO2_total_kg = CO2_total_kg_a[0] # time-independent
    CO2_atmos_kg_a = np.array(case_d['CO2_atmosphere_kg'])
    CO2_atmos_a = np.array(case_d['CO2_atmosphere_bar'])
    CO2_escape_kg_a = CO2_total_kg - CO2_liquid_kg_a - CO2_solid_kg_a - CO2_atmos_kg_a

    # compute total mass (kg) in each reservoir
    H2O_liquid_kg_a = np.array(case_d['H2O_liquid_kg'])
    H2O_solid_kg_a = np.array(case_d['H2O_solid_kg'])
    H2O_total_kg_a = np.array(case_d['H2O_initial_kg'])
    H2O_total_kg = H2O_total_kg_a[0] # time-independent
    H2O_atmos_kg_a = np.array(case_d['H2O_atmosphere_kg'])
    H2O_atmos_a = np.array(case_d['H2O_atmosphere_bar'])
    H2O_escape_kg_a = H2O_total_kg - H2O_liquid_kg_a - H2O_solid_kg_a - H2O_atmos_kg_a

    temperature_surface_a = np.array(case_d['temperature_surface'])
    emissivity_a = np.array(case_d['emissivity'])
    phi_global = np.array(case_d['phi_global'])

    time = np.array(case_d['time'])
    timeMyr_a = time * 1.0E-6

    # write out phi_global versus time, since this is useful
    # to know the mapping for other plotting scripts
    out_a = np.column_stack( (phi_global, time) )
    header = '# phi_global, time'
    np.savetxt(strcase+'_phi_time.dat', out_a )

    xticks = [1.0E-2, 1.0E-1, 1.0E0, 1.0E1, 1.0E2,1.0E3]
    xlabel = 'Time (Myr)'
    xlim = (1.0E-2, 4550)

    red = (0.5,0.1,0.1)
    blue = (0.1,0.1,0.5)
    black = 'black'

    ##########
    # figure a
    ##########
    if 1:
        title = r'\textbf{(a) Pressure and global melt fraction}'
        ylabel = '$p$ (bar)'
        trans = transforms.blended_transform_factory(
            ax0.transData, ax0.transAxes)
        h1, = ax0.semilogx( timeMyr_a, CO2_atmos_a, color=red, linestyle='-', label=r'CO$_2$')
        h2, = ax0.semilogx( timeMyr_a, H2O_atmos_a, color=blue, linestyle='-', label=r'H$_2$O')
        ax0b = ax0.twinx()
        h3, = ax0b.semilogx( timeMyr_a, phi_global, color=black, linestyle='--', label=r'Melt, $\phi_g$')
        handle_l = [h1,h2,h3]
        fig_o.set_myaxes( ax0, title=title, ylabel=ylabel, xlabel=xlabel, xticks=xticks )
        fig_o.set_mylegend( ax0, handle_l, loc='upper center', ncol=1 )
        ax0.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
        ax0.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
        ax0.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax0.set_xlim( *xlim )
        ax0.yaxis.set_label_coords(-0.15,0.5)
        ax0b.set_ylabel( r'$\phi_g$', rotation=0 )
        ax0b.yaxis.set_label_coords(1.1,0.525)

    ##########
    # figure b
    ##########
    if 1:
        title = r'\textbf{(b) Reservoir mass fraction}'
        #h5, = ax1.semilogx( timeMyr_a, mass_liquid_a / mass_mantle, 'k--', label='melt' )
        h1, = ax1.semilogx( timeMyr_a, (CO2_liquid_kg_a+CO2_solid_kg_a) / CO2_total_kg, color=red, linestyle='-', label=r'CO$_2$ interior' )
        h2, = ax1.semilogx( timeMyr_a, CO2_atmos_kg_a / CO2_total_kg, color=red, linestyle='--', label=r'CO$_2$ atmos' )
        #h2b, = ax1.semilogx( timeMyr_a, CO2_escape_kg_a / CO2_total_kg, color=red, linestyle=':', label='Escape' )
        h3, = ax1.semilogx( timeMyr_a, (H2O_liquid_kg_a+H2O_solid_kg_a) / H2O_total_kg, color=blue, linestyle='-', label=r'H$_2$O interior' )
        h4, = ax1.semilogx( timeMyr_a, H2O_atmos_kg_a / H2O_total_kg, color=blue, linestyle='--', label=r'H$_2$O atmos')
        #h4b, = ax1.semilogx( timeMyr_a, H2O_escape_kg_a / H2O_total_kg, color=blue, linestyle=':', label='Atmos' )
        fig_o.set_myaxes( ax1, title=title, ylabel='$x$', xlabel=xlabel,xticks=xticks )
        ax1.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
        ax1.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
        ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax1.set_xlim( *xlim )
        handle_l = [h1,h2,h3,h4]#,h2b]
        fig_o.set_mylegend( ax1, handle_l, loc='center left', ncol=1 )
        ax1.yaxis.set_label_coords(-0.1,0.47)


    ##########
    # figure c
    ##########
    if 1:
        title = r'\textbf{(c) Surface temp and emissivity}'
        ylabel = '$T_s$ (K)'
        yticks = range(200,2701,500)
        h1, = ax2.semilogx( timeMyr_a, temperature_surface_a, 'k-', label=r'Surface temp, $T_s$' )
        ax2b = ax2.twinx()
        h2, = ax2b.loglog( timeMyr_a, emissivity_a, 'k--', label=r'Emissivity, $\epsilon$' )
        fig_o.set_myaxes( ax2, title=title, xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks )
        ax2.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
        ax2.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
        ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax2.set_xlim( *xlim )
        ax2.yaxis.set_label_coords(-0.175,0.48)
        ax2.set_ylim(200,2700)
        ax2b.set_ylim( 4E-4, 2E-3 )
        handle_l = [h1,h2]
        fig_o.set_mylegend( ax2, handle_l, loc='upper right', ncol=1 )
        ax2b.set_ylabel( r'$\epsilon$', rotation=0)
        ax2b.yaxis.set_label_coords(1.1,0.52)

    ##########
    # figure d
    ##########
    if 0:
        title = '(d) Emissivity'
        ylabel = '$\epsilon$'
        ax3.loglog( timeMyr_a, emissivity_a, 'k-' )
        fig_o.set_myaxes( ax3, title=title, xlabel=xlabel, ylabel=ylabel, xticks=xticks )
        ax3.yaxis.set_label_coords(-0.1,0.55)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax3.set_ylim( 1E-4, 1E-2 )
        ax3.set_xlim( 1E-5, 1 )

    fig_o.savefig(7)

#====================================================================
def plot_atmosphere_right_wrong():

    '''plot atmosphere properties for a single case, which should be
       specified by casenum'''

    width = 4.7747 * 3.0 / 2.0
    height = 4.7747 / 2.0
    fig_o = su.FigureData( 1, 3, width, height, 'atmosphere_right_wrong', units='Myr' )
    fig_o.fig.subplots_adjust(wspace=0.4,hspace=0.3)

    ax0 = fig_o.ax[0]#[0]
    ax1 = fig_o.ax[1]#[1]
    ax2 = fig_o.ax[2]#[0]
    #ax3 = fig_o.ax[1][1]

    with open('data.json', 'r') as fp:
        data_d = json.load(fp )

    strcase = 'caseN'
    case_d = data_d[strcase]
    rCO2_atmos_a = np.array(case_d['CO2_atmosphere_bar'])
    rCO2_mixing_ratio_a = np.array(case_d['CO2_mixing_ratio'])
    rH2O_atmos_a = np.array(case_d['H2O_atmosphere_bar'])
    rH2O_mixing_ratio_a = np.array(case_d['H2O_mixing_ratio'])
    rtime = np.array(case_d['time'])
    rtimeMyr_a = rtime* 1.0E-6

    strcase = 'caseNw'
    case_d = data_d[strcase]
    wCO2_atmos_a = np.array(case_d['CO2_atmosphere_bar'])
    wCO2_mixing_ratio_a = np.array(case_d['CO2_mixing_ratio'])
    wH2O_atmos_a = np.array(case_d['H2O_atmosphere_bar'])
    wH2O_mixing_ratio_a = np.array(case_d['H2O_mixing_ratio'])
    wtime = np.array(case_d['time'])
    wtimeMyr_a = wtime * 1.0E-6

    xlabel = 'Time (Myr)'
    xlim = [1.0E-3, 1.0E1]

    red = (0.5,0.1,0.1)
    blue = (0.1,0.1,0.5)
    black = 'black'

    ##########
    # figure a
    ##########
    if 1:
        title = r'\textbf{(a) Partial pressure}'
        ylabel = r'$p$ (bar)'
        #trans = transforms.blended_transform_factory(
        #    ax0.transData, ax0.transAxes)
        h1, = ax0.semilogx( wtimeMyr_a, wCO2_atmos_a, color=red, linestyle='--', label=r'CO$_2$ \textbf{incorrect}')
        h2, = ax0.semilogx( wtimeMyr_a, wH2O_atmos_a, color=blue, linestyle='--', label=r'H$_2$O \textbf{incorrect}')
        h3, = ax0.semilogx( rtimeMyr_a, rCO2_atmos_a, color=red, linestyle='-', label=r'CO$_2$')
        h4, = ax0.semilogx( rtimeMyr_a, rH2O_atmos_a, color=blue, linestyle='-', label=r'H$_2$O')
        ax0.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
        ax0.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
        ax0.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax0.set_xlim( *xlim )
        ax0.set_ylim( 0, 350 )
        handle_l = (h3,h4,h1,h2)
        fig_o.set_myaxes( ax0, title=title, ylabel=ylabel, xlabel=xlabel)
        fig_o.set_mylegend( ax0, handle_l, loc='upper left', ncol=1 )
        ax0.yaxis.set_label_coords(-0.15,0.47)

    ##########
    # figure b
    ##########
    if 1:
        title = r'\textbf{(b) Mixing ratio}'
        ylabel = r'Mixing ratio'
        h5, = ax1.loglog( wtimeMyr_a, wCO2_mixing_ratio_a, color=red, linestyle='--', label=r'CO$_2$ \textbf{incorrect}')
        h6, = ax1.loglog( wtimeMyr_a, wH2O_mixing_ratio_a, color=blue, linestyle='--', label=r'H$_2$O \textbf{incorrect}')
        h7, = ax1.loglog( rtimeMyr_a, rCO2_mixing_ratio_a, color=red, linestyle='-', label=r'CO$_2$')
        h8, = ax1.loglog( rtimeMyr_a, rH2O_mixing_ratio_a, color=blue, linestyle='-', label=r'H$_2$O')
        ax1.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
        ax1.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
        ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax1.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
        ax1.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
        ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax1.set_xlim( *xlim )
        ax1.set_ylim( 1.0E-2,1.0E0 )
        handle_l = (h7,h8,h5,h6)
        fig_o.set_myaxes( ax1, title=title, ylabel=ylabel, xlabel=xlabel, yrotation=90)
        fig_o.set_mylegend( ax1, handle_l, loc='lower right', ncol=1 )
        ax1.yaxis.set_label_coords(-0.25,0.5)

    ##########
    # figure c
    ##########
    if 1:
        title = r'\textbf{(c) Difference (\%)}'
        CO2_pdiff_a = (wCO2_atmos_a-rCO2_atmos_a) / rCO2_atmos_a * 100.0
        H2O_pdiff_a = (wH2O_atmos_a-rH2O_atmos_a) / rH2O_atmos_a * 100.0
        CO2_mdiff_a = (wCO2_mixing_ratio_a-rCO2_mixing_ratio_a) / rCO2_mixing_ratio_a * 100.0
        H2O_mdiff_a = (wH2O_mixing_ratio_a-rH2O_mixing_ratio_a) / rH2O_mixing_ratio_a * 100.0

        ylabel = r'Difference (\%)'
        h9, = ax2.semilogx( wtimeMyr_a, CO2_pdiff_a, color=red, linestyle='-', label=r'CO$_2$ partial' )
        h10, = ax2.semilogx( wtimeMyr_a, H2O_pdiff_a, color=blue, linestyle='-', label=r'H$_2$O partial' )
        h11, = ax2.semilogx( wtimeMyr_a, CO2_mdiff_a, color=red, linestyle='--', label=r'CO$_2$ mixing' )
        h12, = ax2.semilogx( wtimeMyr_a, H2O_mdiff_a, color=blue, linestyle='--', label=r'H$_2$O mixing' )

        ax2.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
        ax2.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))

        ax2.set_xlim( *xlim )
        handle_l = (h9,h10,h11,h12)
        fig_o.set_myaxes( ax2, title=title, ylabel=ylabel, xlabel=xlabel, yrotation=90)
        fig_o.set_mylegend( ax2, handle_l, loc='upper left', ncol=1 )
        ax2.yaxis.set_label_coords(-0.2,0.5)
        ax2.set_ylim(-30,120)

    fig_o.savefig(8)

#====================================================================
def plot_interior_atmosphere():

    figw = 4.7747 # / 2.0
    figh = 4.7747

    # dummy times
    times = '0,0,0,0,0'
    fig_o = su.FigureData( 2, 2, figw, figh, 'case1_case9_static_structure', times, units='Myr' )
    plt.subplots_adjust(hspace=0.0,wspace=0.175)

    ax0 = fig_o.ax[0][0]
    ax1 = fig_o.ax[1][0]
    ax2 = fig_o.ax[0][1]
    ax3 = fig_o.ax[1][1]

    # Bower et al. (2019), Case 1
    #time_l = [0,94400,250000,560000000,4550000000]
    # Case 1m
    # 1.0,0.75,0.3,0.1,0.01
    time_l = [0,94300,250000,500000,2250000]
    temp_l = [2707.070630041155,1966.2749990878704,1728.0118330935393,1167.3880783423733,910.7342822767405]
    case_dir = 'case1m_63_234'
    casenum = '1m'
    plot_interior_atmosphere_subfigure( casenum, fig_o, ax0, ax1, time_l, temp_l, case_dir )
    # Bower et al. (2019), Case 9
    #time_l = [0,3850000,10950000,802000000,4550000000]
    # Case 9m
    # 1.0, 0.75, 0.3, 0.1, 0.01
    time_l = [0,3850000,11150000,16300000,21250000]
    temp_l = [2707.9680676566704,1965.7587030853172,1704.0979064022688,1549.8952368681953,1452.3410862908963]
    case_dir = 'case9m_2558_116'
    casenum = '9m'
    plot_interior_atmosphere_subfigure( casenum, fig_o, ax2, ax3, time_l, temp_l, case_dir )

    fig_o.savefig(9)

#====================================================================
def plot_interior_atmosphere_subfigure( casenum, fig_o, ax0, ax1, time_l, temp_l, case_dir ):

    time = time_l[0]
    strcase = 'Case ' + str(casenum)
    print(strcase)

    filename = '{}/output/{}.json'.format( case_dir, time )
    myjson_o = su.MyJSON( filename )

    xx_pres = myjson_o.get_dict_values(['data','pressure_b'])
    xx_pres *= 1.0E-9
    xx_pres_s = myjson_o.get_dict_values(['data','pressure_s'])
    xx_pres_s *= 1.0E-9
    xx_radius = myjson_o.get_dict_values_internal(['data','radius_b'])
    xx_radius *= 1.0E-3
    xx_depth = xx_radius[0] - xx_radius
    xx_radius_s = myjson_o.get_dict_values(['data','radius_s'])
    xx_radius_s *= 1.0E-3
    xx_depth_s = xx_radius_s[0] - xx_radius_s

    handle_l = [] # handles for legend

    phi_l = ('1.00','0.75','0.30','0.10','0.01')#,0)

    for nn, time in enumerate( time_l ):
        # read json
        filename = '{}/output/{}.json'.format( case_dir, time )
        myjson_o = su.MyJSON( filename )

        color = fig_o.get_color( nn )
        # atmosphere structure
        hatm_interp1d = myjson_o.get_atm_struct_depth_interp1d()
        height_point = hatm_interp1d( 10.0E-3 ) / 1.0E3 # 10 mb
        tatm_interp1d = myjson_o.get_atm_struct_temp_interp1d()
        temp_point = tatm_interp1d( 10.0E-3 ) # 10 mb
        ax0.plot( temp_point, height_point, color=color, marker='_', markersize=8 )

        atmos_pres_a = myjson_o.get_dict_values( ['atmosphere','atm_struct_pressure'] )
        atmos_temp_a = myjson_o.get_dict_values( ['atmosphere','atm_struct_temp'] )
        atmos_height_a = myjson_o.get_dict_values( ['atmosphere','atm_struct_depth'] )
        atmos_height_a *= 1.0E-3
        indices = atmos_height_a < height_point
        atmos_height_a = atmos_height_a[indices]
        atmos_temp_a = atmos_temp_a[indices]
        ax0.plot( atmos_temp_a, atmos_height_a, '-', color=color )

        # get H2O surface partial pressure to check against vapour pressure curve
        H2O_partial_p = myjson_o.get_dict_values( ['atmosphere','H2O','atmosphere_bar'] )

        # use melt fraction to determine mixed region
        rho1D_o = myjson_o.get_rho_interp1d()
        temp1D_o = myjson_o.get_temp_interp1d()
        radius = su.solve_for_planetary_radius( rho1D_o )
        myargs = su.get_myargs_static_structure( rho1D_o )
        z = su.get_static_structure_for_radius( radius, *myargs )
        radius_a = su.get_radius_array_static_structure( radius, *myargs )
        pressure_a = z[:,0]
        temp_a = temp1D_o( pressure_a )
        #label = fig_o.get_legend_label( time )
        depth_a = radius_a[0] - radius_a
        depth_a *= 1.0e-3 # to km
        if nn==0:
            # atmosphere percent change
            height0 = atmos_height_a[0]
            heightn25 = height0 * 0.75
            heightn50 = height0 * 0.50
            heightn75 = height0 * 0.25
            color = 'black'
            ax0.axhline( height0, xmin=0.125, xmax=0.22, color=color, linestyle=':' )
            ax0.text( 1250, height0, '0\%', va='center', ha='left', bbox=dict(facecolor='white',pad=0,edgecolor='none') )
            ax0.axhline( heightn25, xmin=0.18, xmax=0.255, color=color, linestyle=':' )
            ax0.text( 1550, heightn25, '-25\%', va='center', ha='left', bbox=dict(facecolor='white',pad=0,edgecolor='none') )
            ax0.axhline( heightn50, xmin=0.305, xmax=0.38, color=color, linestyle=':' )
            ax0.text( 2150, heightn50, '-50\%', va='center', ha='left', bbox=dict(facecolor='white',pad=0,edgecolor='none') )
            #ax0.axhline( heightn75, xmin=0.43, xmax=0.505, color=color, linestyle=':' )
            #ax0.text( 2750, heightn75, '-75\%', va='center', ha='left', bbox=dict(facecolor='white',pad=0,edgecolor='none') )
            # mantle percent change
            depth0 = depth_a[-1]
            depthn5 = depth0*0.95 # 5% less
            depthn10 = depth0*0.9 # 10% less
            ax1.axhline( depthn10, xmin=0.4, xmax=0.725, color='black', linestyle=':' )
            ax1.text( 2000, depthn10, '-10\%', va='bottom', ha='right', bbox=dict(facecolor='white',pad=0,edgecolor='none') )
            ax1.axhline( depthn5, xmin=0.4, xmax=0.775, color='black', linestyle=':' )
            ax1.text( 2000, depthn5, '-5\%', va='center', ha='right', bbox=dict(facecolor='white',pad=0,edgecolor='none') )
            ax1.axhline( depth0, xmin=0.4, xmax=0.825, color='black', linestyle=':' )
            ax1.text( 2000, depth0, '0\%', va='top', ha='right', bbox=dict(facecolor='white',pad=0,edgecolor='none') )
        color = fig_o.get_color( nn )
        phi_label = phi_l[nn]
        temp_label = str(np.round(temp_l[nn],))[:-2]
        label = str(phi_label) + ' (' + str(temp_label) + ')'
        handle, = ax1.plot( temp_a, depth_a, color=color, label=label )
        ax1.plot( temp_a[-1], depth_a[-1], color=color, marker='_', markersize=8 )

        # below for debugging
        #print( 'time=', time, 'atmos_temp_a[-1]=', atmos_temp_a[-1], 'temp_a[0]=', temp_a[0], 'H2O_partial_p=', H2O_partial_p )
        # issue warning if above vapour pressure curve
        if( atmos_temp_a[-1] < 647.096 and H2O_partial_p > 220.64 ):
            print( 'WARNING: above vapour pressure curve' )

        #ax1.plot( yy, xx_pres, '--', color=color )
        #handle, = ax1.plot( yy*MIX, xx_pres*MIX, '-', color=color, label=label )
        handle_l.append( handle )

    yticks = [0,500,1000,1500,2000,2500,3000,3500]
    ymin = 0.0
    ymax = 3500.0
    xticks = [200,1000,2000,3000,4000,5000]

    if casenum=='1m':
        title = r'\textbf{(a) Case 1c }' + r'($\bf{r_{f,CO_2}=0.1}$)'
    elif casenum=='9m':
        title = r'\textbf{(b) Case 9c }' + r'($\bf{r_{f,CO_2}=0.9}$)'
    ylabel = 'Atmosphere height (km)'
    if casenum=='1m':
        fig_o.set_myaxes( ax0, ylabel=ylabel, title=title, yrotation=90 )
    elif casenum=='9m':
        fig_o.set_myaxes( ax0, title=title )
        ax0.yaxis.set_ticklabels([])
    ax0.set_yticks([25,75,125,175,225], minor=True )

    # titles and axes labels, legends, etc
    units = myjson_o.get_dict_units(['data','temp_b'])

    ylabel= 'Mantle depth (km)'
    if casenum=='1m':
        fig_o.set_myaxes( ax1, ylabel=ylabel, xlabel='$T$ (K)', xticks=xticks, ymin=ymin, ymax=ymax, yticks=yticks, yrotation=90 )
        ax1.yaxis.set_label_coords(-0.25,0.5)
    elif casenum=='9m':
        fig_o.set_myaxes( ax1, xlabel='$T$ (K)', xticks=xticks, ymin=ymin, ymax=ymax, yticks=yticks )
        ax1.yaxis.set_ticklabels([])
    ax1.set_ylim( ymin, ymax )
    ax1.invert_yaxis()
    ax1.set_xticks([600,1500, 2500, 3500, 4500], minor=True)
    ax1.set_yticks([250,750,1250,1750,2250,2750,3250], minor=True)

    ax0.axes.xaxis.set_visible(False)
    ax0.set_xlim( xticks[0], xticks[-1])
    ax0.set_ylim( 0.0, 250.0 )

    #if casenum ==1:
    #TITLE = "Time, Myr ($\phi_g$)"
    TITLE = r'$\phi_g$ ($T_s$)'
    fig_o.set_mylegend( ax0, handle_l, TITLE=TITLE, loc='upper right' )

#====================================================================
def plot_phi_versus_radius():

    figw = 4.7747 / 2.0 # * 3.0 # * 1.1
    figh = 4.7747 / 2.0 # * 1.1

    Rupper = 6900.0E3
    Rlower = 6200.0E3
    ylim_km = (Rlower*1.0E-3,Rupper*1.0E-3)
    diffmajyticks = [-6,-3,0,3]

    ind1 = 0 # 7 # 13 # last time with global melt fraction of 1.0
    ind9 = 0 # 527 # last time with global melt fraction of 1.0

    # dummy for colors
    times = '0,0,0'

    fig_o = su.FigureData( 1, 1, figw, figh, 'radius_evolution_phi', times )
    fig_o.fig.subplots_adjust(wspace=0.7)

    ax = fig_o.ax

    color0 = fig_o.get_color(0)
    color1 = fig_o.get_color(-1)

    ax0 = ax#[0]
    #ax1 = ax[1]
    #ax2 = ax[2]

    case1_filename = 'case1m_63_234/case1m_evolving_radius.dat'
    case1_time, case1_rad, case1_10mb, case1_1mb, case1_temp, case1_phi = np.loadtxt( case1_filename, unpack=True )
    case1_time *= 1.0E-6 # to Myr
    case1_ref = case1_rad[ind1]
    #rcase1_rad = (case1_rad-case1_ref)*100.0/case1_ref
    #rcase1_10mb = (case1_10mb-case1_ref)*100.0/case1_ref
    case1_rad *= 1.0E-3
    case1_10mb *= 1.0E-3

    print( 'case1_ref=', case1_ref )
    case1_upper = (Rupper-case1_ref)/case1_ref*100.0
    print( 'case1_upper=', case1_upper )
    case1_lower = (Rlower-case1_ref)/case1_ref*100.0
    print( 'case1_lower=', case1_lower )

    case9_filename = 'case9m_2558_116/case9m_evolving_radius.dat'
    case9_time, case9_rad, case9_10mb, case9_1mb, case9_temp, case9_phi = np.loadtxt( case9_filename, unpack=True )
    case9_time *= 1.0E-6 # to Myr
    case9_ref = case9_rad[ind9]
    #rcase9_rad = (case9_rad[case9startind:]-case9_ref)*100.0/case9_ref
    #rcase9_10mb = (case9_10mb[case9startind:]-case9_ref)*100.0/case9_ref
    case9_rad *= 1.0E-3
    case9_10mb *= 1.0E-3

    case9_upper = (Rupper-case9_ref)/case9_ref*100.0
    case9_lower = (Rlower-case9_ref)/case9_ref*100.0
    print( 'case9_upper=', case9_upper )
    print( 'case9_lower=', case9_lower )

    handles_l = []

    xlabel = '$T_s$ (K)'
    xlabel2 = '$\phi_g (\%)$'
    xlim = [700,2707]
    xlim2 = [0,100]

    #==========
    # figure a
    #==========
    h1, = ax0.plot( case1_temp, case1_rad, label='Case 1c Surface', linestyle='--', color=color0 )
    handles_l.append( h1 )
    h2, = ax0.plot( case1_temp, case1_10mb, label='Case 1c Total', linestyle='-', color=color0 )
    handles_l.append( h2 )
    h3, = ax0.plot( case9_temp, case9_rad, label='Case 9c Surface', linestyle='--', color=color1 )
    handles_l.append( h3 )
    h4, = ax0.plot( case9_temp, case9_10mb, label='Case 9c Total', linestyle='-', color=color1 )
    handles_l.append( h4 )

    ax0b = ax0.twinx()
    title = r'\textbf{Radius \& surface temperature $T_s$}'
    ylabel = 'Radius (km)'
    ylabel2 = r'Difference to molten mantle (\%)'

    fig_o.set_myaxes( ax0, title=title, ylabel=ylabel, xlabel=xlabel, yrotation=90 )
    ax0.set_xticks( [1000,1500,2000,2500] )
    ax0.set_xticks( [750,1250,1750,2250], minor=True )
    ax0.set_ylim( *ylim_km )

    #ax0b.set_yticks(diffmajyticks)
    #ax0b.set_ylim( case1_lower, case1_upper )
    #ax0b.set_ylim( case9_lower,case9_upper)
    ax0.set_xlim( *xlim )
    ax0b.set_ylim( case1_lower, case1_upper )
    ax0b.set_ylabel( ylabel2 )
    #fig_o.set_mylegend( ax0, handles_l, loc='lower right', ncol=1, facecolor='white', framealpha=1 )#, TITLE=TITLE )

    #==========
    # figure b
    #==========
    if 0:

        handles2_l = []

        h1, = ax1.plot( case1_phi*100, case1_rad, label='Case 1c Surface', color=color0, linestyle='--' )
        handles2_l.append( h1 )
        h2, = ax1.plot( case1_phi*100, case1_10mb, label='Case 1c Total', color=color0, linestyle='-' )
        handles2_l.append( h2 )
        h3, = ax1.plot( case9_phi*100, case9_rad, label='Case 9c Surface', color=color1, linestyle='--' )
        handles2_l.append( h3 )
        h4, = ax1.plot( case9_phi*100, case9_10mb, label='Case 9c Total', color=color1, linestyle='-' )
        handles2_l.append( h4 )
        ax1b = ax1.twinx()

        title = r'\textbf{(b) Radius \& melt fraction $\phi_g$}'
        fig_o.set_myaxes( ax1, title=title, ylabel=ylabel, xlabel=xlabel2, yrotation=90 )
        ax1b.set_yticks(diffmajyticks )
        ax1b.set_ylim( case9_lower, case9_upper )
        ax1b.set_ylabel( ylabel2 )
        ax1.set_xticks( [0,20,40,60,80,100])
        ax1.set_xticks( [10,30,50,70,90], minor=True )

        trans = transforms.blended_transform_factory( ax1.transData, ax1.transAxes)

        ax1.set_xlim( *xlim2 )
        ax1.set_ylim( *ylim_km )
        ax1.yaxis.set_label_coords(-0.275,0.5)
        #ax1b.yaxis.set_label_coords( -0.4, 0.5 )
        fig_o.set_mylegend( ax1, handles2_l, loc='upper left', ncol=1, facecolor='white',framealpha=1 )
        #ax1.yaxis.set_ticklabels([])

    # figure c
    #with open('data.json', 'r') as fp:
    #    data_d = json.load(fp )

    #mu_CO2 = 44.01
    #mu_H2O = 18.01528

    #handles3_l = []

    #time = data_d['case1m']['time']
    #time_myrs = np.array(time) * 1.0E-6
    #CO2_mixing_ratio = np.array(data_d['case1m']['CO2_mixing_ratio'])
    #H2O_mixing_ratio = np.array(data_d['case1m']['H2O_mixing_ratio'])
    #mean_molar = CO2_mixing_ratio*mu_CO2 + H2O_mixing_ratio*mu_H2O
    #label = '1c (0.1)'
    #h1, = ax2.semilogx( time_myrs, mean_molar, color=color0, label=label )
    #handles3_l.append( h1 )

    #time = data_d['case9m']['time']
    #time_myrs = np.array(time) * 1.0E-6
    #CO2_mixing_ratio = np.array(data_d['case9m']['CO2_mixing_ratio'])
    #H2O_mixing_ratio = np.array(data_d['case9m']['H2O_mixing_ratio'])
    #mean_molar = CO2_mixing_ratio*mu_CO2 + H2O_mixing_ratio*mu_H2O
    #label = '9c (0.9)'
    #h2, = ax2.semilogx( time_myrs, mean_molar, color=color1, label=label )
    #handles3_l.append( h2 )

    #ax2.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    #ax2.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    #ax2.set_xlim( *xlim )
    #ax2.yaxis.set_label_coords(-0.2,0.5)
    #TITLE = 'Case ($r_{f,CO_2}$)'
    #fig_o.set_mylegend( ax2, handles3_l, loc='lower left', TITLE=TITLE, ncol=1 )

    #midline = 44.01*0.5+18.01528*0.5
    #ax2.axhline( midline, xmin=0.05, xmax=0.95, color='black', linestyle=':' )
    #ax2.text( 1.01E0, midline+2, 'CO$_2$ dominant', va='bottom', rotation=30 )
    #ax2.text( 1.01E0, midline-2, 'H$_2$O dominant', va='top', rotation=-30 )

    #title = r'\textbf{(c) Atmosphere molar mass}'
    #ylabel = 'Molar mass (g/mol)'
    #fig_o.set_myaxes( ax2, title=title, ylabel=ylabel, xlabel=xlabel, yrotation=90 )

    fig_o.savefig(18)

#====================================================================
def plot_radius_evolution():

    figw = 4.7747 / 2.0 * 3.0 * 1.1
    figh = 4.7747 / 2.0 * 1.1

    Rupper = 6900.0E3
    Rlower = 6200.0E3
    diffmajyticks = [-6,-3,0,3]

    # dummy for colors
    times = '0,0,0'

    fig_o = su.FigureData( 1, 3, figw, figh, 'radius_evolution', times )
    fig_o.fig.subplots_adjust(wspace=0.65)

    ax = fig_o.ax

    color0 = fig_o.get_color(0)
    color1 = fig_o.get_color(-1)

    ax0 = ax[0]
    ax1 = ax[1]
    ax2 = ax[2]

    case1_filename = 'case1m_63_234/case1m_evolving_radius.dat'
    case1_time, case1_rad, case1_10mb, case1_1mb, case1_temp, case1_phi = np.loadtxt( case1_filename, unpack=True )
    case1_time *= 1.0E-6 # to Myr
    case1_ref = case1_rad[0]
    rcase1_rad = (case1_rad-case1_ref)*100.0/case1_ref
    rcase1_10mb = (case1_10mb-case1_ref)*100.0/case1_ref
    case1_rad *= 1.0E-3
    case1_10mb *= 1.0E-3

    case1_upper = (Rupper-case1_ref)/case1_ref*100.0
    case1_lower = (Rlower-case1_ref)/case1_ref*100.0

    case9_filename = 'case9m_2558_116/case9m_evolving_radius.dat'
    case9_time, case9_rad, case9_10mb, case9_1mb, case9_temp, case9_phi = np.loadtxt( case9_filename, unpack=True )
    case9_time *= 1.0E-6 # to Myr
    case9_ref = case9_rad[0]
    rcase9_rad = (case9_rad-case9_ref)*100.0/case9_ref
    rcase9_10mb = (case9_10mb-case9_ref)*100.0/case9_ref
    case9_rad *= 1.0E-3
    case9_10mb *= 1.0E-3

    case9_upper = (Rupper-case9_ref)/case9_ref*100.0
    case9_lower = (Rlower-case9_ref)/case9_ref*100.0

    handles_l = []

    xlabel = 'Time (Myr)'
    xlim = (1.0E-2, 1.0E2)
    ylim_km = (6300, 6900)

    # figure a
    h1, = ax0.semilogx( case1_time, case1_rad, label='Surface', linestyle='--', color=color0 )
    handles_l.append( h1 )
    h2, = ax0.semilogx( case1_time, case1_10mb, label='Total', linestyle='-', color=color0 )
    handles_l.append( h2 )
    ax0b = ax0.twinx()
    #title = r'\textbf{(a) Case 1, Radius}'
    title = r'\textbf{(a) Case 1c }' + r'($\bf{r_{f,CO_2}=0.1}$)'
    ylabel = 'Radius (km)'
    ylabel2 = r'Difference to molten mantle (\%)'

    fig_o.set_myaxes( ax0, title=title, ylabel=ylabel, xlabel=xlabel, yrotation = 90 )
    ax0.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax0.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax0.set_ylim( *ylim_km )
    ax0b.set_yticks(diffmajyticks)
    ax0b.set_ylim( case1_lower, case1_upper )

    phi_cont_l = [0.75,0.30,0.10,0.01]
    phi_cont_label = ['0.75','0.30','0.10','0.01']
    phi_time_l = [] # contains the times for each contour
    for cont in phi_cont_l:
        time_temp_l = su.find_xx_for_yy( case1_time, case1_phi, cont )
        index = su.get_first_non_zero_index( time_temp_l )
        if index is None:
            out = 0.0
        else:
            out = case1_time[index]
        phi_time_l.append( out )

    print( 'case1 phi times=', phi_time_l )

    trans = transforms.blended_transform_factory( ax0.transData, ax0.transAxes)

    for cc, cont in enumerate(phi_cont_l):
        ax0.axvline( phi_time_l[cc], ymin=0.02, ymax=0.95, color='0.25', linestyle=':' )
        label = phi_cont_label[cc] #int(cont*100) # as percent
        #label = str(round(label,2))
        if cc == 3:
            label= r'$\phi_g=$ ' + str(label)
        ha = 'center'
        ax0.text( phi_time_l[cc], 0.8, label, va='top', ha=ha, rotation=90, bbox=dict(facecolor='white', edgecolor='none', pad=2), transform=trans )

    #ax0b.set_yticks([-4.5,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5], minor=True )
    ax0.set_xlim( 1.0E-2,1.0E1 )
    ax0b.set_ylabel( ylabel2 )
    #ax0.yaxis.set_label_coords(-0.2,0.5)
    #ax0b.yaxis.set_label_coords( -0.2, 0.5 )
    fig_o.set_mylegend( ax0, handles_l, loc='upper right', ncol=1, facecolor='white',framealpha=1 )#, TITLE=TITLE )

    #frame = legend.get_frame()
    #frame.set_facecolor('white')

    # figure b
    handles2_l = []

    h3, = ax1.semilogx( case9_time, case9_rad, label='Surface', color=color1, linestyle='--' )
    handles2_l.append( h3 )
    h4, = ax1.semilogx( case9_time, case9_10mb, label='Total', color=color1, linestyle='-' )
    handles2_l.append( h4 )
    ax1b = ax1.twinx()

    #title = r'\textbf{(b) Case 9, Radius}'
    title = r'\textbf{(a) Case 9c }' + r'($\bf{r_{f,CO_2}=0.9}$)'
    fig_o.set_myaxes( ax1, title=title, ylabel=ylabel, xlabel=xlabel, yrotation=90 )
    ax1b.set_yticks(diffmajyticks )
    ax1b.set_ylim( case9_lower, case9_upper )
    ax1b.set_ylabel( ylabel2 )
    ax1.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax1.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))

    phi_cont_l = [0.75,0.30,0.10,0.01]
    phi_cont_label = ['0.75','0.30','0.10','0.01']
    phi_time_l = [] # contains the times for each contour
    for cont in phi_cont_l:
        time_temp_l = su.find_xx_for_yy( case9_time, case9_phi, cont )
        index = su.get_first_non_zero_index( time_temp_l )
        if index is None:
            out = 0.0
        else:
            out = case1_time[index]
        phi_time_l.append( out )

    print( 'case9 phi times=', phi_time_l )

    trans = transforms.blended_transform_factory( ax1.transData, ax1.transAxes)

    for cc, cont in enumerate(phi_cont_l):
        ax1.axvline( phi_time_l[cc], ymin=0.02, ymax=0.95, color='0.25', linestyle=':' )
        label = phi_cont_label[cc]
        if cc == 3:
            label= r'$\phi_g=$ ' + str(label)
        ha = 'center'
        if cc==3:
            ha = 'left'
        ax1.text( phi_time_l[cc], 0.8, label, va='top', ha=ha, rotation=90, bbox=dict(facecolor='white', edgecolor='none', pad=2), transform=trans )

    ax1.set_xlim( 1E0,1E2 )
    ax1.set_ylim( *ylim_km )
    ax1.yaxis.set_label_coords(-0.265,0.5)
    #ax1b.yaxis.set_label_coords( -0.4, 0.5 )
    fig_o.set_mylegend( ax1, handles2_l, loc='upper right', ncol=1, facecolor='white',framealpha=1 )
    #ax1.yaxis.set_ticklabels([])

    # figure c
    with open('data.json', 'r') as fp:
        data_d = json.load(fp )

    mu_CO2 = 44.01
    mu_H2O = 18.01528

    handles3_l = []

    time = data_d['case1m']['time']
    time_myrs = np.array(time) * 1.0E-6
    CO2_mixing_ratio = np.array(data_d['case1m']['CO2_mixing_ratio'])
    H2O_mixing_ratio = np.array(data_d['case1m']['H2O_mixing_ratio'])
    mean_molar = CO2_mixing_ratio*mu_CO2 + H2O_mixing_ratio*mu_H2O
    label = '1c (0.1)'
    h1, = ax2.semilogx( time_myrs, mean_molar, color=color0, label=label )
    handles3_l.append( h1 )

    time = data_d['case9m']['time']
    time_myrs = np.array(time) * 1.0E-6
    CO2_mixing_ratio = np.array(data_d['case9m']['CO2_mixing_ratio'])
    H2O_mixing_ratio = np.array(data_d['case9m']['H2O_mixing_ratio'])
    mean_molar = CO2_mixing_ratio*mu_CO2 + H2O_mixing_ratio*mu_H2O
    label = '9c (0.9)'
    h2, = ax2.semilogx( time_myrs, mean_molar, color=color1, label=label )
    handles3_l.append( h2 )

    ax2.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax2.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax2.set_xlim( *xlim )
    ax2.yaxis.set_label_coords(-0.2,0.5)
    TITLE = 'Case ($r_{f,CO_2}$)'
    fig_o.set_mylegend( ax2, handles3_l, loc='lower left', TITLE=TITLE, ncol=1 )

    midline = 44.01*0.5+18.01528*0.5
    ax2.axhline( midline, xmin=0.05, xmax=0.95, color='black', linestyle=':' )
    ax2.text( 1.01E0, midline+2, 'CO$_2$ dominant', va='bottom', rotation=30 )
    ax2.text( 1.01E0, midline-2, 'H$_2$O dominant', va='top', rotation=-30 )

    title = r'\textbf{(c) Atmosphere molar mass}'
    ylabel = 'Molar mass (g/mol)'
    fig_o.set_myaxes( ax2, title=title, ylabel=ylabel, xlabel=xlabel, yrotation=90 )


    fig_o.savefig(10)

#====================================================================
def plot_emission_spectra():

    emission_dir = '/Volumes/data/spectra'

    figw = 4.7747
    figh = 4.7747

    # dummy for colors
    times = '0,0,0,0,0,0'

    fig_o = su.FigureData( 2, 2, figw, figh, 'emission', times )
    fig_o.fig.subplots_adjust(wspace=0.5,hspace=0.4)

    ax = fig_o.ax

    ax0 = ax[0][0]
    ax1 = ax[0][1]
    ax2 = ax[1][0]
    ax3 = ax[1][1]

    plot_emission_for_case( '1m', 'g', fig_o, ax0 )
    plot_emission_for_case( '9m', 'g', fig_o, ax1 )
    plot_emission_for_case( '1m', 'm', fig_o, ax2 )
    plot_emission_for_case( '9m', 'm', fig_o, ax3 )

    fig_o.savefig(11)

#====================================================================
def plot_emission_for_case( casenum, star, fig_o, ax ):

    emission_dir = '/Volumes/data/spectra_c'

    xlim = (0.7,30)
    ylim = (1.0E-10, 1.0E5)

    ustar = star.upper()

    # case 1
    if casenum == 1:
        times_l = (0,94400,250000,78400000,560000000)#,4550000000)
        phi_l = (1,0.75,0.3,0.1,0.01)#,0)
        if star=='g':
            prefix = '(a) '
        elif star=='m':
            prefix = '(c) '
        title = prefix + 'Case 1 (' + ustar + '-star)'
    # case 9
    elif casenum == 9:
        times_l = (0,3850000,10950000,141000000,802000000)#,4550000000)
        phi_l = (1,0.75,0.3,0.1,0.01)#,0)
        if star=='g':
            prefix = '(b) '
        elif star=='m':
            prefix = '(d) '
        title = prefix + 'Case 9 (' + ustar + '-star)'
    # case 1m
    elif casenum == '1m':
        times_l = (0,94300,250000,500000,2250000)#,4550000000)
        phi_l = (1,0.75,0.3,0.1,0.01)#,0)
        temp_l = [2707.070630041155,1966.2749990878704,1728.0118330935393,1167.3880783423733,910.7342822767405]
        if star=='g':
            prefix = '(a) '
        elif star=='m':
            prefix = '(a) '
        title = prefix + 'Case 1c ($r_{f,CO_2}=0.1$)'# (' + ustar + '-star)'
    # case 9m
    elif casenum == '9m':
        times_l = (0,3850000,11150000,16300000,21250000)#,4550000000)
        phi_l = (1,0.75,0.3,0.1,0.01)#,0)
        temp_l = [2707.9680676566704,1965.7587030853172,1704.0979064022688,1549.8952368681953,1452.3410862908963]
        if star=='g':
            prefix = '(b) '
        elif star=='m':
            prefix = '(b) '
        title = prefix + 'Case 9c ($r_{f,CO_2}=0.9$)'# (' + ustar + '-star)'

    handle_l = []

    for nn, time in enumerate(times_l):
        infilename = 'emission_{}star_case{}_{}yr.dat'.format( star, casenum, time )
        infilename = os.path.join( emission_dir, infilename )
        wavelength, flux_density = np.loadtxt( infilename, unpack=True )
        wavelength, flux_density = resample( wavelength, flux_density, 0.005 )
        color = fig_o.get_color(nn)
        strtime = str( np.around(time*1E-6,2 ) )
        #label = strtime + ' (' + str(phi_l[nn]) + ')'
        phi_label = '%.2f' % phi_l[nn]
        temp_label = str(np.round(temp_l[nn],))[:-2]
        label = str(phi_label) + ' (' + str(temp_label) + ')'
        #h1, = ax.semilogy( wavelength, flux_density, label=label, color=color, linestyle='-', linewidth=0.5 )
        h1, = ax.loglog( wavelength, flux_density, label=label, color=color, linestyle='-', linewidth=0.5 )
        handle_l.append( h1 )

    title = r'\textbf{' + title + '}'
    xlabel = r'Wavelength ($\mu$m)'
    ylabel = r'Flux density (Wm$^{-2}\mu$m$^{-1}$)'
    fig_o.set_myaxes( ax, xlabel=xlabel, ylabel=ylabel, title=title, yrotation=90 )

    #ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(ticker.FixedLocator((1,10,20,30)))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=9) )
    #ax.set_yticks( [1E-10,1E-5,1,1E5] )
    #ax.set_yticks( [1E-9,1E-8,1E-7,1E-6], minor=True )
    #ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(1,2,3,4,5), numticks=20))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    #TITLE = "Time, Myr ($\phi_g$)"
    TITLE = "$\phi_g$ ($T_s$)"
    fig_o.set_mylegend( ax, handle_l, loc='lower right', TITLE=TITLE )

#====================================================================
def plot_flux_ratio_spectra():

    figw = 4.7747
    figh = 4.7747

    # dummy for colors
    times = '0,0,0,0,0,0'

    fig_o = su.FigureData( 2, 2, figw, figh, 'emission_flux_ratio', times )
    fig_o.fig.subplots_adjust(wspace=0.5,hspace=0.4)

    ax = fig_o.ax

    ax0 = ax[0][0]
    ax1 = ax[0][1]
    ax2 = ax[1][0]
    ax3 = ax[1][1]

    plot_flux_ratio_spectra_for_case( '1m', 'g', fig_o, ax0 )
    plot_flux_ratio_spectra_for_case( '9m', 'g', fig_o, ax1 )
    plot_flux_ratio_spectra_for_case( '1m', 'm', fig_o, ax2 )
    plot_flux_ratio_spectra_for_case( '9m', 'm', fig_o, ax3 )

    fig_o.savefig(12)

#====================================================================
def plot_mstar_spectra():

    figw = 4.7747
    figh = 4.7747

    # dummy for colors
    times = '0,0,0,0,0'

    fig_o = su.FigureData( 2, 2, figw, figh, 'mstar_spectra', times )
    fig_o.fig.subplots_adjust(wspace=0.5,hspace=0.4)

    ax = fig_o.ax

    ax0 = ax[0][0]
    ax1 = ax[0][1]
    ax2 = ax[1][0]
    ax3 = ax[1][1]

    plot_emission_for_case( '1m', 'm', fig_o, ax0 )
    plot_emission_for_case( '9m', 'm', fig_o, ax1 )
    plot_flux_ratio_spectra_for_case( '1m', 'm', fig_o, ax2 )
    plot_flux_ratio_spectra_for_case( '9m', 'm', fig_o, ax3 )

    fig_o.savefig(19)

#====================================================================
def plot_flux_ratio_spectra_for_case( casenum, star, fig_o, ax ):

    emission_dir = '/Volumes/data/spectra_c'

    xlim = (0.7,30)

    sigma = 5.670367E-8

    if star=='g':
        # visible (0.5 mu m)
        teff=5800.0 # K
        Rstar = 695510 # km
        ylim = (0, 1.0E3)
    elif star=='m':
        # infra-red (1.1 mu m)
        teff=2560.0 # K
        Rstar = 0.12 * 695510 # km
        ylim = (0, 1.0E3)

    REarth = 6371.0 # km
    geom_fac = (REarth/Rstar)**2.0

    #solar_constant = 1367.0 # W/m^2
    solar_scaling = sigma*teff**4.0

    print( 'solar_scaling=', solar_scaling )

    ustar = star.upper()

    # case 1
    if casenum == 1:
        times_l = (0,94400,250000,78400000,560000000)#,4550000000)
        phi_l = (1,0.75,0.3,0.1,0.01)#,0)
        if star=='g':
            prefix = '(a) '
        elif star=='m':
            prefix = '(c) '
        title = prefix + 'Case 1 (' + ustar + '-star)'
    # case 9
    elif casenum == 9:
        times_l = (0,3850000,10950000,141000000,802000000)#,4550000000)
        phi_l = (1,0.75,0.3,0.1,0.01)#,0)
        if star=='g':
            prefix = '(b) '
        elif star=='m':
            prefix = '(d) '
        title = prefix + 'Case 9 (' + ustar + '-star)'
    # case 1m
    elif casenum == '1m':
        times_l = (0,94300,250000,500000,2250000)#,4550000000)
        phi_l = (1,0.75,0.3,0.1,0.01)#,0)
        if star=='g':
            prefix = '(a) '
        elif star=='m':
            prefix = '(c) '
        title = prefix + 'Case 1c ($r_{f,CO_2}=0.1$)'# (' + ustar + '-star)'
    # case 9m
    elif casenum == '9m':
        times_l = (0,3850000,11150000,16300000,21250000)#,4550000000)
        phi_l = (1,0.75,0.3,0.1,0.01)#,0)
        if star=='g':
            prefix = '(b) '
        elif star=='m':
            prefix = '(d) '
        title = prefix + 'Case 9c ($r_{f,CO_2}=0.9$)'# (' + ustar + '-star)'

    handle_l = []

    # get star spectrum for normalisation
    if star=='m':
        infilename = 'spectrum_trappist-1.dat'
    elif star== 'g':
        infilename = 'spectrum_g2v.dat'
    infilename = os.path.join( emission_dir, infilename )
    star_wavelength, star_flux = np.loadtxt( infilename, unpack=True, usecols=(2,3) )
    star_flux *= solar_scaling
    star_interp1d = interp1d( star_wavelength, star_flux )

    for nn, time in enumerate(times_l):
        infilename = 'emission_{}star_case{}_{}yr.dat'.format( star, casenum, time )
        infilename = os.path.join( emission_dir, infilename )
        wavelength, flux_density = np.loadtxt( infilename, unpack=True )
        wavelength, flux_density = resample( wavelength, flux_density, 0.005 )
        flux_density /= star_interp1d( wavelength )
        flux_density *= geom_fac
        flux_density *= 1.0E6
        color = fig_o.get_color(nn)
        #strtime = str( np.around(time*1E-6,2 ) )
        #label = strtime + ' (' + str(phi_l[nn]) + ')'
        label = '%.2f' % phi_l[nn]
        #h1, = ax.semilogy( wavelength, flux_density, label=label, color=color, linestyle='-', linewidth=0.5 )
        h1, = ax.semilogx( wavelength, flux_density, label=label, color=color, linestyle='-', linewidth=0.5 )
        handle_l.append( h1 )

    title = r'\textbf{' + title + '}'
    xlabel = r'Wavelength ($\mu$m)'
    #ylabel = r'Flux ratio (planet/star)'
    ylabel = r'F$_p$/F$_\star$ (ppm)'
    fig_o.set_myaxes( ax, xlabel=xlabel, ylabel=ylabel, title=title, yrotation=90 )

    #ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(ticker.FixedLocator((1,10,20,30)))
    #ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=9) )
    #ax.set_yticks( [1E-10,1E-5,1,1E5] )
    #ax.set_yticks( [1E-9,1E-8,1E-7,1E-6], minor=True )
    #ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(1,2,3,4,5), numticks=20))
    #ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    #TITLE = "Time, Myr ($\phi_g$)"
    TITLE = "$\phi_g$"
    fig_o.set_mylegend( ax, handle_l, loc='upper left', TITLE=TITLE )

#====================================================================
def plot_transmission_spectra():

    figw = 4.7747
    figh = 4.7747 / 2.0

    # dummy for colors
    times = '0,0,0,0,0,0'

    fig_o = su.FigureData( 1, 2, figw, figh, 'transmission', times )
    fig_o.fig.subplots_adjust(wspace=0.8,hspace=0.4)

    ax = fig_o.ax

    ax0 = ax[0]
    ax1 = ax[1]

    plot_transmission_spectra_for_case( '1m', fig_o, ax0 )
    plot_transmission_spectra_for_case( '9m', fig_o, ax1 )

    fig_o.savefig(13)

#====================================================================
def plot_transmission_spectra_for_case( casenum, fig_o, ax  ):

    emission_dir = '/Volumes/data/spectra_c'

    xlim = (0.7,5)
    # G-star transit depth yaxis range
    ylim1 = (76, 102)

    # case 1
    if casenum == 1:
        times_l = (0,94400,250000,78400000,560000000)#,4550000000)
        phi_l = (1,0.75,0.3,0.1,0.01)#,0)
        prefix = '(a) '
        title = prefix + r'Case 1 ($r_{f,CO_2}$=0.1)'  # (' + ustar + '-star)'
    # case 9
    elif casenum == 9:
        times_l = (0,3850000,10950000,141000000,802000000)#,4550000000)
        phi_l = (1,0.75,0.3,0.1,0.01)#,0)
        prefix = '(b) '
        title = prefix + r'Case 9 ($r_{f,CO_2}$=0.9)' # (' + ustar + '-star)'
    # case 1m
    elif casenum == '1m':
        times_l = (0,94300,250000,500000,2250000)#,4550000000)
        phi_l = (1,0.75,0.3,0.1,0.01)#,0)
        prefix = '(a) '
        title = prefix + 'Case 1c ($r_{f,CO_2}$=0.1)' # '# (' + ustar + '-star)'
    # case 9m
    elif casenum == '9m':
        times_l = (0,3850000,11150000,16300000,21250000)#,4550000000)
        phi_l = (1,0.75,0.3,0.1,0.01)#,0)
        prefix = '(b) '
        title = prefix + 'Case 9c ($r_{f,CO_2}$=0.9)'# (' + ustar + '-star)'

    handle_l = []

    gstar_radius = 695510 # km
    mstar_radius = 0.12 * gstar_radius # km

    for nn, time in enumerate(times_l):
        infilename = 'transmission_case{}_{}yr.dat'.format( casenum, time )
        infilename = os.path.join( emission_dir, infilename )
        wavelength, radius = np.loadtxt( infilename, usecols=(0,1), unpack=True )
        wavelength, radius = resample( wavelength, radius, 0.005 )
        depth = radius**2.0 / gstar_radius**2.0
        depth *= 1.0E6
        color = fig_o.get_color(nn)
        #strtime = str( np.around(time*1E-6,2 ) )
        #label = strtime + ' (' + str(phi_l[nn]) + ')'
        label = str(phi_l[nn])
        h1, = ax.plot( wavelength, depth, label=label, color=color, linestyle='-', linewidth=0.5 )
        handle_l.append( h1 )

    title = r'\textbf{' + title + '}'
    xlabel = r'Wavelength ($\mu$m)'
    #ylabel = r'Flux ratio (planet/star)'
    ylabel = r'Transit depth, G-star (ppm)'
    fig_o.set_myaxes( ax, xlabel=xlabel, ylabel=ylabel, title=title, yrotation=90 )

    # set up other yaxis
    ymin2 = ylim1[0] * gstar_radius**2.0 / mstar_radius**2.0
    ymax2 = ylim1[1] * gstar_radius**2.0 / mstar_radius**2.0
    axb = ax.twinx()
    ylabel = r'Transit depth, M-star (ppm)'
    fig_o.set_myaxes( axb, ylabel=ylabel, yrotation=90 )

    #ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(ticker.FixedLocator((1,2,3,4,5)))
    #ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=9) )
    #ax.set_yticks( [1E-10,1E-5,1,1E5] )
    #ax.set_yticks( [1E-9,1E-8,1E-7,1E-6], minor=True )
    #ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(1,2,3,4,5), numticks=20))
    #ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim1)
    axb.set_ylim( ymin2, ymax2 )
    #TITLE = "Time, Myr ($\phi_g$)"
    TITLE = "$\phi_g$"
    fig_o.set_mylegend( ax, handle_l, loc='lower center', ncol=2, TITLE=TITLE )

#====================================================================
def resample( x, y, increment ):

    interp1d_o = interp1d( x, y )
    xmin = np.min(x)
    xmax = np.max(x)
    newx = np.arange( xmin, xmax, increment )
    newy = interp1d_o( newx )

    return newx, newy

#====================================================================

if __name__ == "__main__":

    # much faster to dump the relevant data to a json and then read
    # from there, faster for debugging, plotting, etc.
    #dump_all_data()

    # uncomment below for generating plots
    #plot_interior_depletion()
    #plot_atmosphere_comparison()
    #plot_pressure_side_by_side()
    #plot_mantle_temperature_at_end()

    # loop over all cases to generate atmosphere PDFs
    #for casenum in ['3m5']:# (1,2,3,'3w',4,5,6,7,'7w',8,9):
    #    plot_atmosphere( casenum )

    #plot_right_versus_wrong()

    #plot_atmosphere_right_wrong()

    #plot_interior_atmosphere()

    #plot_radius_evolution()

    #plot_emission_spectra()

    #plot_flux_ratio_spectra()

    #plot_transmission_spectra()

    plot_mstar_spectra()

    #plot_partial_pressure_versus_depletion()

    #plot_phi_versus_radius()

    plt.show()
