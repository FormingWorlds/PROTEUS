'''
SocRadModel.py
Returns heating rates
MDH 25/01/19
'''

import os
import netCDF4 as net
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import subprocess
import nctools
from subprocess import call
from netCDF4 import Dataset
from atmosphere_column import atmos

path_to_socrates = os.getcwd()+"/socrates/socrates_main"

def radCompSoc(atm, stellar_toa_heating):

    # Write temperature, pressure, and mixing ratios

    templ_list = np.interp(atm.pl[:],atm.p[:],atm.temp[:])
    temp_list =  (templ_list[1:] + templ_list[:-1]) / 2
    pres_list = 100.0*atm.p[:]
    presl_list = 100.0*atm.pl[:]


    # CO2 mixing ratio profile
    co2_mr_list = atm.mixing_ratios[1]
    # Water vapour mixing ratio profile
    q_mr_list = atm.mixing_ratios[0]

    # Write single values
    t_surf = atm.ts
    p_surf = 100.0*atm.p[-1]
    solar_zenith_angle = 0.0
    # solar_toa = 0.0 #zero stellar heating for now
    solar_toa = stellar_toa_heating


    # Write values to netcdf
    nctools.ncout_surf('profile.surf',0,0,1,0.1)
    nctools.ncout2d('profile.tstar',0,0,t_surf,'tstar',longname="Surface Temperature",units='K')
    nctools.ncout2d('profile.pstar',0,0,p_surf,'pstar',longname="Surface Pressure",units='PA')
    nctools.ncout2d('profile.szen',0,0,solar_zenith_angle,'szen',longname="Solar zenith angle",units='Degrees')
    nctools.ncout2d('profile.stoa',0,0,solar_toa,'stoa',longname="Solar Irradiance at TOA",units='WM-2')
    nctools.ncout3d('profile.t',0,0,pres_list,temp_list,'t',longname="Temperature",units='K')
    nctools.ncout3d('profile.tl',0,0,presl_list,templ_list,'tl',longname="Temperature",units='K')
    nctools.ncout3d('profile.p',0,0,pres_list,pres_list,'p',longname="Pressure",units='PA')
    nctools.ncout3d('profile.co2',0,0,pres_list,co2_mr_list,'co2',longname="CO2",units='PPMV')
    nctools.ncout3d('profile.q',0,0,pres_list,q_mr_list,'q',longname="q",units='PPMV')
#    nctools.ncout3d('profile.h2o',0,0,pres_list,q_mr_list,'h2o',longname="h2o",units='PPMV')


    basename = 'profile'
    s = " "


    # seq4 = ("Cl_run_cdf -B", basename,"-s "+path_to_socrates+"/data/spectra/ga7/sp_sw_ga7 -R 1 6 -ch 6 -S -g 2 -C 5")
    seq4 = ("Cl_run_cdf -B", basename,"-s spectral-files/sp_spider_3000K_300 -R 1 300 -ch 300 -S -g 2 -C 5")
    seq5 = ("fmove", basename,"currentsw")
    seq6 = ("Cl_run_cdf -B", basename,"-s spectral-files/sp_spider_3000K_300 -R 1 300 -ch 300 -I -g 2 -C 5")
    seq7 = ("fmove", basename,"currentlw")


    comline1 = s.join(seq4)
    comline2 = s.join(seq5)
    comline3 = s.join(seq6)
    comline4 = s.join(seq7)

    if 1==1:
        os.system(comline1)
        os.system(comline2)
        os.system(comline3)
        os.system(comline4)

    #open netCDF files produced by SOCRATES
    ncfile1 = net.Dataset('currentsw.vflx')
    ncfile2 = net.Dataset('currentsw.sflx')
    ncfile3 = net.Dataset('currentsw.dflx')
    ncfile4 = net.Dataset('currentsw.uflx')
    ncfile5 = net.Dataset('currentsw.nflx')
    ncfile6 = net.Dataset('currentsw.hrts')
    ncfile7 = net.Dataset('currentlw.dflx')
    ncfile8 = net.Dataset('currentlw.nflx')
    ncfile9 = net.Dataset('currentlw.uflx')
    ncfile10 = net.Dataset('currentlw.hrts')

    #create appropriately sized arrays to hold flux data
    p = ncfile1.variables['plev'][:]
    levels = len(p)
    vflx = np.zeros(levels)
    sflx = np.zeros(levels)
    dflx = np.zeros(levels)
    uflx = np.zeros(levels)
    nflx = np.zeros(levels)
    hrts = np.zeros(levels-1)
    dflxlw = np.zeros(levels)
    nflxlw = np.zeros(levels)
    uflxlw = np.zeros(levels)
    hrtslw = np.zeros(levels-1)

    #loop through netCDF variables and populate arrays
    uflxlw = ncfile9.variables['uflx']
    #uflxsw = ncfile4.variables['uflx']
    vflxsw = ncfile1.variables['vflx']

    nflxlw = ncfile8.variables['nflx']
    nflxsw = ncfile5.variables['nflx']

    hrtssw = ncfile6.variables['hrts']
    hrtslw = ncfile10.variables['hrts']

    atm.total_heating = np.squeeze(np.sum(hrtssw[:,:],axis=0) + np.sum(hrtslw[:,:],axis=0))

    # Sum LW flux over all bands
    atm.LW_flux_up = np.sum(uflxlw[:,:],axis=0)[:,0,0]
    atm.LW_spectral_flux_up = uflxlw[:,:,0,0]

    return atm
