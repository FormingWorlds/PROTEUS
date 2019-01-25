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


def radCompSoc(p,T,Tg):

    # Write temperature, pressure, and mixing ratios
    temp_list = T[:]
    templ_list = 0.99*T[:]
    templ_list = np.append(templ_list,1.03*T[-1])
    pres_list = p[:]
    presl_list = np.append(0.99*pres_list,1.03*pres_list[-1])
    

    co2_mr_list = 3e-4*np.ones(len(T))
    q_mr_list = 1e-3*np.ones(len(T))

    
    # Write single values
    t_surf = Tg
    p_surf=p[-1]
    solar_zenith_angle = 0.0
    solar_toa = 1370.
    
    
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
    nctools.ncout3d('profile.q',0,0,pres_list,q_mr_list,'co2',longname="q",units='PPMV')
    
    
    basename = 'profile'
    s = "."
    
    #call SOCRATES for both LW and SW, moving outputfiles and saving as numpy

    
    
    seq4 = ("Cl_run_cdf -B", basename,"-s /Users/markhammond/Work/Projects/1D-RC-SOC/socrates/socrates_1806/data/spectra/ga7/sp_sw_ga7 -R 1 6 -ch 6 -S -g 2 -C 5")
    seq5 = ("fmove", basename,"currentsw")
    seq6 = ("Cl_run_cdf -B", basename,"-s /Users/markhammond/Work/Projects/1D-RC-SOC/socrates/socrates_1806/data/spectra/ga7/sp_lw_ga7 -R 1 9 -ch 9 -I -g 2 -C 5")
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
    
    return hrtssw,hrtslw,uflxlw,nflxsw
    
    
