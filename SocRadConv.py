'''
MDH 28/01/19

Socrates radiative-convective model
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import SocRadModel
from atmosphere_column import atmos
import pandas as pd


# # Font settings
# import matplotlib.pylab as pylab
# import matplotlib.font_manager as fm
# font = fm.FontProperties(family = 'Helvetica', fname = '/Users/tim/Dropbox/work/matplotlib_fonts/Helvetica/Helvetica.ttf')

def surf_Planck_nu(atm):
    h = 6.63e-34
    c = 3.0e8
    kb = 1.38e-23
    B = np.zeros(len(atm.band_centres))
    c1 = 1.191042e-5
    c2 = 1.4387752
    for i in range(len(atm.band_centres)):
        nu = atm.band_centres[i]
        B[i] = (c1*nu**3 / (np.exp(c2*nu/atm.ts)-1))

    B = B * atm.band_widths/1000.0
    return B

def RadConvEqm(output_dir, time_current, runtime_helpfile, stellar_toa_heating, atm_chemistry):
    #--------------------Set radmodel options-------------------
    #---Instantiate the radiation model---

    # Atmosphere struct
    atm = atmos()

    #---Set up pressure array (a global)----
    atm.ps      = runtime_helpfile.iloc[-1]["P_surf"]*1e3 # bar->mbar
    pstart      = atm.ps#*.995
    rat         = (atm.ptop/pstart)**(1./atm.nlev)
    logLevels   = [pstart*rat**i for i in range(atm.nlev+1)]
    logLevels.reverse()
    levels      = [atm.ptop + i*(pstart-atm.ptop)/(atm.nlev-1) for i in range(atm.nlev+1)]
    atm.pl      = np.array(logLevels)
    atm.p       = (atm.pl[1:] + atm.pl[:-1]) / 2

    # print("Pressure calc:")
    # print("Pressure (VULCAN):", atm_chemistry["Pressure"])
    # print("Surface pressure:", p_s)
    # print(atm.p)


    #==============Now do the calculation====================================

    atm.ts          = runtime_helpfile.iloc[-1]["T_surf"]
    atm.Rcp         = 2./7.
    atm.temp        = atm.ts*(atm.p/atm.p[-1])**atm.Rcp  #Initialize on an adiabat
    atm.temp        = np.where(atm.temp<atm.ts/2.,atm.ts/2.,atm.temp)
    # atm.n_species = 2
    atm.n_species   = 7

    # # Water vapour
    # atm.mixing_ratios[0] = 1.e-5
    # # CO2
    # atm.mixing_ratios[1] = 1.e-5

    ## TO DO: change the currently CONSTANT mixing ratios to varying with height
    atm.mixing_ratios[0] = atm_chemistry.iloc[0]["H2O"] # H2O
    atm.mixing_ratios[1] = atm_chemistry.iloc[0]["CO2"] # CO2
    atm.mixing_ratios[2] = atm_chemistry.iloc[0]["H2"]  # H2
    atm.mixing_ratios[3] = atm_chemistry.iloc[0]["CH4"] # CH4
    atm.mixing_ratios[4] = atm_chemistry.iloc[0]["CO"]  # CO
    atm.mixing_ratios[5] = atm_chemistry.iloc[0]["N2"]  # N2
    atm.mixing_ratios[6] = atm_chemistry.iloc[0]["O2"]  # O2

    # Initialise previous OLR and TOA heating to zero
    PrevOLR = 0.
    PrevMaxHeat = 0.
    PrevTemp = 0.*atm.temp[:]

    #---------------------------------------------------------
    #--------------Initializations Done-----------------------
    #--------------Now do the time stepping-------------------
    #---------------------------------------------------------
    matplotlib.rc('axes',edgecolor='k')
    for i in range(0,100):

        atm = steps(atm, stellar_toa_heating)

        #hack!
        # atm.temp[0] = atm.temp[1]

        if i % 5 == 0:
            print("Iteration", i, end =", ")
            # if 1==2:
            #     plt.figure(figsize=(7,4))
            #     plt.semilogy(atm.temp,atm.p)
            #     plt.gca().invert_yaxis()
            #     plt.ylabel('Pressure [mb]')
            #     plt.xlabel('Temperature [K]')
            #     plt.gca().xaxis.label.set_color('white')
            #     plt.tick_params(axis='x', colors='white')
            #     plt.gca().yaxis.label.set_color('white')
            #     plt.tick_params(axis='y', colors='white')
                # plt.show()
            #print("OLR " + str(atm.LW_flux_up[-1]))
            #print("OLR change " + str(atm.LW_flux_up[-1]-PrevOLR))
            # print("Max heating " + str(np.max(atm.total_heating)))
            #print("Max dT " + str(abs(np.max(atm.temp-PrevTemp[:]))))
            
#             # Plot on the fly
#             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
#             ax1.semilogy(atm.temp,atm.p)
#             ax1.invert_yaxis()
#             ax1.set_xlabel('Temperature [K]')
#             ax1.set_ylabel('Pressure [mb]')
#             ax2.plot(atm.band_centres,atm.LW_spectral_flux_up[:,0]/atm.band_widths,'k')
# #            ax2.plot(atm.band_centres,surf_Planck_nu(atm)/atm.band_widths,'k--')
#             ax2.set_xlim([0,8*atm.ts])
#             ax2.set_ylabel('Spectral Flux')
#             ax2.set_xlabel('Wavenumber')
#             ax2.set_title('Spectral OLR')
#             # plt.show()
#             plt.savefig(output_dir+'/TP_profile_'+str(round(time_current))+'.png', bbox_inches="tight")
#             plt.close(fig)

            print("OLR = " + str(PrevOLR)+" W/m^2,", "Max heating = " + str(np.max(atm.total_heating)))

        # Reduce timestep if heating not converging
        if abs(np.max(atm.temp-PrevTemp[:])) < 0.05 or abs(atm.temp[0]-atm.temp[1]) > 3.0:
            #print("reducing timestep")
            atm.dt  = atm.dt*0.99


        if abs(atm.LW_flux_up[-1]-PrevOLR) < 0.1 and abs(np.max(atm.temp-PrevTemp[:])) < 0.5:
           print("break")
           #print(PrevTemp[:]-atm.temp)
           break    # break here

        PrevOLR = atm.LW_flux_up[0]
        PrevMaxHeat = abs(np.max(atm.total_heating))
        PrevTemp[:] = atm.temp[:]

    # # plot equilibrium temperature profile
    # plt.figure()
    # plt.semilogy(atm.temp,atm.p)
    # plt.gca().invert_yaxis()
    # plt.ylabel('Pressure [mb]')
    # plt.xlabel('Temperature [K]')
    # plt.savefig(output_dir+'/T_profile_'+str(round(time_current))+'.pdf', bbox_inches="tight")

    # Write TP and spectral flux profiles for later plotting
    out_a = np.column_stack( ( atm.temp, atm.p*1.e-3 ) ) # K, mbar->bar
    np.savetxt( output_dir+str(int(time_current))+"_atm_TP_profile.dat", out_a )
    out_a = np.column_stack( ( atm.band_centres, atm.LW_spectral_flux_up[:,0]/atm.band_widths ) )
    np.savetxt( output_dir+str(int(time_current))+"_atm_spectral_flux.dat", out_a )

    return atm.LW_flux_up[-1]

# Dry adjustment routine
def dryAdj(atm):
    T = atm.temp
    p = atm.p
    #Rcp is a global
    #Downward pass
    for i in range(len(T)-1):
        T1,p1 = T[i],p[i]
        T2,p2 = T[i+1],p[i+1]
        # Adiabat slope
        pfact = (p1/p2)**atm.Rcp
        # If slope is shallower than adiabat (unstable), adjust it to adiabat
        if T1 < T2*pfact:
            Tbar = .5*(T1+T2) # Equal layer masses
                              # Not quite compatible with how
                              # heating is computed from flux
            T2 = 2.*Tbar/(1.+pfact)
            T1 = T2*pfact
            atm.temp[i] = T1
            atm.temp[i+1] = T2
    #Upward pass
    for i in range(len(T)-2,-1,-1):
        T1,p1 = T[i],p[i]
        T2,p2 = T[i+1],p[i+1]
        pfact = (p1/p2)**atm.Rcp
        if T1 < T2*pfact:
            Tbar = .5*(T1+T2) # Equal layer masses
                              # Not quite compatible with how
                              # heating is computed from flux
            T2 = 2.*Tbar/(1.+pfact)
            T1 = T2*pfact
            atm.temp[i] = T1
            atm.temp[i+1] = T2


#Define function to do time integration for n steps
def steps(atm, stellar_toa_heating):
    atm = SocRadModel.radCompSoc(atm, stellar_toa_heating)
    dT = atm.total_heating*atm.dt
    #Limit the temperature change per step
    dT = np.where(dT>5.,5.,dT)
    dT = np.where(dT<-5.,-5.,dT)
    #Midpoint method time stepping
    #changed call to r.  Also modified to hold Tg fixed
    atm = SocRadModel.radCompSoc(atm, stellar_toa_heating)
    dT = atm.total_heating*atm.dt
    #Limit the temperature change per step
    dT = np.where(dT>5.,5.,dT)
    dT = np.where(dT<-5.,-5.,dT)
    atm.temp += dT
    #
    dTmax = max(abs(dT)) #To keep track of convergence

    #   Do the surface balance
    kturb = .1
    atm.temp[-1] += -atm.dt*kturb*(atm.temp[-1] - atm.ts)
    #Dry adjustment step
    for iadj in range(10):
        dryAdj(atm)
    Tad = atm.temp[-1]*(atm.p/atm.p[-1])**atm.Rcp
    #** Temporary kludge to keep stratosphere from getting too cold
    atm.temp = np.where(atm.temp<50.,50.,atm.temp)  #**KLUDGE
    #
    #Dummies for separate LW and stellar. **FIX THIS**
    fluxStellar = fluxLW = heatStellar = heatLW = np.zeros(atm.nlev)
    return atm
