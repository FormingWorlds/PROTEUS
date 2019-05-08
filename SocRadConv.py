'''
MDH 28/01/19

Socrates radiative-convective model
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import SocRadModel
from atmosphere_column import atmos


def RadConvEqm(Tg):
    #--------------------Set radmodel options-------------------
    #---Instantiate the radiation model---

    atm = atmos()

    #---Set up pressure array (a global)----
    pstart = .995*atm.ps
    rat = (atm.ptop/pstart)**(1./atm.nlev)
    logLevels = [pstart*rat**i for i in range(atm.nlev+1)]
    logLevels.reverse()
    levels = [atm.ptop + i*(pstart-atm.ptop)/(atm.nlev-1) for i in range(atm.nlev+1)]
    atm.pl = np.array(logLevels)
    atm.p = (atm.pl[1:] + atm.pl[:-1]) / 2


    #==============Now do the calculation====================================


    atm.ts = Tg
    atm.Rcp = 2./7.
    atm.temp = atm.ts*(atm.p/atm.p[-1])**atm.Rcp  #Initialize on an adiabat
    atm.temp  = np.where(atm.temp<180.,180,atm.temp)
    atm.n_species = 2

    # Water vapour
    atm.mixing_ratios[0] = 0.0*1.e-4
    # CO2
    atm.mixing_ratios[1] = 1.e-2


    # Initialise previous OLR and TOA heating to zero
    PrevOLR = 0.
    PrevMaxHeat = 0.
    PrevTemp = 0.*atm.temp[:]

    #---------------------------------------------------------
    #--------------Initializations Done-----------------------
    #--------------Now do the time stepping-------------------
    #---------------------------------------------------------
    matplotlib.rc('axes',edgecolor='w')
    for i in range(0,100):

        atm = steps(atm)

        #hack!
        # atm.temp[0] = atm.temp[1]

        if i % 20 == 0:
            print i
            if 1==2:
                plt.figure(figsize=(7,4))
                plt.semilogy(atm.temp,atm.p)
                plt.gca().invert_yaxis()
                plt.ylabel('Pressure (mb)')
                plt.xlabel('Temperature (K)')
                plt.gca().xaxis.label.set_color('white')
                plt.tick_params(axis='x', colors='white')
                plt.gca().yaxis.label.set_color('white')
                plt.tick_params(axis='y', colors='white')
                plt.show()
            #print "OLR " + str(atm.LW_flux_up[-1])
            #print "OLR change " + str(atm.LW_flux_up[-1]-PrevOLR)
            #print "Max heating " + str(np.max(atm.total_heating))
            #print "Max dT " + str(abs(np.max(atm.temp-PrevTemp[:])))


        # Reduce timestep if heating not converging
        if abs(np.max(atm.temp-PrevTemp[:])) < 0.05 or abs(atm.temp[0]-atm.temp[1]) > 3.0:
            #print "reducing timestep"
            atm.dt  = atm.dt*0.99


        if abs(atm.LW_flux_up[-1]-PrevOLR) < 0.1 and abs(np.max(atm.temp-PrevTemp[:])) < 0.5:
           print "break"
           #print PrevTemp[:]-atm.temp
           break    # break here

        PrevOLR = atm.LW_flux_up[-1]
        PrevMaxHeat = abs(np.max(atm.total_heating))
        PrevTemp[:] = atm.temp[:]




    # plot equilibrium temperature profile
    plt.figure()
    plt.semilogy(atm.temp,atm.p)
    plt.gca().invert_yaxis()
    plt.ylabel('Pressure (mb)')
    plt.xlabel('Temperature (K)')
    plt.savefig('Tprofile.pdf',bb_inches='tight')

    return atm.LW_flux_up[-1]



#Dry adjustment routine.
def dryAdj(atm):
    T = atm.temp
    p = atm.p
    #Rcp is a global
    #Downward pass
    for i in range(len(T)-1):
        T1,p1 = T[i],p[i]
        T2,p2 = T[i+1],p[i+1]
        pfact = (p1/p2)**atm.Rcp
        if T1 < T2*pfact:
            Tbar = .5*(T1+T2) #Equal layer masses
                              #Not quite compatible with how
                              #heating is computed from flux
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
            Tbar = .5*(T1+T2) #Equal layer masses
                              #Not quite compatible with how
                              #heating is computed from flux
            T2 = 2.*Tbar/(1.+pfact)
            T1 = T2*pfact
            atm.temp[i] = T1
            atm.temp[i+1] = T2


#Define function to do time integration for n steps
def steps(atm):
    atm = SocRadModel.radCompSoc(atm)
    dT = atm.total_heating*atm.dt
    #Limit the temperature change per step
    dT = np.where(dT>5.,5.,dT)
    dT = np.where(dT<-5.,-5.,dT)
    #Midpoint method time stepping
    #changed call to r.  Also modified to hold Tg fixed
    atm = SocRadModel.radCompSoc(atm)
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
