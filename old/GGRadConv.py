'''
GGRadConv.py
'''



import GreyHeat as Grey
# from ClimateUtilities import *
import math,phys
import planets
import numpy as np
import matplotlib.pyplot as plt

#Set the gravity and thermodynamic constants
Rcp = 2./7.
n=20


#Choose the radiation model for computing the longwave and shortwave heating
radcomp = Grey.radcomp

def RadConvEqm(Tg):
    #--------------------Set radmodel options-------------------
    #---Instantiate the radiation model---
    n = 20
    #

    #Set global constants
    ps = 1000.
    rh = 1.e-30#Relative humidity
    rhbdd = 1.e-30
    dt = 24.*3600. #time step in seconds

    #---Set up pressure array (a global)----
    ptop = 50. #Top pressure in mb (Changed from 1mb in original)
    pstart = .995*ps
    rat = (ptop/pstart)**(1./n)
    logLevels = [pstart*rat**i for i in range(n)]
    logLevels.reverse()
    levels = [ptop + i*(pstart-ptop)/(n-1) for i in range(n)]
    p = numpy.array(logLevels)



    #==============Now do the calculation====================================

    #--------------Initializations-------------------------------------------

    #----------------Set initial time step--------------------------------

    dtime = 1.# 1. # (for CO2 case; gray gas evolves faster)

    #----------------------------------------------------------------------

    #---Temperature and moisture arrays (initialized)
    T = numpy.zeros(n) + 230.


    #--------------Other parameters-------------------------------------------
    doStellarAbs = False

    #Set composition constants (globals)
    #co2 = 300.

    #Ground temperature (held fixed in this computation)
    Tg = 280.
    #---Temperature and moisture arrays (initialized)
    T = Tg*(p/p[-1])**Rcp  #Initialize on an adiabat
    #T = Tg*numpy.ones(len(p))


    #Set composition parameters for the radiation code you are using
    q=np.zeros(n)


    #Grey Gas:
    Grey.tauInf = 2.

    PrevOLR = 0.
    #---------------------------------------------------------
    #--------------Initializations Done-----------------------
    #--------------Now do the time stepping-------------------
    #---------------------------------------------------------
    for i in range(0,50):
        nout = 10*i
        print(dtime)
        #if i%50 == 0 & i > 200:
        #     dtime = .5*dtime
        Tg,Tad,T,flux,fluxStellar,fluxLW,heat,heatStellar,heatLW = steps(Tg,T,p,q,10,dtime)
        print('History step',Tg,flux[-1],max(heat),min(heat))
        #history(nout,caseTag)
        if abs(flux[-1]-PrevOLR) < 1.0:
               break    # break here
        PrevOLR = flux[-1]

    # plot equilibrium temperature profile
    plt.figure()
    plt.semilogy(T,p)
    plt.gca().invert_yaxis()
    plt.ylabel('Pressure (mb)')
    plt.xlabel('Temperature (K)')
    plt.savefig('Tprofile.pdf',bb_inches='tight')


    return flux[-1]



#Dry adjustment routine.
#**ToDo: Modify so it handles non-uniform pressure levels
#        correctly, and conserves integrated enthalpy cp T
#
#Iterative routine for dry convective adjustment
def dryAdj(T,p):
    #Rcp is a global
    #Downward pass
    for i in range(len(T)-1):
        T1,p1 = T[i],p[i]
        T2,p2 = T[i+1],p[i+1]
        pfact = (p1/p2)**Rcp
        if T1 < T2*pfact:
            Tbar = .5*(T1+T2) #Equal layer masses
                              #Not quite compatible with how
                              #heating is computed from flux
            T2 = 2.*Tbar/(1.+pfact)
            T1 = T2*pfact
            T[i] = T1
            T[i+1] = T2
    #Upward pass
    for i in range(len(T)-2,-1,-1):
        T1,p1 = T[i],p[i]
        T2,p2 = T[i+1],p[i+1]
        pfact = (p1/p2)**Rcp
        if T1 < T2*pfact:
            Tbar = .5*(T1+T2) #Equal layer masses
                              #Not quite compatible with how
                              #heating is computed from flux
            T2 = 2.*Tbar/(1.+pfact)
            T1 = T2*pfact
            T[i] = T1
            T[i+1] = T2


#Define function to do time integration for n steps
def steps(Tg,T,p,q,nSteps,dtime):
    for i in range(nSteps):
        #Do smoothing
##        if i%20 == 0:
##            for j in range(1,len(T)-1):
##                T[j] = .25*T[j-1] + .5*T[j] + .25*T[j+1]
        flux,heat = radcomp(p,T,Tg,q)
        dT = heat*dtime
        #Limit the temperature change per step
        dT = numpy.where(dT>5.,5.,dT)
        dT = numpy.where(dT<-5.,-5.,dT)
        #Midpoint method time stepping
        #changed call to r.  Also modified to hold Tg fixed
        flux,heat = radcomp(p,T+.5*dT,Tg,q)
        dT = heat*dtime
        #Limit the temperature change per step
        dT = numpy.where(dT>5.,5.,dT)
        dT = numpy.where(dT<-5.,-5.,dT)
        T += dT
        #
        dTmax = max(abs(dT)) #To keep track of convergence

#   Do the surface balance
        kturb = .1
        T[-1] += -dtime*kturb*(T[-1] - Tg)
        #Dry adjustment step
        for iadj in range(10):
            dryAdj(T,p)
        Tad = T[-1]*(p/p[-1])**Rcp
        #** Temporary kludge to keep stratosphere from getting too cold
        T = numpy.where(T<50.,50.,T)  #**KLUDGE
        #
        #Dummies for separate LW and stellar. **FIX THIS**
        fluxStellar = fluxLW = heatStellar = heatLW = numpy.zeros(n)
    return Tg,Tad,T,flux,fluxStellar,fluxLW,heat,heatStellar,heatLW
