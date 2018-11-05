#Modified for Exoclimes2012.  Uses dry enthalpy conserving convective
#adjustment and implements an energy conserving surface flux scheme
#(Note enthalpy is the right thing to conserve, not dry static energy)
#
#Modified further to make it easier to swap in homebrew or graygas models.
# (8/28/2012, for Beijing lectures)
#
#
#Note: The stellar flux past the shortwave cutoff had to be
#added in to the calculation of incoming flux. This needs
#to be put in the main code-tree.

#ToDo:
#     *Make it possible to use ccm radiation.
#         ->Need radcomp in ccmradFunctions. Return heating in W/kg
#     *make it possible to use the fancy version of miniclimt
#     *Make a script that computes the pure radiative equilibria
#      in Figure {fig:RealGasRadEq}, and also the logarithmic slopes.
#     *Handle the contribution of water vapor to surface pressure
#      in setting up the pressure grid.  (Tricky, because
#      temperature is changing!)  That's important when
#      surface temperatures are much over 300K.  The changing
#      surface pressure makes the time stepping rather tricky.

#Data on section of text which this script is associated with
Chapter = '5'
Section = '**'
Figure = '**'
#

#Following import only needed for ccm radiation computation
#import climt_lite as climt #ccm radiation model (Replace with ccmradFunctions?)

import GreyHeat as Grey
from ClimateUtilities import *
import math,phys
import planets

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

#--------------------Set radmodel options-------------------
#---Instantiate the radiation model---
#r = climt.radiation()
#n = r.nlev
n = 20
#
#Choose the radiation model for computing the longwave and shortwave heating
radcomp = Grey.radcomp

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

#An array of small numbers, useful when we need to pass a near-zero
#water vapor profile to a radiation model. Many rad. models crash if
#given exactly zero moisture
small = numpy.zeros(n) + 1.e-30


#CCM radiative flux and heating. 
#   This is not used for the Grey Gas case
def radcompCCM(p,T,Tg,q):
    #**Changed call to r
    r(p=p,ps=ps,T=T,Ts=Tg,q=q,co2 = co2,o3=small)
    #**Changed indexing of flux and hr
    fluxLW,heatLW = -r.lwflx[:,0,0],r.lwhr[:,0,0]
    fluxStellar,heatStellar = -r.swflx[:,0,0],r.swhr[:,0,0]
    flux = fluxLW+fluxStellar
    if doStellarAbs:    
        heat = heatLW+heatStellar
    else:
        heat = heatLW
    return flux,heat

#Define function to do time integration for n steps
def steps(Tg,T,nSteps,dtime):
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
        #sa
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
    
#------------History function, to write output--------------------------        
def history(n,tag):
    suffix = tag+'%d'%n+'.txt'
    c1 = Curve()
    c1.addCurve(p)
    c1.addCurve(flux,'Flux')
    c1.addCurve(fluxStellar,'Stellar')
    c1.addCurve(fluxLW,'LW')
    c1.switchXY=c1.reverseY = True
    c1.PlotTitle = 'Flux'
    c1.dump(outputDir+'Flux'+suffix)


    c2 = Curve()
    c2.addCurve(p)
    c2.addCurve(heat,'Heating')
    c2.addCurve(heatStellar,'Stellar')
    c2.addCurve(heatLW,'LW')
    c2.switchXY=c2.reverseY = True
    c2.PlotTitle = 'Heating Rate'
    c2.dump(outputDir+'Heat'+suffix)

    c3 = Curve()
    c3.addCurve(p,'p')
    c3.addCurve(T,'T')
    c3.addCurve(Tad,'Tadiabat')
    c3.switchXY=c3.reverseY = True
    c3.dump(outputDir+'T'+suffix)

#==============Now do the calculation====================================

#--------------Initializations-------------------------------------------
    
#----------------Set initial time step--------------------------------
#** Has something changed with climt? this seems to need smaller time step
#   I think maybe heating rate in climt is now K/day, not K/sec
dtime = .1# 1. # (for CO2 case; gray gas evolves faster)
            #Timestep in days; 5 days is the usual for Earthlike case
            #For the radiative convective case, you can get away with
            #using 50 for the first few hundred time steps, then
            #reducing to 5 or less later as the solution converges

#dtime = dtime*24.*3600. (Left over from old version of climt
#----------------------------------------------------------------------

#---Temperature and moisture arrays (initialized)
T = numpy.zeros(n) + 230.
q = small #Not used for grey gas


#Where to put the output
outputDir = 'RCOut/'
caseTag = 'RC'

#--------------Other parameters-------------------------------------------
doStellarAbs = False
#Set the gravity and thermodynamic constants
Rcp = 2./7.
#Set composition constants (globals)
#co2 = 300.

#Ground temperature (held fixed in this computation)
Tg = 280.
#---Temperature and moisture arrays (initialized)
#T = Tg*(p/p[-1])**Rcp  #Initialize on an adiabat
T = Tg*numpy.ones(len(p))
q = small

#
#Initialize the temperature
#c = readTable(outputDir+'TTestTrop80.txt') #For restart from file
#p = c['p']
#T = c['T']
#Tg = T[-1]


#Set composition parameters for the radiation code you are using

#CCMrad:
# **CHANGE (climt): This is specified in the call to r
#r.params.value['co2'] = 300.

#Grey Gas:
Grey.tauInf = 1.


#---------------------------------------------------------
#--------------Initializations Done-----------------------
#--------------Now do the time stepping-------------------
#---------------------------------------------------------
for i in range(0,200):
    nout = 10*i
    print(dtime)
    #if i%50 == 0 & i > 200:
    #     dtime = .5*dtime
    Tg,Tad,T,flux,fluxStellar,fluxLW,heat,heatStellar,heatLW = steps(Tg,T,10,dtime)
    print('History step',Tg,flux[-1],max(heat),min(heat))
    #history(nout,caseTag)

