#===================================================================
#Computes fluxes and heating rates for the grey gas model.
#The fluxes are computed as a function of p/ps, given net optical
#thickness of the atmosphere tauinf .
#
#Modified to return dimensional heat and flux for use with
#the time-stepping radiative-convective model
#===================================================================

#Data on section of text which this script is associated with
#MODIFIED
Chapter = '4.**'
Figure = 'fig:AllTropNetIRFluxGrey'
#
#This is also the solution script for
Problem = '{Workbook:RadBalance2:PressBroadenedHeating}'
#This script can also be modified to use for the problem
# '{Workbook:RadBalance2:StratTropOLRGrey}'


import math,phys
# from ClimateUtilities import *


#Grey gas transmission function.
#tauinf is a global
def Trans(tau1,tau2):
    return math.exp(-abs(tau1-tau2))

#Integrand for upward or downward flux. Note that
#the Schwartzschild integral is written here as an integral
#over p/ps, and correspondingly the gradient of T is written as
#dT/d(p/ps). The solution is written in the form of
#Eq. (4.13) (in First Edition).
#

def integrand(ppsp,params):
    #Without pressure broadening
    if PressureBroadening:
        tau1 = tauInf*(1.-ppsp**2)
        tau2 = tauInf*(1.-params.pps**2)
    else:
        tau1 = tauInf*(1.-ppsp)
        tau2 = tauInf*(1. - params.pps)
    Tfun = params.Tfun
    dTdp = (Tfun(ppsp+.01)-Tfun(ppsp-.01))/.02
    return Trans(tau1,tau2)*4.*phys.sigma*Tfun(ppsp)**3*dTdp

def Iplus(pps,Tfun,Tg):
    params = Dummy()
    params.pps = pps
    params.Tfun = Tfun
    Ts = Tfun(1.)
    limit = min(1.,pps+10./tauInf)
    quad = romberg(integrand,10)
    if PressureBroadening:
        tau = tauInf*(1.-pps**2)
    else:
        tau = tauInf*(1.-pps)
    BddTerm = (phys.sigma*Tg**4 - phys.sigma*Ts**4)*Trans(0.,tau)
    return quad([pps,limit],params,.1)+ phys.sigma*Tfun(pps)**4 +BddTerm

def Iminus(pps,Tfun,Tg):
    params = Dummy()
    params.pps = pps
    params.Tfun = Tfun
    limit = max(0.,pps-10./tauInf)
    quad = romberg(integrand,10)
    Tstrat = Tfun(0.)
    if PressureBroadening:
        tau = tauInf*(1.-pps**2)
    else:
        tau = tauInf*(1.-pps)
    return quad([pps,0.],params,.1)+ phys.sigma*Tfun(pps)**4 - phys.sigma*Tstrat**4*Trans(tau,tauInf)

#Return dimensional flux and heating
#pList is dimensional pressure.
#Moisture q not used for gray gas
#

#Pressure in mb, for consistency with ccmrad
def radcomp(pList,TList,Tg,q):
    ppsL = pList/pList[-1] #p/psurf
    Tfun = interp(ppsL,TList)
    Ip = numpy.array([Iplus(pps,Tfun,Tg) for pps in ppsL])
    Im = numpy.array([Iminus(pps,Tfun,Tg) for pps in ppsL])
    flux = Ip-Im
    heat = -2.*phys.sigma*TList**4 + (Ip+Im)
    #
    #Re-dimensionalize heating to K/day
    if PressureBroadening :
        kappa = 2.*tauInf*g/(100.*pList[-1])
    else:
        kappa = tauInf*g/(100.*pList[-1])
    heat = heat*(kappa/cp)*24.*3600. #K/day
    return flux,heat

#These are all globals
tauInf = 10.
PressureBroadening = True

#Gravity and cp (for diminensionalization)
g = 10.
cp = 1000.
