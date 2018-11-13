#===================================================================
#Computes fluxes and heating rates for the grey gas model.
#The fluxes are computed as a function of p/ps, given net optical
#thickness of the atmosphere tauinf .
#
#Modified to return dimensional heat and flux for use with
#the time-stepping radiative-convective model

#MDH 10/11/18
#Modified to use Socrates
#===================================================================



import math,phys
from ClimateUtilities import *
from SocRadModel import radCompSoc
import numpy as np


#Pressure in mb, for consistency with ccmrad
def radcomp(pList,TList,Tg,q):

    # Socrates heating

    hrtssw,hrtslw,uflxlw,nflxsw = radCompSoc(pList,TList,Tg)
    heat = np.squeeze(np.sum(hrtssw[:,:],axis=0) + np.sum(hrtslw[:,:],axis=0))
    flux = np.sum(uflxlw[:,:],axis=0)


    
    #
    #Re-dimensionalize heating to K/day
    #if PressureBroadening :
    #    kappa = 2.*tauInf*g/(100.*pList[-1])
    #else:
    #    kappa = tauInf*g/(100.*pList[-1])
    #heat = heat*(kappa/cp)*24.*3600. #K/day
    return flux,heat

#These are all globals
tauInf = 10.
PressureBroadening = True

#Gravity and cp (for diminensionalization)
g = 10.
cp = 1000.







