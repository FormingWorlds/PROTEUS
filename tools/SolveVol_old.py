#!/usr/bin/env python3

# Written by Dan Bower. Adapted by Harrison Nicholls.
# See the related issue on the PROTEUS GitHub page:-
# https://github.com/FormingWorlds/PROTEUS/issues/42
# Paper to cite:-
# https://www.sciencedirect.com/science/article/pii/S0012821X22005301

import argparse
import csv
import logging
import numpy as np
import pprint
from itertools import product
from scipy.optimize import fsolve

# Solve for the equilibrium chemistry of a magma ocean atmosphere
# for a given set of solubility and redox relations

#====================================================================
def get_global_parameters():
    """Global parameters in SI units"""

    global_d = {}
    global_d['mantle_mass'] = 4.208261222595111e+24 # kg
    global_d['mantle_melt_fraction'] = 1.0 # fraction of mantle that is molten
    global_d['little_g'] = 9.81 # m/s^2
    global_d['planetary_radius'] = 6371000.0 # m
    global_d['temperature'] = 2000.0 # K
    # add molar mass dict
    global_d['molar_mass_d'] = get_molar_masses()
    return global_d

#====================================================================
def get_molar_masses():
    """Molar masses of atoms and molecules in kg/mol"""

    mass_d = {}
    # atoms
    # below are all given in g/mol, then converted before return
    # at end of this function
    mass_d['H'] = 1.0079
    mass_d['O'] = 15.9994
    mass_d['C'] = 12.0107
    mass_d['N'] = 14.0067
    # molecules
    mass_d['H2'] = mass_d['H']*2
    mass_d['H2O'] = mass_d['H2'] + mass_d['O']
    mass_d['CO'] = mass_d['C'] + mass_d['O']
    mass_d['CO2'] = mass_d['C'] + 2*mass_d['O']
    mass_d['CH4'] = mass_d['C'] + 4*mass_d['H']
    mass_d['N2'] = 2*mass_d['N']
    # convert all to kg/mol
    mass_d = {k:v*1E-3 for k, v in mass_d.items()}
    return mass_d

#====================================================================
class OxygenFugacity:
    """log10 oxygen fugacity as a function of temperature"""

    def __init__(self, model='oneill'):
        self.callmodel = getattr(self, model)

        self.o_coeff_2 = np.log(10)*8.31441

    def __call__(self, T, fO2_shift=0):
        '''Return log10 fO2'''
        return self.callmodel(T) + fO2_shift

    def fischer(self, T):
        '''Fischer et al. (2011) IW'''
        return 6.94059 -28.1808*1E3/T

    def oneill(self, T): 
        '''O'Neill and Eggin (2002) IW'''
        return 2*(-244118+115.559*T-8.474*T*np.log(T))/(self.o_coeff_2*T)

#====================================================================
class ModifiedKeq:
    """Modified equilibrium constant (includes fO2)"""

    def __init__(self, Keq_model, fO2_model='oneill'):
        self.fO2 = OxygenFugacity(fO2_model)
        self.callmodel = getattr(self, Keq_model)

    def __call__(self, T, fO2_shift=0):
        fO2 = self.fO2(T, fO2_shift)
        Keq, fO2_stoich = self.callmodel(T)
        Geq = 10**(Keq-fO2_stoich*fO2)
        return Geq

    def schaefer_CH4(self, T): 
        '''Schaefer log10Keq for CO2 + 2H2 = CH4 + fO2'''
        # second argument returns stoichiometry of O2
        return (-16276/T - 5.4738, 1)

    def schaefer_C(self, T): 
        '''Schaefer log10Keq for CO2 = CO + 0.5 fO2'''
        return (-14787/T + 4.5472, 0.5) 

    def schaefer_H(self, T): 
        '''Schaefer log10Keq for H2O = H2 + 0.5 fO2'''
        return (-12794/T + 2.7768, 0.5) 

    def janaf_C(self, T): 
        '''JANAF log10Keq, 1500 < K < 3000 for CO2 = CO + 0.5 fO2'''
        return (-14467.511400133637/T + 4.348135473316284, 0.5) 

    def janaf_H(self, T): 
        '''JANAF log10Keq, 1500 < K < 3000 for H2O = H2 + 0.5 fO2'''
        return (-13152.477779978302/T + 3.038586383273608, 0.5) 

#====================================================================
class Solubility:
    """Solubility base class.  All p in bar"""

    def __init__(self, composition):
        self.callmodel = getattr(self, composition)

    def power_law(self, p, const, exponent):
        return const*p**exponent

    def __call__(self, p, *args):
        '''Dissolved concentration in ppmw in the melt'''
        return self.callmodel(p, *args)

#====================================================================
class SolubilityH2O(Solubility):
    """H2O solubility models"""

    # below default gives the default model used
    def __init__(self, composition='peridotite'):
        super().__init__(composition)

    def anorthite_diopside(self, p):
        '''Newcombe et al. (2017)'''
        return self.power_law(p, 727, 0.5)

    def peridotite(self, p):
        '''Sossi et al. (2022)'''
        return self.power_law(p, 534, 0.5)

    def basalt_dixon(self, p):
        '''Dixon et al. (1995) refit by Paolo Sossi'''
        return self.power_law(p, 965, 0.5)

    def basalt_wilson(self, p):
        '''Hamilton (1964) and Wilson and Head (1981)'''
        return self.power_law(p, 215, 0.7)

    def lunar_glass(self, p):
        '''Newcombe et al. (2017)'''
        return self.power_law(p, 683, 0.5)

#====================================================================
class SolubilityCH4(Solubility):
    """CH4 solubility models"""

    def __init__(self, composition='basalt_ardia'):
        super().__init__(composition)

    def basalt_ardia(self, p, p_total):
        '''Ardia 2013'''
        p_total *= 1e-4  # Convert to GPa
        p *= 1e-4 # Convert to GPa
        ppmw = p*np.exp(4.93 - (0.000193 * p_total))
        return ppmw
    
#====================================================================
class SolubilityCO(Solubility):
    """CO solubility models"""

    def __init__(self, composition='mafic_armstrong'):
        super().__init__(composition)

    def mafic_armstrong(self, p, p_total):
        '''Armstrong 2015'''
        ppmw = 10 ** (-0.738 + 0.876 * np.log10(p) - 5.44e-5 * p_total)
        return ppmw

#====================================================================
class SolubilityCO2(Solubility):
    """CO2 solubility models"""

    def __init__(self, composition='basalt_dixon'):
        super().__init__(composition)

    def basalt_dixon(self, p, temp):
        '''Dixon et al. (1995)'''
        ppmw = (3.8E-7)*p*np.exp(-23*(p-1)/(83.15*temp))
        ppmw = 1.0E4*(4400*ppmw) / (36.6-44*ppmw)
        return ppmw

#====================================================================
class SolubilityN2(Solubility):
    """N2 solubility models"""

    def __init__(self, composition='libourel'):
        super().__init__(composition)

        # melt composition
        x_SiO2  = 0.56
        x_Al2O3 = 0.11
        x_TiO2  = 0.01
        self.dasfac_2 = np.exp(4.67 + 7.11*x_SiO2 - 13.06*x_Al2O3 - 120.67*x_TiO2)

    def libourel(self, p):
        '''Libourel et al. (2003)'''
        ppmw = self.power_law(p, 0.0611, 1.0)
        return ppmw
    
    def dasgupta(self, p, ptot, temp, fO2_shift):
        '''Dasgupta et al. (2022)'''
        
        # convert bar to GPa
        pb_N2  = p * 1.0e-4  
        pb_tot = ptot * 1.0e-4

        pb_tot = max(pb_tot, 1e-15)

        # calculate N2 concentration in melt
        ppmw  = pb_N2**0.5 * np.exp(5908.0 * pb_tot**0.5/temp - 1.6*fO2_shift)
        ppmw += pb_N2 * self.dasfac_2

        return ppmw 


#====================================================================
def get_partial_pressures(pin, fO2_shift, global_d):
    """Partial pressure of all considered species"""

    # we only need to know pH2O, pCO2, and pN2, since reduced species
    # can be directly determined from equilibrium chemistry

    pH2O, pCO2, pN2 = pin

    # return results in dict, to be explicit about which pressure
    # corresponds to which volatile
    p_d = {}
    p_d['H2O'] = pH2O
    p_d['CO2'] = pCO2
    p_d['N2'] = pN2

    # pH2 from equilibrium chemistry
    gamma = ModifiedKeq('janaf_H')
    gamma = gamma(global_d['temperature'], fO2_shift)
    p_d['H2'] = gamma*pH2O

    # pCO from equilibrium chemistry
    gamma = ModifiedKeq('janaf_C')
    gamma = gamma(global_d['temperature'], fO2_shift)
    p_d['CO'] = gamma*pCO2

    gamma = ModifiedKeq('schaefer_CH4')
    gamma = gamma(global_d['temperature'], fO2_shift)
    p_d['CH4'] = gamma*pCO2*p_d['H2']**2.0

    return p_d

#====================================================================
def get_total_pressure(pin, fO2_shift, global_d):
    """Sum partial pressures to get total pressure"""

    p_d = get_partial_pressures(pin, fO2_shift, global_d)
    ptot = sum(p_d.values())

    return ptot

#====================================================================
def atmosphere_mass(pin, fO2_shift, global_d):
    """Atmospheric mass of volatiles and totals for H, C, O"""

    mass_d = global_d['molar_mass_d']
    p_d = get_partial_pressures(pin, fO2_shift, global_d)
    mu_atm = atmosphere_mean_molar_mass(pin, fO2_shift, global_d)

    mass_atm_d = {}
    for key, value in p_d.items():
        mass_atm_d[key] = value*1.0E5/global_d['little_g']
        mass_atm_d[key] *= 4.0*np.pi*global_d['planetary_radius']**2.0
        mass_atm_d[key] *= mass_d[key]/mu_atm

    # total mass of H
    mass_atm_d['H'] = mass_atm_d['H2'] / mass_d['H2']
    mass_atm_d['H'] += mass_atm_d['H2O'] / mass_d['H2O']
    # note factor 2 below to account for stoichiometry
    mass_atm_d['H'] += mass_atm_d['CH4'] * 2 / mass_d['CH4']
    # below converts moles of H2 to mass of H
    mass_atm_d['H'] *= mass_d['H2']

    # total mass of C
    mass_atm_d['C'] = mass_atm_d['CO'] / mass_d['CO']
    mass_atm_d['C'] += mass_atm_d['CO2'] / mass_d['CO2']
    mass_atm_d['C'] += mass_atm_d['CH4'] / mass_d['CH4']
    mass_atm_d['C'] *= mass_d['C']

    # total mass of N
    mass_atm_d['N'] = mass_atm_d['N2'] 

    # total mass of O
    mass_atm_d['O'] = 0.0
    mass_atm_d['O'] += mass_atm_d['H2O'] / mass_d['H2O']
    mass_atm_d['O'] += mass_atm_d['CO'] / mass_d['CO']
    mass_atm_d['O'] += mass_atm_d['CO2'] / mass_d['CO2'] * 2.0
    mass_atm_d['O'] *= mass_d['O']

    return mass_atm_d

#====================================================================
def atmosphere_mean_molar_mass(pin, fO2_shift, global_d):
    """Mean molar mass of the atmosphere"""

    mass_d = global_d['molar_mass_d']

    p_d = get_partial_pressures(pin, fO2_shift, global_d)
    ptot = get_total_pressure(pin, fO2_shift, global_d)

    mu_atm = 0
    for key, value in p_d.items():
        mu_atm += mass_d[key]*value
    mu_atm /= ptot

    return mu_atm

#====================================================================
def dissolved_mass(pin, fO2_shift, global_d):
    """Volatile masses in the (molten) mantle"""

    mass_int_d = {}

    p_d = get_partial_pressures(pin, fO2_shift, global_d)
    ptot = get_total_pressure(pin, fO2_shift, global_d)

    invalid = (ptot != ptot) or (ptot < 0.0)

    prefactor = 1E-6*global_d['mantle_mass']*global_d['mantle_melt_fraction']
    mass_d = global_d['molar_mass_d']

    # H2O
    sol_H2O = SolubilityH2O() # gets the default solubility model
    ppmw_H2O = sol_H2O(p_d['H2O'])
    mass_int_d['H2O'] = prefactor*ppmw_H2O

    # CO
    sol_CO = SolubilityCO() # gets the default solubility model
    ppmw_CO = sol_CO(p_d["CO"], ptot)
    mass_int_d['CO'] = prefactor*ppmw_CO

    # CH4
    sol_CH4 = SolubilityCH4() # gets the default solubility model
    ppmw_CH4 = sol_CH4(p_d["CH4"], ptot)
    mass_int_d['CH4'] = prefactor*ppmw_CH4

    # CO2
    sol_CO2 = SolubilityCO2() # gets the default solubility model
    ppmw_CO2 = sol_CO2(p_d['CO2'], global_d['temperature'])
    mass_int_d['CO2'] = prefactor*ppmw_CO2

    # N2
    # sol_N2 = SolubilityN2("libourel")  # libourel model without fO2 dependence
    # ppmw_N2 = sol_N2(p_d['N2'])
    # mass_int_d['N2'] = prefactor*ppmw_N2

    # N2
    sol_N2 = SolubilityN2("dasgupta") # calculate fO2-dependent solubility
    ppmw_N2 = sol_N2(p_d['N2'], ptot, global_d['temperature'], fO2_shift)
    mass_int_d['N2'] = prefactor*ppmw_N2

    # now get totals of H, C, N
    mass_int_d['H'] = mass_int_d['H2O']/mass_d['H2O'] + mass_int_d['CH4']*2/mass_d["CH4"]
    mass_int_d['H'] *= mass_d['H2']
    
    mass_int_d['C'] = mass_int_d['CO2']/mass_d['CO2'] + mass_int_d['CO']/mass_d['CO'] + mass_int_d['CH4']/mass_d['CH4']
    mass_int_d['C'] *= mass_d['C']

    mass_int_d['N'] = mass_int_d['N2']

    return mass_int_d

#====================================================================
def func(pin, fO2_shift, global_d, mass_target_d):
    """Function to compute the residual of the mass balance"""

    # pin has three pressures in bar expressed as a tuple
    pH2O, pCO2, pN2 = pin

    # dict of molar masses
    mass_d = global_d['molar_mass_d']

    # get atmospheric masses
    mass_atm_d = atmosphere_mass(pin, fO2_shift, global_d)

    # get (molten) mantle masses
    mass_int_d = dissolved_mass(pin, fO2_shift, global_d)

    # compute residuals
    res_l = []
    for vol in ['H','C','N']:
        # absolute residual
        res = mass_atm_d[vol] + mass_int_d[vol] - mass_target_d[vol]
        # if target is not zero, compute relative residual
        # otherwise, zero target is already solved with zero pressures
        if mass_target_d[vol]:
            res /= mass_target_d[vol]
        res_l.append(res)

    return res_l

#====================================================================
def get_log_rand(rng):
    r = np.random.uniform(low=rng[0], high=rng[1])
    return 10.0**r

def get_initial_pressures(target_d, log=False):
    """Get initial guesses of partial pressures"""

    # all in bar
    if log:
        cH2O = [-7 , +5]  # range in log10 units
        cCO2 = [-8 , +5]
        cN2  = [-10, +5]

        pH2O = get_log_rand(cH2O)
        pCO2 = get_log_rand(cCO2)
        pN2  = get_log_rand(cN2 )
    else:
        pH2O = np.random.uniform(low=1.0e-12, high=1.0)
        pCO2 = np.random.uniform(low=1.0e-12, high=0.9)
        pN2  = np.random.uniform(low=1.0e-12, high=0.5)

    if target_d['H'] == 0:
        pH2O = 0
    if target_d['C'] == 0:
        pCO2 = 0
    if target_d['N'] == 0:
        pN2 = 0

    return pH2O, pCO2, pN2

#====================================================================
def equilibrium_atmosphere(Hydrogen, CH_ratio, fO2_shift, global_d, Nitrogen):
    """Calculate equilibrium chemistry of the atmosphere"""

    H_kg = Hydrogen * 1.0E-6 * global_d['mantle_mass']
    C_kg = CH_ratio * H_kg
    N_kg = Nitrogen * 1.0E-6 * global_d['mantle_mass']
    target_d = {'H': H_kg, 'C': C_kg, 'N': N_kg}

    count = 0
    ier = 0
    count_warn = 5e3
    warned = False
    # could in principle result in an infinite loop, if randomising
    # the ic never finds the physical solution (but in practice,
    # this doesn't seem to happen)
    while ier != 1:

        x0 = get_initial_pressures(target_d, log=True)
        # solve for mass of volatiles in atm and int by finding a solution
        # where the total mass is conserved.
        sol, info, ier, msg = fsolve(func, x0, args=(fO2_shift, global_d, target_d), full_output=True, maxfev=40)
        count += 1

        # sometimes, a solution exists with negative pressures, which
        # is clearly non-physical.  Here, assert we must have positive
        # pressures.
        if any(sol<0):
            # if any negative pressures, report ier!=1
            ier = 0

        if (count > count_warn) and not warned:
            warned = True
            print("    mantle = %g earth masses" % float(global_d['mantle_mass'] / 5.972e24))
            print("    H_ppm  =", Hydrogen)
            print("    C/H    =", CH_ratio)
            print("    fO2-IW =", fO2_shift)
            print("    N_ppm  =", Nitrogen)
            print("    Tsurf  =", global_d['temperature'])
    
    print("    n_fev",info["nfev"],"   n_ini",count)

    p_d = get_partial_pressures(sol, fO2_shift, global_d)
    if warned:
        print(p_d)

    # get residuals for output
    res_l = func(sol, fO2_shift, global_d, target_d)

    # for convenience, add inputs to same dict
    p_d['Hydrogen_ppm'] = Hydrogen
    p_d['CH_ratio'] = CH_ratio
    p_d['fO2_shift'] = fO2_shift
    p_d['Nitrogen_ppm'] = Nitrogen 

    # add total masses of elements
    for k in target_d:
        p_d[k+"_tot_kg"] = target_d[k]

    for key in global_d.keys():
        if key == 'molar_mass_d':
            continue
        p_d[key] = global_d[key]

    ptot = 0.0
    for key in p_d.keys():
        if key in ["H2O","CO2","N2","H2","CO","CH4"]:
            ptot += p_d[key]
    p_d["tot"] = ptot

    atm_tot_mass = 0.0
    for k in ["H2O","CO2","N2","H2","CO","CH4"]:
        # mmr in atmosphere
        p_d[k+"_atm_vmr"] = p_d[k]/p_d["tot"]
        # kg in atmosphere
        this_kg = p_d[k]*1.0E5 * 4.0*np.pi*global_d['planetary_radius']**2.0 / global_d['little_g'] # 1.0E5 because pressures are in bar
        atm_tot_mass += this_kg 
        p_d[k+"_atm_kg"] = this_kg
       
    p_d["tot_atm_kg"] = atm_tot_mass

    atm_masses = atmosphere_mass( (p_d["H2O"], p_d["CO2"], p_d["N2"]) ,fO2_shift, global_d )
    for e in ["H","C","N","O"]:
        p_d[k+"_atm_kg"] = atm_masses[e]
        if e in target_d:
            p_d["%s_atm_kg/%s_tot_kg"%(e,e)] = atm_masses[e]/target_d[e]
            p_d["%s_atm_kg/tot_atm_kg"%(e)]  = atm_masses[e]/atm_tot_mass
    
    # for debugging/checking, add success initial condition
    # that resulted in a converged solution with positive pressures
    # p_d['pH2O_0'] = x0[0]
    # p_d['pCO2_0'] = x0[1]
    # p_d['pN2_0'] = x0[2]

    # also for debugging/checking, report residuals
    p_d['res_H'] = res_l[0]
    p_d['res_C'] = res_l[1]
    p_d['res_N'] = res_l[2]

    return p_d

#====================================================================
def equilibrium_atmosphere_MC():
    """Monte Carlo"""

    global_d = get_global_parameters()

    NN = 2300

    # Samples
    hydrogen_l =        np.random.uniform(5.0,  60,     NN)
    CH_ratio_l =        np.random.uniform(0.1,  1,      NN)
    fO2_shift_l =       np.random.uniform(-5,   3,      NN)
    nitrogen_l =        np.random.uniform(0.5,  9.0,    NN)
    mantle_l =          np.random.uniform(0.2,  5.0,    NN) * global_d['mantle_mass']
    tsurf_l =           np.random.uniform(1500, 3000,   NN)
    # Or,
    # Constant
    # hydrogen_l =        np.ones(NN) * 36.0
    # CH_ratio_l =        np.ones(NN) * 1.0
    # fO2_shift_l =       np.ones(NN) * 2.0 
    # nitrogen_l =        np.ones(NN) * 2.01  # primitive mantle
    # mantle_l =          np.ones(NN) * global_d['mantle_mass']
    # tsurf_l =           np.ones(NN) * 2500.0

    out_l = []

    for ii in range(NN):
        logging.info(f'Simulation number= {ii}')

        Hydrogen = hydrogen_l[ii]
        CH_ratio = CH_ratio_l[ii]
        fO2_shift = fO2_shift_l[ii]
        Nitrogen = nitrogen_l[ii]
        global_d['mantle_mass'] = mantle_l[ii]
        global_d['temperature'] = tsurf_l[ii]

        p_d = equilibrium_atmosphere(Hydrogen, CH_ratio, fO2_shift, global_d, Nitrogen)
        out_l.append(p_d)

    print("Writing results")
    filename = 'equilibrium_atmosphere_MC.csv'
    write_output(filename, out_l)

#====================================================================
def equilibrium_atmosphere_GR():
    """Gridded run"""

    global_d = get_global_parameters()

    
    pl_m = 8.63 * 5.972e24 # kg, known k2-18b mass

    rho = 5515.0 # earth value for interior density [kg m-3]
    pl_r = ( (3 * pl_m) / (4 * np.pi * rho))**(1.0/3)  # get radius at surface, assuming atmosphere mass is small

    global_d['little_g'] = 6.67408e-11 * pl_m / ( pl_r * pl_r )
    global_d['planetary_radius'] = pl_r
    global_d['planetary_mass'] = pl_m

    print(global_d)
    
    # N_ppm calculation..
    # Z    = metallicity
    # A(E) = 12 + log10(n_E / n_H)
    # -> m_N/m_H = mu_N/mu_H * n_N/n_H

    Z   = 1.0

    N_to_H = (14.0/1.0) * 10**(7.83 - 12.0) # mass ratio to hydrogen from Asplund 2009
    C_to_H = (12.0/1.0) * 10**(8.43 - 12.0) # ^


    # Samples
    hydrogen_l =        np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
    CH_ratio_l =        np.array([0.01, 0.05, 0.1, 1.0, 10.0, 100.0]) * C_to_H
    fO2_shift_l =       np.array([-5.0, -2.0, 0.0, 2.0, 4.0])
    mantle_l =          np.array([0.001, 0.01, 0.1, 1.0]) * pl_m
    tsurf_l =           np.array([1500.0, 2000.0, 2500.0, 3000.0])

    # CH_ratio_l =        np.array([0.05, 0.1]) * C_to_H
    # fO2_shift_l =       np.array([-2.0, 0.0])
    # mantle_l =          np.array([0.01, 0.1]) * pl_m
    # tsurf_l =           np.array([2000.0, 2500.0])

    out_l = []

    i = 0
    prod   = product(hydrogen_l, CH_ratio_l, fO2_shift_l, mantle_l, tsurf_l)
    pspace = np.array(list(p for p in prod))
    psize  = len(pspace)

    modprint = 10
    weird_idx = []
    for p in pspace:
        i += 1
        if i%modprint == 0:
            print('Simulation %04d/%04d = %2.2f%%' % (i,psize, i/psize*100.0))

        Hydrogen, CH_ratio, fO2_shift, mm, mt = p
        
        Nitrogen = Hydrogen * N_to_H

        global_d['mantle_mass'] = mm
        global_d['temperature'] = mt

        p_d = equilibrium_atmosphere(Hydrogen, CH_ratio, fO2_shift, global_d, Nitrogen)
        p_d["invalid"] = 0
        for k in ["H","C","N"]:
            if abs(p_d["res_"+k]) > 0.5:
                weird_idx.append(i)
                p_d["invalid"] = 1
                break
                
        out_l.append(p_d)

    print("Invalid cases: " + str(weird_idx))    

    print("Writing results")
    filename = 'equilibrium_atmosphere_GR.csv'
    write_output(filename, out_l)

#==================================================== ================
def write_output(filename, out_l):
    """Write output (list of dictionaries) to a CSV"""

    fieldnames = list(out_l[0].keys())
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(out_l)

#====================================================================
def main():

    # logging
    FORMAT = "[%(levelname)s:%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logging.info('Started')

    # parsing arguments
    parser = argparse.ArgumentParser(description='Melt-vapour equilibrium atmosphere')
    parser.add_argument('-f','--fo2_shift',     help='fO2 shift in log10 units relative to IW', action='store', type=float, default=0)
    parser.add_argument('-c','--ch_ratio',      help='C/H ratio by wt %', action='store', type=float, default=1)
    parser.add_argument('-x','--hydrogen',      help='Hydrogen concentration ppmw', action='store', type=float, default=1)
    parser.add_argument('-m','--monte_carlo',   help='Run Monte Carlo simulation', action='store_true')
    parser.add_argument('-g','--grid',          help='Run grid of simulations', action='store_true')
    parser.add_argument('-n','--nitrogen',      help='Nitrogen concentration ppmw', action='store', type=float, default = 2.8) # 2.8 is the mantle value of N in ppmw

    args = parser.parse_args()
    kwargs = vars(args)

    if args.monte_carlo:
        print("Running monte-carlo simulation")
        equilibrium_atmosphere_MC()
    elif args.grid:
        print("Running gridded simulation")
        equilibrium_atmosphere_GR()
    else:
        global_d = get_global_parameters()
        Hydrogen = kwargs['hydrogen']
        CH_ratio = kwargs['ch_ratio']
        fO2_shift = kwargs['fo2_shift']
        Nitrogen = kwargs['nitrogen']
        p_d = equilibrium_atmosphere(Hydrogen, CH_ratio, fO2_shift, global_d, Nitrogen)
        print('Output')
        pprint.pprint(p_d)

#====================================================================
if __name__ == '__main__':

    main()
