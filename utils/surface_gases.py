from utils.modules_ext import *
from utils.constants import *
from utils.helper import *

log = logging.getLogger("PROTEUS")

# Solve partial pressures functions
# Originally formulated by Dan Bower
# See the related issue on the PROTEUS GitHub page:-
# https://github.com/FormingWorlds/PROTEUS/issues/42
# Paper to cite:-
# https://www.sciencedirect.com/science/article/pii/S0012821X22005301

# Solve for the equilibrium chemistry of a magma ocean atmosphere
# for a given set of solubility and redox relations

class OxygenFugacity:
    """log10 oxygen fugacity as a function of temperature"""

    def __init__(self, model='oneill'):
        self.callmodel = getattr(self, model)

    def __call__(self, T, fO2_shift=0):
        '''Return log10 fO2'''
        return self.callmodel(T) + fO2_shift

    def fischer(self, T):
        '''Fischer et al. (2011) IW'''
        return 6.94059 -28.1808*1E3/T

    def oneill(self, T): 
        '''O'Neill and Eggins (2002) IW'''
        return 2*(-244118+115.559*T-8.474*T*np.log(T))/(np.log(10)*8.31441*T)

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
    
    def janaf_S(self, T): 
        # JANAF log10Keq, 900 < K < 2000 for 0.5 S2 + O2 = SO2
        # https://doi.org/10.1016/j.gca.2022.08.032
        return (18887.0/T - 3.8064, 1) 

class Solubility:
    """Solubility base class.  All p in bar"""

    def __init__(self, composition):
        self.callmodel = getattr(self, composition)

    def power_law(self, p, const, exponent):
        return const*p**exponent

    def __call__(self, p, *args):
        '''Dissolved concentration in ppmw in the melt'''
        return self.callmodel(p, *args)
    
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
        return self.power_law(p, 524, 0.5)

    def basalt_dixon(self, p):
        '''Dixon et al. (1995) refit by Paolo Sossi'''
        return self.power_law(p, 965, 0.5)

    def basalt_wilson(self, p):
        '''Hamilton (1964) and Wilson and Head (1981)'''
        return self.power_law(p, 215, 0.7)

    def lunar_glass(self, p):
        '''Newcombe et al. (2017)'''
        return self.power_law(p, 683, 0.5)


class SolubilityS2(Solubility):
    """S2 solubility models"""

    # below default gives the default model used
    def __init__(self, composition='gaillard'):
        self.fO2_model = OxygenFugacity()
        super().__init__(composition)

    def gaillard(self, p, temp, fO2_shift):
        # Gaillard et al., 2022
        # https://doi.org/10.1016/j.epsl.2021.117255
        # https://ars.els-cdn.com/content/image/1-s2.0-S0012821X21005112-mmc1.pdf

        if p < 1.0e-20:
            return 0.0

        # melt composition [wt%]
        x_FeO  = 10.0

        # calculate fO2 [bar]
        fO2 = 10**self.fO2_model(temp, fO2_shift)

        # calculate log(Ss)
        out = 13.8426 - 26.476e3/temp + 0.124*x_FeO + 0.5*np.log(p/fO2)
        
        # convert to concentration ppmw
        out = np.exp(out) #* 10000.0 

        return out


class SolubilityCO2(Solubility):
    """CO2 solubility models"""

    def __init__(self, composition='basalt_dixon'):
        super().__init__(composition)

    def basalt_dixon(self, p, temp):
        '''Dixon et al. (1995)'''
        ppmw = (3.8E-7)*p*np.exp(-23*(p-1)/(83.15*temp))
        ppmw = 1.0E4*(4400*ppmw) / (36.6-44*ppmw)
        return ppmw


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
    

class SolubilityCO(Solubility):
    """CO solubility models"""

    def __init__(self, composition='mafic_armstrong'):
        super().__init__(composition)

    def mafic_armstrong(self, p, p_total):
        '''Armstrong 2015'''
        ppmw = 10 ** (-0.738 + 0.876 * np.log10(p) - 5.44e-5 * p_total)
        return ppmw

def is_included(gas, COUPLER_options):
    return bool(COUPLER_options[gas+"_included"]>0)

def solvevol_get_partial_pressures(pin, COUPLER_options):
    """Partial pressure of all considered species from oxidised species"""

    # we only need to know pH2O, pCO2, and pN2, since reduced species
    # can be directly determined from equilibrium chemistry

    fO2_shift = COUPLER_options["fO2_shift_IW"]

    # return results in dict, to be explicit about which pressure
    # corresponds to which volatile
    p_d = {}

    # H2O
    p_d['H2O'] = pin['H2O']

    # H2
    if is_included("H2", COUPLER_options):
        gamma = ModifiedKeq('janaf_H')
        gamma = gamma(COUPLER_options['T_outgas'], fO2_shift)
        p_d['H2'] = gamma*pin["H2O"]

    # CO2
    p_d['CO2'] = pin['CO2']

    # CO
    if is_included("CO", COUPLER_options):
        gamma = ModifiedKeq('janaf_C')
        gamma = gamma(COUPLER_options['T_outgas'], fO2_shift)
        p_d['CO'] = gamma*pin["CO2"]
    
    # CH4
    if is_included("H2",COUPLER_options) and is_included("CH4",COUPLER_options):
        gamma = ModifiedKeq('schaefer_CH4')
        gamma = gamma(COUPLER_options['T_outgas'], fO2_shift)
        p_d['CH4'] = gamma*pin["CO2"]*p_d['H2']**2.0

    # N2
    p_d['N2']  = pin['N2']

    # O2 
    fO2_model = OxygenFugacity() 
    p_d['O2'] = 10.0**fO2_model(COUPLER_options['T_outgas'], fO2_shift)

    # S2 
    p_d['S2'] = pin['S2']

    # SO2
    if is_included("SO2", COUPLER_options):
        gamma = ModifiedKeq('janaf_S')
        gamma = gamma(COUPLER_options['T_outgas'], fO2_shift)
        p_d['SO2']  = (gamma*pin['S2']*p_d['O2']**2)**0.5

    return p_d


def calc_mantle_mass(COUPLER_options):
    # Get core's average density using Earth values
    earth_fr = 0.55     # earth core radius fraction
    earth_fm = 0.325    # earth core mass fraction  (https://arxiv.org/pdf/1708.08718.pdf)
    earth_m  = 5.97e24  # kg
    earth_r  = 6.37e6   # m

    core_rho = (3.0 * earth_fm * earth_m) / (4.0 * np.pi * ( earth_fr * earth_r )**3.0 )  # core density [kg m-3]
    log.info("Estimating core density to be %g kg m-3" % core_rho)

    # Calculate mantle mass by subtracting core from total
    core_mass = core_rho * 4.0/3.0 * np.pi * (COUPLER_options["radius"] * COUPLER_options["planet_coresize"] )**3.0
    mantle_mass = COUPLER_options["mass"] - core_mass 
    log.debug("Total mantle mass is %.2e kg" % mantle_mass)
    if (mantle_mass <= 0.0):
        UpdateStatusfile(dirs, 20)
        raise Exception("Something has gone wrong (mantle mass is negative)")
    
    return mantle_mass


def solvevol_get_total_pressure(pin, COUPLER_options):
    """Sum partial pressures to get total pressure"""

    p_d = solvevol_get_partial_pressures(pin, COUPLER_options)
    ptot = sum(p_d.values())

    return ptot

def solvevol_atmosphere_mean_molar_mass(pin, COUPLER_options):
    """Mean molar mass of the atmosphere"""

    p_d = solvevol_get_partial_pressures(pin, COUPLER_options)
    ptot = solvevol_get_total_pressure(pin, COUPLER_options)

    mu_atm = 0
    for key, value in p_d.items():
        mu_atm += molar_mass[key]*value
    mu_atm /= ptot

    return mu_atm

def solvevol_atmosphere_mass(pin, COUPLER_options):
    """Atmospheric mass of volatiles and totals for H, C, and N"""

    p_d = solvevol_get_partial_pressures(pin, COUPLER_options)
    mu_atm = solvevol_atmosphere_mean_molar_mass(pin, COUPLER_options)

    mass_atm_d = {}
    for key, value in p_d.items():
        # 1.0E5 because pressures are in bar
        mass_atm_d[key] = value*1.0E5/COUPLER_options['gravity']
        mass_atm_d[key] *= 4.0*np.pi*COUPLER_options['radius']**2.0
        mass_atm_d[key] *= molar_mass[key]/mu_atm

    # total mass of H
    mass_atm_d['H'] = 2*mass_atm_d['H2O'] / molar_mass['H2O']
    if is_included('H2', COUPLER_options):
        mass_atm_d['H'] += 2*mass_atm_d['H2'] / molar_mass['H2']
    if is_included('CH4', COUPLER_options):
        mass_atm_d['H'] += 4*mass_atm_d['CH4'] / molar_mass['CH4']    # note factor 4 to account for stoichiometry
    # below converts moles of H2 to mass of H
    mass_atm_d['H'] *= molar_mass['H']

    # total mass of C
    mass_atm_d['C'] = mass_atm_d['CO2'] / molar_mass['CO2']
    if is_included("CO", COUPLER_options):
        mass_atm_d['C'] += mass_atm_d['CO'] / molar_mass['CO']
    if is_included("CH4", COUPLER_options):
        mass_atm_d['C'] += mass_atm_d['CH4'] / molar_mass['CH4']
    # below converts moles of C to mass of C
    mass_atm_d['C'] *= molar_mass['C']

    # total mass of N
    mass_atm_d['N'] = mass_atm_d['N2']

    # total mass of O
    mass_atm_d['O'] = mass_atm_d['H2O'] / molar_mass['H2O']
    if is_included("CO", COUPLER_options):
        mass_atm_d['O'] += mass_atm_d['CO'] / molar_mass['CO']
    if is_included("CO2", COUPLER_options):
        mass_atm_d['O'] += mass_atm_d['CO2'] / molar_mass['CO2'] * 2.0
    if is_included("SO2", COUPLER_options):
        mass_atm_d['O'] += mass_atm_d['SO2'] / molar_mass['SO2'] * 2.0
    # below converts moles of O to mass of O
    mass_atm_d['O'] *= molar_mass['O']

    # total mass of S 
    mass_atm_d['S'] = mass_atm_d['S2'] / molar_mass['S2']
    if is_included("SO2", COUPLER_options):
        mass_atm_d['S'] += mass_atm_d['SO2'] / molar_mass['SO2']
    mass_atm_d['S'] *= molar_mass['S']

    return mass_atm_d



def solvevol_dissolved_mass(pin, COUPLER_options):
    """Volatile masses in the (molten) mantle"""

    mass_int_d = {}

    p_d = solvevol_get_partial_pressures(pin, COUPLER_options)
    ptot = solvevol_get_total_pressure(pin, COUPLER_options)

    prefactor = 1E-6*COUPLER_options['mantle_mass']*COUPLER_options['Phi_global']

    keys = list(p_d.keys())

    # H2O
    sol_H2O = SolubilityH2O() # gets the default solubility model
    ppmw_H2O = sol_H2O(p_d['H2O'])
    mass_int_d['H2O'] = prefactor*ppmw_H2O

    # CO2
    sol_CO2 = SolubilityCO2() # gets the default solubility model
    ppmw_CO2 = sol_CO2(p_d['CO2'], COUPLER_options['T_outgas'])
    mass_int_d['CO2'] = prefactor*ppmw_CO2

    # CO
    if is_included("CO", COUPLER_options):
        sol_CO = SolubilityCO() # gets the default solubility model
        ppmw_CO = sol_CO(p_d["CO"], ptot)
        mass_int_d['CO'] = prefactor*ppmw_CO

    # CH4
    if is_included("CH4", COUPLER_options):
        sol_CH4 = SolubilityCH4() # gets the default solubility model
        ppmw_CH4 = sol_CH4(p_d["CH4"], ptot)
        mass_int_d['CH4'] = prefactor*ppmw_CH4

    # N2
    sol_N2 = SolubilityN2("dasgupta") # calculate fO2-dependent solubility
    ppmw_N2 = sol_N2(p_d['N2'], ptot,  COUPLER_options['T_outgas'],  COUPLER_options['fO2_shift_IW'])
    mass_int_d['N2'] = prefactor*ppmw_N2

    # S2
    sol_S2 = SolubilityS2()
    ppmw_S2 = sol_S2(p_d["S2"], COUPLER_options["T_outgas"], COUPLER_options['fO2_shift_IW'])
    mass_int_d['S2'] = prefactor*ppmw_S2

    # now get totals of H, C, N, O, S
    mass_int_d['H'] = mass_int_d['H2O']*2/molar_mass['H2O'] 
    if is_included("CH4", COUPLER_options):
        mass_int_d['H'] += mass_int_d['CH4']*4/molar_mass["CH4"]
    mass_int_d['H'] *= molar_mass['H']
    
    mass_int_d['C'] = mass_int_d['CO2']/molar_mass['CO2']
    if is_included("CO", COUPLER_options):
        mass_int_d['C'] += mass_int_d['CO']/molar_mass['CO'] 
    if is_included("CH4", COUPLER_options):
        mass_int_d['C'] += mass_int_d['CH4']/molar_mass['CH4']
    mass_int_d['C'] *= molar_mass['C']

    mass_int_d['N'] = mass_int_d['N2']

    mass_int_d['O'] = mass_int_d['H2O'] / molar_mass['H2O']
    mass_int_d['O'] += mass_int_d['CO2'] / molar_mass['CO2'] * 2.0
    if is_included("CO", COUPLER_options):
        mass_int_d['O'] += mass_int_d['CO'] / molar_mass['CO']
    mass_int_d['O'] *= molar_mass['O']

    mass_int_d['S'] = mass_int_d['S2']

    return mass_int_d

def solvevol_func(pin_arr, COUPLER_options, mass_target_d):
    """Function to compute the residual of the mass balance given the partial pressures [bar]"""

    pin_dict = {
        "H2O" : pin_arr[0],
        "CO2" : pin_arr[1],
        "N2"  : pin_arr[2],
        "S2" : pin_arr[3]
    }

    # get atmospheric masses
    mass_atm_d = solvevol_atmosphere_mass(pin_dict, COUPLER_options)

    # get (molten) mantle masses
    mass_int_d = solvevol_dissolved_mass(pin_dict, COUPLER_options)
    
    # compute residuals
    res_l = []
    for vol in ['H','C','N','S']:
        # absolute residual
        res = mass_atm_d[vol] + mass_int_d[vol] - mass_target_d[vol]
        # if target is not zero, compute relative residual
        # otherwise, zero target is already solved with zero pressures
        if mass_target_d[vol]:
            res /= mass_target_d[vol]
        res_l.append(res)

    # Debug
    # H_kg = (2*mass_atm_d["H2O"]/molar_mass["H2O"] + 2*mass_atm_d["H2"]/molar_mass["H2"] + 4*mass_atm_d["CH4"]/molar_mass["CH4"]) *molar_mass['H']
    # C_kg = (mass_atm_d["CO2"]/molar_mass["CO2"] + mass_atm_d["CO"]/molar_mass["CO"] + mass_atm_d["CH4"]/molar_mass["CH4"]) * molar_mass['C']
    # N_kg = mass_atm_d["N2"]
    # print("Post:", H_kg, C_kg, N_kg)

    return res_l

def get_log_rand(rng):
    r = np.random.uniform(low=rng[0], high=rng[1])
    return 10.0**r

def solvevol_get_initial_pressures(target_d, log=True):
    """Get initial guesses of partial pressures"""

    # all in bar
    cH2O = [-7 , +5]  # range in log10 units
    cCO2 = [-8 , +5]
    cN2  = [-10, +5]
    cS2  = [-10, +5]

    pH2O = get_log_rand(cH2O)
    pCO2 = get_log_rand(cCO2)
    pN2  = get_log_rand(cN2 )
    pS2  = get_log_rand(cS2)

    if target_d['H'] == 0:
        pH2O = 0.0
    if target_d['C'] == 0:
        pCO2 = 0.0
    if target_d['N'] == 0:
        pN2  = 0.0
    if target_d['S'] == 0:
        pS2 = 0.0

    return pH2O, pCO2, pN2, pS2


def solvevol_get_target_from_params(COUPLER_options):

    N_ocean_moles = COUPLER_options['hydrogen_earth_oceans']
    CH_ratio =      COUPLER_options['CH_ratio']
    Nitrogen =      COUPLER_options['nitrogen_ppmw']
    Sulfur =        COUPLER_options['sulfur_ppmw']

    H_kg = N_ocean_moles * ocean_moles * molar_mass['H2']
    C_kg = CH_ratio * H_kg
    N_kg = Nitrogen * 1.0E-6 * COUPLER_options["mantle_mass"]
    S_kg = Sulfur * 1.0E-6 * COUPLER_options["mantle_mass"]
    target_d = {'H': H_kg, 'C': C_kg, 'N': N_kg, 'S': S_kg}
    return target_d

def solvevol_get_target_from_pressures(COUPLER_options):
    
    target_d = {}

    # store partial pressures for included gases
    pin_dict = {}
    for vol in volatile_species:
        if is_included(vol, COUPLER_options):
            pin_dict[vol] = COUPLER_options[vol+"_initial_bar"]

    # check total pressure 
    p_tot = np.sum(list(pin_dict.values()))
    if p_tot < 1.0e-3:
        raise Exception("Initial surface pressure too low! (%.2e bar)"%p_tot)

    # Check if no sulfur is present
    ptot_S = pin_dict["S2"]
    if is_included("SO2", COUPLER_options):
        ptot_S += pin_dict["SO2"]
    if ptot_S < 1.0e-20:
        target_d['S'] = 0.0

    # Check if no carbon is present
    ptot_C = pin_dict["CO2"]
    if is_included("CO", COUPLER_options):
        ptot_C += pin_dict["CO"]
    if ptot_C < 1.0e-20:
        target_d['C'] = 0.0

    # Check if no nitrogen is present
    ptot_N = pin_dict["N2"]
    if ptot_N < 1.0e-20:
        target_d['N'] = 0.0

    # get dissolved+atmosphere masses from partial pressures
    mass_atm_d = solvevol_atmosphere_mass(pin_dict, COUPLER_options)
    mass_int_d = solvevol_dissolved_mass(pin_dict, COUPLER_options)

    for vol in ['H','C','N','S']:
        if vol in target_d.keys():
            continue
        target_d[vol] = mass_atm_d[vol] + mass_int_d[vol]

    return target_d

def solvevol_equilibrium_atmosphere(target_d, COUPLER_options):
    """Solves for surface partial pressures assuming melt-vapour eqm


    Parameters
    ----------
        COUPLER_options : dict
            Dictionary of coupler options variables

    Returns
    ----------
        partial_pressures : dict
            Dictionary of: volatile partial pressures [Pa], and corresponding reservoir masses [kg]
    """


    log.info("Solving for equilibrium partial pressures at surface")
    log.debug("Target masses: %s"%str(target_d))

    count = 0
    max_attempts = 7000
    ier = 0
    # could in principle result in an infinite loop, if randomising
    # the ic never finds the physical solution (but in practice,
    # this doesn't seem to happen)
    while ier != 1:
        x0 = solvevol_get_initial_pressures(target_d)
        sol, info, ier, msg = fsolve(solvevol_func, x0, args=(COUPLER_options, target_d), full_output=True)
        count += 1
        
        # if any negative pressures, report ier!=1
        if any(sol<0):
            # sometimes, a solution exists with negative pressures, which is clearly non-physical.  Here, assert we must have positive pressures.
            ier = 0

        # check residuals
        this_resid = solvevol_func(sol, COUPLER_options, target_d)
        if np.amax(np.abs(this_resid)) > 1.0:
            ier = 0

        # give up after a while
        if count > max_attempts:
            raise Exception("Could not find solution for volatile abundances (max attempts reached)")

    log.info("Initial guess attempt number = %d" % count)
    log.info("Residuals: " + str(this_resid))

    sol_dict = {
        "H2O" : sol[0],
        "CO2" : sol[1],
        "N2"  : sol[2],
        "S2" : sol[3]
    }

    # Final partial pressures [bar]
    p_d        = solvevol_get_partial_pressures(sol_dict, COUPLER_options)

    # Final masses [kg]
    mass_atm_d = solvevol_atmosphere_mass(p_d, COUPLER_options)
    mass_int_d = solvevol_dissolved_mass(p_d, COUPLER_options)

    # Residuals [relative]
    res_l      = solvevol_func(sol, COUPLER_options, target_d)
    
    # Output dict 
    outdict = {"M_atm":0.0}

    # Initialise and store partial pressures 
    outdict["P_surf"] = 0.0
    for s in volatile_species:

        # Defaults
        outdict[s+"_atm_bar"]   = 0.0      # surface partial pressure [bar]
        outdict[s+"_atm_kg"]    = 0.0      # kg in atmosphere
        outdict[s+"_liquid_kg"] = 0.0      # kg in liquid
        outdict[s+"_solid_kg"]  = 0.0      # kg in solid (not handled here)
        outdict[s+"_total_kg"]  = 0.0      # kg (total)

        # Store partial pressures
        if s in p_d.keys():
            outdict[s+"_atm_bar"] = p_d[s]  # store as bar
            outdict["P_surf"] += outdict[s+"_atm_bar"]

    outdict["O2_atm_bar"] = p_d['O2']

    # Store VMRs (=mole fractions) and total atmosphere
    for s in volatile_species:
        outdict[s+"_mr"] = outdict[s+"_atm_bar"]/outdict["P_surf"]

        log.info("    solvevol: %s = %.1f bar (%.3f VMR)" % (s,outdict[s+"_atm_bar"], outdict[s+"_mr"])) 

    # Store masses of both gases and elements
    all = [s for s in volatile_species]
    all.extend(["H","C","N","O","S"])
    for s in all:
        tot_kg = 0.0

        if s in mass_atm_d.keys():
            outdict[s+"_atm_kg"] = mass_atm_d[s]
            tot_kg += mass_atm_d[s]

        if s in mass_int_d.keys():
            outdict[s+"_liquid_kg"] = mass_int_d[s]
            outdict[s+"_solid_kg"] = 0.0
            tot_kg += mass_int_d[s]

        outdict[s+"_total_kg"] = tot_kg

        if (s in mass_atm_d.keys()) and ( s in mass_int_d.keys()):
            outdict[s+"_outgassed_frac"] = outdict[s+"_atm_kg"]/ outdict[s+"_total_kg"]

    for s in volatile_species:
        outdict["M_atm"] += outdict[s+"_atm_kg"]
    
    # Store residuals 
    outdict["H_res"] = res_l[0]
    outdict["C_res"] = res_l[1]
    outdict["N_res"] = res_l[2]
    outdict["S_res"] = res_l[3]

    return outdict

