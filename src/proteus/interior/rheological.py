# Rheological model from Kervazo et al. (2021)
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.special import erf

@dataclass
class rheo_t():
    dotl:float
    delta:float
    xi:float
    gamma:float
    phist:float

B_ein = 2.5

# Lookup parameters for rheological properties
# Taken from https://doi.org/10.1051/0004-6361/202039433
par_visc  = rheo_t(1.0,  25.7, 1.17e-9, 5.0, 0.569)
par_shear = rheo_t(10.0, 2.10, 7.08e-7, 5.0, 0.597)
par_bulk  = rheo_t(1e9,  2.62, 0.102,   5.0, 0.712)

# Evalulate big Phi at a given layer
def _bigphi(phi:float, par:rheo_t):
    return (1.0-phi)/(1.0-par.phist)

# Evalulate big F at a given layer
def _bigf(phi:float, par:rheo_t):
    numer = np.pi**0.5 * _bigphi(phi, par) * (1.0 + _bigphi(phi, par)**par.gamma)
    denom = 2.0 * (1.0 - par.xi)
    return (1.0-par.xi) * erf(numer/denom)

# Evalulate rheological parameter at a given layer
def eval_rheoparam(phi:float, which:str, phicrit:float):
    match which:
        case 'visc':
            par = par_visc
        case 'shear':
            par = par_shear
        case 'bulk':
            par = par_bulk
        case _:
            raise ValueError(f"Invalid rheological parameter 'f{which}'")
    # Update critical melt fraction
    par.phist = phicrit
    # Evaluate parameter
    numer = 1.0 + _bigphi(phi, par)**par.delta
    denom = (1.0-_bigf(phi, par)) ** (B_ein*(1-par.phist))
    return par.dotl * numer / denom




