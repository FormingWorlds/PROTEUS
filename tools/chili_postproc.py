#!/usr/bin/env python3

# Postprocess output into CHILI-MIP format

# Import modules
from __future__ import annotations

import os
import sys
from glob import glob
from shutil import copyfile, rmtree

import matplotlib as mpl

mpl.use("Agg") # noqa
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import tomllib

from proteus.config import read_config_object
from proteus.utils.constants import R_earth, vol_list
from proteus.utils.plot import get_colour, latexify

# Target times for sampling profile [log years]
tau_target = [3, 4, 5, 7, 9]

# Potential planet names
pl_names = {
    "tr1b" : "trappist1b",
    "tr1e":  "trappist1e",
    "tr1a":  "trappist1alpha",
    "earth": "earth",
    "venus": "venus",
}

def postproc_once(simdir:str, plot:bool=True):
    '''Run postprocessing steps for a single simulation'''
    print(f"    simulation dir: {simdir}")

    # Determine file name
    name = "PLANET"
    for k in pl_names.keys():
        if f"chili_{k}" in simdir:
            name = pl_names[k]

    # Make chili folder
    chilidir = os.path.join(simdir, "chili") + "/"
    if os.path.isdir(chilidir):
        rmtree(chilidir)
    os.mkdir(chilidir)

    # Read simulation helpfile
    hfpath = os.path.join(simdir, "runtime_helpfile.csv")
    if not os.path.isfile(hfpath):
        raise FileNotFoundError(f"Cannot find {hfpath}")
    hf_all = pd.read_csv(hfpath, delimiter=r'\s+')

    # Copy config
    print("    copy config file")
    config_path = os.path.join(simdir, "init_coupler.toml")
    copyfile(config_path, os.path.join(chilidir,f"evolution-proteus-{name}-config.in"))

    # Read config
    config = read_config_object(config_path)

    # Write to expected format
    out = {}
    out["t(yr)"]            = np.array(hf_all["Time"].iloc[:])
    out["T_surf(K)"]        = np.array(hf_all["T_surf"].iloc[:])
    out["T_pot(K)"]         = np.array(hf_all["T_pot"].iloc[:])
    out["phi(vol_frac)"]    = np.array(hf_all["Phi_global_vol"].iloc[:])
    out["massC_solid(kg)"]  = np.array(hf_all["C_kg_solid"].iloc[:])
    out["massC_melt(kg)"]   = np.array(hf_all["C_kg_liquid"].iloc[:])
    out["massC_atm(kg)"]    = np.array(hf_all["C_kg_atm"].iloc[:])
    out["massH_solid(kg)"]  = np.array(hf_all["H_kg_solid"].iloc[:])
    out["massH_melt(kg)"]   = np.array(hf_all["H_kg_liquid"].iloc[:])
    out["massH_atm(kg)"]    = np.array(hf_all["H_kg_atm"].iloc[:])
    out["massO_atm(kg)"]    = np.array(hf_all["O_kg_atm"].iloc[:])
    out["mmw(kg/mol)"]      = np.array(hf_all["atm_kg_per_mol"].iloc[:])
    out["p_surf(bar)"]      = np.array(hf_all["P_surf"].iloc[:])
    for gas in vol_list:
        out[f"p_{gas}(bar)"]       = np.array(hf_all[f"{gas}_bar"].iloc[:])
    out["fO2_liquid(bar)"]  = out["p_O2(bar)"]
    out["fO2_solid(bar)"]   = np.zeros_like(out["t(yr)"])
    out["R_trans(m)"]       = np.array(hf_all["R_obs"].iloc[:])
    out["flux_surf(W/m2)"]  = np.array(hf_all["F_int"].iloc[:])
    out["flux_OLR(W/m2)"]   = np.array(hf_all["F_olr"].iloc[:])
    out["flux_ASR(W/m2)"]   = np.array(hf_all["F_ins"].iloc[:]) * config.orbit.s0_factor * (1-config.atmos_clim.albedo_pl)
    out["thick_surf_bl(m)"] = np.ones_like(out["t(yr)"]) * config.atmos_clim.surface_d


    # Write to scalar CSV
    print("    write CSV file for scalars")
    outpath = os.path.join(chilidir, f"evolution-proteus-{name}-data.csv")
    pd.DataFrame(out).to_csv(outpath, sep=',', index=False, float_format="%.10e")

    # Write TPZ profiles at required times
    print("    write CSV files for profiles")
    for tau in tau_target:
        tau_time = 10**tau
        iclose = np.argmin(np.abs(tau_time - out["t(yr)"]))
        tclose = out["t(yr)"][iclose]
        if (tclose > tau_time*10) or (tclose < tau_time/10):
            print(f"      tau{tau} -> too far from target time, skipping")
            continue
        else:
            print(f"      tau{tau} -> {tclose:.2e} yr")

        ncfile = os.path.join(simdir, "data", f"{tclose:.0f}_atm.nc")
        with nc.Dataset(ncfile) as ds:
            tarr = np.array(ds.variables["tmp"][:])
            parr = np.array(ds.variables["p"][:]  ) * 1e-5
            zarr = np.array(ds.variables["r"][:]  ) - np.amin(ds.variables["r"])
        X = np.array([tarr, parr, zarr]).T
        H = "T(K),p(bar),z(m)"
        csvname = f"evolution-proteus-{name}-tau{tau}-data.csv"
        f = os.path.join(chilidir,csvname)
        np.savetxt(f, X, fmt="%.10e", delimiter=',', header=H)


    # Make plot...
    if plot:
        print("    make plot")
        fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=False)

        # Panel 1 : Surface Temperature
        axes[0, 0].plot(out["t(yr)"], out["T_surf(K)"], linewidth=2)
        axes[0, 0].set_ylabel(r'$\rm T_{surf}$ [K]', fontsize=14)
        axes[0, 0].tick_params(axis='both', labelsize=14)
        axes[0, 0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(500))

        # Panel 2 : Partial pressures
        for g in vol_list:
            axes[0, 1].plot(out["t(yr)"], out[f"p_{g}(bar)"], label=latexify(g), color=get_colour(g), linewidth=2)
        axes[0, 1].set_ylabel(r'$\rm p_{i}$ [bar]', fontsize=14)
        axes[0, 1].set_yscale('symlog', linthresh=1e-6)
        axes[0, 1].tick_params(axis='both', labelsize=14)
        axes[0, 1].legend(loc='best', fontsize=10)

        # Panel 3 : Transit radius
        axes[0, 2].plot(out["t(yr)"], out["R_trans(m)"]/R_earth, linewidth=2)
        axes[0, 2].set_ylabel(r'$\rm R_{trans}$ [R$_{\oplus}$]', fontsize=14)
        axes[0, 2].tick_params(axis='both', labelsize=14)

        # Panel 4 : Potential temperature
        axes[1, 0].plot(out["t(yr)"], out["T_pot(K)"],  linewidth=2)
        axes[1, 0].set_ylabel(r'$\rm T_{pot}$ [K]', fontsize=14)
        axes[1, 0].tick_params(axis='both', labelsize=14)
        axes[1, 0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(500))

        # Panel 5 : Melt fraction
        axes[1, 1].plot(out["t(yr)"], out["phi(vol_frac)"],  linewidth=2)
        axes[1, 1].set_ylabel(r'$\rm \phi$ [volume frac]', fontsize=14)
        axes[1, 1].tick_params(axis='both', labelsize=14)

        # Panel 6 : Mass of volatile in solid and melt
        m_unit = 1e16
        axes[1, 2].plot(out["t(yr)"], out["massC_solid(kg)"]/m_unit, label=r'$\rm C_{solid}$', linewidth=2, color=get_colour('C'))
        axes[1, 2].plot(out["t(yr)"], out["massC_melt(kg)"]/m_unit,  label=r'$\rm C_{melt}$',  linewidth=2, linestyle='--', color=get_colour('C'))
        axes[1, 2].plot(out["t(yr)"], out["massH_solid(kg)"]/m_unit, label=r'$\rm H_{solid}$', linewidth=2, color=get_colour('H'))
        axes[1, 2].plot(out["t(yr)"], out["massH_melt(kg)"]/m_unit,  label=r'$\rm H_{melt}$',  linewidth=2, linestyle='--', color=get_colour('H'))
        axes[1, 2].set_ylabel(r'$m_i$ [$10^{16}$ kg]', fontsize=14)
        axes[1, 2].set_yscale('symlog')
        axes[1, 2].tick_params(axis='both', labelsize=14)
        axes[1, 2].legend(loc='best', fontsize=10)

        # Horizontal lines at Tsurf = 300 K and 1000 K
        axes[0, 0].axhline(300, color='k', linestyle='--', linewidth=1)
        axes[0, 0].text(4.5e2, 0.92*300, r'$\rm T_{surf}$ = 300 K', va='top', ha='left', fontsize=12)
        axes[0, 0].axhline(1000, color='k', linestyle='--', linewidth=1)
        axes[0, 0].text(4.5e2, 0.95*1000, r'$\rm T_{surf}$ = 1000 K', va='top', ha='left', fontsize=12)

        # Horizontal lines at Phi = 0.5 and Phi = 0.01
        axes[1, 1].axhline(0.5, color='k', linestyle='--', linewidth=1)
        axes[1, 1].text(4.5e1, 0.98*0.5, r'$\rm \phi$ = 0.5', va='top', ha='left', fontsize=12)
        axes[1, 1].axhline(0.01, color='k', linestyle='--', linewidth=1)
        axes[1, 1].text(4.5e1, 0.5*0.01, r'$\rm \phi$ = 0.01', va='top', ha='left', fontsize=12)

        # Extra things for plots
        for ax in axes.flatten():

            # reference times
            for tau in tau_target:
                ax.axvline(10**tau, color='silver', linestyle='--', linewidth=1)

            # set scales
            ax.set_xscale('log')
            ax.set_xlabel('Time [yr]', fontsize=14)
            ax.set_xlim(left=10)

        # Save the figure
        fig.suptitle(f'CHILI-MIP, PROTEUS simulation of {name}', fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(chilidir, "chili.pdf"), dpi=300, bbox_inches='tight')

    return name

def postproc_grid(griddir:str):
    '''Postprocess a grid of models'''

    print(f"Postprocessing cases in {griddir}")

    chilidir = os.path.join(griddir, "chili") + "/"
    if os.path.isdir(chilidir):
        rmtree(chilidir)
    os.mkdir(chilidir)

    # Identify cases in grid
    cases = glob(griddir+"/case_*")
    if len(cases) > 0:
        print(f"Found {len(cases)} cases")
    else:
        raise RuntimeError(f"Cannot find cases in {griddir}")

    # Read grid config to work out which cases are high/mid/low in volatiles
    with open(griddir+"/copy.grid.toml", 'rb') as hdl:
        gridtoml = tomllib.load(hdl)
    arr_Hkg = np.unique(gridtoml["delivery.elements.H_kg"]["values"])
    arr_Ckg = np.unique(gridtoml["delivery.elements.C_kg"]["values"])

    # Run them
    N = len(cases)
    for i,c in enumerate(cases):
        print(f"[{i+1:2d}/{N:-2d}]  case:{c.split("_")[-1]}")
        name = postproc_once(c, plot=False)

        # Copy files to common folder
        print("    copy files")
        with open(c+"/init_coupler.toml", 'rb') as hdl:
            thistoml = tomllib.load(hdl)
            Hkg = float(thistoml["delivery"]["elements"]["H_kg"])
            Ckg = float(thistoml["delivery"]["elements"]["C_kg"])
        for fold in glob(c+"/chili/evolution-proteus*"):
            # new file name
            fnew = os.path.basename(fold).split("-")

            # insert "grid-"
            fnew.insert(3,"grid")

            # insert Hkg identifier
            if np.isclose(Hkg, np.amin(arr_Hkg)):
                fnew.insert(4,"Hlow")
            elif np.isclose(Hkg, np.amax(arr_Hkg)):
                fnew.insert(4,"Hhigh")
            else:
                fnew.insert(4,"Hmid")

            # insert Ckg identifier
            if np.isclose(Ckg, np.amin(arr_Ckg)):
                fnew.insert(5,"Clow")
            elif np.isclose(Hkg, np.amax(arr_Ckg)):
                fnew.insert(5,"Chigh")
            else:
                fnew.insert(5,"Cmid")

            fnew = "-".join(fnew)
            fnew = os.path.join(chilidir, fnew)

            copyfile(fold, fnew)

        print("-----------------------------")
        print(" ")

    # Make overview plot of all grid cases
    print("Make overview plot")
    fig,ax = plt.subplots(1,1, figsize=(7,5))
    for fpath in glob(chilidir+"evolution*.csv"):
        if "tau" in fpath:
            continue
        hf_all = pd.read_csv(fpath, delimiter=',')
        x = np.array(hf_all["t(yr)"].iloc[:])
        y = np.array(hf_all["T_surf(K)"].iloc[:])
        l = os.path.basename(fpath).split(".")[0]
        l = " ".join(l.split("-")[4:6])
        ax.plot(x,y,label=l, lw=1.5, alpha=0.8, zorder=4)
    ax.set(xscale='log', xlabel='Time [yr]', ylabel=r'$\rm T_{surf}$ [K]')
    ax.set_title(f'CHILI-MIP, PROTEUS simulation of {name}', fontsize=16)
    ax.set_xlim(left=10.0)
    ax.grid(zorder=-2, alpha=0.8)
    ax.legend(fontsize=10)
    fig.savefig(os.path.join(chilidir,"chili_grid.pdf"), bbox_inches='tight')

    print(" ")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Must provide path to folder")

    argdir = os.path.abspath(sys.argv[1])

    # Folder exists?
    if not os.path.isdir(argdir):
        raise FileNotFoundError(f"Cannot find folder '{argdir}'")

    # Grid or single case?
    if os.path.isdir(argdir+"/case_000000/"):
        print("Identified folder as a grid of models")
        postproc_grid(argdir)
    else:
        print("Identified folder as a single run")
        postproc_once(argdir)
    print("Done")
