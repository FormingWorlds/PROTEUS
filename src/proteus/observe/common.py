from __future__ import annotations

import os
import pandas as pd

OBS_SOURCES=("outgas", "profile", "offchem")
OBS_INSTRUMENTS=("synthesis", )

def get_transit_fpath(outdir:str, instr:str, source:str):
    '''Get the path to the transit spectrum file.'''
    return os.path.join(outdir, "observe", f"transit_{source}_{instr}.csv")

def get_eclipse_fpath(outdir:str, instr:str, source:str):
    '''Get the path to the eclipse spectrum file.'''
    return os.path.join(outdir, "observe", f"eclipse_{source}_{instr}.csv")

def read_transit(outdir:str, instr:str, source:str):
    '''Read the transit spectrum file.'''
    fpath = get_transit_fpath(outdir, instr, source)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Transit spectrum file '{fpath}' not found.")
    return pd.read_csv(fpath, sep=r"\s+")

def read_eclipse(outdir:str, instr:str, source:str):
    '''Read the eclipse spectrum file.'''
    fpath = get_eclipse_fpath(outdir, instr, source)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Eclipse spectrum file '{fpath}' not found.")
    return pd.read_csv(fpath, sep=r"\s+")

