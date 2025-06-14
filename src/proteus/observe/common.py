from __future__ import annotations

import os

import pandas as pd

OBS_SOURCES=("outgas", "profile", "offchem")

def get_transit_fpath(outdir:str, source:str, stage:str):
    '''Get the path to the transit spectrum file.'''
    return os.path.join(outdir, "observe", f"transit_{source}_{stage}.csv")

def get_eclipse_fpath(outdir:str, source:str, stage:str):
    '''Get the path to the eclipse spectrum file.'''
    return os.path.join(outdir, "observe", f"eclipse_{source}_{stage}.csv")

def read_transit(outdir:str, source:str, stage:str):
    '''Read the transit spectrum file.'''
    fpath = get_transit_fpath(outdir, source, stage)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Transit spectrum file '{fpath}' not found.")
    return pd.read_csv(fpath, sep=r"\s+")

def read_eclipse(outdir:str, source:str, stage:str):
    '''Read the eclipse spectrum file.'''
    fpath = get_eclipse_fpath(outdir, source, stage)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Eclipse spectrum file '{fpath}' not found.")
    return pd.read_csv(fpath, sep=r"\s+")
