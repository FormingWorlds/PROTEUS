from __future__ import annotations

import os


def get_transit_fpath(outdir:str):
    return os.path.join(outdir, "data", "obs_synth_transit.csv")

def get_eclipse_fpath(outdir:str):
    return os.path.join(outdir, "data", "obs_synth_eclipse.csv")
