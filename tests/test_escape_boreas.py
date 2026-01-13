from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import pytest

from proteus.config import read_config_object
from proteus.escape.boreas import BOREAS_ELEMS, BOREAS_GASES, _set_boreas_params, run_boreas
from proteus.utils.constants import AU, M_earth, R_earth, element_list
from proteus.utils.helper import get_proteus_dir

log = logging.getLogger(__name__)

def _make_minimal_hf_row() -> dict:
    """Construct `hf_row` with inventories for escape checks."""
    hf_row: dict = {}

    # Orbital/stellar derived quantities
    hf_row["semimajorax"] = 0.05 * AU  # meters

    # Atmospheric state used by escape
    hf_row["T_obs"] = 400.0            # K
    hf_row["F_xuv"] = 10.0             # W m^-2
    hf_row["R_obs"] = 1.0 * R_earth    # m
    hf_row["M_planet"] = 1.0 * M_earth # kg

    # Supply VMRs at the XUV level for all BOREAS-supported gases
    for g in BOREAS_GASES:
        hf_row[f"{g}_vmr_xuv"] = 0.0
    hf_row["H2_vmr_xuv"] = 0.5
    hf_row["CO_vmr_xuv"] = 0.2
    hf_row["H2S_vmr_xuv"] = 0.2
    hf_row["N2_vmr_xuv"] = 0.1

    # Provide bulk/atmospheric elemental inventories used by calc_unfract_fluxes
    # Oxygen is skipped by calc_unfract_fluxes; others are used to form ratios
    # Values are arbitrary but positive and distinct
    for e in element_list:
        hf_row[f"{e}_kg_atm"] = 0.0
        hf_row[f"{e}_kg_total"] = 0.0
    masses_atm = {
        "H": 1.0e17,
        "C": 2.0e17,
        "N": 3.0e17,
        "S": 4.0e17,
    }
    for e, m in masses_atm.items():
        hf_row[f"{e}_kg_atm"] = m
    masses_bulk = {
        "H": 5.0e20,
        "C": 2.5e20,
        "N": 1.0e20,
        "S": 0.5e20,
    }
    for e, m in masses_bulk.items():
        hf_row[f"{e}_kg_total"] = m

    return hf_row


@pytest.fixture(scope="module")
def boreas_config() -> object:
    """Load the demo BOREAS TOML into a validated Config object."""
    config_path = Path("input/demos/boreas.toml")
    cfg = read_config_object(config_path)
    # Ensure module selection
    cfg.escape.module = "boreas"
    return cfg

def test_boreas_params(boreas_config):
    hf_row = _make_minimal_hf_row()

    # make parameters object
    params = _set_boreas_params(boreas_config, hf_row)

    # check it is setup property
    assert params.FXUV == hf_row["F_xuv"] * 1e3
    assert params.albedo == 0.0

def test_boreas_frac_updates_outputs(boreas_config):
    hf_row = _make_minimal_hf_row()
    dirs = {'output':os.path.join(get_proteus_dir(),"output")}

    # Use fractionation mode
    boreas_config.escape.boreas.fractionate = True

    # Execute
    log.debug(boreas_config.escape)
    log.debug(hf_row)
    run_boreas(boreas_config, hf_row, dirs)

    # Bulk outputs exist and have sensible values
    assert "esc_rate_total" in hf_row
    assert "cs_xuv" in hf_row
    assert "R_xuv" in hf_row
    assert "p_xuv" in hf_row

    assert isinstance(hf_row["esc_rate_total"], float) and hf_row["esc_rate_total"] >= 0.0
    assert isinstance(hf_row["cs_xuv"], float) and hf_row["cs_xuv"] > 0.0
    assert isinstance(hf_row["R_xuv"], float) and hf_row["R_xuv"] > 0.0
    assert hf_row["p_xuv"] == 0.0

    # Elemental escape rates keys present; supported elements non-negative
    for e in element_list:
        key = f"esc_rate_{e}"
        # All elements receive a key; unsupported in BOREAS default to 0.0
        assert key in hf_row
        if e in BOREAS_ELEMS:
            assert isinstance(hf_row[key], float) and hf_row[key] >= 0.0
        else:
            assert hf_row[key] == 0.0


def test_boreas_unfrac_conserves_mass_ratios(boreas_config):
    hf_row = _make_minimal_hf_row()
    dirs = {'output':os.path.join(get_proteus_dir(),"output")}

    # Provide a pre-computed total escape rate to split
    # The model will compute bulk escape first; we just validate the splitting
    boreas_config.escape.boreas.fractionate = False

    # Execute
    run_boreas(boreas_config, hf_row, dirs)

    # Mass mixing ratios from the chosen reservoir
    masses = {e: hf_row[f"{e}_kg_atm"] for e in ("H", "C", "N", "S")}
    total = sum(masses.values())
    assert total > 0.0

    # Check conservation of element mass ratios in the unfractionated split
    for e, m in masses.items():
        expected = hf_row["esc_rate_total"] * (m / total)
        assert math.isfinite(expected)
        assert hf_row[f"esc_rate_{e}"] == pytest.approx(expected, rel=1e-12, abs=1e-15)
