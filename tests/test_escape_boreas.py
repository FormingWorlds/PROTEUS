from __future__ import annotations

import math
from pathlib import Path

import pytest

from proteus.config import read_config_object
from proteus.escape.boreas import run_boreas, BOREAS_GASES, BOREAS_ELEMS
from proteus.utils.constants import AU, element_list, R_earth, M_earth


def _make_minimal_hf_row_fractionating() -> dict:
    """Construct a minimal `hf_row` dict suitable for running BOREAS.

    Provides required keys and reasonable physical values.
    """
    hf_row: dict = {}

    # Orbital/stellar derived quantities
    hf_row["semimajorax"] = 0.05 * AU  # meters

    # Atmospheric state used by escape
    hf_row["T_obs"] = 400.0            # K
    hf_row["F_xuv"] = 10.0             # W m^-2
    hf_row["R_obs"] = 1.0 * R_earth    # m
    hf_row["M_planet"] = 1.0 * M_earth # kg

    # Supply VMRs at the XUV level for all BOREAS-supported gases
    # Ensure at least one non-zero so MMW is finite
    # Prefer H2 dominance if available, else H2O/CO2 fallback
    dominant_set = ("H2", "H2O", "CO2")
    dominant = next((g for g in dominant_set if g in BOREAS_GASES), None)
    for g in BOREAS_GASES:
        hf_row[f"{g}_vmr_xuv"] = 0.0
    if dominant is not None:
        hf_row[f"{dominant}_vmr_xuv"] = 0.9
    # Small contribution from a second gas if present to avoid edge singularities
    second = next((g for g in ("CO2", "H2O", "N2") if g in BOREAS_GASES and g != dominant), None)
    if second is not None:
        hf_row[f"{second}_vmr_xuv"] = 0.1

    return hf_row


def _make_minimal_hf_row_unfractionated() -> dict:
    """Construct `hf_row` with inventories for unfractionated escape checks."""
    hf_row = _make_minimal_hf_row_fractionating()

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


def test_run_boreas_fractionating_updates_outputs(boreas_config):
    hf_row = _make_minimal_hf_row_fractionating()
    dirs = {}

    # Use fractionation mode
    boreas_config.escape.boreas.fractionate = True

    # Execute
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


@pytest.mark.parametrize("reservoir", ["outgas", "bulk"])  # validate both paths
def test_run_boreas_unfractionated_conserves_mass_ratios(boreas_config, reservoir):
    hf_row = _make_minimal_hf_row_unfractionated()
    dirs = {}

    # Provide a pre-computed total escape rate to split
    # The model will compute bulk escape first; we just validate the splitting
    boreas_config.escape.boreas.fractionate = False
    boreas_config.escape.reservoir = reservoir

    # Execute
    run_boreas(boreas_config, hf_row, dirs)

    # Mass mixing ratios from the chosen reservoir
    key = "_kg_atm" if reservoir == "outgas" else "_kg_total"
    masses = {e: hf_row[f"{e}{key}"] for e in ("H", "C", "N", "S")}
    total = sum(masses.values())
    assert total > 0.0

    # Oxygen is intentionally skipped in calc_unfract_fluxes
    assert f"esc_rate_O" not in hf_row or hf_row.get("esc_rate_O", 0.0) == 0.0

    # Check conservation of element mass ratios in the unfractionated split
    for e, m in masses.items():
        expected = hf_row["esc_rate_total"] * (m / total)
        assert math.isfinite(expected)
        assert hf_row[f"esc_rate_{e}"] == pytest.approx(expected, rel=1e-12, abs=0.0)
