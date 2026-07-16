"""
Tests for proteus.outgas.lavatmos.compute_silicate_outgassing

These tests mock out external LavAtmos / FastChem interactions by:
- monkeypatching run_lavatmos to produce a minimal boa_chem.dat,
- monkeypatching read_in_element_fracs_normalized to return a controlled element fraction dict,
- replacing species_lib, vol_list and vap_list with small controlled values.

They assert that hf_row is updated appropriately in the happy path, and that
the function raises when the FastChem output directory is missing.
"""

from __future__ import annotations

import math
import os

import pytest

import proteus.outgas.lavatmos as lavatmos_mod


def _make_minimal_hf_row():
    # Provide required hf_row keys used by compute_silicate_outgassing.
    hf_row = {
        # volatile element masses in kg (initial)
        'H_kg_atm': 1.0,
        'C_kg_atm': 1.0,
        'N_kg_atm': 0.0,
        'S_kg_atm': 0.0,
        'O_kg_atm': 0.0,
        # surface properties
        'P_surf': 1.0,  # bar
        'T_magma': 2000.0,  # K
        'atm_kg_per_mol': 0.029,  # kg/mol (29 g/mol)
        # previous atmosphere mass: set >0 so branch uses ratio method
        'M_atm': 1e18,
        'R_int': 6.0e6,
        # book-keeping
        'M_vaps': 0.0,
    }
    return hf_row


def test_compute_silicate_outgassing_happy_path(tmp_path, monkeypatch):
    """
    Happy path:
    - monkeypatch run_lavatmos to write a minimal boa_chem.dat into the fastchem output dir
    - monkeypatch read_in_element_fracs_normalized to return a deterministic element fraction dict
    - set species_lib, vol_list and vap_list to small controlled sets
    - run compute_silicate_outgassing and assert hf_row was updated
    """
    # Prepare a temporary output dir and dirs mapping
    outdir = tmp_path / 'out'
    outdir.mkdir()
    dirs = {'output': str(outdir)}

    # Minimal hf_row
    hf_row = _make_minimal_hf_row()

    # Monkeypatch module vol/vap lists to simple values
    monkeypatch.setattr(lavatmos_mod, 'vol_list', ['CO2'])
    monkeypatch.setattr(lavatmos_mod, 'vap_list', ['H2O'])

    # Construct a minimal species_lib with weights in g/mol (matching the module expectation)
    class DummySpecies:
        def __init__(self, name, fc_name, weight):
            self.name = name
            self.fc_name = fc_name
            self.weight = weight

    # Use common molar masses (g/mol)
    species_lib = {
        'H': DummySpecies('H', 'H', 1.0),
        'C': DummySpecies('C', 'C', 12.0),
        'N': DummySpecies('N', 'N', 14.0),
        'S': DummySpecies('S', 'S', 32.0),
        'O': DummySpecies('O', 'O', 16.0),
        'CO2': DummySpecies('CO2', 'C1O2', 44.0),
        'H2O': DummySpecies('H2O', 'H2O1', 18.0),
        # include O2 for fO2 calc
        'O2': DummySpecies('O2', 'O2', 32.0),
    }
    monkeypatch.setattr(lavatmos_mod, 'species_lib', species_lib)

    # Ensure hf_row has placeholder solid/liquid/mol keys for vol/vap species
    for vol in ['CO2', 'H2O']:
        hf_row.setdefault(vol + '_kg_solid', 0.0)
        hf_row.setdefault(vol + '_kg_liquid', 0.0)
        hf_row.setdefault(vol + '_mol_solid', 0.0)
        hf_row.setdefault(vol + '_mol_liquid', 0.0)

    # Monkeypatch read_in_element_fracs_normalized to a simple distribution (Fe and Mg only)
    element_fracs = {'Fe': 0.7, 'Mg': 0.3, 'O': 0.0}
    monkeypatch.setattr(
        lavatmos_mod,
        'read_in_element_fracs_normalized',
        lambda path: element_fracs,
    )

    # Monkeypatch run_lavatmos to write a minimal boa_chem.dat into fastchem output dir
    def fake_run_lavatmos(dirs_arg, config, hf_row_arg, nfrac, first_iter):
        paths = lavatmos_mod.paths_importer(dirs_arg)
        # paths_importer already created fastchem3_output directory; just write boa_chem.dat
        boa_path = os.path.join(paths.fastchem3_output, 'boa_chem.dat')
        # Create a minimal whitespace-delimited table with Pbar, mu and fc species columns
        with open(boa_path, 'w') as f:
            # columns: Pbar mu <CO2 fc name> <H2O fc name> <O2>
            f.write('Pbar mu C1O2 H2O1 O2\n')
            # Provide values: total pressure 2.0 bar, mu ~ 0.018 kg/mol, vmrs for C1O2 and H2O, small O2
            f.write('2.0 0.018 0.5 0.4999 1e-6\n')

    monkeypatch.setattr(lavatmos_mod, 'run_lavatmos', fake_run_lavatmos)

    # Run the function under test
    lavatmos_mod.compute_silicate_outgassing(dirs, config=None, hf_row=hf_row, first_iter=True)

    # Assertions: check that surface pressure was updated to 2.0 (from boa_chem)
    assert math.isclose(hf_row['P_surf'], 2.0, rel_tol=1e-12)

    # Check that vmr fields were set for CO2 and H2O
    assert 'CO2_vmr' in hf_row and pytest.approx(hf_row['CO2_vmr'], rel=1e-6) == 0.5
    assert 'H2O_vmr' in hf_row and pytest.approx(hf_row['H2O_vmr'], rel=1e-6) == 0.4999

    # Check that kg_atm entries were created and positive
    assert hf_row['CO2_kg_atm'] > 0.0
    assert hf_row['H2O_kg_atm'] > 0.0

    # Check that oxygen fugacity fields were added
    assert 'log10_fO2_vapourise' in hf_row
    assert 'log10_fO2_shift_vapourise' in hf_row

    # mmw logging value present (atm_kg_per_mol should have been unchanged)
    assert 'atm_kg_per_mol' in hf_row


def test_compute_silicate_outgassing_missing_fastchem_output_raises(tmp_path, monkeypatch):
    """
    If the FastChem output directory does not exist (or has no boa_chem.dat),
    compute_silicate_outgassing should call UpdateStatusfile and raise RuntimeError.
    """
    outdir = tmp_path / 'out_missing'
    outdir.mkdir()
    dirs = {'output': str(outdir)}

    hf_row = _make_minimal_hf_row()

    # Keep vol/vap minimal and species_lib present
    monkeypatch.setattr(lavatmos_mod, 'vol_list', ['CO2'])
    monkeypatch.setattr(lavatmos_mod, 'vap_list', ['H2O'])

    class DummySpecies:
        def __init__(self, name, fc_name, weight):
            self.name = name
            self.fc_name = fc_name
            self.weight = weight

    species_lib = {
        'H': DummySpecies('H', 'H', 1.0),
        'C': DummySpecies('C', 'C', 12.0),
        'N': DummySpecies('N', 'N', 14.0),
        'S': DummySpecies('S', 'S', 32.0),
        'O': DummySpecies('O', 'O', 16.0),
        'CO2': DummySpecies('CO2', 'C1O2', 44.0),
        'H2O': DummySpecies('H2O', 'H2O1', 18.0),
    }
    monkeypatch.setattr(lavatmos_mod, 'species_lib', species_lib)

    # Monkeypatch run_lavatmos to NOT create the fastchem output directory or file
    def fake_run_lavatmos_no_output(dirs_arg, config, hf_row_arg, nfrac, first_iter):
        # Intentionally do nothing (no boa_chem.dat)
        return

    monkeypatch.setattr(lavatmos_mod, 'run_lavatmos', fake_run_lavatmos_no_output)

    # Track UpdateStatusfile calls
    called = {'status': False}

    def fake_updatestatus(dirs_arg, code):
        called['status'] = True
        return None

    monkeypatch.setattr(lavatmos_mod, 'UpdateStatusfile', fake_updatestatus)

    # Expect a RuntimeError due to missing fastchem output
    with pytest.raises(RuntimeError):
        lavatmos_mod.compute_silicate_outgassing(
            dirs, config=None, hf_row=hf_row, first_iter=True
        )

    # And UpdateStatusfile should have been called
    assert called['status'] is True
