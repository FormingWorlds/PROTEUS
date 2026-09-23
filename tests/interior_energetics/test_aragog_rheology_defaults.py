"""Verify consistency between PROTEUS Rheology config and Aragog SolidRheologyParams.

Guarantees that the PROTEUS configuration schema for solid-state mantle rheology
matches the single parameter owner in Aragog (SolidRheologyParams) field-for-field
in names, defaults, and types.
"""

from __future__ import annotations

import dataclasses

import attrs
import pytest
from aragog.rheology import SolidRheologyParams

from proteus.config import Config
from proteus.config._interior import Rheology
from proteus.interior_energetics.aragog_phase import build_solid_rheology_params

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_rheology_fields_match_solid_rheology_params():
    """Every field on Rheology must exist on SolidRheologyParams with the same default."""
    aragog_fields = {f.name: f for f in dataclasses.fields(SolidRheologyParams)}
    proteus_fields = {f.name: f for f in attrs.fields(Rheology)}

    assert set(proteus_fields.keys()) == set(aragog_fields.keys()), (
        f'Field mismatch between Rheology and SolidRheologyParams: '
        f'extra={set(proteus_fields.keys()) - set(aragog_fields.keys())}, '
        f'missing={set(aragog_fields.keys()) - set(proteus_fields.keys())}'
    )

    aragog_instance = SolidRheologyParams()
    proteus_instance = Rheology()

    for name in proteus_fields:
        p_val = getattr(proteus_instance, name)
        a_val = getattr(aragog_instance, name)
        if isinstance(p_val, float):
            assert p_val == pytest.approx(a_val), (
                f'Default mismatch for {name}: {p_val} != {a_val}'
            )
        else:
            assert p_val == a_val, f'Default mismatch for {name}: {p_val} != {a_val}'


def test_build_solid_rheology_params_translates_config():
    """build_solid_rheology_params converts Config.interior_energetics.aragog.rheology."""
    cfg = Config()
    rheo = cfg.interior_energetics.aragog.rheology

    solid_params = build_solid_rheology_params(cfg)
    assert isinstance(solid_params, SolidRheologyParams)

    for field_name in attrs.fields_dict(Rheology):
        p_val = getattr(rheo, field_name)
        s_val = getattr(solid_params, field_name)
        if isinstance(p_val, float):
            assert p_val == pytest.approx(s_val)
        else:
            assert p_val == s_val
