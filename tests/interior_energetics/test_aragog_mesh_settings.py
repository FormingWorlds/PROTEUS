"""Aragog mesh end cells and the viscous mixing-length slope in the PROTEUS config.

Module under test: ``interior_energetics.aragog.cmb_cell_thickness``,
``interior_energetics.aragog.surface_cell_thickness`` and
``interior_energetics.aragog.rheology.mlt_bottom_slope`` of ``proteus.config``, their path to
aragog's ``_MeshParameters`` in ``AragogRunner.setup_solver`` and to ``SolidRheologyParams``
in ``solid_rheology_kwargs``. Invariants: the PROTEUS defaults equal the aragog defaults; a
cell thickness is 0, a real number of at least 2 m, or (surface only) "uniform"; the slope
lies in (0, 1]; the values reach aragog unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from proteus.config import read_config_object
from proteus.config._interior import Aragog, Rheology

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]

MINIMAL = Path(__file__).parents[2] / 'input' / 'minimal.toml'


def _aragog_defaults():
    """Field defaults of aragog's _MeshParameters and SolidRheologyParams."""
    import dataclasses

    from aragog.parser import _MeshParameters
    from aragog.rheology import SolidRheologyParams

    mesh = {f.name: f.default for f in dataclasses.fields(_MeshParameters)}
    return (
        mesh['cmb_cell_thickness'],
        mesh['surface_cell_thickness'],
        (SolidRheologyParams().mlt_bottom_slope),
    )


def test_the_proteus_defaults_are_the_aragog_defaults():
    """The 3 PROTEUS defaults equal those of the installed aragog: 1000 m, "uniform", 0.82."""
    cmb, surface, slope = _aragog_defaults()
    a = Aragog()
    assert (a.cmb_cell_thickness, a.surface_cell_thickness) == (cmb, surface)
    assert Rheology().mlt_bottom_slope == slope
    assert (cmb, surface, slope) == (1000.0, 'uniform', 0.82)


def _kept(name, value):
    try:
        return getattr(Aragog(**{name: value}), name) == value
    except (TypeError, ValueError):
        return False


@pytest.mark.parametrize('name', ['cmb_cell_thickness', 'surface_cell_thickness'])
@given(value=st.floats(allow_nan=True, allow_infinity=True) | st.integers(-10, 10**6))
def test_a_cell_thickness_is_0_or_a_finite_number_of_at_least_2_m(name, value):
    """A number is kept exactly when it is 0, or finite and at least 2 m."""
    want = value == 0 or (math.isfinite(value) and value >= 2)
    assert _kept(name, value) is want


@pytest.mark.parametrize('name', ['cmb_cell_thickness', 'surface_cell_thickness'])
@pytest.mark.parametrize(
    'value', [True, False, None, '1000', 'Uniform', 'auto', b'uniform', [1e3]]
)
def test_a_cell_thickness_that_is_not_a_number_or_the_rule_is_refused(name, value):
    """A bool, None, a numeric string, another string, bytes and a list are refused."""
    assert _kept(name, value) is False


def test_uniform_is_kept_at_the_surface_and_refused_at_the_cmb():
    """The rule names the top cell of the uniform mesh; the CMB cell has no such rule."""
    assert _kept('surface_cell_thickness', 'uniform') is True
    assert _kept('cmb_cell_thickness', 'uniform') is False


@given(value=st.floats(allow_nan=True, allow_infinity=True))
def test_the_slope_lies_in_0_to_1(value):
    """mlt_bottom_slope is kept exactly in (0, 1]; NaN, 0 and above 1 are refused."""
    try:
        kept = Rheology(mlt_bottom_slope=value).mlt_bottom_slope == value
    except (TypeError, ValueError):
        kept = False
    assert kept is (0.0 < value <= 1.0)


def test_the_slope_refuses_a_bool():
    """A TOML true is refused, not read as 1."""
    with pytest.raises((TypeError, ValueError), match='mlt_bottom_slope'):
        Rheology(mlt_bottom_slope=True)


def test_a_config_file_sets_the_three_fields(tmp_path):
    """Values in the config file reach the loaded config unchanged."""
    text = MINIMAL.read_text() + (
        '\n[interior_energetics.aragog]\n'
        '    cmb_cell_thickness = 2500.0\n'
        '    surface_cell_thickness = 7000\n'
        '\n[interior_energetics.aragog.rheology]\n'
        '    mlt_bottom_slope = 0.5\n'
    )
    path = tmp_path / 'user.toml'
    path.write_text(text)
    a = read_config_object(path).interior_energetics.aragog
    assert (a.cmb_cell_thickness, a.surface_cell_thickness) == (2500.0, 7000)
    assert a.rheology.mlt_bottom_slope == 0.5


@pytest.mark.parametrize(('cmb', 'surface'), [(2500.0, 7000.0), (0.0, 'uniform'), (0.0, 0.0)])
def test_setup_solver_passes_the_end_cells_to_the_aragog_mesh(tmp_path, cmb, surface):
    """The mesh parameters aragog receives hold the configured end cells."""
    from proteus.interior_energetics.aragog import AragogRunner
    from tests.interior_energetics.test_aragog import (
        _EOS,
        _make_aragog_config,
        _seed_lookup_tables,
    )

    config = _make_aragog_config(struct_module='zalmoxis')
    config.interior_energetics.aragog.cmb_cell_thickness = cmb
    config.interior_energetics.aragog.surface_cell_thickness = surface
    hf_row = {
        'R_int': 6.371e6, 'R_core': 3.48e6, 'gravity': 9.81, 'T_magma': 3000.0,
        'T_eqm': 255.0, 'F_atm': 100.0,
    }  # fmt: skip
    interior_o = MagicMock()
    interior_o.tides = np.zeros(20)
    (tmp_path / 'spider_eos').mkdir()
    interior_o._spider_eos_dir = str(tmp_path / 'spider_eos')
    _seed_lookup_tables(tmp_path)
    (tmp_path / 'interior_lookup_tables' / 'Melting_curves').mkdir(parents=True)
    with (
        patch('proteus.interior_energetics.aragog.FWL_DATA_DIR', tmp_path),
        patch('proteus.interior_energetics.aragog.Parameters') as params,
        patch('proteus.interior_energetics.aragog.EntropySolver'),
        patch('proteus.interior_energetics.aragog._cached_entropy_eos', return_value=_EOS),
    ):
        AragogRunner.setup_solver(config, hf_row, interior_o, str(tmp_path))
    mesh = params.call_args.kwargs['mesh']
    assert mesh.cmb_cell_thickness == cmb
    assert mesh.surface_cell_thickness == surface


def _rheology_config(enabled, slope):
    rheology = SimpleNamespace(enabled=enabled, mlt_bottom_slope=slope)
    aragog = SimpleNamespace(rheology=rheology)
    return SimpleNamespace(interior_energetics=SimpleNamespace(aragog=aragog))


@pytest.mark.parametrize('slope', [0.5, 0.82, 1.0])
def test_the_rheology_keyword_carries_the_slope(slope):
    """With the rheology on, aragog's SolidRheologyParams holds the configured slope and its
    own defaults for the rest; with it off no keyword is passed."""
    from aragog.rheology import SolidRheologyParams

    from proteus.interior_energetics.aragog_phase import solid_rheology_kwargs

    kw = solid_rheology_kwargs(_rheology_config(True, slope))
    assert kw == {'rheology': SolidRheologyParams(enabled=True, mlt_bottom_slope=slope)}
    assert solid_rheology_kwargs(_rheology_config(False, slope)) == {}
