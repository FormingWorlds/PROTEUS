"""Integration test: aragog (real interior) coupled to ZEPHYRUS (real escape).

ZEPHYRUS is hard-coupled to MORS via the ``spada_zephyrus``
cross-validator at ``src/proteus/config/_config.py:25-31``: a
Config with ``escape.module='zephyrus'`` MUST also have
``star.module='mors'`` AND ``star.mors.tracks='spada'``. The
aragog x ZEPHYRUS pair therefore requires star=MORS+spada to
round-trip; aragog provides the interior boundary that responds
to the per-iteration escape mass loss.

Integration-tier scope:

- Schema validators round-trip ``interior_energetics.module=
  'aragog'`` with ``escape.module='zephyrus'`` when paired with
  ``star.module='mors'`` and ``mors.tracks='spada'``.
- The spada_zephyrus cross-validator rejects the same Config
  with ``star.module='dummy'`` or ``mors.tracks='baraffe'``
  (already covered by the AGNI x ZEPHYRUS file; this file
  re-pins the aragog leg so a regression in the interior side
  surfaces here too).
- Aragog backend / core_bc / phase_smoothing / solver_method
  enums are re-pinned as sets (so a regression in the aragog
  side that the AGNI x aragog file would catch also surfaces
  here when escape is enabled).
- Zephyrus.Pxuv (0, 10] bar contract and Zephyrus.efficiency
  [0, 1] contract are re-pinned at the cross-validator layer.
- The wrapper merge guard pins the interior columns (T_magma,
  Phi_global, F_int, F_cmb) and the escape columns
  (esc_rate_total, esc_kg_cumulative, M_vol_initial) in
  GetHelpfileKeys so per-iteration values flow into the helpfile.

The full two-timestep aragog + ZEPHYRUS coupled run is exercised
by the slow-tier ``test_integration_mors_zephyrus.py`` (with
dummy interior) and the slow-tier aragog tests for the interior
leg.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _base_config_kwargs(*, agni: bool = False):
    """Build base Config kwargs for the (aragog, zephyrus, mors+spada)
    combination. ``agni=False`` keeps atmos_clim on the default
    janus/dummy slot; tests for the aragog x zephyrus pair do not
    need AGNI in the loop.
    """
    from proteus.config._atmos_clim import AtmosClim
    from proteus.config._interior import Interior
    from proteus.config._planet import Elements, Planet
    from proteus.config._star import Star

    return dict(
        atmos_clim=AtmosClim(module='agni' if agni else 'janus'),
        interior_energetics=Interior(module='aragog'),
        star=Star(module='mors'),  # default Mors uses spada + phoenix
        planet=Planet(mass_tot=1.0, elements=Elements(O_mode='ic_chemistry')),
    )


# ---------------------------------------------------------------------------
# Schema-validator round-trips for the (aragog, zephyrus, mors+spada) combo.
# ---------------------------------------------------------------------------


def test_aragog_zephyrus_mors_spada_round_trips_through_config():
    """The full hard-coupled triple (interior=aragog, escape=zephyrus,
    star=mors+spada) round-trips through Config without raising.

    Discrimination: a regression in either side that broke schema
    construction would surface here. The asserts confirm each
    module landed where expected.
    """
    from proteus.config import Config
    from proteus.config._escape import Escape

    cfg = Config(escape=Escape(module='zephyrus'), **_base_config_kwargs())
    assert cfg.interior_energetics.module == 'aragog'
    assert cfg.escape.module == 'zephyrus'
    assert cfg.star.module == 'mors'
    assert cfg.star.mors.tracks == 'spada'


def test_aragog_zephyrus_rejects_dummy_star_at_spada_zephyrus_layer():
    """``spada_zephyrus`` fires when escape=zephyrus is paired with
    a non-MORS star. The aragog x zephyrus combination must reject
    the dummy-star configuration even though the aragog interior
    side itself is valid.
    """
    from proteus.config import Config
    from proteus.config._atmos_clim import AtmosClim
    from proteus.config._escape import Escape
    from proteus.config._interior import Interior
    from proteus.config._planet import Elements, Planet
    from proteus.config._star import Star, StarDummy

    kwargs = dict(
        atmos_clim=AtmosClim(module='janus'),
        interior_energetics=Interior(module='aragog'),
        star=Star(module='dummy', dummy=StarDummy(calculate_radius=True)),
        planet=Planet(mass_tot=1.0, elements=Elements(O_mode='ic_chemistry')),
    )
    with pytest.raises(ValueError, match=r'(?i)(MORS|spada)'):
        Config(escape=Escape(module='zephyrus'), **kwargs)


def test_aragog_zephyrus_rejects_baraffe_tracks_at_spada_zephyrus_layer():
    """``spada_zephyrus`` also rejects mors+baraffe even when the
    interior side is aragog.
    """
    from proteus.config import Config
    from proteus.config._atmos_clim import AtmosClim
    from proteus.config._escape import Escape
    from proteus.config._interior import Interior
    from proteus.config._planet import Elements, Planet
    from proteus.config._star import Mors, Star

    kwargs = dict(
        atmos_clim=AtmosClim(module='janus'),
        interior_energetics=Interior(module='aragog'),
        star=Star(module='mors', mors=Mors(tracks='baraffe')),
        planet=Planet(mass_tot=1.0, elements=Elements(O_mode='ic_chemistry')),
    )
    with pytest.raises(ValueError, match=r'(?i)(MORS|spada)'):
        Config(escape=Escape(module='zephyrus'), **kwargs)


# ---------------------------------------------------------------------------
# Aragog-side enum-as-set guards (re-pinned for this pair).
# ---------------------------------------------------------------------------


def test_aragog_enums_pinned_for_zephyrus_pair():
    """Re-pin all four Aragog enums as sets so a regression that
    drifts the interior side surfaces here when escape is enabled
    (the AGNI x aragog pair already pins these, but the matrix
    contract is per-pair coverage of the relevant interior enums).
    """
    import attrs

    from proteus.config._interior import Aragog

    backend_allowed = attrs.fields(Aragog).backend.validator.options
    assert set(backend_allowed) == {'jax', 'numpy'}
    core_bc_allowed = attrs.fields(Aragog).core_bc.validator.options
    assert set(core_bc_allowed) == {
        'quasi_steady',
        'energy_balance',
        'gradient',
        'bower2018',
    }
    smoothing_allowed = attrs.fields(Aragog).phase_smoothing.validator.options
    assert set(smoothing_allowed) == {'tanh', 'cubic_hermite'}
    method_allowed = attrs.fields(Aragog).solver_method.validator.options
    assert set(method_allowed) == {'cvode', 'radau', 'bdf'}


# ---------------------------------------------------------------------------
# Zephyrus-side validator bounds (re-pinned at the cross-validator layer).
# ---------------------------------------------------------------------------


def test_zephyrus_pxuv_upper_bound_under_aragog_pair():
    """``Zephyrus.Pxuv`` upper bound (closed at 10) is checked at the
    cross-validator layer when the rest of the Config is the aragog
    + mors+spada combination. A regression that flipped the
    comparator (<= vs <) would land at the same outcome on the
    boundary either way; pin both the boundary and the just-above
    value.
    """
    from proteus.config import Config
    from proteus.config._escape import Escape, Zephyrus

    # 10.0 round-trips at the upper boundary.
    cfg = Config(
        escape=Escape(module='zephyrus', zephyrus=Zephyrus(Pxuv=10.0)),
        **_base_config_kwargs(),
    )
    assert cfg.escape.zephyrus.Pxuv == pytest.approx(10.0, rel=1e-12)
    # 10.001 raises (just above the closed upper bound).
    with pytest.raises(ValueError, match=r'(?i)pxuv'):
        Config(
            escape=Escape(module='zephyrus', zephyrus=Zephyrus(Pxuv=10.001)),
            **_base_config_kwargs(),
        )


def test_zephyrus_efficiency_endpoints_under_aragog_pair():
    """``Zephyrus.efficiency`` boundary check [0, 1] inclusive at
    both endpoints under the aragog + mors+spada Config.
    """
    from proteus.config import Config
    from proteus.config._escape import Escape, Zephyrus

    for boundary in (0.0, 1.0):
        cfg = Config(
            escape=Escape(module='zephyrus', zephyrus=Zephyrus(efficiency=boundary)),
            **_base_config_kwargs(),
        )
        assert cfg.escape.zephyrus.efficiency == pytest.approx(boundary, abs=1e-12)
    with pytest.raises(ValueError, match=r'(?i)efficiency'):
        Config(
            escape=Escape(module='zephyrus', zephyrus=Zephyrus(efficiency=-0.001)),
            **_base_config_kwargs(),
        )
    with pytest.raises(ValueError, match=r'(?i)efficiency'):
        Config(
            escape=Escape(module='zephyrus', zephyrus=Zephyrus(efficiency=1.001)),
            **_base_config_kwargs(),
        )


# ---------------------------------------------------------------------------
# Wrapper-merge contract: interior + escape columns in GetHelpfileKeys.
# ---------------------------------------------------------------------------


def test_aragog_zephyrus_helpfile_keys_register_interior_and_escape_columns():
    """The wrapper merge guard at ``atmos_clim/wrapper.py:196-198``
    propagates per-iteration columns from both sides into hf_row.
    The schema MUST register interior columns (T_magma, Phi_global,
    F_int, F_cmb) AND escape columns (esc_rate_total,
    esc_kg_cumulative, M_vol_initial).

    Discrimination: every key tested separately so a regression
    that dropped any one fails the per-key loop. ZeroHelpfileRow
    seeds each as float zero.
    """
    from proteus.utils.coupler import GetHelpfileKeys, ZeroHelpfileRow

    keys = GetHelpfileKeys()
    interior_keys = ('T_magma', 'Phi_global', 'F_int', 'F_cmb')
    escape_keys = ('esc_rate_total', 'esc_kg_cumulative', 'M_vol_initial')
    for key in interior_keys + escape_keys:
        assert key in keys, f'{key} must be registered in GetHelpfileKeys()'
    row = ZeroHelpfileRow()
    for key in interior_keys + escape_keys:
        assert row[key] == pytest.approx(0.0, abs=1e-30)
        assert isinstance(row[key], float)
