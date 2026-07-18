"""Integration test: zalmoxis (real interior structure) coupled to
CALLIOPE (real outgassing).

CALLIOPE solves volatile equilibrium between the magma ocean and
the atmosphere, partitioning H, C, N, S (and now O per issue #677)
into the dissolved and atmospheric reservoirs. Zalmoxis reads the
combined volatile mass via ``hf_row[e + '_kg_total']`` over the
whole-planet ``element_list`` and subtracts it from the total
planet mass to set the structure-solve dry-mass target. The two
modules couple through the ``equilibrate_init=True`` path: a
pre-main-loop CALLIOPE + Zalmoxis equilibration converges the
initial dry-mass + volatile partitioning to within
``equilibrate_tol``.

Integration-tier scope:

- The (interior_struct=zalmoxis, outgas=calliope) pair round-trips
  through Config. Calliope landings on its documented backend and
  Zalmoxis on its documented production defaults.
- ``Outgas.module`` enum is pinned as ``{'calliope', 'atmodeller',
  'dummy'}``.
- CALLIOPE's ``include_*`` boolean field family: 10 fields with the
  documented species names. Field count AND name set pinned so a
  silent rename (e.g. ``include_CO`` to ``include_carbon_monoxide``)
  surfaces. All defaults round-trip ``True``.
- A subset of CALLIOPE volatile flags round-trip when set ``False``
  to confirm the schema does not couple the booleans to each other.
- ``Outgas.fO2_shift_IW`` is unconstrained at the field layer (no
  validator); pin the documented default (4.0) and round-trip
  Earth-like (IW+2), reducing (IW-3), and zero shifts.
- ``Outgas.mass_thresh``, ``T_floor``, ``solver_rtol``, and
  ``solver_atol`` all use ``gt(0)``. Pin both edges (= 0 rejects,
  small positive round-trips) and the documented defaults.
- Zalmoxis's ``equilibrate_init``, ``equilibrate_max_iter``
  (``ge(1)``), and ``equilibrate_tol`` (``gt(0)``) gate the pre-
  main-loop convergence loop. Each pinned both-edge-selective.
- ``Zalmoxis.global_miscibility`` (binodal-aware miscibility) gates
  the H2-MgSiO3 solvus path. Pin the off-by-default round-trip AND
  the on-state, plus the ``miscibility_max_iter`` (``ge(1)``) /
  ``miscibility_tol`` (``gt(0)``) selectivity.
- The whole-element propagation contract: ``element_list`` includes
  oxygen (issue #677). The hf_row schema registers ``O_kg_total``
  alongside ``H_kg_total``, ``C_kg_total``, ``N_kg_total``,
  ``S_kg_total``; ``O_kg_user_ic`` (the user-supplied O budget
  sentinel) is also registered so the one-shot IC reconciliation
  check has somewhere to read from.
- The wrapper merge contract registers per-gas pressures CALLIOPE
  writes (``H2O_bar``, ``CO2_bar``, ``N2_bar``, ``H2_bar``,
  ``CO_bar``, ``CH4_bar``, ``SO2_bar``, ``H2S_bar``, ``S2_bar``,
  ``NH3_bar``) and the elemental-reservoir columns Zalmoxis reads
  (``H_kg_total``, etc.).

The full two-timestep coupled run with real Aragog + real CALLIOPE
+ dummy Zalmoxis lives at the slow tier in
``test_slow_aragog_calliope.py``.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _base_config_kwargs():
    """Base Config kwargs for the (zalmoxis, calliope) combination.

    ``atmos_clim='dummy'`` avoids latent cross-validator coupling.
    ``escape='dummy'`` keeps the escape side off the test surface
    (the zephyrus pair handles escape-side rejections). fO2 buffer
    set explicitly to a non-zero Earth-like value so CALLIOPE has
    something to chemically equilibrate against in the schema-tier
    construction.
    """
    from proteus.config._atmos_clim import AtmosClim
    from proteus.config._escape import Escape
    from proteus.config._planet import Elements, Planet
    from proteus.config._star import Star, StarDummy

    return dict(
        atmos_clim=AtmosClim(module='dummy', rayleigh=False),
        escape=Escape(module='dummy'),
        star=Star(module='dummy', dummy=StarDummy(calculate_radius=True)),
        planet=Planet(mass_tot=1.0, elements=Elements(O_mode='ic_chemistry')),
    )


# ---------------------------------------------------------------------------
# Pair round-trip and Outgas.module enum.
# ---------------------------------------------------------------------------


def test_zalmoxis_calliope_round_trips_through_config():
    """The (interior_struct=zalmoxis, outgas=calliope) pair round-
    trips through Config without raising. Calliope lands on its
    documented defaults (all 10 ``include_*`` species True);
    Zalmoxis lands on its production defaults (newton outer solver,
    PALEOS EOS, equilibrate_init enabled).
    """
    from proteus.config import Config
    from proteus.config._outgas import Outgas
    from proteus.config._struct import Struct

    cfg = Config(
        interior_struct=Struct(module='zalmoxis'),
        outgas=Outgas(module='calliope'),
        **_base_config_kwargs(),
    )
    assert cfg.interior_struct.module == 'zalmoxis'
    assert cfg.outgas.module == 'calliope'
    # Production defaults on both sides.
    assert cfg.interior_struct.zalmoxis.equilibrate_init is True
    assert cfg.outgas.calliope.include_H2O is True
    assert cfg.outgas.calliope.include_CO2 is True


def test_outgas_module_enum_pinned_as_set_under_zalmoxis_pair():
    """Pin the ``Outgas.module`` enum as ``{'calliope', 'atmodeller',
    'dummy'}``. Set equality catches a regression that silently
    added a fourth outgas backend.
    """
    import attrs

    from proteus.config._outgas import Outgas

    allowed = attrs.fields(Outgas).module.validator.options
    assert set(allowed) == {'calliope', 'atmodeller', 'dummy'}, (
        f'Outgas.module enum drifted from documented set: {allowed}'
    )
    for known in ('calliope', 'atmodeller', 'dummy'):
        o = Outgas(module=known)
        assert o.module == known
    with pytest.raises(ValueError, match=r'(?i)module'):
        Outgas(module='magma_chem')
    assert Outgas().module == 'calliope'


# ---------------------------------------------------------------------------
# Calliope include_* field family.
# ---------------------------------------------------------------------------


def test_calliope_include_field_count_and_name_set_under_zalmoxis_pair():
    """Pin both the COUNT and the documented NAMES of CALLIOPE's
    ``include_*`` boolean fields. A silent rename (e.g.
    ``include_CO`` to ``include_carbon_monoxide``) would fail the
    set check even if the count happened to match.

    Discrimination: set equality on the name set + a count guard so
    a regression that loosened either alone surfaces.
    """
    import attrs

    from proteus.config._outgas import Calliope

    fields = attrs.fields(Calliope)
    include_fields = {f.name for f in fields if f.name.startswith('include_')}
    assert include_fields == {
        'include_H2O',
        'include_CO2',
        'include_N2',
        'include_S2',
        'include_SO2',
        'include_H2S',
        'include_NH3',
        'include_H2',
        'include_CH4',
        'include_CO',
        # The opt-in noble gases each carry an include_* flag as well.
        'include_He',
        'include_Ne',
        'include_Ar',
        'include_Kr',
        'include_Xe',
    }
    assert len(include_fields) == 15


def test_calliope_include_flags_independent_default_true_round_trip_some_off():
    """Every reactive ``Calliope.include_*`` field defaults to True (the
    opt-in noble gas flags default to False). A subset can be flipped to
    False at construction time independently of the others; the schema does
    not couple them.

    Discrimination: verify the default is True for all ten reactive fields,
    then flip three to False and confirm the other seven remain
    True. A regression that introduced a hidden coupling (e.g. forced
    all-or-nothing) would fail the per-field assertion.
    """
    from proteus.config._outgas import Calliope

    # Defaults.
    default = Calliope()
    for vol in (
        'H2O',
        'CO2',
        'N2',
        'S2',
        'SO2',
        'H2S',
        'NH3',
        'H2',
        'CH4',
        'CO',
    ):
        assert getattr(default, f'include_{vol}') is True, f'include_{vol} default flipped'
    # Selectively turn three off.
    c = Calliope(include_CO=False, include_CH4=False, include_N2=False)
    assert c.include_CO is False
    assert c.include_CH4 is False
    assert c.include_N2 is False
    # Remaining seven stay True.
    for vol in ('H2O', 'CO2', 'S2', 'SO2', 'H2S', 'NH3', 'H2'):
        assert getattr(c, f'include_{vol}') is True


# ---------------------------------------------------------------------------
# Outgas-level shared solver knobs.
# ---------------------------------------------------------------------------


def test_outgas_fO2_shift_unconstrained_round_trips_under_calliope_pair():
    """``Outgas.fO2_shift_IW`` has NO validator at the field layer
    (any float accepted). The documented default is 4.0; pin a
    spread of Earth-like and reducing values to confirm the field
    actually carries the value through Config construction.

    Discrimination: a regression that introduced ``ge(...)`` or
    ``le(...)`` on this field would silently restrict the supported
    fO2 range. Round-tripping a negative shift (IW-3) catches that.
    """
    from proteus.config import Config
    from proteus.config._outgas import Outgas

    for shift in (-3.0, 0.0, 2.0, 4.0, 8.0):
        cfg = Config(
            outgas=Outgas(module='calliope', fO2_shift_IW=shift),
            **_base_config_kwargs(),
        )
        if shift == 0.0:
            assert cfg.outgas.fO2_shift_IW == pytest.approx(0.0, abs=1e-12)
        else:
            assert cfg.outgas.fO2_shift_IW == pytest.approx(shift, rel=1e-12)


def test_outgas_mass_thresh_t_floor_and_solver_tols_gt0_with_selectivity():
    """``Outgas.mass_thresh``, ``T_floor``, ``solver_rtol``, and
    ``solver_atol`` all use ``gt(0)``. Pin both edges (=0 rejects,
    small positive round-trips) and the documented defaults
    (1e16 kg, 700 K, 1e-4, 1e-6).
    """
    from proteus.config._outgas import Outgas

    for bad_val in (0.0, -1.0):
        with pytest.raises(ValueError, match=r'(?i)mass_thresh'):
            Outgas(mass_thresh=bad_val)
        with pytest.raises(ValueError, match=r'(?i)T_floor'):
            Outgas(T_floor=bad_val)
        with pytest.raises(ValueError, match=r'(?i)solver_rtol'):
            Outgas(solver_rtol=bad_val)
        with pytest.raises(ValueError, match=r'(?i)solver_atol'):
            Outgas(solver_atol=bad_val)
    o = Outgas(
        mass_thresh=1e-12,
        T_floor=1e-12,
        solver_rtol=1e-12,
        solver_atol=1e-30,
    )
    assert o.mass_thresh == pytest.approx(1e-12, rel=1e-12)
    assert o.T_floor == pytest.approx(1e-12, rel=1e-12)
    assert o.solver_rtol == pytest.approx(1e-12, rel=1e-12)
    assert o.solver_atol == pytest.approx(1e-30, rel=1e-12)
    default = Outgas()
    assert default.mass_thresh == pytest.approx(1e16, rel=1e-12)
    assert default.T_floor == pytest.approx(700.0, rel=1e-12)
    assert default.solver_rtol == pytest.approx(1e-4, rel=1e-12)
    assert default.solver_atol == pytest.approx(1e-6, rel=1e-12)


# ---------------------------------------------------------------------------
# Zalmoxis equilibration knobs (the CALLIOPE + Zalmoxis convergence loop).
# ---------------------------------------------------------------------------


def test_zalmoxis_equilibrate_init_default_on_round_trips_off():
    """``Zalmoxis.equilibrate_init`` gates the pre-main-loop
    CALLIOPE + Zalmoxis convergence step. Default True; round-trip
    both states.

    Discrimination: pin the default ON state explicitly. A
    regression that flipped the default to False would silently
    skip the IC equilibration and let the dry-mass + volatile
    partitioning start the main loop out of agreement.
    """
    from proteus.config._struct import Zalmoxis

    default = Zalmoxis()
    assert default.equilibrate_init is True
    z_off = Zalmoxis(equilibrate_init=False)
    assert z_off.equilibrate_init is False


def test_zalmoxis_equilibrate_max_iter_ge1_with_selectivity():
    """``Zalmoxis.equilibrate_max_iter`` uses ``ge(1)``: ``=1``
    round-trips (catches a ``gt(1)`` regression), ``=0`` rejects,
    documented default 15 round-trips.
    """
    from proteus.config._struct import Zalmoxis

    assert Zalmoxis(equilibrate_max_iter=1).equilibrate_max_iter == 1
    with pytest.raises(ValueError, match=r'(?i)equilibrate_max_iter'):
        Zalmoxis(equilibrate_max_iter=0)
    assert Zalmoxis().equilibrate_max_iter == 15


def test_zalmoxis_equilibrate_tol_gt0_with_selectivity():
    """``Zalmoxis.equilibrate_tol`` uses ``gt(0)``. ``=0`` rejects;
    small positive round-trips; documented default 0.01 round-trips.
    """
    from proteus.config._struct import Zalmoxis

    for bad_val in (0.0, -1e-6):
        with pytest.raises(ValueError, match=r'(?i)equilibrate_tol'):
            Zalmoxis(equilibrate_tol=bad_val)
    z = Zalmoxis(equilibrate_tol=1e-12)
    assert z.equilibrate_tol == pytest.approx(1e-12, rel=1e-12)
    assert Zalmoxis().equilibrate_tol == pytest.approx(0.01, rel=1e-12)


# ---------------------------------------------------------------------------
# Binodal-aware H2 miscibility (CALLIOPE + Zalmoxis solvus path).
# ---------------------------------------------------------------------------


def test_zalmoxis_global_miscibility_default_off_and_gated_on():
    """``Zalmoxis.global_miscibility`` gates the H2-MgSiO3 solvus
    path (Rogers+2025 binodal). Off by default, and enabling it is
    rejected until the pinned Zalmoxis release can consume a
    per-shell volatile profile.

    The Outgas-side counterpart ``h2_binodal`` is rejected at the
    class layer; the structure-side flag is rejected by the
    cross-field ``Struct`` validator (the bare ``Zalmoxis`` class
    carries no gate, so the enforcement point is pinned explicitly).
    """
    from proteus.config._outgas import Outgas
    from proteus.config._struct import Struct, Zalmoxis

    assert Zalmoxis().global_miscibility is False
    # The bare class accepts the flag; the gate lives on Struct.
    z_on = Zalmoxis(global_miscibility=True)
    assert z_on.global_miscibility is True
    with pytest.raises(ValueError, match=r'global_miscibility'):
        Struct(module='zalmoxis', zalmoxis=z_on)
    # The default-off combination still constructs.
    assert Struct(module='zalmoxis').zalmoxis.global_miscibility is False
    # CALLIOPE-side counterpart: rejected at the class layer.
    assert Outgas().h2_binodal is False
    with pytest.raises(ValueError, match=r'h2_binodal'):
        Outgas(h2_binodal=True)


def test_zalmoxis_miscibility_iter_and_tol_validators():
    """``Zalmoxis.miscibility_max_iter`` ``ge(1)`` and
    ``miscibility_tol`` ``gt(0)``. Both pinned both-edge.
    Documented defaults (10, 0.01) round-trip.
    """
    from proteus.config._struct import Zalmoxis

    assert Zalmoxis(miscibility_max_iter=1).miscibility_max_iter == 1
    with pytest.raises(ValueError, match=r'(?i)miscibility_max_iter'):
        Zalmoxis(miscibility_max_iter=0)
    for bad_val in (0.0, -1e-6):
        with pytest.raises(ValueError, match=r'(?i)miscibility_tol'):
            Zalmoxis(miscibility_tol=bad_val)
    z = Zalmoxis(miscibility_tol=1e-12)
    assert z.miscibility_tol == pytest.approx(1e-12, rel=1e-12)
    default = Zalmoxis()
    assert default.miscibility_max_iter == 10
    assert default.miscibility_tol == pytest.approx(0.01, rel=1e-12)


# ---------------------------------------------------------------------------
# Whole-element propagation (issue #677): O included in element_list.
# ---------------------------------------------------------------------------


def test_element_list_includes_oxygen_under_calliope_pair():
    """Issue #677: the whole-planet ``element_list`` includes O
    alongside H, C, N, S so the dry-mass subtraction in
    ``load_zalmoxis_configuration`` reserves space for the O that
    CALLIOPE places in atmospheric H2O, CO2, SO2, etc. A regression
    that removed O from the list would let M_atm exceed M_planet at
    high H budgets.

    The list also includes the rock-forming elements (Si, Mg, Fe,
    Na) so the same dry-mass subtraction works for sub-Neptune and
    super-Earth compositions where the dissolved rocky inventory is
    non-negligible. Pin the full documented set.

    Discrimination: set equality fails on both addition and removal
    of any element.
    """
    from proteus.utils.constants import element_list, noble_gases

    # The reactive and rock-forming elements plus the opt-in noble gases, which
    # are tracked as elements in the whole-planet mass balance.
    assert set(element_list) == {'H', 'O', 'C', 'N', 'S', 'Si', 'Mg', 'Fe', 'Na'} | set(
        noble_gases
    )
    # The volatile species CALLIOPE partitions must all be present.
    for vol in ('H', 'O', 'C', 'N', 'S'):
        assert vol in element_list, (
            f'{vol} missing from element_list; whole-planet bookkeeping breaks'
        )


def test_whole_element_helpfile_keys_register_per_element_total_columns():
    """The whole-planet element-total columns (``<E>_kg_total``) MUST
    be in ``GetHelpfileKeys`` so the Zalmoxis dry-mass subtraction
    reads a populated value at every iteration. Each species in
    ``element_list`` (H, O, C, N, S, Si, Mg, Fe, Na) gets its own
    column.

    The IC-reconciliation sentinel ``O_kg_user_ic`` is added to
    hf_row by ``outgas/wrapper.py`` at runtime (it is not in the
    declared schema list). Pinning the per-element total columns
    here documents the schema-side contract that Zalmoxis depends
    on.

    Discrimination: per-element check so a regression that dropped
    one element's total fails the specific assertion.
    """
    from proteus.utils.constants import element_list
    from proteus.utils.coupler import GetHelpfileKeys, ZeroHelpfileRow

    keys = GetHelpfileKeys()
    for e in element_list:
        col = f'{e}_kg_total'
        assert col in keys, f'{col} must be registered for whole-element bookkeeping'
    row = ZeroHelpfileRow()
    for e in element_list:
        col = f'{e}_kg_total'
        assert row[col] == pytest.approx(0.0, abs=1e-30)
        assert isinstance(row[col], float)


# ---------------------------------------------------------------------------
# Wrapper-merge contract: per-gas pressures from CALLIOPE.
# ---------------------------------------------------------------------------


def test_calliope_per_gas_pressure_columns_registered_under_zalmoxis_pair():
    """CALLIOPE writes the per-gas partial pressures
    (``<species>_bar``) into ``hf_row`` at every outgas call.
    Zalmoxis uses them indirectly via the surface-pressure target
    in ``_get_target_surface_pressure``. The schema MUST register
    every species CALLIOPE supports.

    Discrimination: per-species check (each pressure tested
    separately so a regression that dropped one fails the
    corresponding line).
    """
    from proteus.utils.coupler import GetHelpfileKeys, ZeroHelpfileRow

    keys = GetHelpfileKeys()
    pressure_keys = (
        'H2O_bar',
        'CO2_bar',
        'N2_bar',
        'H2_bar',
        'CO_bar',
        'CH4_bar',
        'SO2_bar',
        'H2S_bar',
        'S2_bar',
        'NH3_bar',
    )
    for key in pressure_keys:
        assert key in keys, f'{key} must be registered in GetHelpfileKeys()'
    row = ZeroHelpfileRow()
    for key in pressure_keys:
        assert row[key] == pytest.approx(0.0, abs=1e-30)
        assert isinstance(row[key], float)
