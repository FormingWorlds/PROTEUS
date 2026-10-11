"""PROTEUS-side tests for the Zalmoxis ``outer_solver`` config knob.

Covers two angles:

(1) Config schema: ``proteus.config.interior_struct.zalmoxis.outer_solver``
    accepts only 'picard' or 'newton' and defaults to 'newton'. 'picard'
    remains available for explicit opt-in.

(2) Plumbing: ``load_zalmoxis_configuration`` propagates the knob into
    the ``config_params`` dict that ``zalmoxis.solver.main`` consumes,
    and auto-tightens integrator tolerances when 'newton' is selected.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from proteus.config._struct import Zalmoxis

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ----------------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------------


class TestOuterSolverSchema:
    """Config-class validation of outer_solver and Newton-specific knobs."""

    def test_default_outer_solver_is_newton(self):
        """Default ``outer_solver`` is 'newton'. PROTEUS' plumbing also
        tightens integrator tolerances when 'newton' is selected, so the
        default path is end-to-end consistent (see
        ``TestNewtonPathTightensIntegratorTolerances``).
        """
        z = Zalmoxis()
        assert z.outer_solver == 'newton'
        # Discriminator: 'newton' is the only accepted default; a
        # regression that fell back to the legacy 'picard' default
        # would break the end-to-end consistency with the
        # tighten-integrator-tolerance plumbing and would land here.
        assert z.outer_solver != 'picard'

    def test_explicit_picard_accepted(self):
        """An explicit ``outer_solver='picard'`` is accepted by the
        validator; pairs with the default-Newton test above.
        """
        z = Zalmoxis(outer_solver='picard')
        assert z.outer_solver == 'picard'
        # Discriminator: the validator must not silently coerce
        # 'picard' to the default 'newton'. A regression that flipped
        # the default-vs-explicit handling would land here.
        assert z.outer_solver != 'newton'

    def test_explicit_newton_accepted(self):
        """An explicit ``outer_solver='newton'`` is accepted; round-trips
        through the validator and lands on the attribute as-is.
        """
        z = Zalmoxis(outer_solver='newton')
        assert z.outer_solver == 'newton'
        # Discriminator: case-folding or whitespace stripping by the
        # validator could silently rewrite 'newton' to a different
        # casing. Pin the lowercase, exact form.
        assert z.outer_solver == z.outer_solver.lower().strip()

    @pytest.mark.parametrize(
        'bad_value',
        ['Newton', 'PICARD', '', 'broyden', 'levenberg', 'newton ', None, 42],
    )
    def test_invalid_outer_solver_rejected(self, bad_value):
        """attrs in_() validator catches typos, case, and wrong types."""
        with pytest.raises(ValueError):
            Zalmoxis(outer_solver=bad_value)
        # Positive control: the immediately-adjacent good values must
        # both still construct. This rules out a regression that
        # broadened the rejection to also block the legitimate
        # 'newton' and 'picard' tokens.
        assert Zalmoxis(outer_solver='newton').outer_solver == 'newton'
        assert Zalmoxis(outer_solver='picard').outer_solver == 'picard'

    def test_newton_max_iter_default(self):
        """``newton_max_iter`` defaults to 30; tested separately from the
        validator-boundary test below.
        """
        z = Zalmoxis()
        assert z.newton_max_iter == 30
        # Bounded discriminator: 30 must lie strictly above the
        # validator's ge(5) lower bound (otherwise the default sits
        # at the precondition boundary and any noise pushes it
        # below) and well below an absurd ceiling.
        assert 5 < z.newton_max_iter < 1000

    def test_newton_max_iter_minimum_5(self):
        """ge(5) validator: too few iters can't converge anything useful."""
        with pytest.raises(ValueError):
            Zalmoxis(newton_max_iter=4)
        # Boundary: 5 must accept.
        z = Zalmoxis(newton_max_iter=5)
        assert z.newton_max_iter == 5

    def test_newton_tol_default_and_validation(self):
        """``newton_tol`` defaults to 1e-4 and rejects non-positive
        values (zero or negative), so a misconfigured tolerance fails
        loudly at attrs validation time.
        """
        z = Zalmoxis()
        assert z.newton_tol == pytest.approx(1.0e-4, rel=1e-12)
        # gt(0) validator
        with pytest.raises(ValueError):
            Zalmoxis(newton_tol=0)
        with pytest.raises(ValueError):
            Zalmoxis(newton_tol=-1.0e-3)

    def test_newton_integrator_tolerance_defaults(self):
        """Default integrator tolerances are 1e-9 (relative) and 1e-10
        (absolute), which satisfy Zalmoxis' Newton-path precondition
        ``relative_tolerance <= 1e-7``.
        """
        z = Zalmoxis()
        assert z.newton_relative_tolerance == pytest.approx(1.0e-9, rel=1e-12)
        assert z.newton_absolute_tolerance == pytest.approx(1.0e-10, rel=1e-12)


# ----------------------------------------------------------------------
# Plumbing: load_zalmoxis_configuration -> config_params dict
# ----------------------------------------------------------------------


def _make_mock_config(
    outer_solver='picard',
    newton_max_iter=30,
    newton_tol=1.0e-4,
    newton_rel_tol=1.0e-9,
    newton_abs_tol=1.0e-10,
):
    """Build a mock proteus Config that exercises the Zalmoxis-config builder."""
    config = MagicMock()
    config.planet.mass_tot = 1.0  # 1 M_earth
    config.interior_struct.core_frac = 0.325
    config.planet.tsurf_init = 1500.0
    config.planet.tcmb_init = 4000.0
    config.planet.tcenter_init = 5000.0
    config.planet.temperature_mode = 'isothermal'
    config.interior_struct.zalmoxis.core_eos = 'PALEOS:iron'
    config.interior_struct.zalmoxis.mantle_eos = 'PALEOS-2phase:MgSiO3'
    config.interior_struct.zalmoxis.ice_layer_eos = None
    config.interior_struct.zalmoxis.mushy_zone_factor = 0.8
    config.interior_struct.zalmoxis.mantle_mass_fraction = 0.0
    config.interior_struct.zalmoxis.num_levels = 100
    config.interior_struct.zalmoxis.solver_tol_outer = 3.0e-3
    config.interior_struct.zalmoxis.solver_tol_inner = 1.0e-4
    config.interior_struct.zalmoxis.solver_max_iter_outer = 100
    config.interior_struct.zalmoxis.solver_max_iter_inner = 100
    config.interior_struct.zalmoxis.use_jax = False
    config.interior_struct.zalmoxis.use_anderson = False
    config.interior_struct.zalmoxis.dry_mantle = True
    config.interior_struct.zalmoxis.outer_solver = outer_solver
    config.interior_struct.zalmoxis.newton_max_iter = newton_max_iter
    config.interior_struct.zalmoxis.newton_tol = newton_tol
    config.interior_struct.zalmoxis.newton_relative_tolerance = newton_rel_tol
    config.interior_struct.zalmoxis.newton_absolute_tolerance = newton_abs_tol
    config.interior_struct.zalmoxis.global_miscibility = False
    return config


def _stub_get_target_surface_pressure(monkeypatch, value=1.0e5):
    """Patch the surface-pressure helper so the builder doesn't probe outgassing."""
    import proteus.interior_struct.zalmoxis as _mod

    monkeypatch.setattr(_mod, '_get_target_surface_pressure', lambda *a, **kw: value)


def _make_hf_row():
    """Build a minimal hf_row that satisfies load_zalmoxis_configuration's
    volatile-element loop (M_volatiles = sum of *_kg_total over element_list).
    """
    return {
        'M_volatiles': 0.0,
        'H_kg_total': 0.0,
        'C_kg_total': 0.0,
        'N_kg_total': 0.0,
        'S_kg_total': 0.0,
        'Si_kg_total': 0.0,
        'Mg_kg_total': 0.0,
        'Fe_kg_total': 0.0,
        'Na_kg_total': 0.0,
    }


class TestPicardPathDoesNotChangeIntegratorTolerances:
    """``outer_solver='picard'`` must not pass ``relative_tolerance`` or
    ``absolute_tolerance`` in ``config_params``.

    Contract: a Picard run must build a ``config_params`` dict that omits
    the integrator-tolerance keys so Zalmoxis falls back to its
    mass-adaptive defaults (1e-5 / 1e-6). The 'outer_solver' key itself
    is present (=='picard'); only the tolerance overrides must be absent.
    """

    def test_picard_omits_integrator_tolerance_keys(self, monkeypatch):
        """When ``outer_solver='picard'``, ``load_zalmoxis_configuration``
        sets the ``outer_solver`` key but does NOT inject
        ``relative_tolerance`` / ``absolute_tolerance``, so Zalmoxis
        falls back to its mass-adaptive defaults.
        """
        from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

        config = _make_mock_config(outer_solver='picard')
        _stub_get_target_surface_pressure(monkeypatch)
        hf_row = _make_hf_row()

        cp = load_zalmoxis_configuration(config, hf_row)
        assert cp['outer_solver'] == 'picard'
        assert 'relative_tolerance' not in cp, (
            'Picard path must not override integrator tols; got '
            f'relative_tolerance={cp.get("relative_tolerance")}'
        )
        assert 'absolute_tolerance' not in cp


class TestNewtonPathTightensIntegratorTolerances:
    """outer_solver='newton' MUST also pass tightened integrator tols.

    Without these, the in-Zalmoxis Newton fails ValueError at entry
    (precondition: relative_tolerance <= 1e-7).
    """

    def test_newton_passes_relative_tolerance(self, monkeypatch):
        """When ``outer_solver='newton'``, ``load_zalmoxis_configuration``
        injects the tightened integrator tolerances (1e-9 relative,
        1e-10 absolute) plus the Newton-specific iteration cap and tol,
        so the Newton precondition is satisfied automatically.
        """
        from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

        config = _make_mock_config(outer_solver='newton')
        _stub_get_target_surface_pressure(monkeypatch)
        cp = load_zalmoxis_configuration(config, _make_hf_row())

        assert cp['outer_solver'] == 'newton'
        assert cp['relative_tolerance'] == pytest.approx(1.0e-9, rel=1e-12)
        assert cp['absolute_tolerance'] == pytest.approx(1.0e-10, rel=1e-12)
        assert cp['newton_max_iter'] == 30
        assert cp['newton_tol'] == pytest.approx(1.0e-4, rel=1e-12)

    def test_newton_propagates_custom_knobs(self, monkeypatch):
        """Non-default Newton knobs flow through unchanged."""
        from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

        config = _make_mock_config(
            outer_solver='newton',
            newton_max_iter=50,
            newton_tol=5.0e-5,
            newton_rel_tol=1.0e-10,
            newton_abs_tol=1.0e-11,
        )
        _stub_get_target_surface_pressure(monkeypatch)
        cp = load_zalmoxis_configuration(config, _make_hf_row())

        assert cp['newton_max_iter'] == 50
        assert cp['newton_tol'] == pytest.approx(5.0e-5, rel=1e-12)
        assert cp['relative_tolerance'] == pytest.approx(1.0e-10, rel=1e-12)
        assert cp['absolute_tolerance'] == pytest.approx(1.0e-11, rel=1e-12)

    def test_newton_tols_satisfy_zalmoxis_precondition(self, monkeypatch):
        """The default tols must satisfy Zalmoxis' Newton precondition.

        Discriminating: zalmoxis.solver._NEWTON_REQUIRED_REL_TOL is 1e-7;
        if PROTEUS' default newton_relative_tolerance ever drifts above
        this, Newton will ValueError at entry. This test pins the
        invariant.
        """
        from zalmoxis.solver import _NEWTON_REQUIRED_REL_TOL

        from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

        config = _make_mock_config(outer_solver='newton')
        _stub_get_target_surface_pressure(monkeypatch)
        cp = load_zalmoxis_configuration(config, _make_hf_row())

        assert cp['relative_tolerance'] <= _NEWTON_REQUIRED_REL_TOL, (
            f"PROTEUS' default newton_relative_tolerance "
            f'({cp["relative_tolerance"]:.0e}) must be <= '
            f'_NEWTON_REQUIRED_REL_TOL ({_NEWTON_REQUIRED_REL_TOL:.0e}); '
            'otherwise Newton ValueErrors on entry.'
        )
        # Sign / positivity guard: an integrator tolerance must be a
        # strictly positive real. A regression that set the default
        # to 0 or a negative number would still pass the
        # less-than-or-equal check above (0 <= 1e-7) but would crash
        # the SciPy integrator at entry.
        assert cp['relative_tolerance'] > 0.0
        # Scale guard: the default must also be well above the
        # smallest representable double (~5e-324). The Newton
        # precondition is 1e-7; the Zalmoxis-default is 1e-9. A
        # regression that bumped the default below 1e-15 would
        # produce silent integrator stalls instead of useful steps.
        assert cp['relative_tolerance'] >= 1.0e-15


@pytest.mark.unit
@pytest.mark.parametrize('use_anderson', [True, False])
def test_load_zalmoxis_configuration_propagates_use_anderson(monkeypatch, use_anderson):
    """Verify that load_zalmoxis_configuration propagates use_anderson."""
    from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

    config = _make_mock_config()
    config.interior_struct.zalmoxis.use_anderson = use_anderson
    _stub_get_target_surface_pressure(monkeypatch)
    cp = load_zalmoxis_configuration(config, _make_hf_row())
    assert cp['use_anderson'] is use_anderson


# ----------------------------------------------------------------------
# Core mass of the structure target
# ----------------------------------------------------------------------

_BUFFERED = {'H_kg_total': 4.70e20, 'C_kg_total': 2.73e20, 'O_kg_total': 2.117e23}


def _structure_target(
    monkeypatch, hf_extra, outgas='calliope', fO2_source='user_constant', **cfg
):
    """Planet target [kg] and core mass [kg] that the builder hands to Zalmoxis."""
    from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

    config = _make_mock_config()
    config.outgas.module, config.planet.fO2_source = outgas, fO2_source
    config.planet.mass_tot = cfg.get('mass_tot', 1.0)
    config.interior_struct.zalmoxis.dry_mantle = cfg.get('dry_mantle', True)
    _stub_get_target_surface_pressure(monkeypatch)
    params = load_zalmoxis_configuration(config, {**_make_hf_row(), **hf_extra})
    return params['planet_mass'], params['core_mass_fraction'] * params['planet_mass'], params


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_core_does_not_move_with_an_oxygen_inventory_that_the_buffer_derives(monkeypatch):
    """With a fixed fO2 the outgassing derives the oxygen: 2.117e23 kg over a hot magma ocean
    at IW+4, 3.15e21 kg when it has cooled. The planet target falls by the oxygen, and the
    core stays at 0.325 of the planet without its kept volatiles (H and C, 7.43e20 kg): the
    mantle alone gives and takes the oxygen. With core_frac itself as the fraction the core
    would differ by 0.325 of the oxygen change, 6.8e22 kg."""
    from proteus.utils.constants import M_earth

    hot_planet, hot_core, _ = _structure_target(monkeypatch, _BUFFERED)
    cool_planet, cool_core, _ = _structure_target(
        monkeypatch, {**_BUFFERED, 'O_kg_total': 3.15e21}
    )
    expected = 0.325 * (M_earth - 7.43e20)
    assert hot_core == pytest.approx(expected, rel=1e-14)
    assert cool_core == pytest.approx(expected, rel=1e-14)
    assert hot_planet == pytest.approx(M_earth - 7.43e20 - 2.117e23, rel=1e-14)
    assert cool_planet - hot_planet == pytest.approx(2.117e23 - 3.15e21, rel=1e-9)
    assert abs(0.325 * cool_planet - 0.325 * hot_planet) > 6.7e22


@pytest.mark.unit
@pytest.mark.parametrize(
    'outgas, fO2_source',
    [
        ('calliope', 'from_O_budget'),
        ('atmodeller', 'from_O_budget'),
        ('dummy', 'user_constant'),
    ],
)
def test_the_core_fraction_is_the_config_value_when_no_inventory_is_derived(
    monkeypatch, outgas, fO2_source
):
    """When the oxygen is an input of the outgassing (an oxygen budget, or the dummy
    module) every inventory is kept, and the solver gets core_frac itself, bit for bit."""
    _, _, params = _structure_target(monkeypatch, _BUFFERED, outgas, fO2_source)
    assert params['core_mass_fraction'] == 0.325


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'hf_extra, cfg, core_change',
    [
        ({'H_kg_total': 3.70e20, 'M_volatile_change': -1.0e20}, {}, 0.0),
        ({}, {'mass_tot': 1.05}, 0.325 * 0.05 * 5.972e24),
        ({'Si_kg_total': 1.0e21}, {}, 0.0),
    ],
    ids=['escape', 'impact', 'rock_vapour'],
)
def test_the_core_mass_follows_the_rock_and_nothing_else(
    monkeypatch, hf_extra, cfg, core_change
):
    """Hydrogen that escapes lowers the inventory and the volatile ledger by the same mass,
    so the core stays; an impact that adds 0.05 Earth masses of rock adds 0.325 of it to the
    core; and rock vapour is mantle mass, not a kept volatile."""
    _, base, _ = _structure_target(monkeypatch, _BUFFERED)
    _, core, _ = _structure_target(monkeypatch, {**_BUFFERED, **hf_extra}, **cfg)
    assert core - base == pytest.approx(core_change, rel=2e-3, abs=1e12)


@pytest.mark.unit
def test_an_inventory_that_leaves_no_mantle_is_refused(monkeypatch):
    """An oxygen inventory of 4.1e24 kg leaves 1.87e24 kg for the interior, less than the
    core of 1.94e24 kg: the builder refuses the structure with the masses in the message."""
    with pytest.raises(ValueError, match='leaves no mantle: 1.87'):
        _structure_target(monkeypatch, {**_BUFFERED, 'O_kg_total': 4.1e24})


_WET = {'H_kg_atm': 1.0e20, 'C_kg_atm': 2.0e20, 'O_kg_atm': 2.079e23, 'H_kg_total': 6.0e21}


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_a_wet_mantle_core_is_computed_in_the_init_stage_and_kept_after(monkeypatch):
    """With a wet mantle the structure target takes off the atmosphere only (2.082e23 kg
    here), and on a row of the init stage the core is 0.325 of the planet without the
    atmosphere mass of its kept elements (H and C, 3e20 kg; the derived oxygen is mantle
    mass). After the init stage the core is read from the row: when 5e21 kg of water have
    outgassed, the planet target is lower by that mass and the core is the kept one, to
    the last bit, not 0.325 of the new target."""
    from proteus.utils.constants import M_earth

    planet, core, params = _structure_target(monkeypatch, _WET, dry_mantle=False)
    assert planet == pytest.approx(M_earth - 2.082e23, rel=1e-14)
    assert core == pytest.approx(0.325 * (M_earth - 3.0e20), rel=1e-14)
    assert params['core_mass_target'] == pytest.approx(core, rel=1e-15)
    # A row of the init stage computes also when the row before left a value.
    _, again, _ = _structure_target(
        monkeypatch, {**_WET, 'Time': 0.0, 'M_core_target': 1.5e24}, dry_mantle=False
    )
    assert again == pytest.approx(core, rel=1e-15)

    kept = params['core_mass_target']
    later = {**_WET, 'H_kg_atm': 5.1e21, 'Time': 250.0, 'M_core_target': kept}
    planet_later, core_later, params_later = _structure_target(
        monkeypatch, later, dry_mantle=False
    )
    assert planet - planet_later == pytest.approx(5.0e21, rel=1e-9)
    assert params_later['core_mass_target'] == kept
    assert core_later == pytest.approx(core, rel=1e-14)
    assert abs(0.325 * (M_earth - 5.3e21) - core) > 1.0e21


@pytest.mark.unit
@pytest.mark.parametrize('stored', [None, 0.0], ids=['column_absent', 'zero_fill'])
def test_a_wet_resume_without_the_core_target_computes_it_with_a_warning(
    monkeypatch, caplog, stored
):
    """A helpfile row that predates the column, absent or filled with 0, gives a wet run
    after its init stage the core of the present row, with a warning that names it."""
    from proteus.utils.constants import M_earth

    row = {**_WET, 'Time': 250.0, **({} if stored is None else {'M_core_target': stored})}
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        _, core, _ = _structure_target(monkeypatch, row, dry_mantle=False)
    assert core == pytest.approx(0.325 * (M_earth - 3.0e20), rel=1e-14)
    assert 'holds no M_core_target' in caplog.text


@pytest.mark.unit
def test_a_dry_mantle_core_is_computed_on_every_row(monkeypatch):
    """With a dry mantle the core target is the value of the present row also after the
    init stage: a stored value of another size is not read."""
    row = {**_BUFFERED, 'Time': 250.0, 'M_core_target': 1.5e24}
    _, base, _ = _structure_target(monkeypatch, _BUFFERED)
    _, core, params = _structure_target(monkeypatch, row)
    assert params['core_mass_target'] == pytest.approx(base, rel=1e-15)
    assert core == pytest.approx(base, rel=1e-15)
