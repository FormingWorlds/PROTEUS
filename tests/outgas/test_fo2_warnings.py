"""Unit tests for ``proteus.outgas.fo2_warnings``.

The warnings read the helpfile and never change it. Oxygen derived by a fixed fO2 leaves the
Zalmoxis structure mass; a rise of it above ``O_RISE_WARN_FRAC`` of the mantle since the first
finite row warns once per crossing. The IW buffer evaluated outside its calibrated temperature
range warns once per run on each side of the range.

Invariants tested:
  - the oxygen warning fires iff the rise crosses the threshold from below, on either side of it
  - the column the warning reads is the one the structure target removes
  - the buffer warning fires once per side, on the first row outside the range that ran an outgassing call
  - a run resumed from its helpfile warns as an uninterrupted one
  - non-finite or non-positive inputs never raise and never warn

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip('hypothesis')

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from proteus.outgas import fo2_warnings
from proteus.outgas.fo2_warnings import (
    BUFFER_T_RANGE_K,
    O_RISE_WARN_FRAC,
    check_fo2_warnings,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

LOGGER = 'fwl.proteus.outgas.fo2_warnings'
O_MSG = 'Oxygen derived by the fixed fO2'
T_MSG = 'outside the calibrated temperature range'
M_MANTLE = 3.46e25  # kg, mantle of an 8.6 M_E planet
O0 = 1.6e23  # kg, derived O at the first row
QUICK = Path(__file__).parents[2] / 'input' / 'minimal_quick.toml'


@pytest.fixture(autouse=True)
def _fresh_calliope_default():
    """Each test reads the CALLIOPE default buffer again."""
    fo2_warnings._calliope_default_buffer.cache_clear()
    yield
    fo2_warnings._calliope_default_buffer.cache_clear()


def _config(
    module='calliope',
    source='user_constant',
    struct='zalmoxis',
    dry=True,
    T_floor=700.0,
    update=1e9,
    accretion=None,
):
    return SimpleNamespace(
        outgas=SimpleNamespace(module=module, T_floor=T_floor, fO2_shift_IW=2.0),
        planet=SimpleNamespace(fO2_source=source),
        accretion=SimpleNamespace(module=accretion),
        interior_struct=SimpleNamespace(
            module=struct,
            zalmoxis=None
            if update is None
            else SimpleNamespace(dry_mantle=dry, update_interval=update),
        ),
    )


def _rows(rises, T=2500.0, O_atm=None, o0=O0, mantle=M_MANTLE, called=None):
    """Helpfile with O_kg_total = o0 + rise * mantle per row, the given T_magma and outgas_called."""
    T = [T] * len(rises) if not isinstance(T, list) else T
    return pd.DataFrame(
        {
            'Time': [float(i) for i in range(len(rises))],
            'O_kg_total': [o0 + r * mantle for r in rises],
            'O_kg_atm': O_atm if O_atm is not None else [1e22] * len(rises),
            'M_mantle': [mantle] * len(rises),
            'T_magma': T,
            'outgas_called': called if called is not None else [1.0] * len(rises),
        }
    )


def _run(config, hf_all, caplog, start=1):
    """Call the check after each appended row from row ``start``, as the main loop does."""
    caplog.clear()
    with caplog.at_level('WARNING', logger=LOGGER):
        for k in range(start, len(hf_all) + 1):
            check_fo2_warnings(config, hf_all.iloc[:k])
    return caplog.text.count(O_MSG), caplog.text.count(T_MSG)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('last', 'n'),
    [(1.01 * O_RISE_WARN_FRAC, 1), (0.99 * O_RISE_WARN_FRAC, 0)],
    ids=['just_above', 'just_below'],
)
def test_oxygen_warning_fires_only_above_the_threshold(caplog, last, n):
    """The rise warns strictly above O_RISE_WARN_FRAC of the mantle, on both sides of it."""
    assert O_RISE_WARN_FRAC == pytest.approx(1e-3, rel=0)
    n_o, n_t = _run(_config(), _rows([0.0, 0.5 * O_RISE_WARN_FRAC, last]), caplog)
    assert n_o == n
    assert n_t == 0


@pytest.mark.physics_invariant
def test_oxygen_threshold_is_strict_at_the_exact_value(caplog):
    """A rise of exactly 1e-3 is quiet; a row above it after a row at it warns."""
    exact = _rows(
        [0.0, 0.001], o0=0.0, mantle=1000.0
    )  # O = 1.0 kg; 1.0 / 1000.0 is 1e-3 exactly
    assert exact['O_kg_total'].iloc[1] == pytest.approx(1.0, rel=0, abs=0)
    assert exact['O_kg_total'].iloc[1] / 1000.0 == O_RISE_WARN_FRAC
    assert _run(_config(), exact, caplog)[0] == 0
    assert _run(_config(), _rows([0.0, 0.001, 0.002], o0=0.0, mantle=1000.0), caplog)[0] == 1


@pytest.mark.physics_invariant
def test_oxygen_warning_once_per_crossing(caplog):
    """Rows above warn once; a fall below and a new rise warn again."""
    f = O_RISE_WARN_FRAC
    n_o, n_t = _run(_config(), _rows([0.0, 2 * f, 3 * f, 4 * f, 0.5 * f, 2 * f, 5 * f]), caplog)
    assert n_o == 2
    assert n_t == 0


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('dry', 'outside'),
    [
        (
            True,
            'dry_mantle = true all of it is outside the structure mass, and the mantle loses that mass',
        ),
        (
            False,
            f'only O_kg_atm is outside the structure mass (its change since that row: {0.005 * M_MANTLE:.3e} kg), and',
        ),
    ],
    ids=['dry_mantle', 'wet_mantle'],
)
def test_oxygen_warning_names_values_and_the_mass_outside_the_structure(caplog, dry, outside):
    """The message gives the O_kg_total rise, its fraction of the mantle and the mass outside the structure."""
    O_atm = [1e22, 1e22 + 0.005 * M_MANTLE]
    assert _run(_config(dry=dry), _rows([0.0, 0.02], O_atm=O_atm), caplog)[0] == 1
    assert f'rose by {0.02 * M_MANTLE:.3e} kg' in caplog.text
    assert f'2.00 % of the mantle mass of that row ({M_MANTLE:.3e} kg)' in caplog.text
    assert 'outgas.fO2_shift_IW = +2.00' in caplog.text
    assert outside in caplog.text


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'config',
    [
        _config(source='from_O_budget'),
        _config(module='dummy'),
        _config(struct='spider'),
        _config(update=0.0),
        _config(struct='spider', update=None),
    ],
    ids=[
        'o_budget_sets_fo2',
        'dummy_outgassing',
        'spider_structure',
        'structure_never_resolved',
        'no_zalmoxis_settings',
    ],
)
def test_no_oxygen_warning_when_no_derived_o_leaves_the_structure(caplog, config):
    """Without a fixed fO2, the Zalmoxis structure or a structure solve in the run, no mass leaves."""
    n_o, n_t = _run(config, _rows([0.0, 0.1, 0.2]), caplog)
    assert n_o == 0
    assert n_t == 0


@pytest.mark.physics_invariant
def test_an_impact_resolves_the_structure_without_update_interval(caplog):
    """With update_interval 0 an accretion module still re-solves the structure, so the rise warns."""
    assert _run(_config(update=0.0, accretion='dummy'), _rows([0.0, 0.01]), caplog)[0] == 1
    assert _run(_config(update=0.0, accretion=None), _rows([0.0, 0.01]), caplog)[0] == 0


@pytest.mark.physics_invariant
def test_wet_mantle_warns_on_the_derived_oxygen_not_on_degassing(caplog):
    """With dry_mantle off, O_kg_atm that rises by degassing at a constant O_kg_total is quiet;
    a rise of the derived O_kg_total warns."""
    degassing = _rows([0.0, 0.0], O_atm=[1e22, 1e22 + 0.01 * M_MANTLE])
    assert _run(_config(dry=False), degassing, caplog)[0] == 0
    derived = _rows([0.0, 0.01], O_atm=[1e22, 1e22])
    assert _run(_config(dry=False), derived, caplog)[0] == 1


@pytest.mark.physics_invariant
def test_rise_is_relative_to_the_mantle_mass_of_the_baseline_row(caplog):
    """An impact that doubles the mantle does not hide a rise above the threshold of the first row,
    and a mantle that halves does not raise one below it."""
    f = O_RISE_WARN_FRAC
    grown = _rows([0.0, 1.5 * f])
    grown.loc[1, 'M_mantle'] = 2 * M_MANTLE
    assert _run(_config(update=0.0, accretion='dummy'), grown, caplog)[0] == 1
    assert f'rose by {1.5 * f * M_MANTLE:.3e} kg' in caplog.text
    assert f'0.15 % of the mantle mass of that row ({M_MANTLE:.3e} kg)' in caplog.text
    shrunk = _rows([0.0, 0.75 * f])
    shrunk.loc[1, 'M_mantle'] = 0.5 * M_MANTLE
    assert _run(_config(update=0.0, accretion='dummy'), shrunk, caplog)[0] == 0


@pytest.mark.physics_invariant
@pytest.mark.parametrize('dry', [True, False], ids=['dry_mantle', 'wet_mantle'])
def test_the_warned_column_is_the_one_the_structure_target_removes(dry):
    """Mass added to the warned column leaves the structure target; mass in the other column does not."""
    from proteus.config import read_config_object
    from proteus.interior_struct.zalmoxis import _structure_target_masses

    config = read_config_object(QUICK)
    config.interior_struct.zalmoxis.dry_mantle = dry
    col, other = ('O_kg_total', 'O_kg_atm') if dry else ('O_kg_atm', 'O_kg_total')
    row = {'O_kg_total': 2e21, 'O_kg_atm': 1e21, 'H_kg_total': 3e20, 'H_kg_atm': 2e20}
    dm = 7e20
    base = _structure_target_masses(config, row)
    moved = _structure_target_masses(config, {**row, col: row[col] + dm})
    assert moved[1] - base[1] == pytest.approx(dm, rel=1e-12)
    assert moved[2] == base[2]  # the kept part holds no O
    assert _structure_target_masses(config, {**row, other: row[other] + 0.5 * dm})[1] == base[1]


@pytest.mark.physics_invariant
def test_real_config_reaches_both_warnings(caplog):
    """A config loaded from input/ has every setting the checks read, and both warnings fire."""
    from proteus.config import read_config_object

    config = read_config_object(QUICK)
    config.interior_struct.zalmoxis.update_interval = 1e9
    n_o, n_t = _run(config, _rows([0.0, 0.01], T=[2000.0, 3500.0]), caplog)
    assert (n_o, n_t) == (1, 1)
    assert 'calliope is evaluated at 3500.0 K' in caplog.text


@pytest.mark.physics_invariant
@pytest.mark.parametrize('resume_at', [2, 3, 5])
def test_resumed_run_warns_as_an_uninterrupted_run(caplog, resume_at):
    """A process that starts from a stored helpfile emits the warnings of the rows after it."""
    f = O_RISE_WARN_FRAC
    hf = _rows(
        [0.0, 2 * f, 3 * f, 0.5 * f, 0.4 * f, 2 * f], T=[3500.0, 3600, 2500, 2500, 3200, 900]
    )
    full = []
    with caplog.at_level('WARNING', logger=LOGGER):
        for k in range(1, len(hf) + 1):
            caplog.clear()
            check_fo2_warnings(_config(), hf.iloc[:k])
            full.append((caplog.text.count(O_MSG), caplog.text.count(T_MSG)))
    resumed = []
    with caplog.at_level('WARNING', logger=LOGGER):
        for k in range(resume_at + 1, len(hf) + 1):
            caplog.clear()
            check_fo2_warnings(_config(), hf.iloc[:k])
            resumed.append((caplog.text.count(O_MSG), caplog.text.count(T_MSG)))
    assert resumed == full[resume_at:]
    assert sum(o for o, _ in full) == 2
    assert sum(t for _, t in full) == 2


@pytest.mark.physics_invariant
def test_baseline_is_the_first_finite_row_and_a_gap_row_keeps_the_crossing(caplog):
    """A NaN first row does not silence the run; a NaN row between two rows above does not warn
    twice; a NaN, zero or negative mantle mass in the baseline row warns on no row."""
    hf = _rows([0.0, 0.0, 0.02])
    hf.loc[0, 'O_kg_total'] = math.nan
    assert _run(_config(), hf, caplog)[0] == 1
    hf = _rows([0.0, 0.02, 0.02, 0.03])
    hf.loc[2, 'O_kg_total'] = math.nan
    assert _run(_config(), hf, caplog)[0] == 1
    for bad in (math.nan, 0.0, -M_MANTLE):
        hf = _rows([0.0, 0.02])
        hf.loc[0, 'M_mantle'] = bad
        assert _run(_config(), hf, caplog)[0] == 0, bad


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('module', 'T', 'n'),
    [
        ('calliope', 3000.1, 1),
        ('calliope', 3000.0, 0),
        ('calliope', 999.9, 1),
        ('calliope', 1000.0, 0),
        ('atmodeller', 3000.1, 1),
        ('atmodeller', 999.9, 0),
    ],
    ids=[
        'calliope_hot',
        'calliope_top_edge',
        'calliope_cold',
        'calliope_bottom_edge',
        'atmodeller_hot',
        'atmodeller_cold_has_low_branch',
    ],
)
def test_buffer_warning_at_the_range_edges(caplog, module, T, n):
    """The buffer warns outside 1000-3000 K for calliope and above 3000 K for atmodeller; edges are inside."""
    assert BUFFER_T_RANGE_K[module][1] == pytest.approx(3000.0, rel=0)
    _, n_t = _run(_config(module=module), _rows([0.0, 0.0], T=[2000.0, T]), caplog)
    assert n_t == n
    if n:
        assert 'Hirschmann (2021, GCA 313, 74)' in caplog.text
        assert f'{max(T, 700.0):.1f} K' in caplog.text
        assert ('1000-3000 K' if module == 'calliope' else '(at most 3000 K)') in caplog.text


@pytest.mark.physics_invariant
def test_buffer_warns_once_per_side_of_the_range(caplog):
    """A hot start, a return inside, a second hot row and a cold row give 2 messages: hot, then cold."""
    _, n_t = _run(
        _config(), _rows([0.0] * 5, T=[3980.0, 4100.0, 2500.0, 3500.0, 800.0]), caplog
    )
    assert n_t == 2
    assert caplog.text.index('3980.0 K') < caplog.text.index('800.0 K')


@pytest.mark.physics_invariant
def test_buffer_warning_counts_only_rows_that_ran_an_outgassing_call(caplog):
    """Rows without an outgassing call are quiet and do not block the first outgassed row outside."""
    hf = _rows([0.0] * 4, T=[2500.0, 900.0, 850.0, 800.0], called=[1.0, 0.0, 0.0, 0.0])
    assert _run(_config(), hf, caplog)[1] == 0
    T = [2500.0, 900.0, 850.0, 3500.0, 3600.0]
    hf = _rows([0.0] * 5, T=T, called=[1.0, 0.0, 1.0, 0.0, 1.0])
    assert _run(_config(), hf, caplog)[1] == 2
    assert '850.0 K' in caplog.text
    assert '3600.0 K' in caplog.text
    hf = _rows([0.0] * 2, T=[2500.0, 900.0]).drop(columns='outgas_called')
    assert _run(_config(), hf, caplog)[1] == 0  # a row of an older run reads as no call


@pytest.mark.physics_invariant
def test_buffer_warning_needs_the_hirschmann_default_of_calliope(caplog, monkeypatch):
    """A CALLIOPE whose default buffer is Fischer (2011) has no stated range: no buffer warning."""
    import calliope.oxygen_fugacity as fug

    monkeypatch.setattr(fug, 'DEFAULT_FO2_MODEL', 'fischer')
    assert _run(_config(), _rows([0.0, 0.0], T=[2000.0, 3500.0]), caplog)[1] == 0
    assert (
        _run(_config(module='atmodeller'), _rows([0.0, 0.0], T=[2000.0, 3500.0]), caplog)[1]
        == 1
    )
    monkeypatch.setattr(fug, 'DEFAULT_FO2_MODEL', 'hirschmann')
    fo2_warnings._calliope_default_buffer.cache_clear()
    assert _run(_config(), _rows([0.0, 0.0], T=[2000.0, 3500.0]), caplog)[1] == 1


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('source', 'effect', 'other'),
    [
        ('user_constant', 'its fO2 there is an extrapolation', 'fO2_shift_IW_derived'),
        (
            'from_O_budget',
            'the reported fO2_shift_IW_derived is relative to an extrapolated buffer',
            'its fO2 there',
        ),
    ],
)
def test_buffer_warning_names_what_the_extrapolation_affects(caplog, source, effect, other):
    """A fixed fO2 is itself extrapolated; under from_O_budget only the reported offset is."""
    assert _run(_config(source=source), _rows([0.0, 0.0], T=[2000.0, 3500.0]), caplog)[1] == 1
    assert 'IW buffer of calliope is evaluated at 3500.0 K' in caplog.text
    assert effect in caplog.text
    assert other not in caplog.text


@pytest.mark.physics_invariant
def test_calliope_without_a_named_default_buffer_skips_the_check_with_one_line(
    caplog, monkeypatch
):
    """A CALLIOPE that names no DEFAULT_FO2_MODEL logs one line for the run and no buffer warning."""
    import calliope.oxygen_fugacity as fug

    monkeypatch.delattr(fug, 'DEFAULT_FO2_MODEL')
    assert _run(_config(), _rows([0.0] * 3, T=[3500.0, 3600.0, 800.0]), caplog)[1] == 0
    assert caplog.text.count('names no DEFAULT_FO2_MODEL') == 1
    assert (
        _run(_config(module='atmodeller'), _rows([0.0, 0.0], T=[2000.0, 3500.0]), caplog)[1]
        == 1
    )


@pytest.mark.physics_invariant
@pytest.mark.parametrize('T', [1000.0, 1250.0, 2000.0, 3000.0, 4000.0, 6000.0, 8100.0])
def test_calliope_default_buffer_equals_atmodeller_from_1000_k(T):
    """Both modules evaluate the Hirschmann (2021) IW at 1 bar from 1000 K to 8100 K."""
    from calliope.oxygen_fugacity import OxygenFugacity

    pytest.importorskip('atmodeller')
    from atmodeller.thermodata import IronWustiteBuffer

    atm = float(IronWustiteBuffer().log10_fugacity_buffer(T, 1.0))
    assert OxygenFugacity()(T) == pytest.approx(atm, rel=1e-12, abs=1e-12)
    assert BUFFER_T_RANGE_K['calliope'][0] == pytest.approx(1000.0, rel=0)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(('T', 'diff'), [(800.0, -0.1013), (8200.0, -42.745)])
def test_outside_1000_to_8100_k_the_two_buffers_differ(T, diff):
    """Below 1000 K atmodeller uses its low-temperature buffer, above about 8150 K the hcp iron
    branch of the 2021 fit, at least 42 dex lower; CALLIOPE extrapolates the fcc/bcc branch."""
    from calliope.oxygen_fugacity import OxygenFugacity

    pytest.importorskip('atmodeller')
    from atmodeller.thermodata import IronWustiteBuffer

    atm = float(IronWustiteBuffer().log10_fugacity_buffer(T, 1.0))
    calliope = OxygenFugacity()(T)
    assert atm - calliope == pytest.approx(diff, abs=1e-3)
    assert atm < calliope


@pytest.mark.physics_invariant
def test_buffer_temperature_is_the_floored_magma_temperature(caplog):
    """Both modules evaluate the buffer at max(T_magma, T_floor): a floor inside the range is quiet."""
    _, n_t = _run(_config(T_floor=1200.0), _rows([0.0, 0.0], T=[500.0, 600.0]), caplog)
    assert n_t == 0
    _, n_t = _run(_config(T_floor=700.0), _rows([0.0, 0.0], T=[500.0, 600.0]), caplog)
    assert n_t == 1
    assert '700.0 K' in caplog.text


@pytest.mark.physics_invariant
def test_empty_helpfile_missing_columns_and_unknown_module_are_quiet(caplog):
    """No rows, rows without the columns, and a module without a buffer range give no warning or error."""
    with caplog.at_level('WARNING', logger=LOGGER):
        check_fo2_warnings(_config(), pd.DataFrame())
        check_fo2_warnings(
            _config(), pd.DataFrame(columns=['T_magma', 'O_kg_total', 'M_mantle'])
        )
        check_fo2_warnings(_config(), pd.DataFrame({'Time': [0.0, 1.0]}))
        check_fo2_warnings(_config(module='lavatmos'), _rows([0.0, 0.0], T=[3500.0, 3600.0]))
    assert caplog.text.count(O_MSG) == 0
    assert caplog.text.count(T_MSG) == 0


_odd = st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.just(0.0), st.just(-1.0))


@pytest.mark.physics_invariant
@settings(
    max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(o_first=_odd, o_last=_odd, m_mantle=_odd, T=_odd)
@example(o_first=0.0, o_last=0.0, m_mantle=M_MANTLE, T=math.inf)
@example(o_first=math.nan, o_last=1e22, m_mantle=-1.0, T=2000.0)
def test_odd_row_values_never_raise_and_warn_only_on_finite_inputs(
    caplog, o_first, o_last, m_mantle, T
):
    """NaN, inf, 0 and negative values never raise; each warning needs finite inputs past its limit."""
    hf = pd.DataFrame(
        {
            'O_kg_total': [o_first, o_last],
            'O_kg_atm': [1e22, 1e22],
            'M_mantle': [M_MANTLE, m_mantle],
            'T_magma': [2000.0, T],
            'outgas_called': [1.0, 1.0],
        }
    )
    caplog.clear()
    with caplog.at_level('WARNING', logger=LOGGER):
        check_fo2_warnings(_config(), hf)  # a log-only check must not raise
    msgs = [r.getMessage() for r in caplog.records]
    base, m_base = (o_first, M_MANTLE) if math.isfinite(o_first) else (o_last, m_mantle)
    finite = all(math.isfinite(v) for v in (o_last, base, m_base))
    rise = (o_last - base) / m_base if finite and m_base > 0 else math.nan
    assert any(O_MSG in m for m in msgs) == (rise > O_RISE_WARN_FRAC)
    T_used = max(T, 700.0)
    assert any(T_MSG in m for m in msgs) == (
        math.isfinite(T) and not 1000.0 <= T_used <= 3000.0
    )


def test_run_outgassing_records_a_solve_that_runs(tmp_path):
    """A solve that runs leaves outgas_called at 1, also when the row arrived at 0."""
    from unittest.mock import MagicMock, patch

    from proteus.outgas.wrapper import run_outgassing
    from proteus.utils.constants import gas_list

    config = MagicMock()
    config.outgas.module = 'calliope'
    config.planet.fO2_source = 'user_constant'
    config.interior_struct.zalmoxis.global_miscibility = False
    config.outgas.h2_binodal = False
    config.outgas.fO2_shift_IW = 2.0
    hf_row = {f'{s}_{k}': 0.0 for s in gas_list for k in ('kg_atm', 'vmr', 'bar')}
    hf_row.update(P_surf=0.0, atm_kg_per_mol=0.018, outgas_called=0.0)
    with patch('proteus.outgas.calliope.calc_surface_pressures') as solve:
        run_outgassing({'output': str(tmp_path)}, config, hf_row, initial=False)
    solve.assert_called_once()
    assert hf_row['outgas_called'] == pytest.approx(1.0, rel=0, abs=0)
    assert hf_row['fO2_shift_IW_derived'] == pytest.approx(2.0, rel=0, abs=0)


@pytest.mark.physics_invariant
def test_resume_from_a_helpfile_without_outgas_called_repeats_the_buffer_warning_once(
    tmp_path, caplog
):
    """An older helpfile reads back with outgas_called zero-filled, so its hot rows read as no call
    and the first hot row after the resume warns once more."""
    from proteus.utils.coupler import (
        CreateHelpfileFromDict,
        ExtendHelpfile,
        ReadHelpfileFromCSV,
        WriteHelpfileToCSV,
        ZeroHelpfileRow,
    )

    rows = []
    for i, T in enumerate([3500.0, 3600.0, 3700.0]):
        row = ZeroHelpfileRow()
        row.update(
            Time=float(i), T_magma=T, M_mantle=M_MANTLE, O_kg_total=O0, outgas_called=1.0
        )
        rows.append(row)
    hf = CreateHelpfileFromDict(rows[0])
    for row in rows[1:]:
        hf = ExtendHelpfile(hf, row)
    assert _run(_config(), hf, caplog)[1] == 1  # uninterrupted: the first hot row only
    WriteHelpfileToCSV(str(tmp_path), hf.iloc[:2])
    csv = tmp_path / 'runtime_helpfile.csv'
    older = pd.read_csv(csv, sep=r'\s+').drop(columns='outgas_called')
    older.to_csv(csv, sep='\t', index=False)
    stored = ReadHelpfileFromCSV(str(tmp_path))
    assert stored['outgas_called'].tolist() == [0.0, 0.0]
    resumed = ExtendHelpfile(stored, rows[2])
    assert _run(_config(), resumed, caplog, start=3)[1] == 1
    assert 'evaluated at 3700.0 K' in caplog.text


@pytest.mark.physics_invariant
def test_a_skipped_atmodeller_solve_records_no_outgassing_call(caplog):
    """Under from_O_budget with no budget above the threshold atmodeller evaluates no buffer, so
    the row records no call and a hot temperature gives no buffer warning."""
    from proteus.config import read_config_object
    from proteus.outgas.wrapper import run_outgassing
    from proteus.utils.coupler import ZeroHelpfileRow

    pytest.importorskip('atmodeller')
    config = read_config_object(QUICK)
    object.__setattr__(config.outgas, 'module', 'atmodeller')
    object.__setattr__(config.planet, 'fO2_source', 'from_O_budget')
    row = ZeroHelpfileRow()
    row.update(T_magma=3500.0, outgas_called=1.0)
    run_outgassing({'output': ''}, config, row, initial=False)
    assert math.isnan(row['fO2_shift_IW_derived'])
    assert row['outgas_called'] == 0.0
    hf = pd.DataFrame([{**row, 'T_magma': 2000.0, 'outgas_called': 1.0}, row])
    assert _run(config, hf, caplog)[1] == 0
