"""Unit tests for ``proteus.outgas.fo2_warnings``.

The warnings read the helpfile and never change it. Oxygen derived by a fixed fO2 leaves the
Zalmoxis structure mass; a rise of it above ``O_RISE_WARN_FRAC`` of the mantle since the first
row warns once per crossing. The IW buffer evaluated outside its calibrated temperature range
warns once per run.

Invariants tested:
  - the oxygen warning fires iff the rise crosses the threshold from below, on either side of it
  - the buffer warning fires once, on the first row outside the range, for the module's range
  - a run resumed from its helpfile warns as an uninterrupted one
  - non-finite or non-positive inputs never raise and never warn

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip('hypothesis')

from hypothesis import given, settings
from hypothesis import strategies as st

from proteus.outgas.fo2_warnings import (
    BUFFER_T_RANGE_K,
    O_RISE_WARN_FRAC,
    check_fo2_warnings,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

LOGGER = 'fwl.proteus.outgas.fo2_warnings'
O_MSG = 'Oxygen derived by the fixed fO2'
T_MSG = 'outside the calibrated range'
M_MANTLE = 3.46e25  # kg, mantle of an 8.6 M_E planet
O0 = 1.6e23  # kg, derived O at the first row


def _config(
    module='calliope', source='user_constant', struct='zalmoxis', dry=True, T_floor=700.0
):
    return SimpleNamespace(
        outgas=SimpleNamespace(module=module, T_floor=T_floor, fO2_shift_IW=2.0),
        planet=SimpleNamespace(fO2_source=source),
        interior_struct=SimpleNamespace(
            module=struct, zalmoxis=SimpleNamespace(dry_mantle=dry)
        ),
    )


def _rows(rises, T=2500.0, O_atm=None):
    """Helpfile with O_kg_total = O0 + rise * M_mantle per row and a fixed T_magma."""
    T = [T] * len(rises) if not isinstance(T, list) else T
    return pd.DataFrame(
        {
            'Time': [float(i) for i in range(len(rises))],
            'O_kg_total': [O0 + r * M_MANTLE for r in rises],
            'O_kg_atm': O_atm if O_atm is not None else [1e22] * len(rises),
            'M_mantle': [M_MANTLE] * len(rises),
            'T_magma': T,
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
def test_oxygen_warning_once_per_crossing(caplog):
    """Rows above warn once; a fall below and a new rise warn again."""
    f = O_RISE_WARN_FRAC
    n_o, _ = _run(_config(), _rows([0.0, 2 * f, 3 * f, 4 * f, 0.5 * f, 2 * f, 5 * f]), caplog)
    assert n_o == 2
    assert 'leaves the structure mass' in caplog.text


@pytest.mark.physics_invariant
def test_oxygen_warning_names_values_and_setting(caplog):
    """The message gives the derived mass, its fraction of the mantle and the fO2 setting."""
    _run(_config(), _rows([0.0, 0.02]), caplog)
    assert f'{0.02 * M_MANTLE:.3e} kg' in caplog.text
    assert '2.00 % of the mantle' in caplog.text
    assert 'outgas.fO2_shift_IW = +2.00' in caplog.text


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'config',
    [
        _config(source='from_O_budget'),
        _config(module='dummy'),
        _config(struct='spider'),
    ],
    ids=['o_budget_sets_fo2', 'dummy_outgassing', 'spider_structure'],
)
def test_no_oxygen_warning_when_no_derived_o_leaves_the_structure(caplog, config):
    """Without a fixed fO2 or the Zalmoxis structure no mass leaves the mantle this way."""
    n_o, n_t = _run(config, _rows([0.0, 0.1, 0.2]), caplog)
    assert n_o == 0
    assert n_t == 0


@pytest.mark.physics_invariant
def test_wet_mantle_reads_the_atmosphere_oxygen(caplog):
    """With dry_mantle off only O_kg_atm leaves the structure: a rise of O_kg_total alone is quiet."""
    quiet = _rows([0.0, 0.1], O_atm=[1e22, 1e22])
    assert _run(_config(dry=False), quiet, caplog)[0] == 0
    loud = _rows([0.0, 0.0], O_atm=[1e22, 1e22 + 0.01 * M_MANTLE])
    assert _run(_config(dry=False), loud, caplog)[0] == 1


@pytest.mark.physics_invariant
@pytest.mark.parametrize('resume_at', [2, 3, 5])
def test_resumed_run_warns_as_an_uninterrupted_run(caplog, resume_at):
    """A process that starts from a stored helpfile emits the warnings of the rows after it."""
    f = O_RISE_WARN_FRAC
    hf = _rows(
        [0.0, 2 * f, 3 * f, 0.5 * f, 0.4 * f, 2 * f], T=[3500.0, 3600, 2500, 2500, 3200, 2500]
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
    assert sum(t for _, t in full) == 1


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('module', 'T', 'n'),
    [
        ('calliope', 3000.1, 1),
        ('calliope', 2999.9, 0),
        ('calliope', 999.9, 1),
        ('calliope', 1000.1, 0),
        ('atmodeller', 3000.1, 1),
        ('atmodeller', 999.9, 0),
    ],
    ids=[
        'calliope_hot',
        'calliope_inside_top',
        'calliope_cold',
        'calliope_inside_bottom',
        'atmodeller_hot',
        'atmodeller_cold_has_low_branch',
    ],
)
def test_buffer_warning_at_the_range_edges(caplog, module, T, n):
    """The buffer warns outside 1000-3000 K for calliope and above 3000 K for atmodeller."""
    assert BUFFER_T_RANGE_K[module][1] == pytest.approx(3000.0, rel=0)
    _, n_t = _run(_config(module=module), _rows([0.0, 0.0], T=[2000.0, T]), caplog)
    assert n_t == n
    if n:
        assert 'Hirschmann (2021, GCA 313, 74)' in caplog.text
        assert f'{max(T, 700.0):.1f} K' in caplog.text


@pytest.mark.physics_invariant
def test_buffer_warning_once_per_run(caplog):
    """Rows outside, inside and outside again give one message, at the first row outside."""
    _, n_t = _run(
        _config(), _rows([0.0] * 5, T=[3980.0, 4100.0, 2500.0, 3500.0, 3600.0]), caplog, start=1
    )
    assert n_t == 1
    assert '3980.0 K' in caplog.text


@pytest.mark.physics_invariant
def test_buffer_temperature_is_the_floored_magma_temperature(caplog):
    """Both modules evaluate the buffer at max(T_magma, T_floor): a floor inside the range is quiet."""
    _, n_t = _run(_config(T_floor=1200.0), _rows([0.0, 0.0], T=[500.0, 600.0]), caplog)
    assert n_t == 0
    _, n_t = _run(_config(T_floor=700.0), _rows([0.0, 0.0], T=[500.0, 600.0]), caplog)
    assert n_t == 1
    assert '700.0 K' in caplog.text


@pytest.mark.physics_invariant
def test_empty_helpfile_and_missing_columns_are_quiet(caplog):
    """No rows, or rows without the oxygen and temperature columns, give no warning and no error."""
    with caplog.at_level('WARNING', logger=LOGGER):
        check_fo2_warnings(_config(), pd.DataFrame())
        check_fo2_warnings(_config(), pd.DataFrame({'Time': [0.0, 1.0]}))
    assert caplog.text.count(O_MSG) == 0
    assert caplog.text.count(T_MSG) == 0


_odd = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True),
    st.just(0.0),
    st.just(-1.0),
    st.booleans(),
)


@pytest.mark.physics_invariant
@settings(max_examples=200, deadline=None)
@given(o_last=_odd, m_mantle=_odd, T=_odd)
def test_odd_row_values_never_raise_and_warn_only_on_a_finite_rise(o_last, m_mantle, T):
    """NaN, inf, 0, negative and boolean column values never raise; the oxygen warning needs a finite rise."""
    hf = pd.DataFrame(
        {
            'O_kg_total': [O0, o_last],
            'O_kg_atm': [1e22, 1e22],
            'M_mantle': [M_MANTLE, m_mantle],
            'T_magma': [2000.0, T],
        }
    )
    import logging

    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger(LOGGER)
    logger.addHandler(handler)
    try:
        check_fo2_warnings(_config(), hf)
    finally:
        logger.removeHandler(handler)
    o_warned = any(O_MSG in r.getMessage() for r in records)
    ok = all(isinstance(v, float) and math.isfinite(v) for v in (o_last, m_mantle))
    rise = (o_last - O0) / m_mantle if ok and m_mantle > 0 else math.nan
    assert o_warned == (rise > O_RISE_WARN_FRAC)
    if o_warned:
        assert m_mantle > 0
