"""The aragog top_node option over its input domain."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from proteus.config._interior import Aragog

pytestmark = pytest.mark.unit


def test_the_top_node_defaults_to_linear_and_takes_limited():
    assert Aragog().top_node == 'linear'
    assert Aragog(top_node='limited').top_node == 'limited'


@given(
    value=st.one_of(
        st.text().filter(lambda v: v not in ('linear', 'limited')),
        st.booleans(),
        st.integers(),
        st.floats(),
        st.none(),
    )
)
def test_another_top_node_value_is_refused(value):
    """Another word or spelling, a TOML true, a number or nothing is refused."""
    with pytest.raises(ValueError):
        Aragog(top_node=value)
