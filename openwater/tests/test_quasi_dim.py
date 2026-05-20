"""Tests for the quasi_dim module (Phase 0 — pure module, no integration)."""
import os
import tempfile

import pandas as pd
import pytest

from openwater import quasi_dim as qd_mod
from openwater.quasi_dim import (
    QuasiDimension,
    QuasiDimRegistry,
    CycleError,
    NameCollisionError,
    PartialCoverageError,
    UnknownDimensionError,
    from_csv,
    from_dict,
    from_series,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_from_dict_basic():
    qd = from_dict({1: 'North', 2: 'North', 3: 'South'},
                   name='reporting_catchment', keyed_by='SC')
    assert qd.name == 'reporting_catchment'
    assert qd.keyed_by == 'SC'
    assert list(qd.mapping.index) == [1, 2, 3]
    assert list(qd.mapping.values) == ['North', 'North', 'South']
    assert qd.mapping.index.name == 'SC'
    assert qd.mapping.name == 'reporting_catchment'
    assert not qd.has_default


def test_from_series_infers_name_and_keyed_by():
    s = pd.Series(['A', 'A', 'B'], index=pd.Index([1, 2, 3], name='SC'),
                  name='grp')
    qd = from_series(s)
    assert qd.name == 'grp'
    assert qd.keyed_by == 'SC'


def test_from_series_overrides():
    s = pd.Series(['A', 'B'], index=[1, 2])
    qd = from_series(s, name='zone', keyed_by='SC')
    assert qd.name == 'zone'
    assert qd.keyed_by == 'SC'


def test_from_series_missing_name_raises():
    s = pd.Series(['A', 'B'], index=[1, 2])
    with pytest.raises(ValueError, match='name'):
        from_series(s)


def test_from_series_missing_keyed_by_raises():
    s = pd.Series(['A', 'B'], index=[1, 2], name='grp')
    with pytest.raises(ValueError, match='keyed_by'):
        from_series(s)


def test_from_csv(tmp_path):
    csv = tmp_path / 'rc.csv'
    csv.write_text("SC,reporting_catchment\n1,North\n2,North\n3,South\n")
    qd = from_csv(str(csv), key='SC', value='reporting_catchment')
    assert qd.name == 'reporting_catchment'
    assert qd.keyed_by == 'SC'
    assert qd.mapping.loc[2] == 'North'


def test_from_csv_overrides(tmp_path):
    csv = tmp_path / 'rc.csv'
    csv.write_text("a,b\n1,X\n2,Y\n")
    qd = from_csv(str(csv), key='a', value='b', name='zone', keyed_by='SC')
    assert qd.name == 'zone'
    assert qd.keyed_by == 'SC'


def test_from_csv_missing_column(tmp_path):
    csv = tmp_path / 'rc.csv'
    csv.write_text("SC,foo\n1,X\n")
    with pytest.raises(ValueError, match="reporting_catchment"):
        from_csv(str(csv), key='SC', value='reporting_catchment')


def test_from_csv_missing_file():
    with pytest.raises(FileNotFoundError):
        from_csv('/no/such/file.csv', key='a', value='b')


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_duplicate_index_rejected():
    s = pd.Series(['North', 'South'], index=[1, 1])
    with pytest.raises(ValueError, match='unique'):
        QuasiDimension('rc', 'SC', s)


def test_unhashable_value_rejected():
    s = pd.Series([[1, 2], [3]], index=[1, 2])
    with pytest.raises(TypeError, match='hashable'):
        QuasiDimension('rc', 'SC', s)


def test_self_keyed_rejected():
    s = pd.Series(['A'], index=[1])
    with pytest.raises(CycleError):
        QuasiDimension('foo', 'foo', s)


def test_empty_name_rejected():
    s = pd.Series(['A'], index=[1])
    with pytest.raises(ValueError):
        QuasiDimension('', 'SC', s)
    with pytest.raises(ValueError):
        QuasiDimension('foo', '', s)


def test_external_mutation_isolated():
    s = pd.Series(['A', 'B'], index=[1, 2], name='grp')
    qd = QuasiDimension('grp', 'SC', s)
    s.iloc[0] = 'CHANGED'
    assert qd.mapping.iloc[0] == 'A'


# ---------------------------------------------------------------------------
# project / partial coverage
# ---------------------------------------------------------------------------

def test_project_basic():
    qd = from_dict({1: 'North', 2: 'North', 3: 'South'}, 'rc', 'SC')
    vals = pd.Series([1, 3, 2])
    out = qd.project(vals)
    assert list(out) == ['North', 'South', 'North']
    assert out.name == 'rc'


def test_project_partial_coverage_raises():
    qd = from_dict({1: 'North', 2: 'North'}, 'rc', 'SC')
    with pytest.raises(PartialCoverageError, match=r"\[3\]"):
        qd.project(pd.Series([1, 3]))


def test_project_with_default():
    qd = from_dict({1: 'North'}, 'rc', 'SC', default='Unassigned')
    out = qd.project(pd.Series([1, 2, 3]))
    assert list(out) == ['North', 'Unassigned', 'Unassigned']


# ---------------------------------------------------------------------------
# Registry — basic
# ---------------------------------------------------------------------------

def _registry(real=('SC', 'CGU', 'Cons')):
    real_set = set(real)
    return QuasiDimRegistry(real_dim_names=lambda: real_set)


def test_registry_add_and_lookup():
    reg = _registry()
    qd = from_dict({1: 'N', 2: 'S'}, 'rc', 'SC')
    reg.add(qd)
    assert 'rc' in reg
    assert reg['rc'] is qd
    assert reg.names() == ['rc']
    assert len(reg) == 1
    assert reg.is_quasi('rc')
    assert not reg.is_quasi('SC')


def test_registry_persistence_flag():
    reg = _registry()
    reg.add(from_dict({1: 'N'}, 'rc', 'SC'))
    reg.add(from_dict({1: 'A'}, 'lu', 'SC'), persist=True)
    assert reg.is_persisted('lu')
    assert not reg.is_persisted('rc')


def test_registry_collision_with_real_dim():
    reg = _registry()
    with pytest.raises(NameCollisionError, match='real dimension'):
        reg.add(from_dict({1: 'X'}, 'SC', 'CGU'))


def test_registry_collision_with_existing_quasi():
    reg = _registry()
    reg.add(from_dict({1: 'N'}, 'rc', 'SC'))
    with pytest.raises(NameCollisionError, match='already registered'):
        reg.add(from_dict({1: 'X'}, 'rc', 'SC'))


def test_registry_remove():
    reg = _registry()
    reg.add(from_dict({1: 'N'}, 'rc', 'SC'))
    reg.remove('rc')
    assert 'rc' not in reg


def test_registry_remove_unknown():
    reg = _registry()
    with pytest.raises(KeyError):
        reg.remove('nope')


def test_registry_remove_blocked_by_dependent():
    reg = _registry()
    reg.add(from_dict({1: 'N', 2: 'S'}, 'rc', 'SC'))
    reg.add(from_dict({'N': 'NE', 'S': 'SW'}, 'rr', 'rc'))
    with pytest.raises(qd_mod.QuasiDimensionError, match='depended on'):
        reg.remove('rc')
    reg.remove('rr')
    reg.remove('rc')  # now allowed


def test_registry_re_register_after_remove():
    reg = _registry()
    reg.add(from_dict({1: 'N'}, 'rc', 'SC'))
    reg.remove('rc')
    reg.add(from_dict({1: 'S'}, 'rc', 'SC'))
    assert reg['rc'].mapping.loc[1] == 'S'


# ---------------------------------------------------------------------------
# Registry — cycle detection
# ---------------------------------------------------------------------------

def test_direct_cycle_via_existing_chain():
    reg = _registry()
    reg.add(from_dict({1: 'N'}, 'a', 'SC'))
    reg.add(from_dict({'N': 'X'}, 'b', 'a'))
    # registering c keyed by b, then trying to make a keyed by c would cycle.
    # We can't re-add 'a', but we can simulate the case by trying to add a
    # quasi-dim whose chain loops back into itself via forward references.
    # Simpler direct case: keyed_by points to a name that resolves back.
    reg.add(from_dict({'X': 'P'}, 'c', 'b'))
    # Now attempting to register a quasi-dim 'a2' keyed by 'c' that we then
    # also point 'a' at would cycle — but 'a' already exists. Use forward
    # references: add d keyed by e, e keyed by d.
    reg.add(from_dict({1: 'q'}, 'd', 'e'))  # forward ref allowed
    with pytest.raises(CycleError):
        reg.add(from_dict({1: 'r'}, 'e', 'd'))


def test_forward_reference_allowed_until_resolution():
    reg = _registry()
    # 'rc' refers to 'mid' which doesn't exist yet — allowed at add time.
    reg.add(from_dict({'A': 'X'}, 'rc', 'mid'))
    # but resolving fails until 'mid' is added
    with pytest.raises(UnknownDimensionError):
        reg.resolve_chain('rc')
    reg.add(from_dict({1: 'A'}, 'mid', 'SC'))
    real, composed = reg.resolve_chain('rc')
    assert real == 'SC'
    assert composed.mapping.loc[1] == 'X'


# ---------------------------------------------------------------------------
# Registry — chain composition
# ---------------------------------------------------------------------------

def test_resolve_chain_single_link():
    reg = _registry()
    reg.add(from_dict({1: 'N', 2: 'S'}, 'rc', 'SC'))
    real, composed = reg.resolve_chain('rc')
    assert real == 'SC'
    assert composed.name == 'rc'
    assert composed.keyed_by == 'SC'
    assert composed.mapping.loc[1] == 'N'
    assert composed.mapping.loc[2] == 'S'


def test_resolve_chain_two_links():
    reg = _registry()
    reg.add(from_dict({1: 'N', 2: 'N', 3: 'S'}, 'rc', 'SC'))
    reg.add(from_dict({'N': 'NE', 'S': 'SW'}, 'rr', 'rc'))
    real, composed = reg.resolve_chain('rr')
    assert real == 'SC'
    assert composed.keyed_by == 'SC'
    assert composed.mapping.loc[1] == 'NE'
    assert composed.mapping.loc[3] == 'SW'


def test_resolve_chain_three_links():
    reg = _registry()
    reg.add(from_dict({1: 'A', 2: 'B'}, 'l1', 'SC'))
    reg.add(from_dict({'A': 'X', 'B': 'Y'}, 'l2', 'l1'))
    reg.add(from_dict({'X': 'P', 'Y': 'Q'}, 'l3', 'l2'))
    real, composed = reg.resolve_chain('l3')
    assert real == 'SC'
    assert composed.mapping.loc[1] == 'P'
    assert composed.mapping.loc[2] == 'Q'


def test_resolve_chain_partial_coverage_propagates():
    reg = _registry()
    reg.add(from_dict({1: 'N', 2: 'S'}, 'rc', 'SC'))
    # 'rr' missing the 'S' bucket and has no default
    reg.add(from_dict({'N': 'NE'}, 'rr', 'rc'))
    with pytest.raises(PartialCoverageError):
        reg.resolve_chain('rr')


def test_resolve_chain_default_propagates():
    reg = _registry()
    reg.add(from_dict({1: 'N', 2: 'S'}, 'rc', 'SC'))
    reg.add(from_dict({'N': 'NE'}, 'rr', 'rc', default='Other'))
    real, composed = reg.resolve_chain('rr')
    assert composed.mapping.loc[1] == 'NE'
    assert composed.mapping.loc[2] == 'Other'
    assert composed.has_default
    assert composed.default == 'Other'


def test_resolve_chain_on_real_dim_rejected():
    reg = _registry()
    with pytest.raises(ValueError):
        reg.resolve_chain('SC')


def test_resolve_chain_unknown_name():
    reg = _registry()
    with pytest.raises(KeyError):
        reg.resolve_chain('nope')
