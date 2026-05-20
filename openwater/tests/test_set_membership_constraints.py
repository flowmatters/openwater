"""Phase 1: set-membership constraints in config._matches_constraints and
OpenwaterResults filtering.

Covers:
* config._constraint_matches / _matches_constraints — scalar vs iterable, string-is-scalar.
* results._is_set_constraint / _index_run_map — multi-list cartesian product.
* OpenwaterResults.time_series / table — end-to-end on a small synthetic HDF5.
* time_series filter_tags / **kwargs overlap detection.
"""
import os
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest

from openwater import config as config_mod
from openwater import results as results_mod
from openwater.results import OpenwaterResults


# ---------------------------------------------------------------------------
# config._constraint_matches / _matches_constraints
# ---------------------------------------------------------------------------

def test_constraint_matches_scalar():
    assert config_mod._constraint_matches('A', 'A')
    assert not config_mod._constraint_matches('A', 'B')
    assert config_mod._constraint_matches(7, 7)


def test_constraint_matches_set_membership():
    assert config_mod._constraint_matches('A', ['A', 'B'])
    assert config_mod._constraint_matches('A', ('A', 'B'))
    assert config_mod._constraint_matches('A', {'A', 'B'})
    assert config_mod._constraint_matches('A', frozenset(['A', 'B']))
    assert config_mod._constraint_matches(2, np.array([1, 2, 3]))
    assert not config_mod._constraint_matches('C', ['A', 'B'])


def test_constraint_matches_string_not_iterated_characterwise():
    # 'AB' is a scalar — must not match 'A' via character iteration.
    assert not config_mod._constraint_matches('A', 'AB')
    assert config_mod._constraint_matches('AB', 'AB')


def test_matches_constraints_mixed():
    present = {'SC': 1, 'CGU': 'Ag', 'Cons': 'TSS'}
    assert config_mod._matches_constraints({'SC': [1, 2], 'Cons': 'TSS'}, present)
    assert not config_mod._matches_constraints({'SC': [3, 4]}, present)
    assert config_mod._matches_constraints({}, present)
    assert config_mod._matches_constraints(None, present)


def test_matches_constraints_missing_key():
    assert not config_mod._matches_constraints({'absent': 1}, {'SC': 1})


# ---------------------------------------------------------------------------
# results._is_set_constraint / _index_run_map
# ---------------------------------------------------------------------------

def test_is_set_constraint():
    assert results_mod._is_set_constraint([1, 2])
    assert results_mod._is_set_constraint((1, 2))
    assert results_mod._is_set_constraint({1, 2})
    assert results_mod._is_set_constraint(np.array([1, 2]))
    assert not results_mod._is_set_constraint(1)
    assert not results_mod._is_set_constraint('SC1')
    assert not results_mod._is_set_constraint(None)


def test_index_run_map_basic_slicing_unchanged():
    rm = np.arange(24).reshape(2, 3, 4)
    out = results_mod._index_run_map(rm, [slice(None), 1, slice(None)])
    assert out.shape == (2, 4)
    np.testing.assert_array_equal(out, rm[:, 1, :])


def test_index_run_map_single_list_in_place():
    rm = np.arange(24).reshape(2, 3, 4)
    out = results_mod._index_run_map(rm, [slice(None), [0, 2], slice(None)])
    np.testing.assert_array_equal(out, rm[:, [0, 2], :])


def test_index_run_map_multi_list_is_cartesian():
    # The whole point of this helper: two list indices must Cartesian-product,
    # not broadcast pairwise.
    rm = np.arange(24).reshape(2, 3, 4)
    out = results_mod._index_run_map(rm, [[0, 1], [0, 2], slice(None)])
    # Pairwise broadcasting would give shape (2, 4): rm[(0,1),(0,2),:].
    # We want shape (2, 2, 4) — every combination.
    assert out.shape == (2, 2, 4)
    np.testing.assert_array_equal(out[0, 0], rm[0, 0])
    np.testing.assert_array_equal(out[0, 1], rm[0, 2])
    np.testing.assert_array_equal(out[1, 0], rm[1, 0])
    np.testing.assert_array_equal(out[1, 1], rm[1, 2])


def test_index_run_map_multi_list_with_int_and_slice():
    rm = np.arange(60).reshape(2, 3, 2, 5)
    out = results_mod._index_run_map(rm, [[0, 1], 1, [0, 1], slice(None)])
    # Expected: every (i,1,k,:) for i in {0,1}, k in {0,1}.
    assert out.shape == (2, 1, 2, 5)
    np.testing.assert_array_equal(out[0, 0, 0], rm[0, 1, 0])
    np.testing.assert_array_equal(out[1, 0, 1], rm[1, 1, 1])


# ---------------------------------------------------------------------------
# End-to-end on a synthetic HDF5
# ---------------------------------------------------------------------------

class _FakeModelType:
    name = 'FakeModel'
    description = {
        'Inputs': ['rainfall'],
        'Outputs': ['runoff'],
        'States': [],
        'Parameters': [],
    }


@pytest.fixture
def synthetic_results(monkeypatch, tmp_path):
    '''Build a minimal model/results HDF5 pair on disk and an OpenwaterResults.

    Model: FakeModel with two dims (SC of size 3, CGU of size 2). Every cell
    is mapped — six runs total. The output value of run k at every timestep
    is k * 10, so it's trivial to assert which runs were included.
    '''
    # Make the fake type discoverable via openwater.nodes.<name>.
    from openwater import nodes as node_types
    monkeypatch.setattr(node_types, 'FakeModel', _FakeModelType, raising=False)

    model_path = tmp_path / 'model.h5'
    results_path = tmp_path / 'results.h5'
    n_sc, n_cgu = 3, 2
    n_runs = n_sc * n_cgu
    n_timesteps = 4

    # Run-id layout: row-major over (SC, CGU).
    run_map = np.arange(n_runs).reshape(n_sc, n_cgu)

    with h5py.File(model_path, 'w') as f:
        dims = f.create_group('DIMENSIONS')
        dims.create_dataset('SC', data=np.array(['SC1', 'SC2', 'SC3'], dtype='S'))
        dims.create_dataset('CGU', data=np.array(['Ag', 'Forest'], dtype='S'))
        models = f.create_group('MODELS')
        fm = models.create_group('FakeModel')
        ds = fm.create_dataset('map', data=run_map.astype(np.int64))
        ds.attrs['DIMS'] = np.array([b'SC', b'CGU'])
        meta = f.create_group('META')
        meta.create_dataset(
            'timeperiod',
            data=np.array([
                f'2020-01-0{i+1}T00:00:00' for i in range(n_timesteps)
            ], dtype='S'),
        )

    with h5py.File(results_path, 'w') as f:
        models = f.create_group('MODELS')
        fm = models.create_group('FakeModel')
        # outputs shape: (n_runs, n_outputs=1, n_timesteps)
        out = np.zeros((n_runs, 1, n_timesteps), dtype=np.float64)
        for k in range(n_runs):
            out[k, 0, :] = k * 10.0
        fm.create_dataset('outputs', data=out)

    res = OpenwaterResults(str(model_path), str(results_path))
    yield res
    res.close()


def test_time_series_scalar_constraint(synthetic_results):
    # filter to CGU=Ag (CGU index 0) — runs 0, 2, 4 — columns by SC.
    df = synthetic_results.time_series('FakeModel', 'runoff', 'SC',
                                       aggregator='mean', CGU='Ag')
    assert list(df.columns) == ['SC1', 'SC2', 'SC3']
    # Values per column are the single run's value (no aggregation needed).
    assert df['SC1'].iloc[0] == 0.0
    assert df['SC2'].iloc[0] == 20.0
    assert df['SC3'].iloc[0] == 40.0


def test_time_series_set_constraint(synthetic_results):
    df = synthetic_results.time_series('FakeModel', 'runoff', 'SC',
                                       aggregator='mean',
                                       CGU=['Ag', 'Forest'])
    # Each SC column averages both CGUs: mean of (k, k+1)*10 for the two CGUs.
    # SC1: runs 0,1 → mean 5; SC2: 2,3 → 25; SC3: 4,5 → 45.
    assert df['SC1'].iloc[0] == 5.0
    assert df['SC2'].iloc[0] == 25.0
    assert df['SC3'].iloc[0] == 45.0


def test_time_series_set_equals_repeated_scalars(synthetic_results):
    # filter SC to {SC1, SC3} reporting by CGU — equivalent to a manual union.
    df_set = synthetic_results.time_series('FakeModel', 'runoff', 'CGU',
                                           aggregator='mean',
                                           SC=['SC1', 'SC3'])
    # By CGU: Ag includes runs 0 (SC1,Ag) and 4 (SC3,Ag) → mean 20.
    #          Forest includes runs 1 (SC1,F) and 5 (SC3,F) → mean 30.
    assert df_set['Ag'].iloc[0] == 20.0
    assert df_set['Forest'].iloc[0] == 30.0


def test_time_series_single_element_set_matches_scalar(synthetic_results):
    df_scalar = synthetic_results.time_series('FakeModel', 'runoff', 'SC',
                                              aggregator='mean', CGU='Ag')
    df_set = synthetic_results.time_series('FakeModel', 'runoff', 'SC',
                                           aggregator='mean', CGU=['Ag'])
    pd.testing.assert_frame_equal(df_scalar, df_set)


def test_table_set_constraint(synthetic_results):
    # rows=SC, columns=CGU; restrict to SC in {SC1,SC3}. Each cell is one run.
    tbl = synthetic_results.table('FakeModel', 'runoff', 'SC', 'CGU',
                                  temporal_aggregator='mean',
                                  SC=['SC1', 'SC3'])
    # With the SC constraint, only those rows have meaningful values; the
    # current implementation still emits all three rows (the constraint
    # narrows nothing on the rows axis because rows IS the SC dim). The
    # constrained rows should equal the unconstrained value; SC2 still
    # appears with its underlying mean.
    assert tbl.loc['SC1', 'Ag'] == 0.0
    assert tbl.loc['SC3', 'Forest'] == 50.0


def test_filter_tags_kwargs_overlap_raises(synthetic_results):
    with pytest.raises(ValueError, match='both filter_tags and kwargs'):
        synthetic_results.time_series('FakeModel', 'runoff', 'SC',
                                      aggregator='mean',
                                      CGU='Ag',
                                      filter_tags={'CGU': 'Forest'})


def test_invalid_set_value_raises(synthetic_results):
    with pytest.raises(Exception, match='Invalid value'):
        synthetic_results.time_series('FakeModel', 'runoff', 'SC',
                                      aggregator='mean',
                                      CGU=['Ag', 'NonExistent'])
