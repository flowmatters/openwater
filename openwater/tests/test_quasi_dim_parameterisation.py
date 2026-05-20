"""Phase 4: quasi-dim integration in parameterisation classes."""
import h5py
import numpy as np
import pandas as pd
import pytest

from openwater import config as config_mod
from openwater import quasi_dim as qd_mod
from openwater.config import (
    DataframeInput,
    DataframeInputs,
    DictParameteriser,
    ParameterTableAssignment,
    SingleTimeseriesInput,
    _matches_constraints,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ModelDesc:
    '''Minimal model_desc object used by the parameteriser methods.'''
    def __init__(self, name='FakeModel', parameters=None, states=None,
                 inputs=None, outputs=None):
        self.name = name
        self.description = {
            'Parameters': [{'Name': p} for p in (parameters or [])],
            'States': states or [],
            'Inputs': inputs or [],
            'Outputs': outputs or [],
        }


def _make_resolver(real, registered):
    real_set = set(real)
    reg = qd_mod.QuasiDimRegistry(real_dim_names=lambda: real_set)
    for qd in registered:
        reg.add(qd)
    return qd_mod.QuasiDimResolver(
        real_dim_names=lambda: real_set, registry=reg)


def _nodes_and_df(node_specs):
    '''node_specs: list of dicts (must contain _run_idx, _model, tags).
    Returns (nodes_dict, nodes_df).
    '''
    nodes = {f'n{i}': dict(spec) for i, spec in enumerate(node_specs)}
    nodes_df = pd.DataFrame(list(nodes.values()))
    return nodes, nodes_df


def _h5_param_group(tmp_path, n_params, n_cells, n_states=0):
    '''Create an in-memory HDF5 group with the param dataset shapes the
    parameterisers expect.'''
    f = h5py.File(tmp_path / 'tmp.h5', 'w')
    grp = f.create_group('model')
    grp.create_dataset('parameters', shape=(n_params, n_cells), dtype=np.float64)
    if n_states:
        grp.create_dataset('states', shape=(n_cells, n_states), dtype=np.float64)
    return f, grp


# ---------------------------------------------------------------------------
# extend_nodes_df
# ---------------------------------------------------------------------------

def test_extend_nodes_df_adds_projected_column():
    resolver = _make_resolver(
        real=['SC'],
        registered=[qd_mod.from_dict({1: 'N', 2: 'N', 3: 'S'}, 'rc', 'SC')],
    )
    df = pd.DataFrame({'SC': [1, 3, 2], '_run_idx': [0, 1, 2]})
    out = resolver.extend_nodes_df(df, ['rc'])
    assert list(out.columns) == ['SC', '_run_idx', 'rc']
    assert list(out['rc']) == ['N', 'S', 'N']
    # Original untouched
    assert 'rc' not in df.columns


def test_extend_nodes_df_missing_real_dim_raises():
    resolver = _make_resolver(
        real=['SC'],
        registered=[qd_mod.from_dict({1: 'N'}, 'rc', 'SC')],
    )
    df = pd.DataFrame({'CGU': ['Ag']})
    with pytest.raises(qd_mod.QuasiDimensionError, match='not present in nodes_df'):
        resolver.extend_nodes_df(df, ['rc'])


# ---------------------------------------------------------------------------
# _matches_constraints with resolver
# ---------------------------------------------------------------------------

def test_matches_constraints_with_resolver_expands_quasi_dim():
    resolver = _make_resolver(
        real=['SC'],
        registered=[qd_mod.from_dict({1: 'N', 2: 'N', 3: 'S'}, 'rc', 'SC')],
    )
    assert _matches_constraints({'rc': 'N'}, {'SC': 1}, resolver=resolver)
    assert _matches_constraints({'rc': 'N'}, {'SC': 2}, resolver=resolver)
    assert not _matches_constraints({'rc': 'N'}, {'SC': 3}, resolver=resolver)


def test_matches_constraints_resolver_none_unchanged():
    # Pre-Phase-4 behaviour preserved when resolver is omitted.
    assert _matches_constraints({'SC': 1}, {'SC': 1})
    assert not _matches_constraints({'rc': 'N'}, {'SC': 1})  # no resolver: rc unknown


# ---------------------------------------------------------------------------
# DictParameteriser
# ---------------------------------------------------------------------------

def test_dict_parameteriser_with_quasi_dim_constraint(tmp_path):
    resolver = _make_resolver(
        real=['SC'],
        registered=[qd_mod.from_dict({1: 'N', 2: 'N', 3: 'S'}, 'rc', 'SC')],
    )
    nodes, nodes_df = _nodes_and_df([
        {'_model': 'FakeModel', '_run_idx': 0, 'SC': 1},
        {'_model': 'FakeModel', '_run_idx': 1, 'SC': 2},
        {'_model': 'FakeModel', '_run_idx': 2, 'SC': 3},
    ])
    f, grp = _h5_param_group(tmp_path, n_params=1, n_cells=3)
    try:
        dp = DictParameteriser(
            parameter='dwc',
            key_format='${SC}',
            model='FakeModel',
            parameters={'1': 5.0, '2': 7.0, '3': 9.0},
            constraints={'rc': 'N'},
        )
        md = _ModelDesc('FakeModel', parameters=['dwc'])
        dp.parameterise(md, grp, None, {'SC': [1, 2, 3]}, nodes, nodes_df,
                        resolver=resolver)
        # Only SC=1 and SC=2 (the N-mapped) should be written.
        result = grp['parameters'][0, :]
        assert result[0] == 5.0
        assert result[1] == 7.0
        assert result[2] == 0.0  # default — SC=3 maps to S, filtered out
    finally:
        f.close()


def test_dict_parameteriser_back_compat_no_resolver(tmp_path):
    nodes, nodes_df = _nodes_and_df([
        {'_model': 'FakeModel', '_run_idx': 0, 'SC': 1},
        {'_model': 'FakeModel', '_run_idx': 1, 'SC': 2},
    ])
    f, grp = _h5_param_group(tmp_path, n_params=1, n_cells=2)
    try:
        dp = DictParameteriser(
            parameter='dwc',
            key_format='${SC}',
            model='FakeModel',
            parameters={'1': 5.0, '2': 7.0},
            constraints={'SC': 1},
        )
        md = _ModelDesc('FakeModel', parameters=['dwc'])
        dp.parameterise(md, grp, None, {'SC': [1, 2]}, nodes, nodes_df)
        assert grp['parameters'][0, 0] == 5.0
        assert grp['parameters'][0, 1] == 0.0
    finally:
        f.close()


# ---------------------------------------------------------------------------
# ParameterTableAssignment._parameterise_nd
# ---------------------------------------------------------------------------

def test_parameter_table_with_quasi_dim_column(tmp_path):
    '''CSV keyed by reporting_catchment (one row per RC) broadcasts to SC rows.'''
    resolver = _make_resolver(
        real=['SC'],
        registered=[qd_mod.from_dict(
            {1: 'North', 2: 'North', 3: 'South'}, 'rc', 'SC')],
    )
    nodes, nodes_df = _nodes_and_df([
        {'_model': 'FakeModel', '_run_idx': 0, 'SC': 1},
        {'_model': 'FakeModel', '_run_idx': 1, 'SC': 2},
        {'_model': 'FakeModel', '_run_idx': 2, 'SC': 3},
    ])
    df = pd.DataFrame({
        'rc': ['North', 'South'],
        'dwc': [0.5, 1.5],
    })
    f, grp = _h5_param_group(tmp_path, n_params=1, n_cells=3)
    try:
        pta = ParameterTableAssignment(df, model='FakeModel')
        md = _ModelDesc('FakeModel', parameters=['dwc'])
        pta.parameterise(md, grp, None, {'SC': [1, 2, 3]}, nodes, nodes_df,
                         resolver=resolver)
        result = grp['parameters'][0, :]
        assert result[0] == 0.5   # SC1 → North
        assert result[1] == 0.5   # SC2 → North
        assert result[2] == 1.5   # SC3 → South
    finally:
        f.close()


def test_parameter_table_real_and_quasi_columns(tmp_path):
    '''Table with both a real dim and a quasi-dim column — joins on both.'''
    resolver = _make_resolver(
        real=['SC', 'CGU'],
        registered=[qd_mod.from_dict(
            {1: 'North', 2: 'South'}, 'rc', 'SC')],
    )
    nodes, nodes_df = _nodes_and_df([
        {'_model': 'FakeModel', '_run_idx': 0, 'SC': 1, 'CGU': 'Ag'},
        {'_model': 'FakeModel', '_run_idx': 1, 'SC': 1, 'CGU': 'Forest'},
        {'_model': 'FakeModel', '_run_idx': 2, 'SC': 2, 'CGU': 'Ag'},
    ])
    df = pd.DataFrame({
        'rc': ['North', 'North', 'South'],
        'CGU': ['Ag', 'Forest', 'Ag'],
        'dwc': [1.0, 2.0, 3.0],
    })
    f, grp = _h5_param_group(tmp_path, n_params=1, n_cells=3)
    try:
        pta = ParameterTableAssignment(df, model='FakeModel')
        md = _ModelDesc('FakeModel', parameters=['dwc'])
        pta.parameterise(md, grp, None,
                         {'SC': [1, 2], 'CGU': ['Ag', 'Forest']},
                         nodes, nodes_df, resolver=resolver)
        result = grp['parameters'][0, :]
        assert result[0] == 1.0  # SC1 (North), Ag
        assert result[1] == 2.0  # SC1 (North), Forest
        assert result[2] == 3.0  # SC2 (South), Ag
    finally:
        f.close()


# ---------------------------------------------------------------------------
# ParameterTableAssignment._parameterise_2d
# ---------------------------------------------------------------------------

def test_parameter_table_2d_with_quasi_row_dim(tmp_path):
    resolver = _make_resolver(
        real=['SC', 'Cons'],
        registered=[qd_mod.from_dict({1: 'N', 2: 'S'}, 'rc', 'SC')],
    )
    # 2D table: rows = rc, columns = Cons
    df = pd.DataFrame({
        'TSS': [10.0, 20.0],
        'TN':  [11.0, 22.0],
    }, index=['N', 'S'])

    nodes, nodes_df = _nodes_and_df([
        {'_model': 'FakeModel', '_run_idx': 0, 'SC': 1, 'Cons': 'TSS'},
        {'_model': 'FakeModel', '_run_idx': 1, 'SC': 2, 'Cons': 'TN'},
    ])
    f, grp = _h5_param_group(tmp_path, n_params=1, n_cells=2)
    try:
        pta = ParameterTableAssignment(
            df, model='FakeModel', parameter='dwc',
            row_dim='rc', column_dim='Cons',
        )
        md = _ModelDesc('FakeModel', parameters=['dwc'])
        pta.parameterise(md, grp, None,
                         {'SC': [1, 2], 'Cons': ['TSS', 'TN']},
                         nodes, nodes_df, resolver=resolver)
        result = grp['parameters'][0, :]
        assert result[0] == 10.0  # SC1 → rc=N, Cons=TSS
        assert result[1] == 22.0  # SC2 → rc=S, Cons=TN
    finally:
        f.close()


# ---------------------------------------------------------------------------
# SingleTimeseriesInput
# ---------------------------------------------------------------------------

def test_single_timeseries_input_with_quasi_dim_tag(tmp_path):
    resolver = _make_resolver(
        real=['SC'],
        registered=[qd_mod.from_dict({1: 'N', 2: 'S', 3: 'N'}, 'rc', 'SC')],
    )
    nodes, nodes_df = _nodes_and_df([
        {'_model': 'FakeModel', '_run_idx': 0, 'SC': 1},
        {'_model': 'FakeModel', '_run_idx': 1, 'SC': 2},
        {'_model': 'FakeModel', '_run_idx': 2, 'SC': 3},
    ])
    n_timesteps = 4
    series = np.array([1.0, 2.0, 3.0, 4.0])
    f = h5py.File(tmp_path / 'sti.h5', 'w')
    grp = f.create_group('model')
    grp.create_dataset('inputs', shape=(3, 1, n_timesteps), dtype=np.float64)
    try:
        sti = SingleTimeseriesInput(series, 'rainfall',
                                    model='FakeModel', rc='N')
        md = _ModelDesc('FakeModel', inputs=['rainfall'])
        sti.parameterise(md, grp, None, {'SC': [1, 2, 3]}, nodes, nodes_df,
                         resolver=resolver)
        # rc='N' → SC in {1, 3}. Series written to run_idx 0 and 2.
        assert np.array_equal(grp['inputs'][0, 0, :], series)
        assert np.array_equal(grp['inputs'][1, 0, :], np.zeros(n_timesteps))
        assert np.array_equal(grp['inputs'][2, 0, :], series)
    finally:
        f.close()


# ---------------------------------------------------------------------------
# DataframeInput.get_series via the resolver path
# ---------------------------------------------------------------------------

def test_dataframe_input_get_series_with_quasi_dim_constraint():
    resolver = _make_resolver(
        real=['SC'],
        registered=[qd_mod.from_dict({1: 'N', 2: 'S'}, 'rc', 'SC')],
    )
    df = pd.DataFrame({'rain-1': [1.0, 2.0], 'rain-2': [3.0, 4.0]})
    di = DataframeInput(
        dataframe=df,
        column_format='rain-${SC}',
        model='FakeModel',
        constraint_tags={'rc': 'N'},
    )
    out = di.get_series(resolver=resolver, SC=1)
    assert np.array_equal(out, np.array([1.0, 2.0]))
    out = di.get_series(resolver=resolver, SC=2)
    assert out is None  # SC=2 → S, doesn't match rc=N


# ---------------------------------------------------------------------------
# Chained quasi-dim
# ---------------------------------------------------------------------------

def test_chained_quasi_dim_in_dict_parameteriser(tmp_path):
    rc = qd_mod.from_dict({1: 'N', 2: 'N', 3: 'S'}, 'rc', 'SC')
    rr = qd_mod.from_dict({'N': 'NE', 'S': 'SW'}, 'rr', 'rc')
    resolver = _make_resolver(real=['SC'], registered=[rc, rr])
    nodes, nodes_df = _nodes_and_df([
        {'_model': 'FakeModel', '_run_idx': 0, 'SC': 1},
        {'_model': 'FakeModel', '_run_idx': 1, 'SC': 3},
    ])
    f, grp = _h5_param_group(tmp_path, n_params=1, n_cells=2)
    try:
        dp = DictParameteriser(
            parameter='dwc',
            key_format='${SC}',
            model='FakeModel',
            parameters={'1': 5.0, '3': 9.0},
            constraints={'rr': 'NE'},
        )
        md = _ModelDesc('FakeModel', parameters=['dwc'])
        dp.parameterise(md, grp, None, {'SC': [1, 3]}, nodes, nodes_df,
                        resolver=resolver)
        # rr='NE' → rc='N' → SC=1. Only node 0 written.
        assert grp['parameters'][0, 0] == 5.0
        assert grp['parameters'][0, 1] == 0.0
    finally:
        f.close()
