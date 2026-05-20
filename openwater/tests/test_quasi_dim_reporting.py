"""Phase 3: quasi-dim integration in OpenwaterResults.time_series / table."""
import h5py
import numpy as np
import pandas as pd
import pytest

from openwater import quasi_dim as qd_mod
from openwater.results import OpenwaterResults


class _FakeModelType:
    name = 'FakeModel'
    description = {
        'Inputs': ['rainfall'],
        'Outputs': ['runoff'],
        'States': [],
        'Parameters': [],
    }


def _write_synthetic(model_path, results_path, quasi_dims=None):
    '''Synthetic FakeModel with dims SC (4) x CGU (2). Run id k yields the
    constant value k * 10 at every timestep, so cell values are predictable.

    quasi_dims: optional list of QuasiDimension to persist.
    '''
    sc_values = ['SC1', 'SC2', 'SC3', 'SC4']
    cgu_values = ['Ag', 'Forest']
    n_sc, n_cgu = len(sc_values), len(cgu_values)
    n_runs = n_sc * n_cgu
    n_timesteps = 3
    run_map = np.arange(n_runs).reshape(n_sc, n_cgu)

    with h5py.File(model_path, 'w') as f:
        dims = f.create_group('DIMENSIONS')
        dims.create_dataset('SC', data=np.array([s.encode() for s in sc_values]))
        dims.create_dataset('CGU', data=np.array([s.encode() for s in cgu_values]))
        models = f.create_group('MODELS')
        fm = models.create_group('FakeModel')
        ds = fm.create_dataset('map', data=run_map.astype(np.int64))
        ds.attrs['DIMS'] = np.array([b'SC', b'CGU'])
        meta = f.create_group('META')
        meta.create_dataset('timeperiod',
                            data=np.array([f'2020-01-0{i+1}T00:00:00'
                                           for i in range(n_timesteps)],
                                          dtype='S'))
        if quasi_dims:
            real_set = {'SC', 'CGU'}
            reg = qd_mod.QuasiDimRegistry(real_dim_names=lambda: real_set)
            for qd in quasi_dims:
                reg.add(qd, persist=True)
            reg.write_to_h5(meta)

    with h5py.File(results_path, 'w') as f:
        models = f.create_group('MODELS')
        fm = models.create_group('FakeModel')
        out = np.zeros((n_runs, 1, n_timesteps), dtype=np.float64)
        for k in range(n_runs):
            out[k, 0, :] = k * 10.0
        fm.create_dataset('outputs', data=out)


@pytest.fixture
def results_with_rc(monkeypatch, tmp_path):
    '''Results with one quasi-dim: reporting_catchment keyed by SC.'''
    from openwater import nodes as node_types
    monkeypatch.setattr(node_types, 'FakeModel', _FakeModelType, raising=False)

    rc = qd_mod.from_dict({
        'SC1': 'North', 'SC2': 'North',
        'SC3': 'South', 'SC4': 'South',
    }, name='reporting_catchment', keyed_by='SC')

    model_path = tmp_path / 'model.h5'
    results_path = tmp_path / 'results.h5'
    _write_synthetic(model_path, results_path, quasi_dims=[rc])

    res = OpenwaterResults(str(model_path), str(results_path))
    yield res
    res.close()


@pytest.fixture
def results_with_chain(monkeypatch, tmp_path):
    '''reporting_catchment(SC -> N/S) and reporting_region(rc -> NE/SW).'''
    from openwater import nodes as node_types
    monkeypatch.setattr(node_types, 'FakeModel', _FakeModelType, raising=False)

    rc = qd_mod.from_dict({
        'SC1': 'North', 'SC2': 'North',
        'SC3': 'South', 'SC4': 'South',
    }, name='reporting_catchment', keyed_by='SC')
    rr = qd_mod.from_dict({
        'North': 'NorthEast', 'South': 'SouthWest',
    }, name='reporting_region', keyed_by='reporting_catchment')

    model_path = tmp_path / 'model.h5'
    results_path = tmp_path / 'results.h5'
    _write_synthetic(model_path, results_path, quasi_dims=[rc, rr])

    res = OpenwaterResults(str(model_path), str(results_path))
    yield res
    res.close()


# ---------------------------------------------------------------------------
# time_series
# ---------------------------------------------------------------------------

def test_time_series_group_by_quasi_dim(results_with_rc):
    '''columns=quasi-dim: each column aggregates the underlying real-dim cells.'''
    df = results_with_rc.time_series('FakeModel', 'runoff',
                                     'reporting_catchment', aggregator='mean')
    # Expected per-run values for SC*CGU runs (row-major over (SC, CGU)):
    # SC1: runs 0,1 (vals 0,10) ; SC2: 2,3 (20,30) ; SC3: 4,5 (40,50) ; SC4: 6,7 (60,70)
    # North aggregates SC1+SC2 → mean of (0,10,20,30) = 15.
    # South aggregates SC3+SC4 → mean of (40,50,60,70) = 55.
    assert set(df.columns) == {'North', 'South'}
    assert df['North'].iloc[0] == 15.0
    assert df['South'].iloc[0] == 55.0
    assert df.columns.name == 'reporting_catchment'


def test_time_series_group_by_quasi_dim_sum(results_with_rc):
    df = results_with_rc.time_series('FakeModel', 'runoff',
                                     'reporting_catchment', aggregator='sum')
    assert df['North'].iloc[0] == 0 + 10 + 20 + 30
    assert df['South'].iloc[0] == 40 + 50 + 60 + 70


def test_time_series_filter_by_quasi_dim_real_columns(results_with_rc):
    '''Constrain by quasi-dim, report by underlying real dim.'''
    df = results_with_rc.time_series('FakeModel', 'runoff', 'SC',
                                     aggregator='mean',
                                     reporting_catchment='North')
    # Only SC1, SC2 (both map to North). Each column averages its two CGUs.
    assert set(df.columns) == {'SC1', 'SC2'}
    assert df['SC1'].iloc[0] == (0 + 10) / 2.0
    assert df['SC2'].iloc[0] == (20 + 30) / 2.0


def test_time_series_filter_by_quasi_dim_set(results_with_rc):
    df = results_with_rc.time_series('FakeModel', 'runoff', 'CGU',
                                     aggregator='sum',
                                     reporting_catchment=['North', 'South'])
    # All SCs included; sum by CGU:
    # Ag = runs 0,2,4,6 → 0+20+40+60 = 120
    # Forest = runs 1,3,5,7 → 10+30+50+70 = 160
    assert df['Ag'].iloc[0] == 120
    assert df['Forest'].iloc[0] == 160


def test_time_series_chained_quasi_columns(results_with_chain):
    df = results_with_chain.time_series('FakeModel', 'runoff',
                                        'reporting_region', aggregator='mean')
    # NorthEast aggregates all of North → SC1,SC2 → vals 0,10,20,30 → mean 15
    # SouthWest aggregates all of South → vals 40,50,60,70 → mean 55
    assert set(df.columns) == {'NorthEast', 'SouthWest'}
    assert df['NorthEast'].iloc[0] == 15.0
    assert df['SouthWest'].iloc[0] == 55.0


def test_time_series_chained_quasi_filter(results_with_chain):
    df = results_with_chain.time_series('FakeModel', 'runoff', 'SC',
                                        aggregator='mean',
                                        reporting_region='NorthEast')
    # NorthEast → North → SC1, SC2
    assert set(df.columns) == {'SC1', 'SC2'}


def test_time_series_multilevel_columns_real_then_quasi(results_with_rc):
    '''Mixed list: [real, quasi]. Resulting MultiIndex names follow original order.'''
    df = results_with_rc.time_series('FakeModel', 'runoff',
                                     ['CGU', 'reporting_catchment'],
                                     aggregator='mean')
    assert isinstance(df.columns, pd.MultiIndex)
    assert df.columns.names == ['CGU', 'reporting_catchment']
    # (Ag, North) → SC1,SC2 Ag → runs 0,2 → mean (0+20)/2 = 10
    assert df[('Ag', 'North')].iloc[0] == 10.0
    # (Forest, South) → SC3,SC4 Forest → runs 5,7 → mean (50+70)/2 = 60
    assert df[('Forest', 'South')].iloc[0] == 60.0


def test_time_series_unknown_constraint_raises(results_with_rc):
    with pytest.raises(Exception):
        results_with_rc.time_series('FakeModel', 'runoff', 'SC',
                                    aggregator='mean',
                                    not_a_dim='x')


def test_time_series_empty_quasi_constraint_raises(results_with_rc):
    with pytest.raises(qd_mod.QuasiDimensionError, match='no values'):
        results_with_rc.time_series('FakeModel', 'runoff', 'SC',
                                    aggregator='mean',
                                    reporting_catchment='Nowhere')


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------

def test_table_rows_quasi(results_with_rc):
    tbl = results_with_rc.table('FakeModel', 'runoff',
                                rows='reporting_catchment',
                                columns='CGU',
                                temporal_aggregator='mean',
                                aggregator='mean')
    # North/Ag = mean(SC1.Ag, SC2.Ag) = mean(0, 20) = 10
    # South/Forest = mean(SC3.F, SC4.F) = mean(50, 70) = 60
    assert tbl.loc['North', 'Ag'] == 10.0
    assert tbl.loc['South', 'Forest'] == 60.0
    assert tbl.index.name == 'reporting_catchment'


def test_table_columns_quasi(results_with_rc):
    tbl = results_with_rc.table('FakeModel', 'runoff',
                                rows='CGU',
                                columns='reporting_catchment',
                                temporal_aggregator='mean',
                                aggregator='sum')
    # Ag column under North = sum(SC1.Ag, SC2.Ag) = 0 + 20 = 20
    # Forest under South = sum(SC3.F, SC4.F) = 50 + 70 = 120
    assert tbl.loc['Ag', 'North'] == 20.0
    assert tbl.loc['Forest', 'South'] == 120.0


def test_table_both_axes_quasi_different_real_dims(monkeypatch, tmp_path):
    '''Rows quasi(SC) + columns quasi(CGU) — separate real dims, OK.'''
    from openwater import nodes as node_types
    monkeypatch.setattr(node_types, 'FakeModel', _FakeModelType, raising=False)
    rc = qd_mod.from_dict({
        'SC1': 'N', 'SC2': 'N', 'SC3': 'S', 'SC4': 'S',
    }, 'rc', 'SC')
    lu = qd_mod.from_dict({'Ag': 'Crop', 'Forest': 'Crop'}, 'lu', 'CGU')
    model_path = tmp_path / 'model.h5'
    results_path = tmp_path / 'results.h5'
    _write_synthetic(model_path, results_path, quasi_dims=[rc, lu])
    res = OpenwaterResults(str(model_path), str(results_path))
    try:
        tbl = res.table('FakeModel', 'runoff', rows='rc', columns='lu',
                        temporal_aggregator='sum', aggregator='sum')
        assert tbl.index.name == 'rc'
        assert tbl.columns.name == 'lu'
        # Crop column under N is sum of all 4 runs in SC1,SC2 = 0+10+20+30 = 60
        # Each run's value (constant over time) is multiplied by n_timesteps=3 by temporal sum.
        assert tbl.loc['N', 'Crop'] == (0 + 10 + 20 + 30) * 3
        assert tbl.loc['S', 'Crop'] == (40 + 50 + 60 + 70) * 3
    finally:
        res.close()


def test_table_rows_and_columns_same_real_dim_raises(results_with_rc):
    with pytest.raises(ValueError, match='same real dimension'):
        results_with_rc.table('FakeModel', 'runoff',
                              rows='reporting_catchment',
                              columns='SC',
                              temporal_aggregator='mean')


# ---------------------------------------------------------------------------
# Constraint intersection (real + quasi targeting same real dim)
# ---------------------------------------------------------------------------

def test_constraint_intersection_real_and_quasi_same_dim(results_with_rc):
    '''SC constraint + reporting_catchment constraint must intersect.'''
    # reporting_catchment='North' → SC in {SC1, SC2}
    # SC=['SC2','SC3'] → intersection = {SC2}
    df = results_with_rc.time_series('FakeModel', 'runoff', 'SC',
                                     aggregator='mean',
                                     SC=['SC2', 'SC3'],
                                     reporting_catchment='North')
    assert set(df.columns) == {'SC2'}


def test_constraint_intersection_empty_raises(results_with_rc):
    # reporting_catchment='North' → SC in {SC1,SC2}; SC=['SC3'] → empty intersect
    with pytest.raises(qd_mod.QuasiDimensionError, match='empty intersection'):
        results_with_rc.time_series('FakeModel', 'runoff', 'SC',
                                    aggregator='mean',
                                    SC=['SC3'],
                                    reporting_catchment='North')


# ---------------------------------------------------------------------------
# Default propagation in regrouping
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# OpenwaterResults.add_quasi_dim / remove_quasi_dim (dynamic, session-only)
# ---------------------------------------------------------------------------

@pytest.fixture
def results_without_quasi(monkeypatch, tmp_path):
    '''Synthetic results with no quasi-dims persisted in the model file.'''
    from openwater import nodes as node_types
    monkeypatch.setattr(node_types, 'FakeModel', _FakeModelType, raising=False)
    model_path = tmp_path / 'model.h5'
    results_path = tmp_path / 'results.h5'
    _write_synthetic(model_path, results_path)
    res = OpenwaterResults(str(model_path), str(results_path))
    yield res, model_path
    res.close()


def test_results_add_quasi_dim_then_group_by(results_without_quasi):
    res, _ = results_without_quasi
    res.add_quasi_dim({
        'SC1': 'North', 'SC2': 'North',
        'SC3': 'South', 'SC4': 'South',
    }, name='reporting_catchment', keyed_by='SC')

    df = res.time_series('FakeModel', 'runoff',
                         'reporting_catchment', aggregator='mean')
    assert set(df.columns) == {'North', 'South'}
    assert df['North'].iloc[0] == 15.0


def test_results_add_quasi_dim_then_filter(results_without_quasi):
    res, _ = results_without_quasi
    res.add_quasi_dim({
        'SC1': 'North', 'SC2': 'North',
        'SC3': 'South', 'SC4': 'South',
    }, name='rc', keyed_by='SC')

    df = res.time_series('FakeModel', 'runoff', 'SC',
                         aggregator='mean', rc='North')
    assert set(df.columns) == {'SC1', 'SC2'}


def test_results_add_quasi_dim_does_not_touch_model_file(results_without_quasi):
    res, model_path = results_without_quasi
    mtime_before = model_path.stat().st_mtime
    res.add_quasi_dim({'SC1': 'N'}, name='rc', keyed_by='SC')
    assert model_path.stat().st_mtime == mtime_before


def test_results_remove_quasi_dim(results_without_quasi):
    res, _ = results_without_quasi
    res.add_quasi_dim({'SC1': 'N', 'SC2': 'S'}, name='rc', keyed_by='SC')
    assert 'rc' in res.quasi_dims()
    res.remove_quasi_dim('rc')
    assert 'rc' not in res.quasi_dims()


def test_results_remove_persisted_session_only(monkeypatch, tmp_path):
    '''Removing a persisted quasi-dim through the results view is session-only.'''
    from openwater import nodes as node_types
    monkeypatch.setattr(node_types, 'FakeModel', _FakeModelType, raising=False)
    rc = qd_mod.from_dict({'SC1': 'N', 'SC2': 'S'}, 'rc', 'SC')
    model_path = tmp_path / 'model.h5'
    results_path = tmp_path / 'results.h5'
    _write_synthetic(model_path, results_path, quasi_dims=[rc])

    res = OpenwaterResults(str(model_path), str(results_path))
    try:
        assert 'rc' in res.quasi_dims()
        res.remove_quasi_dim('rc')
        assert res.quasi_dims() == []
    finally:
        res.close()

    # Disk copy survives — removal was session-only.
    res2 = OpenwaterResults(str(model_path), str(results_path))
    try:
        assert 'rc' in res2.quasi_dims()
    finally:
        res2.close()


def test_results_add_quasi_dim_polymorphic_sources(results_without_quasi, tmp_path):
    '''All four source types are accepted, sharing dispatch with ModelGraph.'''
    res, _ = results_without_quasi

    res.add_quasi_dim({'SC1': 'N'}, name='from_dict', keyed_by='SC')

    s = pd.Series(['X', 'Y'], index=pd.Index(['SC1', 'SC2'], name='SC'),
                  name='from_series')
    res.add_quasi_dim(s)

    csv = tmp_path / 'from_csv.csv'
    csv.write_text("SC,from_csv\nSC1,A\nSC2,B\n")
    res.add_quasi_dim(str(csv), key='SC', value='from_csv')

    qd = qd_mod.from_dict({'SC1': 'P'}, 'from_qd', 'SC')
    res.add_quasi_dim(qd)

    assert set(res.quasi_dims()) == {
        'from_dict', 'from_series', 'from_csv', 'from_qd'}


def test_partial_coverage_default_in_regroup(monkeypatch, tmp_path):
    '''Quasi-dim with default groups unmapped real-dim values under the default.'''
    from openwater import nodes as node_types
    monkeypatch.setattr(node_types, 'FakeModel', _FakeModelType, raising=False)
    rc = qd_mod.from_dict({
        'SC1': 'North', 'SC2': 'North',
    }, name='reporting_catchment', keyed_by='SC', default='Unassigned')
    model_path = tmp_path / 'model.h5'
    results_path = tmp_path / 'results.h5'
    _write_synthetic(model_path, results_path, quasi_dims=[rc])
    res = OpenwaterResults(str(model_path), str(results_path))
    try:
        df = res.time_series('FakeModel', 'runoff',
                             'reporting_catchment', aggregator='sum')
        assert set(df.columns) == {'North', 'Unassigned'}
        # Unassigned aggregates SC3, SC4 runs → 40+50+60+70 = 220
        assert df['Unassigned'].iloc[0] == 220
    finally:
        res.close()
