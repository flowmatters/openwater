"""Tests for results.py — temporal subsetting, aggregators, grouped tables."""
import numpy as np
import pandas as pd
import h5py
import pytest

import openwater.results as results_mod
from openwater.results import (
    OpenwaterResults,
    OpenwaterSplitResults,
    resolve_temporal_agg,
)


class DummyModelDescription:
    name = 'DummyModel'
    description = {
        'Inputs': [{'Name': 'rainfall', 'Units': 'mm'}],
        'Outputs': [{'Name': 'runoff', 'Units': 'mm'}],
    }


CATCHMENTS = ['c1', 'c2']
HRUS = ['h1', 'h2']
# 2020-01-01 .. 2021-12-31 (2020 is a leap year): 366 + 365 = 731 days
DATES = pd.date_range('2020-01-01', '2021-12-31', freq='D')


@pytest.fixture(autouse=True)
def dummy_model(monkeypatch):
    """results.py resolves model descriptions via getattr(node_types, name)."""
    monkeypatch.setattr(
        results_mod.node_types, 'DummyModel', DummyModelDescription, raising=False)


def write_model_h5(path, dates):
    with h5py.File(path, 'w') as f:
        f.create_dataset('/DIMENSIONS/catchment',
                         data=np.array(CATCHMENTS, dtype='S'))
        f.create_dataset('/DIMENSIONS/hru', data=np.array(HRUS, dtype='S'))
        # run_map[catchment_idx, hru_idx] -> run index 0..3
        ds = f.create_dataset('/MODELS/DummyModel/map',
                              data=np.arange(4).reshape(2, 2))
        ds.attrs['DIMS'] = np.array([b'catchment', b'hru'])
        f.create_dataset(
            '/META/timeperiod',
            data=np.array([d.isoformat() for d in dates], dtype='S'))


def write_results_h5(path, n_time):
    """Run r has the constant value (r + 1.0) at every timestep."""
    outputs = np.zeros((4, 1, n_time))
    for r in range(4):
        outputs[r, 0, :] = r + 1.0
    with h5py.File(path, 'w') as f:
        f.create_dataset('/MODELS/DummyModel/outputs', data=outputs)


@pytest.fixture
def results(tmp_path):
    model_path = str(tmp_path / 'model.h5')
    results_path = str(tmp_path / 'outputs.h5')
    write_model_h5(model_path, DATES)
    write_results_h5(results_path, len(DATES))
    r = OpenwaterResults(model_path, results_path)
    yield r
    r.close()


class TestResolveTemporalAgg:
    def test_named_aggregators(self):
        a = np.array([[1.0, 2.0, 3.0, 4.0]])
        assert resolve_temporal_agg('mean')(a)[0] == pytest.approx(2.5)
        assert resolve_temporal_agg('sum')(a)[0] == pytest.approx(10.0)
        assert resolve_temporal_agg('min')(a)[0] == pytest.approx(1.0)
        assert resolve_temporal_agg('max')(a)[0] == pytest.approx(4.0)

    def test_percentile(self):
        a = np.arange(101, dtype=float).reshape(1, 101)
        assert resolve_temporal_agg('p10')(a)[0] == pytest.approx(10.0)
        assert resolve_temporal_agg('p99.5')(a)[0] == pytest.approx(99.5)
        assert resolve_temporal_agg('p0')(a)[0] == pytest.approx(0.0)
        assert resolve_temporal_agg('p100')(a)[0] == pytest.approx(100.0)

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            resolve_temporal_agg('median-ish')
        with pytest.raises(KeyError):
            resolve_temporal_agg('p101')
        with pytest.raises(KeyError):
            resolve_temporal_agg(None)
        with pytest.raises(KeyError):
            resolve_temporal_agg('')


class TestTimeSeriesSubsetting:
    def test_no_subset_unchanged(self, results):
        ts = results.time_series('DummyModel', 'runoff', 'catchment', hru='h1')
        assert len(ts) == 731
        assert list(ts.columns) == ['c1', 'c2']
        assert ts['c1'].iloc[0] == pytest.approx(1.0)
        assert ts['c2'].iloc[0] == pytest.approx(3.0)  # run_map[1,0] == 2

    def test_time_period_subset(self, results):
        ts = results.time_series(
            'DummyModel', 'runoff', 'catchment', hru='h1',
            time_period=('2020-03-01', '2020-03-31'))
        assert len(ts) == 31
        assert ts.index[0] == pd.Timestamp('2020-03-01')
        assert ts.index[-1] == pd.Timestamp('2020-03-31')

    def test_open_ended_period(self, results):
        ts = results.time_series(
            'DummyModel', 'runoff', 'catchment', hru='h1',
            time_period=('2021-01-01', None))
        assert len(ts) == 365

    def test_months_filter(self, results):
        ts = results.time_series(
            'DummyModel', 'runoff', 'catchment', hru='h1', months=[1])
        assert len(ts) == 62  # Jan 2020 + Jan 2021
        assert set(ts.index.month) == {1}

    def test_subset_without_time_index_raises(self, tmp_path):
        model_path = str(tmp_path / 'model.h5')
        results_path = str(tmp_path / 'outputs.h5')
        with h5py.File(model_path, 'w') as f:
            f.create_dataset('/DIMENSIONS/catchment',
                             data=np.array(CATCHMENTS, dtype='S'))
            f.create_dataset('/DIMENSIONS/hru', data=np.array(HRUS, dtype='S'))
            ds = f.create_dataset('/MODELS/DummyModel/map',
                                  data=np.arange(4).reshape(2, 2))
            ds.attrs['DIMS'] = np.array([b'catchment', b'hru'])
            # no META/timeperiod
        write_results_h5(results_path, 10)
        r = OpenwaterResults(model_path, results_path)
        try:
            with pytest.raises(ValueError):
                r.time_series('DummyModel', 'runoff', 'catchment', hru='h1',
                              time_period=('2020-01-01', '2020-01-05'))
        finally:
            r.close()


class TestTableSubsetting:
    def test_default_mean_unchanged(self, results):
        tbl = results.table('DummyModel', 'runoff', rows='catchment',
                            columns='hru')
        # constant series: mean == run value
        assert tbl.loc['c1', 'h1'] == pytest.approx(1.0)
        assert tbl.loc['c2', 'h2'] == pytest.approx(4.0)

    def test_sum_over_time_period(self, results):
        tbl = results.table(
            'DummyModel', 'runoff', rows='catchment', columns='hru',
            temporal_aggregator='sum',
            time_period=('2020-03-01', '2020-03-31'))
        assert tbl.loc['c1', 'h1'] == pytest.approx(31.0)
        assert tbl.loc['c2', 'h1'] == pytest.approx(3.0 * 31)

    def test_percentile_aggregator(self, results):
        tbl = results.table('DummyModel', 'runoff', rows='catchment',
                            columns='hru', temporal_aggregator='p50')
        assert tbl.loc['c2', 'h2'] == pytest.approx(4.0)

    def test_months_filter_sum(self, results):
        tbl = results.table(
            'DummyModel', 'runoff', rows='catchment', columns='hru',
            temporal_aggregator='sum', months=[1])
        assert tbl.loc['c1', 'h1'] == pytest.approx(62.0)


class TestGroupedTable:
    def test_year_sum(self, results):
        tbl = results.grouped_table(
            'DummyModel', 'runoff', columns='catchment',
            temporal_grouping='year', temporal_aggregator='sum', hru='h1')
        assert list(tbl.index) == [2020, 2021]
        assert tbl.loc[2020, 'c1'] == pytest.approx(366.0)
        assert tbl.loc[2021, 'c2'] == pytest.approx(3.0 * 365)

    def test_water_year_sum(self, results):
        tbl = results.grouped_table(
            'DummyModel', 'runoff', columns='catchment',
            temporal_grouping='water_year', temporal_aggregator='sum',
            hru='h1')
        # WY2019: 2020-01-01..2020-06-30 (182d); WY2020: full year (365d);
        # WY2021: 2021-07-01..2021-12-31 (184d). Labelled by starting year.
        assert list(tbl.index) == [2019, 2020, 2021]
        assert tbl.loc[2019, 'c1'] == pytest.approx(182.0)
        assert tbl.loc[2020, 'c1'] == pytest.approx(365.0)
        assert tbl.loc[2021, 'c1'] == pytest.approx(184.0)

    def test_month_of_year_mean(self, results):
        tbl = results.grouped_table(
            'DummyModel', 'runoff', columns='catchment',
            temporal_grouping='month_of_year', temporal_aggregator='mean',
            hru='h1')
        assert list(tbl.index) == list(range(1, 13))
        assert tbl.loc[6, 'c2'] == pytest.approx(3.0)

    def test_month_grouping_with_period(self, results):
        tbl = results.grouped_table(
            'DummyModel', 'runoff', columns='catchment',
            temporal_grouping='month', temporal_aggregator='sum', hru='h1',
            time_period=('2020-01-01', '2020-03-31'))
        assert list(tbl.index) == ['2020-01', '2020-02', '2020-03']
        assert tbl.loc['2020-02', 'c1'] == pytest.approx(29.0)

    def test_unknown_grouping_raises(self, results):
        with pytest.raises(ValueError):
            results.grouped_table(
                'DummyModel', 'runoff', columns='catchment',
                temporal_grouping='fortnight', hru='h1')


@pytest.fixture
def split_results(tmp_path):
    """Two splits: 2020 (366 days) and 2021 (365 days)."""
    parts = []
    boundaries = [('2020-01-01', '2020-12-31'), ('2021-01-01', '2021-12-31')]
    for i, (start, end) in enumerate(boundaries):
        dates = pd.date_range(start, end, freq='D')
        model_path = str(tmp_path / f'model{i}.h5')
        results_path = str(tmp_path / f'outputs{i}.h5')
        write_model_h5(model_path, dates)
        write_results_h5(results_path, len(dates))
        parts.append(OpenwaterResults(model_path, results_path))
    sr = OpenwaterSplitResults(parts, time_period=DATES)
    yield sr
    sr.close()


class TestSplitResults:
    def test_existing_mean_table_unchanged(self, split_results):
        tbl = split_results.table('DummyModel', 'runoff', rows='catchment',
                                  columns='hru')
        assert tbl.loc['c1', 'h1'] == pytest.approx(1.0)

    def test_time_series_subset_across_boundary(self, split_results):
        ts = split_results.time_series(
            'DummyModel', 'runoff', 'catchment', hru='h1',
            time_period=('2020-12-01', '2021-01-31'))
        assert len(ts) == 62

    def test_table_sum_subset_across_boundary(self, split_results):
        tbl = split_results.table(
            'DummyModel', 'runoff', rows='catchment', columns='hru',
            temporal_aggregator='sum',
            time_period=('2020-12-01', '2021-01-31'))
        assert tbl.loc['c1', 'h1'] == pytest.approx(62.0)
        assert tbl.loc['c2', 'h2'] == pytest.approx(4.0 * 62)

    def test_table_percentile(self, split_results):
        tbl = split_results.table('DummyModel', 'runoff', rows='catchment',
                                  columns='hru', temporal_aggregator='p50')
        assert tbl.loc['c2', 'h1'] == pytest.approx(3.0)

    def test_grouped_table_year(self, split_results):
        tbl = split_results.grouped_table(
            'DummyModel', 'runoff', columns='catchment',
            temporal_grouping='year', temporal_aggregator='sum', hru='h1')
        assert tbl.loc[2020, 'c1'] == pytest.approx(366.0)
        assert tbl.loc[2021, 'c1'] == pytest.approx(365.0)
