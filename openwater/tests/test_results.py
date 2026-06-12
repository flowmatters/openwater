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
