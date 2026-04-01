"""Tests for split.py — temporal splitting of OpenWater models."""
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import h5py

from openwater.split import split_time_series, split_model


# ---------------------------------------------------------------------------
# Tests for split_time_series (existing, expanded)
# ---------------------------------------------------------------------------

class TestSplitTimeSeriesWithBreaks:
    def _make_grp(self, ts_length):
        return {'DummyModel': {'inputs': np.zeros((1, 1, ts_length))}}

    def test_explicit_breaks(self):
        grp = self._make_grp(10000)
        BREAKS = [
            [100, 1000, 5000],
            [0, 100, 1000, 5000],
            [100, 1000, 5000, 10000],
            [0, 100, 1000, 5000, 10000],
        ]
        for breaks in BREAKS:
            windows = split_time_series(grp, 10, breaks)
            assert len(windows) == 4
            assert windows[0] == (0, 100)
            assert windows[1] == (100, 1000)
            assert windows[2] == (1000, 5000)
            assert windows[3] == (5000, 10000)

    def test_degenerate_breaks(self):
        grp = self._make_grp(10000)
        BREAKS = [[], [0], [10000], [0, 10000]]
        for breaks in BREAKS:
            windows = split_time_series(grp, 11, breaks)
            assert len(windows) == 1
            assert windows[0] == (0, 10000)

    def test_equal_splits(self):
        grp = self._make_grp(10000)
        windows = split_time_series(grp, 11, None)
        assert len(windows) == 11
        assert windows[0] == (0, 909)
        assert windows[1] == (909, 1818)
        assert windows[10] == (9090, 10000)

    def test_single_split(self):
        grp = self._make_grp(100)
        windows = split_time_series(grp, 1, None)
        assert len(windows) == 1
        assert windows[0] == (0, 100)

    def test_no_inputs_returns_none(self):
        grp = {'DummyModel': {}}
        result = split_time_series(grp, 5, None)
        assert result is None

    def test_windows_cover_full_range(self):
        grp = self._make_grp(1000)
        windows = split_time_series(grp, 7, None)
        assert windows[0][0] == 0
        assert windows[-1][1] == 1000
        for i in range(len(windows) - 1):
            assert windows[i][1] == windows[i + 1][0]


# ---------------------------------------------------------------------------
# Tests for split_model
# ---------------------------------------------------------------------------

def _build_simple_model_h5(path, n_timesteps=100):
    """Build a minimal model HDF5 for split_model testing."""
    f = h5py.File(path, 'w')

    meta = f.create_group('META')
    meta.create_dataset('models', data=np.array(['ModelA'], dtype='S10'))
    meta.create_dataset('timeperiod', data=np.array(
        [f'2020-01-{d+1:02d}' for d in range(n_timesteps)], dtype='S20'
    ))

    dims = f.create_group('DIMENSIONS')
    dims.create_dataset('catchment', data=np.array(['c1'], dtype='S10'))

    links = np.zeros((0, 10), dtype=np.uint32)
    f.create_dataset('LINKS', data=links)

    models = f.create_group('MODELS')
    ma = models.create_group('ModelA')
    ma.create_dataset('batches', data=np.array([1], dtype=np.uint32))
    m = ma.create_dataset('map', data=np.array([0], dtype=np.int64))
    m.attrs['DIMS'] = np.array([b'catchment'])
    m.attrs['PROCESSES'] = np.array([b'rr'])
    ma.create_dataset('parameters', data=np.array([[1.0]]))
    ma.create_dataset('states', data=np.array([[0.0]]))
    # Inputs: 1 node, 1 input variable, n_timesteps
    inputs_data = np.arange(n_timesteps, dtype=np.float64).reshape(1, 1, n_timesteps)
    ma.create_dataset('inputs', data=inputs_data)

    f.close()


class TestSplitModel:
    def test_split_creates_structure_file(self, tmp_path):
        orig = str(tmp_path / 'orig.h5')
        _build_simple_model_h5(orig, n_timesteps=100)

        structure = str(tmp_path / 'structure.h5')
        split_model(orig, structure)

        with h5py.File(structure, 'r') as f:
            assert 'DIMENSIONS' in f
            assert 'LINKS' in f
            assert 'META' in f
            assert 'MODELS' in f
            assert 'ModelA' in f['MODELS']
            assert 'batches' in f['MODELS']['ModelA']
            assert 'map' in f['MODELS']['ModelA']

    def test_split_creates_separate_param_file(self, tmp_path):
        orig = str(tmp_path / 'orig.h5')
        _build_simple_model_h5(orig, n_timesteps=100)

        structure = str(tmp_path / 'structure.h5')
        params = str(tmp_path / 'params.h5')
        split_model(orig, structure, parameters=params)

        with h5py.File(params, 'r') as f:
            assert 'MODELS' in f
            assert 'ModelA' in f['MODELS']
            assert 'parameters' in f['MODELS']['ModelA']

    def test_split_creates_windowed_inputs(self, tmp_path):
        orig = str(tmp_path / 'orig.h5')
        _build_simple_model_h5(orig, n_timesteps=100)

        structure = str(tmp_path / 'structure.h5')
        inputs_tmpl = str(tmp_path / 'inputs.h5')
        split_model(orig, structure, inputs=inputs_tmpl, input_windows=[50])

        # Should create inputs-0.h5 and inputs-1.h5
        with h5py.File(str(tmp_path / 'inputs-0.h5'), 'r') as f:
            data = f['MODELS']['ModelA']['inputs'][...]
            assert data.shape[2] == 50  # first 50 timesteps

        with h5py.File(str(tmp_path / 'inputs-1.h5'), 'r') as f:
            data = f['MODELS']['ModelA']['inputs'][...]
            assert data.shape[2] == 50  # last 50 timesteps

    def test_split_input_values_correct(self, tmp_path):
        orig = str(tmp_path / 'orig.h5')
        _build_simple_model_h5(orig, n_timesteps=100)

        structure = str(tmp_path / 'structure.h5')
        inputs_tmpl = str(tmp_path / 'inputs.h5')
        split_model(orig, structure, inputs=inputs_tmpl, input_windows=[30])

        with h5py.File(str(tmp_path / 'inputs-0.h5'), 'r') as f:
            data = f['MODELS']['ModelA']['inputs'][0, 0, :]
            np.testing.assert_array_equal(data, np.arange(30, dtype=np.float64))

        with h5py.File(str(tmp_path / 'inputs-1.h5'), 'r') as f:
            data = f['MODELS']['ModelA']['inputs'][0, 0, :]
            np.testing.assert_array_equal(data, np.arange(30, 100, dtype=np.float64))

    def test_split_single_window_no_suffix(self, tmp_path):
        orig = str(tmp_path / 'orig.h5')
        _build_simple_model_h5(orig, n_timesteps=100)

        structure = str(tmp_path / 'structure.h5')
        inputs_path = str(tmp_path / 'inputs.h5')
        split_model(orig, structure, inputs=inputs_path, split_ts=1)

        # Single window: should use the exact filename, no -0 suffix
        assert os.path.exists(inputs_path)

    def test_split_preserves_states(self, tmp_path):
        orig = str(tmp_path / 'orig.h5')
        _build_simple_model_h5(orig, n_timesteps=100)

        structure = str(tmp_path / 'structure.h5')
        states = str(tmp_path / 'states.h5')
        split_model(orig, structure, init_states=states)

        with h5py.File(states, 'r') as f:
            assert 'states' in f['MODELS']['ModelA']
            s = f['MODELS']['ModelA']['states'][...]
            assert s.shape == (1, 1)
            assert s[0, 0] == 0.0
