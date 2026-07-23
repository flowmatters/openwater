'''
Tests for changing/extending a model's time period (ModelFile.retime) and for
applying partial, date-aligned timeseries via DataframeInputs(align='dates').

These tests build a minimal model file by hand and register a throwaway model
type, so they do not require the compiled openwater core library.
'''
import os
import numpy as np
import pandas as pd
import pytest
import h5py

import openwater.nodes as node_types
import openwater.config as config
from openwater.config import (FillRules, align_source_indices,
                              build_resized_input_array, DataframeInputs)
from openwater.template import ModelFile, LINK_TABLE_COLUMNS

MODEL = 'RetimeTestModel'
INPUTS = ['rainfall', 'pet']


@pytest.fixture(autouse=True)
def _register_model():
    node_types._create_model_type(MODEL, {
        'Inputs': INPUTS,
        'Outputs': ['runoff'],
        'States': [],
        'Parameters': [],
    })


def _write_model_file(path, time_period, inputs):
    '''inputs: ndarray (n_cells, n_inputs, n_timesteps).'''
    n_cells = inputs.shape[0]
    with h5py.File(path, 'w') as f:
        meta = f.create_group('META')
        meta.create_dataset('models', data=[np.bytes_(MODEL)],
                            dtype='S%d' % len(MODEL))
        dates = np.array([ts.isoformat() for ts in pd.DatetimeIndex(time_period)],
                         dtype=h5py.special_dtype(vlen=str))
        meta.create_dataset('timeperiod', data=dates)

        dims = f.create_group('DIMENSIONS')
        dims.create_dataset('catchment', data=np.arange(1, n_cells + 1))

        models = f.create_group('MODELS')
        grp = models.create_group(MODEL)
        m = grp.create_dataset('map', dtype=np.int32,
                               data=np.arange(n_cells, dtype=np.int32))
        m.attrs['DIMS'] = [np.bytes_('catchment')]
        m.attrs['PROCESSES'] = [np.bytes_('test')]
        grp.create_dataset('inputs', data=inputs, dtype=np.float64)

        f.create_dataset('LINKS', dtype=np.uint32,
                         shape=[0, len(LINK_TABLE_COLUMNS)])
    return path


def _model_path(tmp_path):
    return str(tmp_path / 'retime_model.h5')


# --- pure helpers ----------------------------------------------------------

def test_fill_rules_resolution():
    rules = (FillRules(default=0.0)
             .set('ffill', variable='rainfall')
             .set(3.0, variable='pet')
             .set(9.0, variable='pet', model=MODEL))
    assert rules.rule_for(MODEL, 'rainfall') == ('ffill',)
    assert rules.rule_for('OtherModel', 'rainfall') == ('ffill',)
    assert rules.rule_for('OtherModel', 'pet') == ('value', 3.0)
    # model+variable beats variable-only
    assert rules.rule_for(MODEL, 'pet') == ('value', 9.0)
    # falls back to default
    assert rules.rule_for(MODEL, 'unknown') == ('value', 0.0)


def test_fill_rules_zero_alias_and_model_object():
    rules = FillRules(default='zero')
    assert rules.rule_for('m', 'v') == ('value', 0.0)
    # model may be passed as an object with a .name
    md = node_types.RetimeTestModel
    rules.set(1.0, variable='pet', model=md)
    assert rules.rule_for(MODEL, 'pet') == ('value', 1.0)


def test_align_source_indices():
    old = pd.date_range('2000-01-01', periods=5, freq='D')
    # extend at end by 2 days
    new = pd.date_range('2000-01-01', periods=7, freq='D')
    np.testing.assert_array_equal(align_source_indices(old, new),
                                  [0, 1, 2, 3, 4, -1, -1])
    # prepend 2 days
    new = pd.date_range('1999-12-30', periods=7, freq='D')
    np.testing.assert_array_equal(align_source_indices(old, new),
                                  [-1, -1, 0, 1, 2, 3, 4])
    # trim to middle
    new = pd.date_range('2000-01-02', periods=2, freq='D')
    np.testing.assert_array_equal(align_source_indices(old, new), [1, 2])


def test_build_resized_array_zero_default():
    old = np.arange(2 * 2 * 3, dtype=np.float64).reshape(2, 2, 3)
    src = np.array([0, 1, 2, -1, -1])          # extend by 2
    out = build_resized_input_array(old, src, INPUTS, MODEL, FillRules(default=0.0))
    assert out.shape == (2, 2, 5)
    np.testing.assert_array_equal(out[:, :, :3], old)
    np.testing.assert_array_equal(out[:, :, 3:], 0.0)


def test_build_resized_array_value_and_ffill():
    old = np.zeros((1, 2, 3), dtype=np.float64)
    old[0, 0, :] = [10, 11, 12]                # rainfall
    old[0, 1, :] = [5, 6, 7]                   # pet
    src = np.array([0, 1, 2, -1, -1])
    rules = (FillRules(default=0.0)
             .set('ffill', variable='rainfall')
             .set(99.0, variable='pet'))
    out = build_resized_input_array(old, src, INPUTS, MODEL, rules)
    # rainfall trailing gap holds last value
    np.testing.assert_array_equal(out[0, 0, :], [10, 11, 12, 12, 12])
    # pet filled with the specific value
    np.testing.assert_array_equal(out[0, 1, :], [5, 6, 7, 99, 99])


def test_build_resized_array_ffill_leading_holds_first():
    old = np.zeros((1, 1, 3), dtype=np.float64)
    old[0, 0, :] = [10, 11, 12]
    src = np.array([-1, -1, 0, 1, 2])          # prepend 2
    rules = FillRules(default=0.0).set('ffill', variable='rainfall')
    out = build_resized_input_array(old, src, ['rainfall'], MODEL, rules)
    np.testing.assert_array_equal(out[0, 0, :], [10, 10, 10, 11, 12])


# --- ModelFile.retime ------------------------------------------------------

def test_retime_extends_and_fills(tmp_path):
    old_period = pd.date_range('2000-01-01', periods=4, freq='D')
    inputs = np.zeros((2, 2, 4), dtype=np.float64)
    inputs[:, 0, :] = [[1, 2, 3, 4], [10, 20, 30, 40]]     # rainfall
    inputs[:, 1, :] = [[0.1, 0.2, 0.3, 0.4], [1, 1, 1, 1]] # pet
    path = _write_model_file(_model_path(tmp_path), old_period, inputs)

    new_period = pd.date_range('2000-01-01', periods=6, freq='D')  # +2 days
    rules = FillRules(default=0.0).set('ffill', variable='rainfall')

    mf = ModelFile(path)
    mf.retime(new_period, fill_rules=rules)
    mf.close()

    with h5py.File(path, 'r') as f:
        arr = f['MODELS'][MODEL]['inputs'][...]
        tp = [d.decode() if isinstance(d, bytes) else d
              for d in f['META']['timeperiod'][...]]
    assert arr.shape == (2, 2, 6)
    # carried over
    np.testing.assert_array_equal(arr[:, :, :4], inputs)
    # rainfall ffill holds last value
    np.testing.assert_array_equal(arr[:, 0, 4:], [[4, 4], [40, 40]])
    # pet default zero fill
    np.testing.assert_array_equal(arr[:, 1, 4:], [[0, 0], [0, 0]])
    assert tp[-1] == new_period[-1].isoformat()
    assert len(tp) == 6


def test_retime_prepend_holds_first_value(tmp_path):
    old_period = pd.date_range('2000-01-03', periods=3, freq='D')
    inputs = np.zeros((1, 2, 3), dtype=np.float64)
    inputs[0, 0, :] = [7, 8, 9]
    inputs[0, 1, :] = [1, 2, 3]
    path = _write_model_file(_model_path(tmp_path), old_period, inputs)

    new_period = pd.date_range('2000-01-01', periods=5, freq='D')  # prepend 2
    rules = FillRules(default=0.0).set('ffill', variable='rainfall')

    mf = ModelFile(path)
    mf.retime(new_period, fill_rules=rules)
    mf.close()

    with h5py.File(path, 'r') as f:
        arr = f['MODELS'][MODEL]['inputs'][...]
    # rainfall leading gap holds first value; pet leading gap defaults to zero
    np.testing.assert_array_equal(arr[0, 0, :], [7, 7, 7, 8, 9])
    np.testing.assert_array_equal(arr[0, 1, :], [0, 0, 1, 2, 3])


def test_retime_trim(tmp_path):
    old_period = pd.date_range('2000-01-01', periods=5, freq='D')
    inputs = np.zeros((1, 2, 5), dtype=np.float64)
    inputs[0, 0, :] = [1, 2, 3, 4, 5]
    path = _write_model_file(_model_path(tmp_path), old_period, inputs)

    new_period = pd.date_range('2000-01-02', periods=2, freq='D')

    mf = ModelFile(path)
    mf.retime(new_period)
    mf.close()

    with h5py.File(path, 'r') as f:
        arr = f['MODELS'][MODEL]['inputs'][...]
        tp = [d.decode() if isinstance(d, bytes) else d
              for d in f['META']['timeperiod'][...]]
    assert arr.shape == (1, 2, 2)
    np.testing.assert_array_equal(arr[0, 0, :], [2, 3])
    assert len(tp) == 2


def test_retime_updates_open_modelfile_time_period(tmp_path):
    old_period = pd.date_range('2000-01-01', periods=4, freq='D')
    inputs = np.zeros((1, 2, 4), dtype=np.float64)
    path = _write_model_file(_model_path(tmp_path), old_period, inputs)
    new_period = pd.date_range('2000-01-01', periods=6, freq='D')

    mf = ModelFile(path)
    mf.retime(new_period)
    assert len(mf.time_period) == 6
    mf.close()


# --- date-aligned subset application ---------------------------------------

def test_dated_subset_applies_only_overlap(tmp_path):
    period = pd.date_range('2000-01-01', periods=6, freq='D')
    inputs = np.zeros((2, 2, 6), dtype=np.float64)
    inputs[:, 0, :] = 1.0    # rainfall baseline
    inputs[:, 1, :] = 9.0    # pet baseline (should be untouched)
    path = _write_model_file(_model_path(tmp_path), period, inputs)

    # New rainfall covering only days 3-4 (positions 2,3), for both catchments.
    sub = pd.date_range('2000-01-03', periods=2, freq='D')
    df = pd.DataFrame({'rain-1': [100.0, 200.0], 'rain-2': [300.0, 400.0]},
                      index=sub)

    dfi = DataframeInputs()
    dfi.inputter(df, 'rainfall', 'rain-${catchment}', align='dates')

    mf = ModelFile(path)
    mf._parameteriser = dfi
    mf.write()
    mf.close()

    with h5py.File(path, 'r') as f:
        arr = f['MODELS'][MODEL]['inputs'][...]
    # catchment 1 (run idx 0): only positions 2,3 changed
    np.testing.assert_array_equal(arr[0, 0, :], [1, 1, 100, 200, 1, 1])
    np.testing.assert_array_equal(arr[1, 0, :], [1, 1, 300, 400, 1, 1])
    # pet completely untouched
    np.testing.assert_array_equal(arr[:, 1, :], 9.0)


def test_invalid_align_mode_raises():
    dfi = DataframeInputs()
    df = pd.DataFrame({'rain-1': [1.0, 2.0]},
                      index=pd.date_range('2000-01-01', periods=2, freq='D'))
    # a typo like 'date' must fail loudly rather than silently applying positionally
    with pytest.raises(ValueError):
        dfi.inputter(df, 'rainfall', 'rain-${catchment}', align='date')


def test_dated_subset_ignores_out_of_range(tmp_path):
    period = pd.date_range('2000-01-01', periods=4, freq='D')
    inputs = np.zeros((1, 2, 4), dtype=np.float64)
    inputs[0, 0, :] = 1.0
    path = _write_model_file(_model_path(tmp_path), period, inputs)

    # series straddles the end of the model period; last row is out of range
    sub = pd.date_range('2000-01-04', periods=3, freq='D')
    df = pd.DataFrame({'rain-1': [50.0, 60.0, 70.0]}, index=sub)

    dfi = DataframeInputs()
    dfi.inputter(df, 'rainfall', 'rain-${catchment}', align='dates')

    mf = ModelFile(path)
    mf._parameteriser = dfi
    mf.write()
    mf.close()

    with h5py.File(path, 'r') as f:
        arr = f['MODELS'][MODEL]['inputs'][...]
    # only the in-range value (2000-01-04, position 3) is applied
    np.testing.assert_array_equal(arr[0, 0, :], [1, 1, 1, 50])
