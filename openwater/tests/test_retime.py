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
                              build_resized_input_array, DataframeInputs,
                              climatology, recycle)
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


# --- repack ----------------------------------------------------------------

def _snapshot(path):
    '''Everything in the file that repack must preserve.'''
    with h5py.File(path, 'r') as f:
        return dict(
            attrs=dict(f.attrs),
            inputs=f['MODELS'][MODEL]['inputs'][...],
            map=f['MODELS'][MODEL]['map'][...],
            map_attrs={k: list(v) for k, v in f['MODELS'][MODEL]['map'].attrs.items()},
            timeperiod=[d.decode() if isinstance(d, bytes) else d
                        for d in f['META']['timeperiod'][...]],
            catchment=f['DIMENSIONS']['catchment'][...],
            links=f['LINKS'][...],
        )


def _assert_same(a, b):
    assert a['attrs'] == b['attrs']
    np.testing.assert_array_equal(a['inputs'], b['inputs'])
    np.testing.assert_array_equal(a['map'], b['map'])
    assert a['map_attrs'] == b['map_attrs']
    assert a['timeperiod'] == b['timeperiod']
    np.testing.assert_array_equal(a['catchment'], b['catchment'])
    np.testing.assert_array_equal(a['links'], b['links'])


def _sized_model(tmp_path, n_timesteps=2000, n_cells=200, name='retime_model.h5'):
    period = pd.date_range('2000-01-01', periods=n_timesteps, freq='D')
    inputs = np.arange(n_cells * len(INPUTS) * n_timesteps,
                       dtype=np.float64).reshape(n_cells, len(INPUTS), n_timesteps)
    return _write_model_file(str(tmp_path / name), period, inputs), period


def _punch_hole(path, n=500_000):
    '''Leave an unreachable extent in the middle of the file.

    HDF5 truncates free space at the *end* of a file, so the deleted dataset has
    to be followed by something still live for the hole to persist.
    '''
    with h5py.File(path, 'r+') as f:
        f.create_dataset('scratch', data=np.zeros(n, dtype=np.float64))
        f.create_dataset('after_scratch', data=np.zeros(4, dtype=np.float64))
    size_with_scratch = os.path.getsize(path)
    with h5py.File(path, 'r+') as f:
        del f['scratch']
    return size_with_scratch


def test_repack_preserves_everything(tmp_path):
    path, _ = _sized_model(tmp_path, n_timesteps=50, n_cells=4)
    with h5py.File(path, 'r+') as f:
        f.attrs['openwater_version'] = 'test-version'
        f.attrs['signature_hash'] = 'abc123'
    before = _snapshot(path)

    mf = ModelFile(path)
    assert mf.repack() is mf          # chainable, like retime
    mf.close()

    _assert_same(before, _snapshot(path))


def test_repack_leaves_model_file_usable(tmp_path):
    path, period = _sized_model(tmp_path, n_timesteps=50, n_cells=4)
    mf = ModelFile(path)
    mf.repack()

    assert len(mf.time_period) == len(period)
    assert mf.time_period[-1] == period[-1]
    assert mf.dims_for_model(MODEL) == ['catchment']
    # the reopened handle points at the repacked file and is live
    assert mf._h5f['MODELS'][MODEL]['inputs'].shape == (4, len(INPUTS), 50)
    mf.close()


def test_repack_reclaims_freed_space(tmp_path):
    path, _ = _sized_model(tmp_path, n_timesteps=50, n_cells=4)
    size_with_scratch = _punch_hole(path)

    # deleting the dataset does not give the space back...
    assert os.path.getsize(path) == size_with_scratch

    mf = ModelFile(path)
    mf.repack()
    mf.close()

    # ...but repacking does
    assert os.path.getsize(path) < size_with_scratch / 2


def test_repack_reclaims_freed_space_without_touching_contents(tmp_path):
    path, _ = _sized_model(tmp_path, n_timesteps=50, n_cells=4)
    before = _snapshot(path)
    _punch_hole(path)

    mf = ModelFile(path)
    mf.repack()
    mf.close()

    _assert_same(before, _snapshot(path))
    with h5py.File(path, 'r') as f:
        assert 'scratch' not in f
        assert 'after_scratch' in f       # live objects survive


def test_repack_after_retime_shrinks_towards_the_new_data_size(tmp_path):
    path, period = _sized_model(tmp_path)
    original_size = os.path.getsize(path)

    new_period = pd.date_range(period[0], periods=len(period) + 200, freq='D')
    mf = ModelFile(path)
    mf.retime(new_period)
    retimed_size = os.path.getsize(path)
    after_retime = _snapshot(path)

    mf.repack()
    mf.close()
    packed_size = os.path.getsize(path)

    # never larger, and close to what the extra timesteps actually warrant
    assert packed_size <= retimed_size
    assert packed_size < 1.2 * original_size * len(new_period) / len(period)
    # and the contents are untouched
    _assert_same(after_retime, _snapshot(path))


def test_retime_repack_flag_matches_manual_repack(tmp_path):
    period = pd.date_range('2000-01-01', periods=500, freq='D')
    inputs = np.arange(20 * len(INPUTS) * len(period),
                       dtype=np.float64).reshape(20, len(INPUTS), len(period))
    new_period = pd.date_range(period[0], periods=len(period) + 100, freq='D')

    manual = _write_model_file(str(tmp_path / 'manual.h5'), period, inputs)
    mf = ModelFile(manual)
    mf.retime(new_period)
    mf.repack()
    mf.close()

    flagged = _write_model_file(str(tmp_path / 'flagged.h5'), period, inputs)
    mf = ModelFile(flagged)
    assert mf.retime(new_period, repack=True) is mf
    mf.close()

    assert os.path.getsize(flagged) == os.path.getsize(manual)
    _assert_same(_snapshot(manual), _snapshot(flagged))


def test_retime_does_not_repack_by_default(tmp_path):
    path, period = _sized_model(tmp_path, n_timesteps=50, n_cells=4)
    new_period = pd.date_range(period[0], periods=len(period) + 10, freq='D')

    calls = []
    original = ModelFile.repack
    try:
        ModelFile.repack = lambda self: calls.append(1) or self
        mf = ModelFile(path)
        mf.retime(new_period)
        assert calls == []
        mf.retime(new_period, repack=True)
        assert calls == [1]
        mf.close()
    finally:
        ModelFile.repack = original


def test_repack_failure_leaves_original_intact(tmp_path, monkeypatch):
    path, _ = _sized_model(tmp_path, n_timesteps=50, n_cells=4)
    before = _snapshot(path)
    size_before = os.path.getsize(path)

    def boom(*args, **kwargs):
        raise OSError('no space left on device')
    monkeypatch.setattr(os, 'replace', boom)

    mf = ModelFile(path)
    with pytest.raises(OSError):
        mf.repack()

    # the original is untouched, the half-written copy is cleaned up, and the
    # ModelFile is still usable
    assert os.path.getsize(path) == size_before
    assert not os.path.exists(path + '.repacking')
    assert len(mf.time_period) == 50
    mf.close()
    _assert_same(before, _snapshot(path))


# --- climatology / recycle fill rules ---------------------------------------

def _resize(old_arr, old_period, new_period, rules, var_names=None):
    src = align_source_indices(old_period, new_period)
    return build_resized_input_array(old_arr, src, var_names or INPUTS, MODEL,
                                     rules, new_period=new_period)


def test_mean_fill_uses_the_whole_covered_record():
    old_period = pd.date_range('2000-01-01', periods=4, freq='D')
    new_period = pd.date_range('2000-01-01', periods=6, freq='D')
    old = np.zeros((2, 2, 4))
    old[:, 0, :] = [[1, 2, 3, 4], [10, 20, 30, 40]]

    out = _resize(old, old_period, new_period, FillRules('mean'))

    np.testing.assert_array_equal(out[:, 0, :4], old[:, 0, :])   # untouched
    np.testing.assert_allclose(out[:, 0, 4:], [[2.5, 2.5], [25, 25]])


def test_monthly_mean_fill_uses_the_matching_calendar_month():
    old_period = pd.date_range('2000-01-01', '2000-12-31', freq='D')
    new_period = pd.date_range('2000-01-01', '2001-12-31', freq='D')
    old = np.zeros((1, 2, len(old_period)))
    old[0, 0, :] = old_period.month * 1.0        # value == month number

    out = _resize(old, old_period, new_period, FillRules('monthly_mean'))

    filled = pd.Series(out[0, 0, :], index=new_period)['2001']
    # every 2001 day takes the mean of the same month in 2000, i.e. the month number
    np.testing.assert_allclose(filled.to_numpy(), filled.index.month)


def test_daily_mean_fill_averages_matching_day_of_year():
    old_period = pd.date_range('2001-01-01', '2002-12-31', freq='D')  # two non-leap years
    new_period = pd.date_range('2001-01-01', '2003-12-31', freq='D')
    old = np.zeros((1, 2, len(old_period)))
    series = pd.Series(0.0, index=old_period)
    series['2001'] = 10.0
    series['2002'] = 20.0
    old[0, 0, :] = series.to_numpy()

    out = _resize(old, old_period, new_period, FillRules('daily_mean'))

    filled = pd.Series(out[0, 0, :], index=new_period)['2003']
    np.testing.assert_allclose(filled.to_numpy(), 15.0)          # mean of 10 and 20


def test_recycle_reproduces_the_reference_year_exactly():
    old_period = pd.date_range('2001-01-01', '2002-12-31', freq='D')
    new_period = pd.date_range('2001-01-01', '2003-12-31', freq='D')
    old = np.zeros((1, 2, len(old_period)))
    rng = np.random.RandomState(0)
    reference = rng.rand(365)
    old[0, 0, :] = np.concatenate([rng.rand(365), reference])    # 2001 noise, 2002 reference

    out = _resize(old, old_period, new_period,
                  FillRules(recycle(('2002-01-01', '2002-12-31'))))

    filled = pd.Series(out[0, 0, :], index=new_period)['2003']
    np.testing.assert_allclose(filled.to_numpy(), reference)


def test_recycle_window_can_be_an_explicit_date_index():
    old_period = pd.date_range('2001-01-01', '2002-12-31', freq='D')
    new_period = pd.date_range('2001-01-01', '2003-12-31', freq='D')
    old = np.zeros((1, 2, len(old_period)))
    old[0, 0, :] = np.concatenate([np.full(365, 1.0), np.full(365, 7.0)])

    window = pd.date_range('2002-01-01', '2002-12-31', freq='D')
    out = _resize(old, old_period, new_period, FillRules(recycle(window)))

    np.testing.assert_allclose(pd.Series(out[0, 0, :], index=new_period)['2003'], 7.0)


def test_recycle_falls_back_to_28_feb_for_a_leap_day():
    # reference year has no 29 Feb; the extension does
    old_period = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    new_period = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    old = np.zeros((1, 2, len(old_period)))
    series = pd.Series(1.0, index=old_period)
    series['2023-02-28'] = 99.0
    old[0, 0, :] = series.to_numpy()

    out = _resize(old, old_period, new_period,
                  FillRules(recycle(('2023-01-01', '2023-12-31'))))

    filled = pd.Series(out[0, 0, :], index=new_period)
    assert filled['2024-02-29'] == 99.0          # borrowed from 28 Feb
    assert filled['2024-02-28'] == 99.0


def test_climatology_over_a_window_ignores_data_outside_it():
    old_period = pd.date_range('2001-01-01', '2002-12-31', freq='D')
    new_period = pd.date_range('2001-01-01', '2003-12-31', freq='D')
    old = np.zeros((1, 2, len(old_period)))
    old[0, 0, :] = np.concatenate([np.full(365, 1000.0), np.full(365, 5.0)])

    out = _resize(old, old_period, new_period,
                  FillRules(climatology(by=None, over=('2002-01-01', '2002-12-31'))))

    np.testing.assert_allclose(pd.Series(out[0, 0, :], index=new_period)['2003'], 5.0)


def test_climatology_fills_leading_gaps_too():
    old_period = pd.date_range('2002-01-01', '2002-12-31', freq='D')
    new_period = pd.date_range('2001-01-01', '2002-12-31', freq='D')
    old = np.zeros((1, 2, len(old_period)))
    old[0, 0, :] = old_period.month * 1.0

    out = _resize(old, old_period, new_period, FillRules('monthly_mean'))

    filled = pd.Series(out[0, 0, :], index=new_period)['2001']
    np.testing.assert_allclose(filled.to_numpy(), filled.index.month)


def test_climatology_window_missing_data_warns_and_uses_whole_record(caplog):
    old_period = pd.date_range('2001-01-01', periods=10, freq='D')
    new_period = pd.date_range('2001-01-01', periods=12, freq='D')
    old = np.zeros((1, 2, 10))
    old[0, 0, :] = 4.0

    with caplog.at_level('WARNING'):
        out = _resize(old, old_period, new_period,
                      FillRules(climatology(by=None, over=('1990-01-01', '1990-12-31'))))

    assert 'covers none of the existing data' in caplog.text
    np.testing.assert_allclose(out[0, 0, 10:], 4.0)


def test_climatology_rule_without_dates_raises():
    old = np.zeros((1, 2, 4))
    src = np.array([0, 1, 2, 3, -1, -1])
    with pytest.raises(ValueError, match='needs the dates'):
        build_resized_input_array(old, src, INPUTS, MODEL, FillRules('monthly_mean'))


def test_fill_rule_aliases_and_factories_normalise():
    assert FillRules('mean').rule_for(MODEL, 'x') == ('climatology', None, None)
    assert FillRules('monthly').rule_for(MODEL, 'x') == ('climatology', 'month', None)
    assert FillRules('day_of_year').rule_for(MODEL, 'x') == ('climatology', 'day', None)
    assert climatology(by='month') == ('climatology', 'month', None)
    assert recycle(('2000-01-01', '2000-12-31'))[:2] == ('climatology', 'day')
    with pytest.raises(ValueError):
        climatology(by='fortnightly')
    with pytest.raises(ValueError):
        climatology(over=('2000-01-01', '2000-06-30', '2001-01-01'))
    with pytest.raises(ValueError):
        recycle(None)


def test_climatology_rules_mix_with_other_rules_per_variable(tmp_path):
    old_period = pd.date_range('2001-01-01', '2002-12-31', freq='D')
    new_period = pd.date_range('2001-01-01', '2003-12-31', freq='D')
    inputs = np.zeros((1, 2, len(old_period)))
    inputs[0, 0, :] = 3.0                                   # rainfall
    inputs[0, 1, :] = old_period.month * 1.0                # pet
    path = _write_model_file(_model_path(tmp_path), old_period, inputs)

    rules = (FillRules(default=0.0)
             .set('ffill', variable='rainfall')
             .set('monthly_mean', variable='pet'))

    mf = ModelFile(path)
    mf.retime(new_period, fill_rules=rules)
    mf.close()

    with h5py.File(path, 'r') as f:
        arr = f['MODELS'][MODEL]['inputs'][...]
    filled = pd.Series(arr[0, 1, :], index=new_period)['2003']
    np.testing.assert_allclose(arr[0, 0, :], 3.0)                        # ffill
    np.testing.assert_allclose(filled.to_numpy(), filled.index.month)    # monthly mean


# --- input_summary ---------------------------------------------------------

def test_input_summary_reports_every_stored_input(tmp_path):
    period = pd.date_range('2000-01-01', periods=6, freq='D')
    inputs = np.zeros((3, 2, 6))
    inputs[0, 0, :] = [1, 2, 3, 4, 5, 6]     # varying
    inputs[1, 0, :] = 5.0                    # active but constant
    #  cell 2 rainfall, and all of pet, left at zero

    path = _write_model_file(_model_path(tmp_path), period, inputs)
    mf = ModelFile(path)
    summary = mf.input_summary()
    mf.close()

    assert list(summary.columns) == ['model', 'input', 'cells', 'link_fed',
                                     'active_cells', 'time_varying_cells']
    rain = summary[summary.input == 'rainfall'].iloc[0]
    assert rain.model == MODEL
    assert rain.cells == 3
    assert not rain.link_fed
    assert rain.active_cells == 2
    assert rain.time_varying_cells == 1

    pet = summary[summary.input == 'pet'].iloc[0]
    assert pet.active_cells == 0


def test_input_summary_stored_only_drops_empty_and_link_fed(tmp_path):
    period = pd.date_range('2000-01-01', periods=6, freq='D')
    inputs = np.zeros((2, 2, 6))
    inputs[:, 0, :] = 1.0                    # rainfall active, pet empty
    path = _write_model_file(_model_path(tmp_path), period, inputs)

    mf = ModelFile(path)
    assert len(mf.input_summary()) == 2
    stored = mf.input_summary(stored_only=True)
    mf.close()

    assert list(stored.input) == ['rainfall']       # pet is empty
    assert list(stored.index) == [0]                # index reset


def test_input_summary_flags_link_fed_inputs(tmp_path):
    period = pd.date_range('2000-01-01', periods=6, freq='D')
    inputs = np.ones((2, 2, 6))
    path = _write_model_file(_model_path(tmp_path), period, inputs)

    # one link into pet (input index 1) of node 0
    with h5py.File(path, 'r+') as f:
        del f['LINKS']
        row = np.zeros((1, len(LINK_TABLE_COLUMNS)), dtype=np.uint32)
        row[0, LINK_TABLE_COLUMNS.index('dest_var')] = INPUTS.index('pet')
        f.create_dataset('LINKS', dtype=np.uint32, data=row)

    mf = ModelFile(path)
    summary = mf.input_summary().set_index('input')
    stored = mf.input_summary(stored_only=True)
    mf.close()

    assert not summary.loc['rainfall', 'link_fed']
    assert summary.loc['pet', 'link_fed']
    assert list(stored.input) == ['rainfall']       # link-fed rows are dropped


def test_input_summary_skips_models_without_stored_inputs(tmp_path):
    period = pd.date_range('2000-01-01', periods=6, freq='D')
    path = _write_model_file(_model_path(tmp_path), period, np.zeros((2, 2, 6)))
    with h5py.File(path, 'r+') as f:
        del f['MODELS'][MODEL]['inputs']

    mf = ModelFile(path)
    summary = mf.input_summary()
    mf.close()

    assert summary.empty
    assert list(summary.columns) == ['model', 'input', 'cells', 'link_fed',
                                     'active_cells', 'time_varying_cells']
