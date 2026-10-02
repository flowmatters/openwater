'''Tests for the model reporting API.

Reports leave the engine as bare integers -- a run index, a timestep index, a
code and up to three numbers. These tests are about the translation back into
dates, place names and messages, and about refusing to do that translation when
it would be wrong.
'''
import numpy as np
import pandas as pd
import pytest
import h5py

from openwater.results import (OpenwaterResults, REPORT_COLUMNS,
                               REPORT_COUNT_COLUMNS, _render_report)

CATCHMENTS = ['sc0', 'sc1', 'sc2']
N_TIMESTEPS = 10
SIGNATURE = 'abcd1234'


def write_model_file(path, signature=SIGNATURE, with_time=True, with_map=True):
    with h5py.File(path, 'w') as f:
        if signature is not None:
            f.attrs['signature_hash'] = signature

        meta = f.create_group('META')
        meta.create_dataset('models', data=[np.bytes_('StorageRouting')], dtype='S14')
        if with_time:
            dates = np.array([(np.datetime64('2020-01-01') + np.timedelta64(d, 'D')).astype(str) + 'T00:00:00'
                              for d in range(N_TIMESTEPS)], dtype=h5py.special_dtype(vlen=str))
            meta.create_dataset('timeperiod', data=dates)

        dims = f.create_group('DIMENSIONS')
        dims.create_dataset('catchment', data=[np.bytes_(c) for c in CATCHMENTS], dtype='S8')

        g = f.create_group('MODELS/StorageRouting')
        g.create_dataset('batches', data=np.array([len(CATCHMENTS)], dtype=np.int32))
        if with_map:
            run_map = g.create_dataset('map', data=np.arange(len(CATCHMENTS), dtype=np.int32))
            run_map.attrs['DIMS'] = [np.bytes_('catchment')]
    return str(path)


def write_results_file(path, runs, timesteps, codes, severities, args, counts,
                       signature=SIGNATURE, names=None, messages=None,
                       code_severity=None):
    names = names or ['massBalanceNotConverged', 'nanInput']
    messages = messages or ['routing mass balance error {0} m^3 exceeds limit {1} m^3',
                            'inflow {0} or lateral {1} is not a number']
    code_severity = code_severity if code_severity is not None else [1, 2]

    with h5py.File(path, 'w') as f:
        reports = f.create_group('REPORTS')
        if signature is not None:
            reports.create_dataset('signature', data=[np.bytes_(signature)], dtype='S8')

        g = reports.create_group('StorageRouting')
        g.create_dataset('names', data=[np.bytes_(n) for n in names],
                         dtype='S%d' % max(len(n) for n in names))
        g.create_dataset('messages', data=[np.bytes_(m) for m in messages],
                         dtype='S%d' % max(len(m) for m in messages))
        g.create_dataset('code_severity', data=np.array(code_severity, dtype=np.int32))
        g.create_dataset('counts', data=np.array(counts, dtype=np.int32))
        if runs is not None:
            g.create_dataset('run', data=np.array(runs, dtype=np.int32))
            g.create_dataset('timestep', data=np.array(timesteps, dtype=np.int32))
            g.create_dataset('code', data=np.array(codes, dtype=np.int32))
            g.create_dataset('severity', data=np.array(severities, dtype=np.int32))
            g.create_dataset('args', data=np.array(args, dtype=np.float64))
    return str(path)


@pytest.fixture
def results(tmp_path):
    '''Two cells report: run 1 three times, run 2 once. Run 0 stays silent.'''
    model = write_model_file(tmp_path / 'model.h5')
    res = write_results_file(
        tmp_path / 'results.h5',
        runs=[1, 1, 1, 2],
        timesteps=[2, 3, 4, 0],
        codes=[0, 0, 0, 0],
        severities=[1, 1, 1, 1],
        args=[[-161800.0, 0.001, 0.0],
              [-415674.0, 0.001, 0.0],
              [-463775.0, 0.001, 0.0],
              [-1000.0, 0.001, 0.0]],
        # Counts are complete: run 1 tripped 138 times, only three were kept.
        counts=[[0, 0], [138, 0], [1, 0]])
    return OpenwaterResults(model, res)


def test_reports_translate_runs_and_timesteps(results):
    df = results.reports()

    assert len(df) == 4
    assert list(df.columns[:len(REPORT_COLUMNS)]) == REPORT_COLUMNS
    assert 'catchment' in df.columns

    # Run index 1 is the second catchment, timestep 2 is the third day.
    assert list(df['catchment']) == ['sc1', 'sc1', 'sc1', 'sc2']
    assert df['date'].iloc[0] == pd.Timestamp('2020-01-03')
    assert df['date'].iloc[3] == pd.Timestamp('2020-01-01')
    assert list(df['timestep']) == [2, 3, 4, 0]


def test_reports_render_messages(results):
    df = results.reports()
    assert df['message'].iloc[0] == 'routing mass balance error -161800 m^3 exceeds limit 0.001 m^3'
    assert list(df['report'].unique()) == ['massBalanceNotConverged']
    assert list(df['severity'].unique()) == ['warning']


def test_reports_filter_by_dimension(results):
    assert len(results.reports(catchment='sc1')) == 3
    assert len(results.reports(catchment='sc2')) == 1
    assert len(results.reports(catchment='sc0')) == 0
    assert len(results.reports(catchment=['sc1', 'sc2'])) == 4


def test_reports_filter_by_severity(results):
    assert len(results.reports(severity='warning')) == 4
    assert len(results.reports(severity='error')) == 0
    assert len(results.reports(severity='all')) == 4

    with pytest.raises(ValueError):
        results.reports(severity='chatty')


def test_reports_filter_by_model(results):
    assert len(results.reports('StorageRouting')) == 4
    assert len(results.reports('Storage')) == 0


def test_counts_are_complete_where_details_are_capped(results):
    '''The point of having both accessors: three records kept, 138 occurrences.'''
    details = results.reports(catchment='sc1')
    counts = results.report_counts(catchment='sc1')

    assert len(details) == 3
    assert len(counts) == 1
    assert counts['occurrences'].iloc[0] == 138
    assert counts['report'].iloc[0] == 'massBalanceNotConverged'
    assert list(counts.columns[:len(REPORT_COUNT_COLUMNS)]) == REPORT_COUNT_COLUMNS


def test_report_counts_omit_silent_cells(results):
    counts = results.report_counts()
    assert set(counts['catchment']) == {'sc1', 'sc2'}


def test_configuration_faults_have_no_date(tmp_path):
    '''Timestep -1 means a configuration fault, which must not be given a date.'''
    model = write_model_file(tmp_path / 'model.h5')
    res = write_results_file(
        tmp_path / 'results.h5',
        runs=[0], timesteps=[-1], codes=[1], severities=[2],
        args=[[2.0, 0.0, 0.0]],
        counts=[[0, 1], [0, 0], [0, 0]])

    df = OpenwaterResults(model, res).reports(severity='error')
    assert len(df) == 1
    assert pd.isnull(df['date'].iloc[0])
    assert df['timestep'].iloc[0] == -1
    assert df['catchment'].iloc[0] == 'sc0'
    assert df['severity'].iloc[0] == 'error'


def test_signature_mismatch_raises(tmp_path):
    model = write_model_file(tmp_path / 'model.h5', signature='11112222')
    res = write_results_file(
        tmp_path / 'results.h5',
        runs=[1], timesteps=[0], codes=[0], severities=[1],
        args=[[1.0, 2.0, 0.0]], counts=[[0, 0], [1, 0], [0, 0]],
        signature='33334444')

    results = OpenwaterResults(model, res)
    with pytest.raises(Exception, match='wrong places and dates'):
        results.reports()

    # ...but the caller can insist.
    assert len(results.reports(check_signature=False)) == 1


def test_missing_signature_on_model_file_warns_but_reads(tmp_path, caplog):
    model = write_model_file(tmp_path / 'model.h5', signature=None)
    res = write_results_file(
        tmp_path / 'results.h5',
        runs=[1], timesteps=[0], codes=[0], severities=[1],
        args=[[1.0, 2.0, 0.0]], counts=[[0, 0], [1, 0], [0, 0]])

    df = OpenwaterResults(model, res).reports()
    assert len(df) == 1
    assert 'cannot confirm' in caplog.text


def test_reports_without_a_run_map_warn_and_omit_places(tmp_path, caplog):
    '''A model file that cannot say where a run is should still yield messages.'''
    model = write_model_file(tmp_path / 'model.h5', with_map=False)
    res = write_results_file(
        tmp_path / 'results.h5',
        runs=[1], timesteps=[0], codes=[0], severities=[1],
        args=[[1.0, 2.0, 0.0]], counts=[[0, 0], [1, 0], [0, 0]])

    df = OpenwaterResults(model, res).reports()
    assert len(df) == 1
    assert 'catchment' not in df.columns
    assert df['message'].iloc[0].startswith('routing mass balance error 1')
    assert 'dimension labels' in caplog.text


def test_reports_without_a_time_period_warn_and_omit_dates(tmp_path, caplog):
    model = write_model_file(tmp_path / 'model.h5', with_time=False)
    res = write_results_file(
        tmp_path / 'results.h5',
        runs=[1], timesteps=[0], codes=[0], severities=[1],
        args=[[1.0, 2.0, 0.0]], counts=[[0, 0], [1, 0], [0, 0]])

    df = OpenwaterResults(model, res).reports()
    assert len(df) == 1
    assert pd.isnull(df['date'].iloc[0])
    assert df['timestep'].iloc[0] == 0


def test_results_without_reports_return_an_empty_frame(tmp_path):
    '''Results written before reporting existed must keep working.'''
    model = write_model_file(tmp_path / 'model.h5')
    res_path = tmp_path / 'results.h5'
    with h5py.File(res_path, 'w') as f:
        f.create_group('MODELS')

    results = OpenwaterResults(model, str(res_path))
    df = results.reports()
    assert len(df) == 0
    assert list(df.columns) == REPORT_COLUMNS
    assert len(results.report_counts()) == 0


def test_silent_model_writes_counts_but_no_details(tmp_path):
    '''"Checked, all clear" must be distinguishable from "never ran".'''
    model = write_model_file(tmp_path / 'model.h5')
    res = write_results_file(
        tmp_path / 'results.h5',
        runs=None, timesteps=None, codes=None, severities=None, args=None,
        counts=[[0, 0], [0, 0], [0, 0]])

    results = OpenwaterResults(model, res)
    assert len(results.reports()) == 0
    assert len(results.report_counts()) == 0


def test_render_report_formats_like_the_engine():
    assert _render_report('{0} vs {1}', [-161800.23, 0.001, 0.0], 2) == '-161800 vs 0.001'
    assert _render_report('no placeholders', [1.0, 2.0, 3.0], 0) == 'no placeholders'
    assert _render_report('{0} {1} {2}', [1.5, 2.5, 3.5], 3) == '1.5 2.5 3.5'
