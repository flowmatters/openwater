"""Tests for version stamps and recorded variable names in model and results files."""
import numpy as np
import h5py
import pytest

import openwater.lib as lib
import openwater.results as results_mod
from openwater import releases, stamps
from openwater.results import OpenwaterResults

from .test_results import DATES, write_model_h5, write_results_h5, dummy_model  # noqa: F401

LIBRARY_VERSION = '1.0.3+abc1234.6d3bb266'
LIBRARY_SIGNATURE = '6d3bb266'


@pytest.fixture(autouse=True)
def loaded_library(monkeypatch):
    """Pretend a particular openwater-core build is loaded."""
    monkeypatch.setattr(lib, 'get_core_version', lambda: LIBRARY_VERSION)
    monkeypatch.setattr(lib, 'get_core_signature_hash', lambda: LIBRARY_SIGNATURE)
    results_mod._warned_signatures.clear()


def stamp_as_ow_sim(path, version, signature, **extra):
    """Stamp a results file the way ow-sim does."""
    with h5py.File(path, 'a') as f:
        f.attrs['openwater_version'] = version
        f.attrs['signature_hash'] = signature
        f.attrs['created_by'] = 'ow-sim'
        f.attrs['created_timestamp'] = '2026-10-02T06:00:00.000000'
        for k, v in extra.items():
            f.attrs[k] = v


def open_results(tmp_path, stamp=None, **kwargs):
    model_path = str(tmp_path / 'model.h5')
    results_path = str(tmp_path / 'outputs.h5')
    write_model_h5(model_path, DATES)
    write_results_h5(results_path, len(DATES))
    if stamp is not None:
        stamp_as_ow_sim(results_path, *stamp)
    return OpenwaterResults(model_path, results_path, **kwargs)


class TestResultsStampCheck:
    def test_matching_signature_logs_nothing(self, tmp_path, caplog):
        with caplog.at_level('WARNING', logger='openwater.results'):
            r = open_results(tmp_path, stamp=('1.0.3+fe85ae1.6d3bb266', LIBRARY_SIGNATURE))
            r.close()
        assert caplog.text == ''

    def test_mismatched_signature_warns_naming_both_versions(self, tmp_path, caplog):
        with caplog.at_level('WARNING', logger='openwater.results'):
            r = open_results(tmp_path, stamp=('1.0.1+2939cdc.74cdc5ba', '74cdc5ba'))
            r.close()
        assert 'Results signature mismatch' in caplog.text
        assert '1.0.1+2939cdc.74cdc5ba' in caplog.text
        assert LIBRARY_VERSION in caplog.text
        assert 'use_for_model' in caplog.text

    def test_mismatch_is_reported_once_across_files(self, tmp_path, caplog):
        (tmp_path / 'a').mkdir()
        (tmp_path / 'b').mkdir()
        with caplog.at_level('WARNING', logger='openwater.results'):
            for sub in ['a', 'b']:
                r = open_results(tmp_path / sub, stamp=('1.0.1+2939cdc.74cdc5ba', '74cdc5ba'))
                r.close()
        assert caplog.text.count('Results signature mismatch') == 1

    def test_unstamped_results_log_nothing_on_open(self, tmp_path, caplog):
        with caplog.at_level('WARNING', logger='openwater.results'):
            r = open_results(tmp_path)
            r.close()
        assert caplog.text == ''

    def test_no_library_means_no_check(self, tmp_path, caplog, monkeypatch):
        monkeypatch.setattr(lib, 'get_core_signature_hash', lambda: 'unknown')
        with caplog.at_level('WARNING', logger='openwater.results'):
            r = open_results(tmp_path, stamp=('1.0.1+2939cdc.74cdc5ba', '74cdc5ba'))
            r.close()
        assert caplog.text == ''

    def test_version_exposes_the_stamp(self, tmp_path):
        model_path = str(tmp_path / 'model.h5')
        results_path = str(tmp_path / 'outputs.h5')
        write_model_h5(model_path, DATES)
        write_results_h5(results_path, len(DATES))
        stamp_as_ow_sim(results_path, LIBRARY_VERSION, LIBRARY_SIGNATURE,
                        model_signature_hash='74cdc5ba')
        r = OpenwaterResults(model_path, results_path)
        try:
            version = r.version()
            assert version['openwater_version'] == LIBRARY_VERSION
            assert version['signature_hash'] == LIBRARY_SIGNATURE
            assert version['created_by'] == 'ow-sim'
            assert version['model_signature_hash'] == '74cdc5ba'
        finally:
            r.close()

    def test_unstamped_results_have_an_empty_version(self, tmp_path):
        r = open_results(tmp_path)
        try:
            assert r.version() == {}
        finally:
            r.close()

    def test_release_helpers_read_a_results_file(self, tmp_path):
        results_path = str(tmp_path / 'outputs.h5')
        write_results_h5(results_path, len(DATES))
        stamp_as_ow_sim(results_path, '1.0.1+2939cdc.74cdc5ba', '74cdc5ba')
        version, signature = releases.model_file_version(results_path)[:2]
        assert version == '1.0.1+2939cdc.74cdc5ba'
        assert signature == '74cdc5ba'


class TestReadingByName:
    """A results file that names its variables is read by name, whatever is loaded."""

    def _open(self, tmp_path, names, n_outputs=2):
        model_path = str(tmp_path / 'model.h5')
        results_path = str(tmp_path / 'outputs.h5')
        write_model_h5(model_path, DATES)
        outputs = np.zeros((4, n_outputs, len(DATES)))
        for slot in range(n_outputs):
            outputs[:, slot, :] = 100.0 * (slot + 1)
        with h5py.File(results_path, 'w') as f:
            ds = f.create_dataset('/MODELS/DummyModel/outputs', data=outputs)
            if names is not None:
                ds.attrs['VARIABLES'] = np.array([n.encode() for n in names])
        return OpenwaterResults(model_path, results_path)

    def test_variable_is_found_where_the_file_says_it_is(self, tmp_path):
        # The loaded description has runoff as its only output, in slot 0. This
        # file was written by a build that had another output ahead of it.
        r = self._open(tmp_path, ['quickflow', 'runoff'])
        try:
            ts = r.time_series('DummyModel', 'runoff', 'catchment', hru='h1')
            assert ts['c1'].iloc[0] == pytest.approx(200.0)
        finally:
            r.close()

    def test_variables_the_loaded_library_has_never_heard_of_are_readable(self, tmp_path):
        r = self._open(tmp_path, ['quickflow', 'runoff'])
        try:
            ts = r.time_series('DummyModel', 'quickflow', 'catchment', hru='h1')
            assert ts['c1'].iloc[0] == pytest.approx(100.0)
        finally:
            r.close()

    def test_variables_for_lists_what_the_file_holds(self, tmp_path):
        r = self._open(tmp_path, ['quickflow', 'runoff'])
        try:
            assert r.variables_for('DummyModel') == ['rainfall', 'quickflow', 'runoff']
        finally:
            r.close()

    def test_a_variable_the_file_does_not_hold_is_an_error(self, tmp_path):
        # 'runoff' is in the loaded description, but not in this file. Reading
        # slot 0 because the description says so would return something else.
        r = self._open(tmp_path, ['quickflow', 'slowflow'])
        try:
            with pytest.raises(Exception) as e:
                r.time_series('DummyModel', 'runoff', 'catchment', hru='h1')
            assert 'quickflow' in str(e.value) and 'slowflow' in str(e.value)
        finally:
            r.close()

    def test_names_that_do_not_fit_the_dataset_are_ignored(self, tmp_path, caplog):
        r = self._open(tmp_path, ['runoff'], n_outputs=2)
        try:
            with caplog.at_level('WARNING', logger='openwater.stamps'):
                with pytest.raises(results_mod.VariableMismatchError):
                    r.time_series('DummyModel', 'runoff', 'catchment', hru='h1')
            assert 'ignoring the names' in caplog.text
        finally:
            r.close()

    def test_inputs_are_read_by_name_from_the_model_file(self, tmp_path):
        model_path = str(tmp_path / 'model.h5')
        results_path = str(tmp_path / 'outputs.h5')
        write_model_h5(model_path, DATES)
        write_results_h5(results_path, len(DATES))
        inputs = np.zeros((4, 2, len(DATES)))
        inputs[:, 0, :] = 5.0
        inputs[:, 1, :] = 7.0
        with h5py.File(model_path, 'a') as f:
            ds = f.create_dataset('/MODELS/DummyModel/inputs', data=inputs)
            ds.attrs['VARIABLES'] = np.array([b'pet', b'rainfall'])
        r = OpenwaterResults(model_path, results_path)
        try:
            ts = r.time_series('DummyModel', 'rainfall', 'catchment', hru='h1')
            assert ts['c1'].iloc[0] == pytest.approx(7.0)
        finally:
            r.close()


class TestVariableNames:
    def test_round_trip(self, tmp_path):
        with h5py.File(tmp_path / 'f.h5', 'w') as f:
            ds = f.create_dataset('/MODELS/M/outputs', shape=(3, 2, 5))
            assert stamps.variable_names(ds) is None
            assert stamps.set_variable_names(ds, ['outflow', 'storage'])
            assert stamps.variable_names(ds) == ['outflow', 'storage']

    def test_parameters_are_named_along_their_first_axis(self, tmp_path):
        with h5py.File(tmp_path / 'f.h5', 'w') as f:
            ds = f.create_dataset('/MODELS/M/parameters', shape=(2, 7))
            assert stamps.set_variable_names(ds, ['k', 'x'])
            assert stamps.variable_names(ds) == ['k', 'x']

    def test_names_are_not_written_when_the_count_is_wrong(self, tmp_path):
        with h5py.File(tmp_path / 'f.h5', 'w') as f:
            ds = f.create_dataset('/MODELS/M/outputs', shape=(3, 3, 5))
            assert not stamps.set_variable_names(ds, ['outflow', 'storage'])
            assert 'VARIABLES' not in ds.attrs

    def test_a_model_with_no_states(self, tmp_path):
        with h5py.File(tmp_path / 'f.h5', 'w') as f:
            ds = f.create_dataset('/MODELS/M/states', shape=(3, 0))
            assert stamps.set_variable_names(ds, [])
            assert stamps.variable_names(ds) == []

    def test_table_parameters_get_a_name_per_row(self):
        desc = {
            'Dimensions': ['nPts'],
            'Parameters': [
                {'Name': 'nPts', 'Dimensions': []},
                {'Name': 'levels', 'Dimensions': ['nPts']},
                {'Name': 'scale', 'Dimensions': []},
            ],
        }
        # One cell, a table of three points: nPts, levels[0..2], scale
        parameters = np.array([[3.0], [0.0], [5.0], [10.0], [1.5]])
        assert stamps.parameter_row_names(desc, parameters) == [
            'nPts', 'levels[0]', 'levels[1]', 'levels[2]', 'scale']


class TestDerivedFilesKeepTheirMetadata:
    def test_stamp_and_names_are_carried_to_a_derived_file(self, tmp_path):
        with h5py.File(tmp_path / 'src.h5', 'w') as src, h5py.File(tmp_path / 'dest.h5', 'w') as dest:
            src.attrs['openwater_version'] = '1.0.1+2939cdc.74cdc5ba'
            src.attrs['signature_hash'] = '74cdc5ba'
            src.attrs['created_by'] = 'openwater-py 0.1'
            stamps.set_variable_names(src.create_dataset('/MODELS/M/inputs', shape=(4, 2, 10)), ['rainfall', 'pet'])
            stamps.set_variable_names(src.create_dataset('/MODELS/M/states', shape=(4, 1)), ['store'])
            src.create_dataset('/MODELS/Unnamed/inputs', shape=(4, 1, 10))

            # A clipped copy: fewer cells and timesteps, same variables.
            dest.create_dataset('/MODELS/M/inputs', shape=(2, 2, 5))
            dest.create_dataset('/MODELS/M/states', shape=(2, 1))
            dest.create_dataset('/MODELS/Unnamed/inputs', shape=(2, 1, 5))

            stamps.copy_model_metadata(src, dest)

            assert stamps.read_stamp(dest) == {
                'openwater_version': '1.0.1+2939cdc.74cdc5ba',
                'signature_hash': '74cdc5ba',
                'created_by': 'openwater-py 0.1',
            }
            assert stamps.variable_names(dest['/MODELS/M/inputs']) == ['rainfall', 'pet']
            assert stamps.variable_names(dest['/MODELS/M/states']) == ['store']
            assert stamps.variable_names(dest['/MODELS/Unnamed/inputs']) is None

    def test_an_unstamped_source_leaves_the_copy_unstamped(self, tmp_path):
        with h5py.File(tmp_path / 'src.h5', 'w') as src, h5py.File(tmp_path / 'dest.h5', 'w') as dest:
            stamps.copy_model_metadata(src, dest)
            assert stamps.read_stamp(dest) == {}
