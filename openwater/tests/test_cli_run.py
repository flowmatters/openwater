"""Tests for the ow-run CLI."""
import argparse
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from openwater.cli_run import (
    _parse_param,
    _parse_data,
    _apply_param_overrides,
    _apply_time_override,
    _write_results,
    build_parser,
    main,
)
from openwater.text_model import (
    ModelDefinition,
    ModelResults,
    NodeConfig,
    load_model_from_dict,
)
from openwater.template import OWTemplate


# Reuse mock model types from test_text_model
class _MockModelType:
    def __init__(self, name, inputs=None, outputs=None, states=None, parameters=None):
        self.name = name
        self.description = {
            'Inputs': inputs or [],
            'Outputs': outputs or [],
            'States': states or [],
            'Parameters': parameters or [],
        }


MockSimhyd = _MockModelType(
    'Simhyd',
    inputs=['rainfall', 'pet'],
    outputs=['runoff', 'baseflow'],
    states=['SoilMoistureStore', 'GroundwaterStore'],
    parameters=[
        {'Name': 'baseflowCoefficient', 'Default': 0.0, 'Dimensions': []},
        {'Name': 'perviousFraction', 'Default': 0.9, 'Dimensions': []},
    ],
)

MODEL_REGISTRY = {'Simhyd': MockSimhyd}


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

class TestParseParam:
    def test_float_value(self):
        node, param, value = _parse_param('S1.perviousFraction=0.8')
        assert node == 'S1'
        assert param == 'perviousFraction'
        assert value == 0.8

    def test_int_value(self):
        node, param, value = _parse_param('S1.count=5')
        assert node == 'S1'
        assert param == 'count'
        assert value == 5
        assert isinstance(value, int)

    def test_string_value(self):
        node, param, value = _parse_param('S1.label=hello')
        assert value == 'hello'

    def test_missing_equals_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="expected"):
            _parse_param('S1.perviousFraction')

    def test_missing_dot_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="node"):
            _parse_param('perviousFraction=0.8')

    def test_dotted_node_name(self):
        node, param, value = _parse_param('sub.S1.perviousFraction=0.8')
        assert node == 'sub.S1'
        assert param == 'perviousFraction'


class TestParseData:
    def test_basic(self):
        name, path = _parse_data('climate=new_data.csv')
        assert name == 'climate'
        assert path == 'new_data.csv'

    def test_path_with_equals(self):
        name, path = _parse_data('src=a=b.csv')
        assert name == 'src'
        assert path == 'a=b.csv'

    def test_missing_equals_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="expected"):
            _parse_data('climate')


# ---------------------------------------------------------------------------
# Override application tests
# ---------------------------------------------------------------------------

class TestApplyParamOverrides:
    def test_override_existing_param(self):
        model_def = ModelDefinition(
            template=OWTemplate(),
            node_configs={'S1': NodeConfig(parameters={'perviousFraction': 0.9})},
            data_source_paths={},
            time_period=None,
            base_dir=None,
        )
        _apply_param_overrides(model_def, [('S1', 'perviousFraction', 0.8)])
        assert model_def.node_configs['S1'].parameters['perviousFraction'] == 0.8

    def test_override_new_node(self):
        model_def = ModelDefinition(
            template=OWTemplate(),
            node_configs={},
            data_source_paths={},
            time_period=None,
            base_dir=None,
        )
        _apply_param_overrides(model_def, [('S1', 'perviousFraction', 0.8)])
        assert model_def.node_configs['S1'].parameters['perviousFraction'] == 0.8

    def test_multiple_overrides(self):
        model_def = ModelDefinition(
            template=OWTemplate(),
            node_configs={'S1': NodeConfig(parameters={'a': 1})},
            data_source_paths={},
            time_period=None,
            base_dir=None,
        )
        _apply_param_overrides(model_def, [
            ('S1', 'a', 2),
            ('S1', 'b', 3),
            ('S2', 'c', 4),
        ])
        assert model_def.node_configs['S1'].parameters == {'a': 2, 'b': 3}
        assert model_def.node_configs['S2'].parameters == {'c': 4}


class TestApplyTimeOverride:
    def test_both_start_and_end(self):
        model_def = ModelDefinition(
            template=OWTemplate(), node_configs={},
            data_source_paths={}, time_period=('2010-01-01', '2010-12-31'),
            base_dir=None,
        )
        _apply_time_override(model_def, '2012-01-01', '2012-12-31')
        assert model_def.time_period == ('2012-01-01', '2012-12-31')

    def test_start_only(self):
        model_def = ModelDefinition(
            template=OWTemplate(), node_configs={},
            data_source_paths={}, time_period=('2010-01-01', '2010-12-31'),
            base_dir=None,
        )
        _apply_time_override(model_def, '2012-01-01', None)
        assert model_def.time_period == ('2012-01-01', '2010-12-31')

    def test_no_override(self):
        model_def = ModelDefinition(
            template=OWTemplate(), node_configs={},
            data_source_paths={}, time_period=('2010-01-01', '2010-12-31'),
            base_dir=None,
        )
        _apply_time_override(model_def, None, None)
        assert model_def.time_period == ('2010-01-01', '2010-12-31')


# ---------------------------------------------------------------------------
# Result writing tests
# ---------------------------------------------------------------------------

class TestWriteResults:
    def test_writes_csvs(self, tmp_path):
        results = ModelResults(
            outputs={
                'S1': pd.DataFrame({'runoff': [1.0, 2.0]}, index=pd.date_range('2010-01-01', periods=2)),
                'S2': pd.DataFrame({'runoff': [3.0, 4.0]}, index=pd.date_range('2010-01-01', periods=2)),
            },
            states={
                'S1': pd.DataFrame({'SoilMoistureStore': [100.0]}),
                'S2': pd.DataFrame(),
            },
            time_period=('2010-01-01', '2010-01-02'),
        )
        dest = str(tmp_path / 'out')
        _write_results(results, dest)

        assert os.path.isfile(os.path.join(dest, 'S1_outputs.csv'))
        assert os.path.isfile(os.path.join(dest, 'S2_outputs.csv'))
        assert os.path.isfile(os.path.join(dest, 'S1_states.csv'))
        # S2 states was empty, should not be written
        assert not os.path.isfile(os.path.join(dest, 'S2_states.csv'))

        df = pd.read_csv(os.path.join(dest, 'S1_outputs.csv'), index_col=0)
        assert list(df['runoff']) == [1.0, 2.0]

    def test_creates_directory(self, tmp_path):
        dest = str(tmp_path / 'nested' / 'deep' / 'out')
        results = ModelResults(outputs={}, states={}, time_period=None)
        _write_results(results, dest)
        assert os.path.isdir(dest)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_required_args(self):
        parser = build_parser()
        args = parser.parse_args(['model.yaml', 'results/'])
        assert args.model_file == 'model.yaml'
        assert args.output_dir == 'results/'
        assert args.param == []
        assert args.data == []

    def test_param_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            'model.yaml', 'results/',
            '--param', 'S1.x=1',
            '--param', 'S2.y=2',
        ])
        assert args.param == ['S1.x=1', 'S2.y=2']

    def test_data_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            'model.yaml', 'results/',
            '--data', 'climate=c.csv',
        ])
        assert args.data == ['climate=c.csv']

    def test_time_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            'model.yaml', 'results/',
            '--start', '2012-01-01',
            '--end', '2012-12-31',
        ])
        assert args.start == '2012-01-01'
        assert args.end == '2012-12-31'

    def test_init_states_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            'model.yaml', 'results/',
            '--init-states', 'prev_run/',
        ])
        assert args.init_states == 'prev_run/'

    def test_log_level_default(self):
        parser = build_parser()
        args = parser.parse_args(['model.yaml', 'results/'])
        assert args.log_level == 'warning'

    def test_log_level_flag(self):
        parser = build_parser()
        args = parser.parse_args(['model.yaml', 'results/', '--log-level', 'debug'])
        assert args.log_level == 'debug'


# ---------------------------------------------------------------------------
# Integration test (mocked model execution)
# ---------------------------------------------------------------------------

class TestMainIntegration:
    def _make_model(self, tmp_path, data=None):
        if data is None:
            data = {
                'ow_model': '1.0',
                'label': 'CLI Test',
                'time_period': {'start': '2010-01-01', 'end': '2010-01-03'},
                'nodes': [{'name': 'S1', 'model': 'Simhyd', 'parameters': {'baseflowCoefficient': 0.3}}],
            }
        return load_model_from_dict(data, model_registry=MODEL_REGISTRY, base_dir=str(tmp_path))

    def _mock_results(self):
        return ModelResults(
            outputs={'S1': pd.DataFrame({'runoff': [1.0, 2.0, 3.0], 'baseflow': [0.1, 0.2, 0.3]},
                                        index=pd.date_range('2010-01-01', periods=3))},
            states={'S1': pd.DataFrame({'SoilMoistureStore': [100.0], 'GroundwaterStore': [50.0]})},
            time_period=('2010-01-01', '2010-01-03'),
        )

    def test_end_to_end(self, tmp_path):
        model_yaml = tmp_path / 'model.yaml'
        model_yaml.write_text('dummy')  # won't be read due to mocking
        out_dir = str(tmp_path / 'results')

        real_model = self._make_model(tmp_path)

        with patch('openwater.cli_run.load_model', return_value=real_model), \
             patch('openwater.cli_run.run_model', return_value=self._mock_results()):
            main([str(model_yaml), out_dir, '--param', 'S1.perviousFraction=0.8'])

        # Verify parameter override was applied
        assert real_model.node_configs['S1'].parameters['perviousFraction'] == 0.8

        # Verify output files exist
        assert os.path.isfile(os.path.join(out_dir, 'S1_outputs.csv'))
        assert os.path.isfile(os.path.join(out_dir, 'S1_states.csv'))

    def test_data_source_override(self, tmp_path):
        model_yaml = tmp_path / 'model.yaml'
        model_yaml.write_text('dummy')
        out_dir = str(tmp_path / 'results')

        real_model = self._make_model(tmp_path, {
            'ow_model': '1.0',
            'data_sources': {'rainfall': 'original.csv'},
            'nodes': [{'name': 'S1', 'model': 'Simhyd'}],
        })

        mock_results = ModelResults(
            outputs={'S1': pd.DataFrame({'runoff': [0.0]})},
            states={},
            time_period=None,
        )

        with patch('openwater.cli_run.load_model', return_value=real_model), \
             patch('openwater.cli_run.run_model', return_value=mock_results):
            main([str(model_yaml), out_dir, '--data', 'rainfall=new_rain.csv'])

        assert real_model.data_source_paths['rainfall'] == 'new_rain.csv'

    def test_time_override(self, tmp_path):
        model_yaml = tmp_path / 'model.yaml'
        model_yaml.write_text('dummy')
        out_dir = str(tmp_path / 'results')

        real_model = self._make_model(tmp_path)
        mock_results = ModelResults(outputs={}, states={}, time_period=None)

        with patch('openwater.cli_run.load_model', return_value=real_model), \
             patch('openwater.cli_run.run_model', return_value=mock_results):
            main([str(model_yaml), out_dir, '--start', '2012-01-01', '--end', '2012-06-30'])

        assert real_model.time_period == ('2012-01-01', '2012-06-30')

    def test_init_states_from_dir(self, tmp_path):
        model_yaml = tmp_path / 'model.yaml'
        model_yaml.write_text('dummy')
        out_dir = str(tmp_path / 'results')

        # Create a states directory with CSV files
        states_dir = tmp_path / 'prev_states'
        states_dir.mkdir()
        pd.DataFrame({
            'SoilMoistureStore': [100.0, 200.0],
            'GroundwaterStore': [10.0, 30.0],
        }).to_csv(states_dir / 'S1_states.csv')

        real_model = self._make_model(tmp_path)
        mock_results = ModelResults(outputs={}, states={}, time_period=None)

        with patch('openwater.cli_run.load_model', return_value=real_model), \
             patch('openwater.cli_run.run_model', return_value=mock_results):
            main([str(model_yaml), out_dir, '--init-states', str(states_dir)])

        # Last row should be loaded
        assert real_model.node_configs['S1'].initial_states['SoilMoistureStore'] == 200.0
        assert real_model.node_configs['S1'].initial_states['GroundwaterStore'] == 30.0

    def test_info_logging_includes_timings(self, tmp_path, caplog):
        model_yaml = tmp_path / 'model.yaml'
        model_yaml.write_text('dummy')
        out_dir = str(tmp_path / 'results')

        real_model = self._make_model(tmp_path)

        import logging
        with caplog.at_level(logging.INFO, logger='openwater.cli_run'), \
             patch('openwater.cli_run.load_model', return_value=real_model), \
             patch('openwater.cli_run.run_model', return_value=self._mock_results()):
            main([str(model_yaml), out_dir, '--log-level', 'info'])

        messages = caplog.text
        assert 'Loaded model from' in messages
        assert 'Model execution completed' in messages
        assert 'Total time' in messages
        # Timings should appear as e.g. "(0.001s)"
        assert 's)' in messages
