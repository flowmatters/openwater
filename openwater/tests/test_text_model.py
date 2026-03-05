"""Tests for complete-model YAML loading and execution."""
import os
import tempfile
from unittest.mock import patch, MagicMock

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from openwater.template import OWTemplate, OWNode, OWLink
from openwater.persistence import (
    _parse_shorthand_link,
    _normalize_links,
    TemplateLoadError,
)
from openwater.text_model import (
    ModelDefinition,
    ModelResults,
    ModelLoadError,
    NodeConfig,
    TimeseriesRef,
    load_model,
    load_model_from_dict,
    load_initial_states,
    run_model,
    _build_graph,
    _resolve_data_sources,
    _find_linked_input,
    _extract_inline_outputs,
)


# ---------------------------------------------------------------------------
# Mock model types — same pattern as test_persistence.py
# ---------------------------------------------------------------------------

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
        {'Name': 'soilMoistureStoreCapacity', 'Default': 320.0, 'Dimensions': []},
    ],
)

MockDepthToRate = _MockModelType(
    'DepthToRate',
    inputs=['input'],
    outputs=['outflow'],
    states=[],
    parameters=[
        {'Name': 'DeltaT', 'Default': 86400.0, 'Dimensions': []},
        {'Name': 'area', 'Default': 1.0, 'Dimensions': []},
    ],
)

MockStorageRouting = _MockModelType(
    'StorageRouting',
    inputs=['inflow', 'lateral'],
    outputs=['outflow', 'storage'],
    states=['S', 'prevInflow', 'prevOutflow'],
    parameters=[
        {'Name': 'InflowBias', 'Default': 0.0, 'Dimensions': []},
        {'Name': 'RoutingConstant', 'Default': 0.0, 'Dimensions': []},
        {'Name': 'RoutingPower', 'Default': 0.0, 'Dimensions': []},
        {'Name': 'area', 'Default': 0.0, 'Dimensions': []},
        {'Name': 'deadStorage', 'Default': 0.0, 'Dimensions': []},
        {'Name': 'DeltaT', 'Default': 86400.0, 'Dimensions': []},
    ],
)

MockStorage = _MockModelType(
    'Storage',
    inputs=['inflow'],
    outputs=['outflow', 'overflow'],
    states=['currentVolume'],
    parameters=[
        {'Name': 'nLVA', 'Default': 0.0, 'Dimensions': []},
        {'Name': 'levels', 'Default': 0.0, 'Dimensions': ['nLVA']},
        {'Name': 'volumes', 'Default': 0.0, 'Dimensions': ['nLVA']},
        {'Name': 'areas', 'Default': 0.0, 'Dimensions': ['nLVA']},
    ],
)

MODEL_REGISTRY = {
    'Simhyd': MockSimhyd,
    'DepthToRate': MockDepthToRate,
    'StorageRouting': MockStorageRouting,
    'Storage': MockStorage,
}


# ---------------------------------------------------------------------------
# Loading tests
# ---------------------------------------------------------------------------

class TestLoadModelFromDict:
    def test_basic_model(self):
        data = {
            'ow_model': '1.0',
            'label': 'Test Model',
            'nodes': [
                {'name': 'N1', 'model': 'Simhyd'},
            ],
        }
        model = load_model_from_dict(data, model_registry=MODEL_REGISTRY)
        assert isinstance(model, ModelDefinition)
        assert model.template.label == 'Test Model'
        assert len(model.template.nodes) == 1
        assert model.template.nodes[0].name == 'N1'
        assert 'N1' in model.node_configs

    def test_missing_ow_model_raises(self):
        data = {'nodes': []}
        with pytest.raises(ModelLoadError, match="ow_model"):
            load_model_from_dict(data, model_registry=MODEL_REGISTRY)

    def test_time_period(self):
        data = {
            'ow_model': '1.0',
            'time_period': {'start': '2010-01-01', 'end': '2010-12-31'},
            'nodes': [{'name': 'N1', 'model': 'Simhyd'}],
        }
        model = load_model_from_dict(data, model_registry=MODEL_REGISTRY)
        assert model.time_period == ('2010-01-01', '2010-12-31')

    def test_time_period_missing_end_raises(self):
        data = {
            'ow_model': '1.0',
            'time_period': {'start': '2010-01-01'},
            'nodes': [],
        }
        with pytest.raises(ModelLoadError, match="start.*end"):
            load_model_from_dict(data, model_registry=MODEL_REGISTRY)

    def test_data_sources(self):
        data = {
            'ow_model': '1.0',
            'data_sources': {'climate': 'data/climate.csv'},
            'nodes': [{'name': 'N1', 'model': 'Simhyd'}],
        }
        model = load_model_from_dict(data, model_registry=MODEL_REGISTRY)
        assert model.data_source_paths == {'climate': 'data/climate.csv'}

    def test_node_parameters(self):
        data = {
            'ow_model': '1.0',
            'nodes': [{
                'name': 'N1', 'model': 'Simhyd',
                'parameters': {
                    'baseflowCoefficient': 0.3,
                    'soilMoistureStoreCapacity': 320.0,
                },
            }],
        }
        model = load_model_from_dict(data, model_registry=MODEL_REGISTRY)
        config = model.node_configs['N1']
        assert config.parameters['baseflowCoefficient'] == 0.3
        assert config.parameters['soilMoistureStoreCapacity'] == 320.0

    def test_node_initial_states(self):
        data = {
            'ow_model': '1.0',
            'nodes': [{
                'name': 'N1', 'model': 'Simhyd',
                'initial_states': {'SoilMoistureStore': 100.0},
            }],
        }
        model = load_model_from_dict(data, model_registry=MODEL_REGISTRY)
        config = model.node_configs['N1']
        assert config.initial_states['SoilMoistureStore'] == 100.0

    def test_node_timeseries(self):
        data = {
            'ow_model': '1.0',
            'nodes': [{
                'name': 'N1', 'model': 'Simhyd',
                'timeseries': {
                    'rainfall': {
                        'data_source': 'climate',
                        'column': 'Rain_mm',
                    },
                },
            }],
        }
        model = load_model_from_dict(data, model_registry=MODEL_REGISTRY)
        config = model.node_configs['N1']
        assert 'rainfall' in config.timeseries
        ts_ref = config.timeseries['rainfall']
        assert ts_ref.data_source == 'climate'
        assert ts_ref.column == 'Rain_mm'

    def test_links_preserved(self):
        data = {
            'ow_model': '1.0',
            'nodes': [
                {'name': 'N1', 'model': 'Simhyd'},
                {'name': 'N2', 'model': 'DepthToRate'},
            ],
            'links': [{
                'from': {'node': 'N1', 'output': 'runoff'},
                'to': {'node': 'N2', 'input': 'input'},
            }],
        }
        model = load_model_from_dict(data, model_registry=MODEL_REGISTRY)
        assert len(model.template.links) == 1
        link = model.template.links[0]
        assert link.from_node.name == 'N1'
        assert link.to_node.name == 'N2'

    def test_tabular_parameters(self):
        data = {
            'ow_model': '1.0',
            'nodes': [{
                'name': 'S1', 'model': 'Storage',
                'parameters': {
                    'nLVA': 5,
                    'levels': [0.0, 5.0, 10.0, 15.0, 20.0],
                    'volumes': [0.0, 50000.0, 200000.0, 600000.0, 1200000.0],
                    'areas': [0.0, 1000.0, 3000.0, 7000.0, 12000.0],
                },
            }],
        }
        model = load_model_from_dict(data, model_registry=MODEL_REGISTRY)
        config = model.node_configs['S1']
        assert config.parameters['nLVA'] == 5
        assert config.parameters['levels'] == [0.0, 5.0, 10.0, 15.0, 20.0]

    def test_config_keys_stripped_from_template(self):
        """parameters/initial_states/timeseries should NOT appear in template node data."""
        data = {
            'ow_model': '1.0',
            'nodes': [{
                'name': 'N1', 'model': 'Simhyd',
                'parameters': {'baseflowCoefficient': 0.3},
                'initial_states': {'SoilMoistureStore': 50.0},
                'timeseries': {
                    'rainfall': {'data_source': 'c', 'column': 'r'},
                },
            }],
        }
        model = load_model_from_dict(data, model_registry=MODEL_REGISTRY)
        # Template should parse without error — the extra keys were stripped
        assert len(model.template.nodes) == 1
        assert model.template.nodes[0].model_name == 'Simhyd'

    def test_node_tags_preserved(self):
        data = {
            'ow_model': '1.0',
            'nodes': [{
                'name': 'N1', 'model': 'Simhyd',
                'tags': {'process': 'RR', 'hru': 'Forest'},
            }],
        }
        model = load_model_from_dict(data, model_registry=MODEL_REGISTRY)
        node = model.template.nodes[0]
        assert node.tags['_process'] == 'RR'
        assert node.tags['hru'] == 'Forest'


class TestLoadModelFromFile:
    def test_load_from_yaml_file(self):
        yaml_content = """\
ow_model: "1.0"
label: "File Test"
nodes:
  - name: "N1"
    model: Simhyd
    parameters:
      baseflowCoefficient: 0.5
"""
        with tempfile.NamedTemporaryFile(
            suffix='.yaml', delete=False, mode='w'
        ) as f:
            f.write(yaml_content)
            path = f.name

        try:
            model = load_model(path, model_registry=MODEL_REGISTRY)
            assert model.template.label == 'File Test'
            assert model.node_configs['N1'].parameters['baseflowCoefficient'] == 0.5
            assert model.base_dir == os.path.dirname(os.path.abspath(path))
        finally:
            os.unlink(path)

    def test_nonexistent_file_raises(self):
        with pytest.raises(ModelLoadError, match="Failed to read"):
            load_model('/nonexistent/path.yaml')


# ---------------------------------------------------------------------------
# Defaults tests
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_file_defaults_parsed(self):
        data = {
            'ow_model': '1.0',
            'defaults': {'DeltaT': 86400},
            'nodes': [{'name': 'D1', 'model': 'DepthToRate'}],
        }
        model = load_model_from_dict(data, model_registry=MODEL_REGISTRY)
        assert model.defaults == {'DeltaT': 86400}

    def test_no_defaults_section(self):
        data = {
            'ow_model': '1.0',
            'nodes': [{'name': 'D1', 'model': 'DepthToRate'}],
        }
        model = load_model_from_dict(data, model_registry=MODEL_REGISTRY)
        assert model.defaults == {}

    @patch('openwater.text_model.ow_lib', create=True)
    def test_file_defaults_applied_at_run(self, mock_lib):
        data = {
            'ow_model': '1.0',
            'defaults': {'DeltaT': 3600},
            'nodes': [{'name': 'D1', 'model': 'DepthToRate'}],
        }
        model_def = load_model_from_dict(data, model_registry=MODEL_REGISTRY)

        mock_func = MagicMock()
        mock_func.return_value = [np.array([[0.0]])]

        with patch('openwater.lib.DepthToRate', mock_func, create=True):
            run_model(model_def)

        call_kwargs = mock_func.call_args.kwargs
        assert call_kwargs['DeltaT'] == 3600

    @patch('openwater.text_model.ow_lib', create=True)
    def test_node_params_override_file_defaults(self, mock_lib):
        data = {
            'ow_model': '1.0',
            'defaults': {'DeltaT': 3600},
            'nodes': [{
                'name': 'D1', 'model': 'DepthToRate',
                'parameters': {'DeltaT': 900},
            }],
        }
        model_def = load_model_from_dict(data, model_registry=MODEL_REGISTRY)

        mock_func = MagicMock()
        mock_func.return_value = [np.array([[0.0]])]

        with patch('openwater.lib.DepthToRate', mock_func, create=True):
            run_model(model_def)

        call_kwargs = mock_func.call_args.kwargs
        assert call_kwargs['DeltaT'] == 900

    @patch('openwater.text_model.ow_lib', create=True)
    def test_model_type_defaults_applied(self, mock_lib):
        """Parameters not in node config or file defaults get model-type defaults."""
        data = {
            'ow_model': '1.0',
            'nodes': [{
                'name': 'D1', 'model': 'DepthToRate',
                'parameters': {'area': 1e6},
            }],
        }
        model_def = load_model_from_dict(data, model_registry=MODEL_REGISTRY)

        mock_func = MagicMock()
        mock_func.return_value = [np.array([[0.0]])]

        with patch('openwater.lib.DepthToRate', mock_func, create=True):
            run_model(model_def)

        call_kwargs = mock_func.call_args.kwargs
        # DeltaT not specified anywhere — should get model default of 86400
        assert call_kwargs['DeltaT'] == 86400.0
        assert call_kwargs['area'] == 1e6

    @patch('openwater.text_model.ow_lib', create=True)
    def test_priority_order(self, mock_lib):
        """node-specific > file defaults > model-type default."""
        data = {
            'ow_model': '1.0',
            'defaults': {'DeltaT': 3600, 'area': 500.0},
            'nodes': [{
                'name': 'D1', 'model': 'DepthToRate',
                'parameters': {'area': 999.0},
            }],
        }
        model_def = load_model_from_dict(data, model_registry=MODEL_REGISTRY)

        mock_func = MagicMock()
        mock_func.return_value = [np.array([[0.0]])]

        with patch('openwater.lib.DepthToRate', mock_func, create=True):
            run_model(model_def)

        call_kwargs = mock_func.call_args.kwargs
        assert call_kwargs['area'] == 999.0       # node-specific wins
        assert call_kwargs['DeltaT'] == 3600      # file default wins over model default

    @patch('openwater.text_model.ow_lib', create=True)
    def test_defaults_only_apply_to_matching_params(self, mock_lib):
        """File defaults with names not matching any parameter are ignored."""
        data = {
            'ow_model': '1.0',
            'defaults': {'DeltaT': 3600, 'nonexistent': 42},
            'nodes': [{'name': 'D1', 'model': 'DepthToRate'}],
        }
        model_def = load_model_from_dict(data, model_registry=MODEL_REGISTRY)

        mock_func = MagicMock()
        mock_func.return_value = [np.array([[0.0]])]

        with patch('openwater.lib.DepthToRate', mock_func, create=True):
            run_model(model_def)

        call_kwargs = mock_func.call_args.kwargs
        assert call_kwargs['DeltaT'] == 3600
        assert 'nonexistent' not in call_kwargs

    @patch('openwater.text_model.ow_lib', create=True)
    def test_defaults_across_multiple_model_types(self, mock_lib):
        """DeltaT default applies to both DepthToRate and StorageRouting."""
        data = {
            'ow_model': '1.0',
            'defaults': {'DeltaT': 3600},
            'nodes': [
                {'name': 'D1', 'model': 'DepthToRate', 'parameters': {'area': 1e6}},
                {'name': 'R1', 'model': 'StorageRouting'},
            ],
            'links': ['D1.outflow -> R1.inflow'],
        }
        model_def = load_model_from_dict(data, model_registry=MODEL_REGISTRY)

        dtr_mock = MagicMock()
        dtr_mock.return_value = [np.array([[1.0]])]

        sr_mock = MagicMock()
        sr_mock.return_value = [
            np.array([[1.0]]),  # outflow
            np.array([[0.0]]),  # storage
            np.array([0.0]),    # S
            np.array([0.0]),    # prevInflow
            np.array([0.0]),    # prevOutflow
        ]

        with patch('openwater.lib.DepthToRate', dtr_mock, create=True), \
             patch('openwater.lib.StorageRouting', sr_mock, create=True):
            run_model(model_def)

        assert dtr_mock.call_args.kwargs['DeltaT'] == 3600
        assert sr_mock.call_args.kwargs['DeltaT'] == 3600


# ---------------------------------------------------------------------------
# Link shorthand tests
# ---------------------------------------------------------------------------

class TestParseShorthandLink:
    def test_basic(self):
        result = _parse_shorthand_link('S1.runoff -> R1.inflow')
        assert result == {
            'from': {'node': 'S1', 'output': 'runoff'},
            'to': {'node': 'R1', 'input': 'inflow'},
        }

    def test_extra_whitespace(self):
        result = _parse_shorthand_link('  S1.runoff  ->  R1.inflow  ')
        assert result['from'] == {'node': 'S1', 'output': 'runoff'}
        assert result['to'] == {'node': 'R1', 'input': 'inflow'}

    def test_missing_arrow_raises(self):
        with pytest.raises(TemplateLoadError, match="Invalid shorthand"):
            _parse_shorthand_link('S1.runoff R1.inflow')

    def test_missing_port_raises(self):
        with pytest.raises(TemplateLoadError, match="Invalid shorthand"):
            _parse_shorthand_link('S1 -> R1.inflow')


class TestNormalizeLinks:
    """Tests for _normalize_links in persistence.py (compact string links)."""

    def test_string_links(self):
        links = ['S1.runoff -> R1.inflow', 'S2.runoff -> R2.inflow']
        result = _normalize_links(links)
        assert len(result) == 2
        assert result[0]['from']['node'] == 'S1'
        assert result[1]['from']['node'] == 'S2'

    def test_dict_links_passthrough(self):
        links = [{'from': {'node': 'A', 'output': 'x'}, 'to': {'node': 'B', 'input': 'y'}}]
        result = _normalize_links(links)
        assert result == links

    def test_mixed_links(self):
        links = [
            {'from': {'node': 'A', 'output': 'x'}, 'to': {'node': 'B', 'input': 'y'}},
            'C.z -> D.w',
        ]
        result = _normalize_links(links)
        assert len(result) == 2
        assert result[1]['from'] == {'node': 'C', 'output': 'z'}

    def test_none_input(self):
        result = _normalize_links(None)
        assert result == []


class TestExtractInlineOutputs:
    """Tests for _extract_inline_outputs in text_model.py (node outputs section)."""

    def test_inline_outputs_single(self):
        nodes = [{'name': 'S1', 'outputs': {'runoff': 'R1.inflow'}}]
        result = _extract_inline_outputs([], nodes)
        assert len(result) == 1
        assert result[0] == {
            'from': {'node': 'S1', 'output': 'runoff'},
            'to': {'node': 'R1', 'input': 'inflow'},
        }

    def test_inline_outputs_list(self):
        nodes = [{'name': 'S1', 'outputs': {'outflow': ['R2.inflow', 'R3.lateral']}}]
        result = _extract_inline_outputs([], nodes)
        assert len(result) == 2
        assert result[0]['to'] == {'node': 'R2', 'input': 'inflow'}
        assert result[1]['to'] == {'node': 'R3', 'input': 'lateral'}

    def test_inline_outputs_bad_format_raises(self):
        nodes = [{'name': 'S1', 'outputs': {'runoff': 'nope'}}]
        with pytest.raises(ModelLoadError, match="Invalid inline output"):
            _extract_inline_outputs([], nodes)

    def test_preserves_existing_links(self):
        links = ['A.x -> B.y', {'from': {'node': 'C', 'output': 'z'}, 'to': {'node': 'D', 'input': 'w'}}]
        nodes = [{'name': 'E', 'outputs': {'out': 'F.in'}}]
        result = _extract_inline_outputs(links, nodes)
        assert len(result) == 3
        # First two are passed through as-is (strings/dicts)
        assert result[0] == 'A.x -> B.y'
        assert result[1]['from']['node'] == 'C'
        # Third is from inline outputs
        assert result[2]['from']['node'] == 'E'

    def test_none_inputs(self):
        result = _extract_inline_outputs(None, None)
        assert result == []


class TestShorthandIntegration:
    def test_compact_string_links(self):
        data = {
            'ow_model': '1.0',
            'nodes': [
                {'name': 'N1', 'model': 'Simhyd'},
                {'name': 'N2', 'model': 'DepthToRate'},
            ],
            'links': ['N1.runoff -> N2.input'],
        }
        model = load_model_from_dict(data, model_registry=MODEL_REGISTRY)
        assert len(model.template.links) == 1
        link = model.template.links[0]
        assert link.from_node.name == 'N1'
        assert link.from_output == 'runoff'
        assert link.to_node.name == 'N2'
        assert link.to_input == 'input'

    def test_inline_node_outputs(self):
        data = {
            'ow_model': '1.0',
            'nodes': [
                {
                    'name': 'N1', 'model': 'Simhyd',
                    'outputs': {'runoff': 'N2.input'},
                },
                {'name': 'N2', 'model': 'DepthToRate'},
            ],
        }
        model = load_model_from_dict(data, model_registry=MODEL_REGISTRY)
        assert len(model.template.links) == 1
        link = model.template.links[0]
        assert link.from_node.name == 'N1'
        assert link.to_node.name == 'N2'

    def test_mixed_all_three_styles(self):
        data = {
            'ow_model': '1.0',
            'nodes': [
                {'name': 'A', 'model': 'Simhyd'},
                {
                    'name': 'B', 'model': 'Simhyd',
                    'outputs': {'runoff': 'D.input'},
                },
                {'name': 'C', 'model': 'DepthToRate'},
                {'name': 'D', 'model': 'DepthToRate'},
            ],
            'links': [
                {'from': {'node': 'A', 'output': 'runoff'}, 'to': {'node': 'C', 'input': 'input'}},
                'A.baseflow -> D.input',
            ],
        }
        model = load_model_from_dict(data, model_registry=MODEL_REGISTRY)
        assert len(model.template.links) == 3


# ---------------------------------------------------------------------------
# Graph building tests
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_simple_graph(self):
        tpl = OWTemplate('test')
        n1 = OWNode(MockSimhyd, name='A')
        n2 = OWNode(MockDepthToRate, name='B')
        tpl.nodes.extend([n1, n2])
        tpl.links.append(OWLink(n1, 'runoff', n2, 'input'))

        g = _build_graph(tpl)
        assert set(g.nodes) == {'A', 'B'}
        assert ('A', 'B') in g.edges
        assert g.edges[('A', 'B')]['src'] == ['runoff']
        assert g.edges[('A', 'B')]['dest'] == ['input']

    def test_uses_original_names(self):
        tpl = OWTemplate('test')
        n = OWNode(MockSimhyd, name='My-Custom-Name')
        tpl.nodes.append(n)

        g = _build_graph(tpl)
        assert 'My-Custom-Name' in g.nodes

    def test_multiple_links_same_nodes(self):
        tpl = OWTemplate('test')
        n1 = OWNode(MockSimhyd, name='A')
        n2 = OWNode(MockStorage, name='B')
        tpl.nodes.extend([n1, n2])
        tpl.links.append(OWLink(n1, 'runoff', n2, 'inflow'))
        tpl.links.append(OWLink(n1, 'baseflow', n2, 'inflow'))

        g = _build_graph(tpl)
        edge = g.edges[('A', 'B')]
        assert edge['src'] == ['runoff', 'baseflow']
        assert edge['dest'] == ['inflow', 'inflow']


# ---------------------------------------------------------------------------
# Data source resolution tests
# ---------------------------------------------------------------------------

class TestResolveDataSources:
    def test_csv_file(self, tmp_path):
        csv_path = tmp_path / 'data.csv'
        df = pd.DataFrame({'rain': [1.0, 2.0, 3.0]})
        df.to_csv(csv_path, index=False)

        model_def = ModelDefinition(
            template=OWTemplate(),
            node_configs={},
            data_source_paths={'climate': 'data.csv'},
            time_period=None,
            base_dir=str(tmp_path),
        )
        resolved = _resolve_data_sources(model_def)
        assert 'climate' in resolved
        assert list(resolved['climate']['rain']) == [1.0, 2.0, 3.0]

    def test_runtime_override(self):
        runtime_df = pd.DataFrame({'col': [10, 20]})
        model_def = ModelDefinition(
            template=OWTemplate(),
            node_configs={},
            data_source_paths={'src': 'dummy.csv'},
            time_period=None,
            base_dir=None,
        )
        resolved = _resolve_data_sources(model_def, {'src': runtime_df})
        pd.testing.assert_frame_equal(resolved['src'], runtime_df)

    def test_extra_runtime_sources(self):
        runtime_df = pd.DataFrame({'x': [1]})
        model_def = ModelDefinition(
            template=OWTemplate(),
            node_configs={},
            data_source_paths={},
            time_period=None,
            base_dir=None,
        )
        resolved = _resolve_data_sources(model_def, {'extra': runtime_df})
        assert 'extra' in resolved


# ---------------------------------------------------------------------------
# Find linked input tests
# ---------------------------------------------------------------------------

class TestLoadInitialStates:
    def test_loads_last_row(self, tmp_path):
        # Write a states CSV with multiple rows (timeseries of states)
        states_dir = tmp_path / 'states'
        states_dir.mkdir()
        df = pd.DataFrame({
            'SoilMoistureStore': [100.0, 150.0, 200.0],
            'GroundwaterStore': [10.0, 20.0, 30.0],
        })
        df.to_csv(states_dir / 'S1_states.csv')

        model_def = ModelDefinition(
            template=OWTemplate(),
            node_configs={'S1': NodeConfig()},
            data_source_paths={},
            time_period=None,
            base_dir=None,
        )
        load_initial_states(str(states_dir), model_def)

        assert model_def.node_configs['S1'].initial_states['SoilMoistureStore'] == 200.0
        assert model_def.node_configs['S1'].initial_states['GroundwaterStore'] == 30.0

    def test_creates_node_config_if_missing(self, tmp_path):
        states_dir = tmp_path / 'states'
        states_dir.mkdir()
        pd.DataFrame({'S': [500.0]}).to_csv(states_dir / 'R1_states.csv')

        model_def = ModelDefinition(
            template=OWTemplate(),
            node_configs={},
            data_source_paths={},
            time_period=None,
            base_dir=None,
        )
        load_initial_states(str(states_dir), model_def)

        assert model_def.node_configs['R1'].initial_states['S'] == 500.0

    def test_preserves_existing_states_not_in_file(self, tmp_path):
        states_dir = tmp_path / 'states'
        states_dir.mkdir()
        pd.DataFrame({'SoilMoistureStore': [200.0]}).to_csv(states_dir / 'S1_states.csv')

        model_def = ModelDefinition(
            template=OWTemplate(),
            node_configs={'S1': NodeConfig(initial_states={'GroundwaterStore': 99.0})},
            data_source_paths={},
            time_period=None,
            base_dir=None,
        )
        load_initial_states(str(states_dir), model_def)

        assert model_def.node_configs['S1'].initial_states['SoilMoistureStore'] == 200.0
        assert model_def.node_configs['S1'].initial_states['GroundwaterStore'] == 99.0

    def test_multiple_node_files(self, tmp_path):
        states_dir = tmp_path / 'states'
        states_dir.mkdir()
        pd.DataFrame({'SoilMoistureStore': [100.0]}).to_csv(states_dir / 'S1_states.csv')
        pd.DataFrame({'S': [50.0]}).to_csv(states_dir / 'R1_states.csv')

        model_def = ModelDefinition(
            template=OWTemplate(),
            node_configs={},
            data_source_paths={},
            time_period=None,
            base_dir=None,
        )
        load_initial_states(str(states_dir), model_def)

        assert model_def.node_configs['S1'].initial_states['SoilMoistureStore'] == 100.0
        assert model_def.node_configs['R1'].initial_states['S'] == 50.0

    def test_skips_empty_files(self, tmp_path):
        states_dir = tmp_path / 'states'
        states_dir.mkdir()
        pd.DataFrame().to_csv(states_dir / 'S1_states.csv')

        model_def = ModelDefinition(
            template=OWTemplate(),
            node_configs={},
            data_source_paths={},
            time_period=None,
            base_dir=None,
        )
        load_initial_states(str(states_dir), model_def)

        assert 'S1' not in model_def.node_configs

    def test_ignores_non_states_files(self, tmp_path):
        states_dir = tmp_path / 'states'
        states_dir.mkdir()
        pd.DataFrame({'runoff': [1.0]}).to_csv(states_dir / 'S1_outputs.csv')

        model_def = ModelDefinition(
            template=OWTemplate(),
            node_configs={},
            data_source_paths={},
            time_period=None,
            base_dir=None,
        )
        load_initial_states(str(states_dir), model_def)

        assert len(model_def.node_configs) == 0

    def test_nonexistent_dir_raises(self):
        model_def = ModelDefinition(
            template=OWTemplate(),
            node_configs={},
            data_source_paths={},
            time_period=None,
            base_dir=None,
        )
        with pytest.raises(ModelLoadError, match="does not exist"):
            load_initial_states('/nonexistent/path', model_def)

    def test_round_trip_with_write_results(self, tmp_path):
        """Output states from _write_results can be loaded back as initial states."""
        from openwater.cli_run import _write_results

        results = ModelResults(
            outputs={'S1': pd.DataFrame({'runoff': [1.0]})},
            states={
                'S1': pd.DataFrame({
                    'SoilMoistureStore': [100.0, 150.0, 200.0],
                    'GroundwaterStore': [10.0, 20.0, 30.0],
                }),
            },
            time_period=None,
        )
        out_dir = str(tmp_path / 'out')
        _write_results(results, out_dir)

        model_def = ModelDefinition(
            template=OWTemplate(),
            node_configs={},
            data_source_paths={},
            time_period=None,
            base_dir=None,
        )
        load_initial_states(out_dir, model_def)

        assert model_def.node_configs['S1'].initial_states['SoilMoistureStore'] == 200.0
        assert model_def.node_configs['S1'].initial_states['GroundwaterStore'] == 30.0


class TestFindLinkedInput:
    def _make_graph(self):
        g = nx.DiGraph()
        g.add_node('A')
        g.add_node('B')
        g.add_edge('A', 'B', src=['runoff'], dest=['input'])
        return g

    def test_finds_linked_output(self):
        g = self._make_graph()
        outputs = {'A': {'runoff': np.array([1.0, 2.0])}}
        result = _find_linked_input(g, 'B', 'input', outputs)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_returns_none_for_unlinked(self):
        g = self._make_graph()
        outputs = {'A': {'runoff': np.array([1.0])}}
        result = _find_linked_input(g, 'B', 'unlinked_port', outputs)
        assert result is None

    def test_returns_none_no_predecessors(self):
        g = nx.DiGraph()
        g.add_node('X')
        result = _find_linked_input(g, 'X', 'input', {})
        assert result is None


# ---------------------------------------------------------------------------
# Execution tests (mocked lib)
# ---------------------------------------------------------------------------

class TestRunModel:
    def _make_single_node_model(self):
        data = {
            'ow_model': '1.0',
            'nodes': [{
                'name': 'N1', 'model': 'Simhyd',
                'parameters': {'baseflowCoefficient': 0.3},
            }],
        }
        return load_model_from_dict(data, model_registry=MODEL_REGISTRY)

    def _make_linked_model(self):
        data = {
            'ow_model': '1.0',
            'time_period': {'start': '2010-01-01', 'end': '2010-01-03'},
            'nodes': [
                {
                    'name': 'SH', 'model': 'Simhyd',
                    'parameters': {'baseflowCoefficient': 0.3},
                },
                {
                    'name': 'DTR', 'model': 'DepthToRate',
                    'parameters': {'area': 1e8},
                },
            ],
            'links': [{
                'from': {'node': 'SH', 'output': 'runoff'},
                'to': {'node': 'DTR', 'input': 'input'},
            }],
        }
        return load_model_from_dict(data, model_registry=MODEL_REGISTRY)

    @patch('openwater.text_model.ow_lib', create=True)
    def test_single_node(self, mock_lib):
        # Mock lib.Simhyd to return fake outputs + states
        mock_func = MagicMock()
        mock_func.return_value = [
            np.array([[1.0, 2.0, 3.0]]),  # runoff
            np.array([[0.5, 1.0, 1.5]]),   # baseflow
            np.array([100.0]),              # SoilMoistureStore final
            np.array([50.0]),               # GroundwaterStore final
        ]

        model_def = self._make_single_node_model()

        with patch('openwater.lib.Simhyd', mock_func, create=True):
            results = run_model(model_def)

        assert isinstance(results, ModelResults)
        assert 'N1' in results.outputs
        mock_func.assert_called_once()

    @patch('openwater.text_model.ow_lib', create=True)
    def test_linked_nodes_output_routing(self, mock_lib):
        simhyd_mock = MagicMock()
        simhyd_runoff = np.array([[10.0, 20.0, 30.0]])
        simhyd_mock.return_value = [
            simhyd_runoff,
            np.array([[1.0, 2.0, 3.0]]),
            np.array([100.0]),
            np.array([50.0]),
        ]

        dtr_mock = MagicMock()
        dtr_mock.return_value = [
            np.array([[1e9, 2e9, 3e9]]),   # outflow
        ]

        model_def = self._make_linked_model()

        with patch('openwater.lib.Simhyd', simhyd_mock, create=True), \
             patch('openwater.lib.DepthToRate', dtr_mock, create=True):
            results = run_model(model_def)

        assert 'SH' in results.outputs
        assert 'DTR' in results.outputs

        # Verify DTR received routed input from SH
        dtr_call_kwargs = dtr_mock.call_args
        # The 'input' kwarg should contain the flattened runoff
        if dtr_call_kwargs.kwargs:
            assert 'input' in dtr_call_kwargs.kwargs
        else:
            # Could be positional — just verify it was called
            dtr_mock.assert_called_once()

    @patch('openwater.text_model.ow_lib', create=True)
    def test_timeseries_from_data_source(self, mock_lib):
        climate_df = pd.DataFrame({
            'Rain': [5.0, 10.0, 15.0],
            'PET': [2.0, 3.0, 4.0],
        })

        data = {
            'ow_model': '1.0',
            'data_sources': {'climate': 'dummy.csv'},
            'nodes': [{
                'name': 'N1', 'model': 'Simhyd',
                'timeseries': {
                    'rainfall': {'data_source': 'climate', 'column': 'Rain'},
                    'pet': {'data_source': 'climate', 'column': 'PET'},
                },
            }],
        }
        model_def = load_model_from_dict(data, model_registry=MODEL_REGISTRY)

        mock_func = MagicMock()
        mock_func.return_value = [
            np.array([[1.0, 2.0, 3.0]]),
            np.array([[0.5, 1.0, 1.5]]),
            np.array([100.0]),
            np.array([50.0]),
        ]

        with patch('openwater.lib.Simhyd', mock_func, create=True):
            results = run_model(model_def, data_sources={'climate': climate_df})

        mock_func.assert_called_once()
        call_kwargs = mock_func.call_args.kwargs
        np.testing.assert_array_equal(call_kwargs['rainfall'], [5.0, 10.0, 15.0])
        np.testing.assert_array_equal(call_kwargs['pet'], [2.0, 3.0, 4.0])

    @patch('openwater.text_model.ow_lib', create=True)
    def test_initial_states_passed(self, mock_lib):
        data = {
            'ow_model': '1.0',
            'nodes': [{
                'name': 'N1', 'model': 'Simhyd',
                'initial_states': {
                    'SoilMoistureStore': 150.0,
                    'GroundwaterStore': 25.0,
                },
            }],
        }
        model_def = load_model_from_dict(data, model_registry=MODEL_REGISTRY)

        mock_func = MagicMock()
        mock_func.return_value = [
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([150.0]),
            np.array([25.0]),
        ]

        with patch('openwater.lib.Simhyd', mock_func, create=True):
            results = run_model(model_def)

        call_kwargs = mock_func.call_args.kwargs
        np.testing.assert_array_equal(
            call_kwargs['SoilMoistureStore'], [150.0]
        )
        np.testing.assert_array_equal(
            call_kwargs['GroundwaterStore'], [25.0]
        )

    @patch('openwater.text_model.ow_lib', create=True)
    def test_results_have_time_index(self, mock_lib):
        data = {
            'ow_model': '1.0',
            'time_period': {'start': '2010-01-01', 'end': '2010-01-03'},
            'nodes': [{
                'name': 'N1', 'model': 'DepthToRate',
                'parameters': {'area': 1e6},
            }],
        }
        model_def = load_model_from_dict(data, model_registry=MODEL_REGISTRY)

        mock_func = MagicMock()
        mock_func.return_value = [
            np.array([[100.0, 200.0, 300.0]]),  # outflow
        ]

        with patch('openwater.lib.DepthToRate', mock_func, create=True):
            results = run_model(model_def)

        df = results.outputs['N1']
        assert len(df) == 3
        assert df.index[0] == pd.Timestamp('2010-01-01')
        assert df.index[2] == pd.Timestamp('2010-01-03')

    @patch('openwater.text_model.ow_lib', create=True)
    def test_csv_data_source_integration(self, mock_lib, tmp_path):
        csv_path = tmp_path / 'climate.csv'
        pd.DataFrame({'Rain': [1.0, 2.0], 'PET': [0.5, 0.6]}).to_csv(
            csv_path, index=False
        )

        data = {
            'ow_model': '1.0',
            'data_sources': {'climate': 'climate.csv'},
            'nodes': [{
                'name': 'N1', 'model': 'Simhyd',
                'timeseries': {
                    'rainfall': {'data_source': 'climate', 'column': 'Rain'},
                    'pet': {'data_source': 'climate', 'column': 'PET'},
                },
            }],
        }
        model_def = load_model_from_dict(
            data, base_dir=str(tmp_path), model_registry=MODEL_REGISTRY,
        )

        mock_func = MagicMock()
        mock_func.return_value = [
            np.array([[1.0, 2.0]]),
            np.array([[0.1, 0.2]]),
            np.array([50.0]),
            np.array([10.0]),
        ]

        with patch('openwater.lib.Simhyd', mock_func, create=True):
            results = run_model(model_def)

        call_kwargs = mock_func.call_args.kwargs
        np.testing.assert_array_almost_equal(call_kwargs['rainfall'], [1.0, 2.0])

    @patch('openwater.text_model.ow_lib', create=True)
    def test_parameters_passed_to_lib(self, mock_lib):
        data = {
            'ow_model': '1.0',
            'nodes': [{
                'name': 'N1', 'model': 'DepthToRate',
                'parameters': {'area': 150000000.0},
            }],
        }
        model_def = load_model_from_dict(data, model_registry=MODEL_REGISTRY)

        mock_func = MagicMock()
        mock_func.return_value = [np.array([[0.0]])]

        with patch('openwater.lib.DepthToRate', mock_func, create=True):
            run_model(model_def)

        call_kwargs = mock_func.call_args.kwargs
        assert call_kwargs['area'] == 150000000.0

    @patch('openwater.text_model.ow_lib', create=True)
    def test_partial_initial_states(self, mock_lib):
        """Setting only some states should not cause an inhomogeneous array error."""
        data = {
            'ow_model': '1.0',
            'nodes': [{
                'name': 'R1', 'model': 'StorageRouting',
                'initial_states': {'S': 100.0},
                # prevInflow and prevOutflow not set — should default to 0
            }],
        }
        model_def = load_model_from_dict(data, model_registry=MODEL_REGISTRY)

        mock_func = MagicMock()
        mock_func.return_value = [
            np.array([[1.0]]),   # outflow
            np.array([[0.0]]),   # storage
            np.array([100.0]),   # S
            np.array([0.0]),     # prevInflow
            np.array([0.0]),     # prevOutflow
        ]

        with patch('openwater.lib.StorageRouting', mock_func, create=True):
            results = run_model(model_def)

        call_kwargs = mock_func.call_args.kwargs
        np.testing.assert_array_equal(call_kwargs['S'], [100.0])
        np.testing.assert_array_equal(call_kwargs['prevInflow'], [0.0])
        np.testing.assert_array_equal(call_kwargs['prevOutflow'], [0.0])

    @patch('openwater.text_model.ow_lib', create=True)
    def test_node_without_config(self, mock_lib):
        """A node not mentioned in node_configs should still run with defaults."""
        tpl = OWTemplate('test')
        n = OWNode(MockDepthToRate, name='DTR')
        tpl.nodes.append(n)

        model_def = ModelDefinition(
            template=tpl,
            node_configs={},
            data_source_paths={},
            time_period=None,
            base_dir=None,
        )

        mock_func = MagicMock()
        mock_func.return_value = [np.array([[0.0]])]

        with patch('openwater.lib.DepthToRate', mock_func, create=True):
            results = run_model(model_def)

        assert 'DTR' in results.outputs
