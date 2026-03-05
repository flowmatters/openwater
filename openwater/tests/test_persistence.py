"""Tests for YAML persistence of OpenWater templates."""
import os
import tempfile

import pytest

from openwater.template import OWTemplate, OWNode, OWLink
from openwater.persistence import (
    template_to_dict,
    template_to_yaml,
    template_to_yaml_string,
    dict_to_template,
    template_from_yaml,
    template_from_yaml_string,
    TemplateLoadError,
    SCHEMA_VERSION,
)


# ---------------------------------------------------------------------------
# Mock model types — avoids needing ow-inspect / compiled binaries
# ---------------------------------------------------------------------------

class _MockModelType:
    """Minimal model type object matching what OWNode.set_model_type expects."""
    def __init__(self, name, inputs=None, outputs=None):
        self.name = name
        self.description = {
            'Inputs': inputs or [],
            'Outputs': outputs or [],
            'States': [],
            'Parameters': [],
        }


MockSimhyd = _MockModelType(
    'Simhyd',
    inputs=['rainfall', 'pet'],
    outputs=['runoff', 'baseflow'],
)

MockDepthToRate = _MockModelType(
    'DepthToRate',
    inputs=['input'],
    outputs=['outflow'],
)

MockRouting = _MockModelType(
    'Routing',
    inputs=['inflow', 'lateral'],
    outputs=['outflow'],
)

MODEL_REGISTRY = {
    'Simhyd': MockSimhyd,
    'DepthToRate': MockDepthToRate,
    'Routing': MockRouting,
}


# ---------------------------------------------------------------------------
# Helpers for building test templates
# ---------------------------------------------------------------------------

def _make_simple_template():
    """Create a simple template with 2 nodes, 1 link, 1 input, 1 output."""
    tpl = OWTemplate('Simple HRU')
    n1 = OWNode(MockSimhyd, name='Simhyd-RR-ForestHRU', _process='RR', hru='ForestHRU')
    n2 = OWNode(MockDepthToRate, name='DepthToRate-Scale', _process='Scaling')
    tpl.nodes.append(n1)
    tpl.nodes.append(n2)
    tpl.links.append(OWLink(n1, 'runoff', n2, 'input'))
    tpl.inputs.append((n1, 'rainfall', 'rainfall', {}))
    tpl.outputs.append((n2, 'outflow', 'runoff', {}))
    return tpl


def _make_nested_template():
    """Create a template with a nested child."""
    parent = OWTemplate('Parent')
    n_route = OWNode(MockRouting, name='Router', _process='Routing')
    parent.nodes.append(n_route)
    parent.inputs.append((n_route, 'inflow', 'inflow', {}))
    parent.outputs.append((n_route, 'outflow', 'outflow', {}))

    child = _make_simple_template()
    parent.nested.append(child)
    return parent


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestSerializationEmptyTemplate:
    def test_empty_template(self):
        tpl = OWTemplate()
        data = template_to_dict(tpl)
        assert data['ow_template'] == SCHEMA_VERSION
        assert 'nodes' not in data
        assert 'links' not in data
        assert 'nested' not in data
        assert 'inputs' not in data
        assert 'outputs' not in data

    def test_empty_template_with_label(self):
        tpl = OWTemplate('Test Label')
        data = template_to_dict(tpl)
        assert data['label'] == 'Test Label'


class TestSerializationNodes:
    def test_node_basic(self):
        tpl = OWTemplate('test')
        n = OWNode(MockSimhyd, name='MyNode')
        tpl.nodes.append(n)
        data = template_to_dict(tpl)
        assert len(data['nodes']) == 1
        node_data = data['nodes'][0]
        assert node_data['name'] == 'MyNode'
        assert node_data['model'] == 'Simhyd'

    def test_node_tags_serialization(self):
        tpl = OWTemplate('test')
        n = OWNode(MockSimhyd, name='N1', _process='RR', hru='Forest')
        tpl.nodes.append(n)
        data = template_to_dict(tpl)
        tags = data['nodes'][0]['tags']
        # _process renamed to process
        assert tags['process'] == 'RR'
        assert tags['hru'] == 'Forest'
        # _model should be stripped (redundant with model field)
        assert '_model' not in tags
        assert 'model' not in tags  # model is a separate top-level field

    def test_runtime_tags_stripped(self):
        tpl = OWTemplate('test')
        n = OWNode(MockSimhyd, name='N1')
        # Simulate runtime tags being set
        n.tags['_run_idx'] = 5
        n.tags['_generation'] = 2
        tpl.nodes.append(n)
        data = template_to_dict(tpl)
        tags = data['nodes'][0].get('tags', {})
        assert '_run_idx' not in tags
        assert '_generation' not in tags
        assert '_model' not in tags

    def test_node_metadata(self):
        tpl = OWTemplate('test')
        n = OWNode(MockSimhyd, name='N1')
        tpl.nodes.append(n)
        node_meta = {'N1': {'position': {'x': 100, 'y': 50}}}
        data = template_to_dict(tpl, node_metadata=node_meta)
        assert data['nodes'][0]['metadata'] == {'position': {'x': 100, 'y': 50}}


class TestSerializationLinks:
    def test_link(self):
        tpl = _make_simple_template()
        data = template_to_dict(tpl)
        assert len(data['links']) == 1
        link = data['links'][0]
        assert link['from']['node'] == 'Simhyd-RR-ForestHRU'
        assert link['from']['output'] == 'runoff'
        assert link['to']['node'] == 'DepthToRate-Scale'
        assert link['to']['input'] == 'input'


class TestSerializationPortBindings:
    def test_inputs(self):
        tpl = _make_simple_template()
        data = template_to_dict(tpl)
        assert len(data['inputs']) == 1
        inp = data['inputs'][0]
        assert inp['alias'] == 'rainfall'
        assert inp['node'] == 'Simhyd-RR-ForestHRU'
        assert inp['port'] == 'rainfall'

    def test_outputs(self):
        tpl = _make_simple_template()
        data = template_to_dict(tpl)
        assert len(data['outputs']) == 1
        out = data['outputs'][0]
        assert out['alias'] == 'runoff'
        assert out['node'] == 'DepthToRate-Scale'
        assert out['port'] == 'outflow'

    def test_port_binding_with_tags(self):
        tpl = OWTemplate('test')
        n = OWNode(MockSimhyd, name='N1', _process='RR')
        tpl.nodes.append(n)
        tpl.inputs.append((n, 'rainfall', 'rain', {'_process': 'RR', 'zone': 'coastal'}))
        data = template_to_dict(tpl)
        tags = data['inputs'][0]['tags']
        assert tags['process'] == 'RR'
        assert tags['zone'] == 'coastal'


class TestSerializationNested:
    def test_nested_inline(self):
        tpl = _make_nested_template()
        data = template_to_dict(tpl)
        assert len(data['nested']) == 1
        nested_entry = data['nested'][0]
        assert 'inline' in nested_entry
        inner = nested_entry['inline']
        assert inner['label'] == 'Simple HRU'
        assert len(inner['nodes']) == 2


class TestSerializationMetadata:
    def test_template_metadata(self):
        tpl = OWTemplate('test')
        data = template_to_dict(tpl, template_metadata={'description': 'A test'})
        assert data['metadata'] == {'description': 'A test'}


# ---------------------------------------------------------------------------
# Deserialization tests
# ---------------------------------------------------------------------------

class TestDeserializationBasic:
    def test_empty_template(self):
        data = {'ow_template': '1.0'}
        tpl, node_meta, tpl_meta = dict_to_template(data, model_registry=MODEL_REGISTRY)
        assert isinstance(tpl, OWTemplate)
        assert tpl.label == ''
        assert tpl.nodes == []
        assert tpl.links == []
        assert tpl.nested == []
        assert tpl.inputs == []
        assert tpl.outputs == []

    def test_label(self):
        data = {'ow_template': '1.0', 'label': 'My Template'}
        tpl, _, _ = dict_to_template(data, model_registry=MODEL_REGISTRY)
        assert tpl.label == 'My Template'

    def test_template_metadata_returned(self):
        data = {'ow_template': '1.0', 'metadata': {'desc': 'test'}}
        _, _, tpl_meta = dict_to_template(data, model_registry=MODEL_REGISTRY)
        assert tpl_meta == {'desc': 'test'}


class TestDeserializationNodes:
    def test_basic_node(self):
        data = {
            'ow_template': '1.0',
            'nodes': [{'name': 'N1', 'model': 'Simhyd'}],
        }
        tpl, _, _ = dict_to_template(data, model_registry=MODEL_REGISTRY)
        assert len(tpl.nodes) == 1
        assert tpl.nodes[0].name == 'N1'
        assert tpl.nodes[0].model_name == 'Simhyd'

    def test_node_with_tags(self):
        data = {
            'ow_template': '1.0',
            'nodes': [{
                'name': 'N1', 'model': 'Simhyd',
                'tags': {'process': 'RR', 'hru': 'Forest'},
            }],
        }
        tpl, _, _ = dict_to_template(data, model_registry=MODEL_REGISTRY)
        node = tpl.nodes[0]
        assert node.tags['_process'] == 'RR'
        assert node.tags['hru'] == 'Forest'

    def test_node_metadata(self):
        data = {
            'ow_template': '1.0',
            'nodes': [{
                'name': 'N1', 'model': 'Simhyd',
                'metadata': {'position': {'x': 10, 'y': 20}},
            }],
        }
        _, node_meta, _ = dict_to_template(data, model_registry=MODEL_REGISTRY)
        assert node_meta['N1'] == {'position': {'x': 10, 'y': 20}}


class TestDeserializationLinks:
    def test_link(self):
        data = {
            'ow_template': '1.0',
            'nodes': [
                {'name': 'N1', 'model': 'Simhyd'},
                {'name': 'N2', 'model': 'DepthToRate'},
            ],
            'links': [{
                'from': {'node': 'N1', 'output': 'runoff'},
                'to': {'node': 'N2', 'input': 'input'},
            }],
        }
        tpl, _, _ = dict_to_template(data, model_registry=MODEL_REGISTRY)
        assert len(tpl.links) == 1
        link = tpl.links[0]
        assert link.from_node.name == 'N1'
        assert link.from_output == 'runoff'
        assert link.to_node.name == 'N2'
        assert link.to_input == 'input'


class TestDeserializationCompactLinks:
    def test_compact_string_link(self):
        data = {
            'ow_template': '1.0',
            'nodes': [
                {'name': 'N1', 'model': 'Simhyd'},
                {'name': 'N2', 'model': 'DepthToRate'},
            ],
            'links': ['N1.runoff -> N2.input'],
        }
        tpl, _, _ = dict_to_template(data, model_registry=MODEL_REGISTRY)
        assert len(tpl.links) == 1
        link = tpl.links[0]
        assert link.from_node.name == 'N1'
        assert link.from_output == 'runoff'
        assert link.to_node.name == 'N2'
        assert link.to_input == 'input'

    def test_mixed_link_styles(self):
        data = {
            'ow_template': '1.0',
            'nodes': [
                {'name': 'A', 'model': 'Simhyd'},
                {'name': 'B', 'model': 'DepthToRate'},
                {'name': 'C', 'model': 'Routing'},
            ],
            'links': [
                {'from': {'node': 'A', 'output': 'runoff'}, 'to': {'node': 'B', 'input': 'input'}},
                'B.outflow -> C.inflow',
            ],
        }
        tpl, _, _ = dict_to_template(data, model_registry=MODEL_REGISTRY)
        assert len(tpl.links) == 2
        assert tpl.links[0].from_node.name == 'A'
        assert tpl.links[1].from_node.name == 'B'
        assert tpl.links[1].to_node.name == 'C'

    def test_compact_link_via_yaml_string(self):
        yaml_str = """\
ow_template: "1.0"
nodes:
  - name: X
    model: Simhyd
  - name: Y
    model: DepthToRate
links:
  - X.runoff -> Y.input
"""
        tpl, _, _ = template_from_yaml_string(yaml_str, model_registry=MODEL_REGISTRY)
        assert len(tpl.links) == 1
        assert tpl.links[0].from_node.name == 'X'
        assert tpl.links[0].to_input == 'input'


class TestDeserializationPortBindings:
    def test_inputs(self):
        data = {
            'ow_template': '1.0',
            'nodes': [{'name': 'N1', 'model': 'Simhyd'}],
            'inputs': [{'alias': 'rain', 'node': 'N1', 'port': 'rainfall'}],
        }
        tpl, _, _ = dict_to_template(data, model_registry=MODEL_REGISTRY)
        assert len(tpl.inputs) == 1
        node, port, alias, tags = tpl.inputs[0]
        assert node.name == 'N1'
        assert port == 'rainfall'
        assert alias == 'rain'

    def test_outputs(self):
        data = {
            'ow_template': '1.0',
            'nodes': [{'name': 'N2', 'model': 'DepthToRate'}],
            'outputs': [{'alias': 'flow', 'node': 'N2', 'port': 'outflow'}],
        }
        tpl, _, _ = dict_to_template(data, model_registry=MODEL_REGISTRY)
        assert len(tpl.outputs) == 1
        node, port, alias, tags = tpl.outputs[0]
        assert node.name == 'N2'
        assert port == 'outflow'
        assert alias == 'flow'

    def test_port_tags(self):
        data = {
            'ow_template': '1.0',
            'nodes': [{'name': 'N1', 'model': 'Simhyd'}],
            'inputs': [{
                'alias': 'rain', 'node': 'N1', 'port': 'rainfall',
                'tags': {'process': 'Climate', 'zone': 'A'},
            }],
        }
        tpl, _, _ = dict_to_template(data, model_registry=MODEL_REGISTRY)
        _, _, _, tags = tpl.inputs[0]
        assert tags['_process'] == 'Climate'
        assert tags['zone'] == 'A'


class TestDeserializationNested:
    def test_inline_nested(self):
        data = {
            'ow_template': '1.0',
            'label': 'Parent',
            'nodes': [{'name': 'R', 'model': 'Routing'}],
            'nested': [{
                'inline': {
                    'ow_template': '1.0',
                    'label': 'Child',
                    'nodes': [{'name': 'C1', 'model': 'Simhyd'}],
                },
            }],
        }
        tpl, _, _ = dict_to_template(data, model_registry=MODEL_REGISTRY)
        assert len(tpl.nested) == 1
        child = tpl.nested[0]
        assert child.label == 'Child'
        assert len(child.nodes) == 1
        assert child.nodes[0].name == 'C1'

    def test_file_reference(self):
        child_data = {
            'ow_template': '1.0',
            'label': 'FromFile',
            'nodes': [{'name': 'F1', 'model': 'Simhyd'}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            from ruamel.yaml import YAML
            yaml = YAML()
            child_path = os.path.join(tmpdir, 'child.yaml')
            with open(child_path, 'w') as f:
                yaml.dump(child_data, f)

            parent_data = {
                'ow_template': '1.0',
                'label': 'Parent',
                'nested': [{'template': 'child.yaml'}],
            }
            tpl, _, _ = dict_to_template(
                parent_data, base_dir=tmpdir, model_registry=MODEL_REGISTRY
            )
            assert len(tpl.nested) == 1
            assert tpl.nested[0].label == 'FromFile'

    def test_file_reference_without_base_dir_raises(self):
        data = {
            'ow_template': '1.0',
            'nested': [{'template': 'child.yaml'}],
        }
        with pytest.raises(TemplateLoadError, match="Cannot resolve file reference"):
            dict_to_template(data, model_registry=MODEL_REGISTRY)

    def test_cycle_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from ruamel.yaml import YAML
            yaml = YAML()
            # File A references File B, File B references File A
            a_path = os.path.join(tmpdir, 'a.yaml')
            b_path = os.path.join(tmpdir, 'b.yaml')

            a_data = {
                'ow_template': '1.0',
                'label': 'A',
                'nested': [{'template': 'b.yaml'}],
            }
            b_data = {
                'ow_template': '1.0',
                'label': 'B',
                'nested': [{'template': 'a.yaml'}],
            }
            with open(a_path, 'w') as f:
                yaml.dump(a_data, f)
            with open(b_path, 'w') as f:
                yaml.dump(b_data, f)

            with pytest.raises(TemplateLoadError, match="Cycle detected"):
                template_from_yaml(a_path, model_registry=MODEL_REGISTRY)


# ---------------------------------------------------------------------------
# Validation / error tests
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_missing_version(self):
        with pytest.raises(TemplateLoadError, match="ow_template"):
            dict_to_template({'nodes': []})

    def test_duplicate_node_names(self):
        data = {
            'ow_template': '1.0',
            'nodes': [
                {'name': 'N1', 'model': 'Simhyd'},
                {'name': 'N1', 'model': 'DepthToRate'},
            ],
        }
        with pytest.raises(TemplateLoadError, match="Duplicate node name"):
            dict_to_template(data, model_registry=MODEL_REGISTRY)

    def test_link_bad_from_node(self):
        data = {
            'ow_template': '1.0',
            'nodes': [{'name': 'N1', 'model': 'Simhyd'}],
            'links': [{
                'from': {'node': 'MISSING', 'output': 'runoff'},
                'to': {'node': 'N1', 'input': 'rainfall'},
            }],
        }
        with pytest.raises(TemplateLoadError, match="unknown node 'MISSING'"):
            dict_to_template(data, model_registry=MODEL_REGISTRY)

    def test_link_bad_to_node(self):
        data = {
            'ow_template': '1.0',
            'nodes': [{'name': 'N1', 'model': 'Simhyd'}],
            'links': [{
                'from': {'node': 'N1', 'output': 'runoff'},
                'to': {'node': 'MISSING', 'input': 'input'},
            }],
        }
        with pytest.raises(TemplateLoadError, match="unknown node 'MISSING'"):
            dict_to_template(data, model_registry=MODEL_REGISTRY)

    def test_input_bad_node_ref(self):
        data = {
            'ow_template': '1.0',
            'nodes': [{'name': 'N1', 'model': 'Simhyd'}],
            'inputs': [{'alias': 'rain', 'node': 'MISSING', 'port': 'rainfall'}],
        }
        with pytest.raises(TemplateLoadError, match="unknown node 'MISSING'"):
            dict_to_template(data, model_registry=MODEL_REGISTRY)

    def test_output_bad_node_ref(self):
        data = {
            'ow_template': '1.0',
            'nodes': [{'name': 'N1', 'model': 'Simhyd'}],
            'outputs': [{'alias': 'flow', 'node': 'MISSING', 'port': 'runoff'}],
        }
        with pytest.raises(TemplateLoadError, match="unknown node 'MISSING'"):
            dict_to_template(data, model_registry=MODEL_REGISTRY)

    def test_node_missing_model(self):
        data = {
            'ow_template': '1.0',
            'nodes': [{'name': 'N1'}],
        }
        with pytest.raises(TemplateLoadError, match="missing 'model'"):
            dict_to_template(data, model_registry=MODEL_REGISTRY)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def _assert_templates_equal(self, original, restored):
        assert original.label == restored.label
        assert len(original.nodes) == len(restored.nodes)
        for orig_n, rest_n in zip(original.nodes, restored.nodes):
            assert orig_n.name == rest_n.name
            assert orig_n.model_name == rest_n.model_name
            # Compare non-runtime tags
            orig_tags = {k: v for k, v in orig_n.tags.items()
                         if k not in ('_model', '_run_idx', '_generation')}
            rest_tags = {k: v for k, v in rest_n.tags.items()
                         if k not in ('_model', '_run_idx', '_generation')}
            assert orig_tags == rest_tags

        assert len(original.links) == len(restored.links)
        for orig_l, rest_l in zip(original.links, restored.links):
            assert orig_l.from_node.name == rest_l.from_node.name
            assert orig_l.from_output == rest_l.from_output
            assert orig_l.to_node.name == rest_l.to_node.name
            assert orig_l.to_input == rest_l.to_input

        assert len(original.inputs) == len(restored.inputs)
        for orig_i, rest_i in zip(original.inputs, restored.inputs):
            assert orig_i[0].name == rest_i[0].name  # node
            assert orig_i[1] == rest_i[1]  # port
            assert orig_i[2] == rest_i[2]  # alias

        assert len(original.outputs) == len(restored.outputs)
        for orig_o, rest_o in zip(original.outputs, restored.outputs):
            assert orig_o[0].name == rest_o[0].name
            assert orig_o[1] == rest_o[1]
            assert orig_o[2] == rest_o[2]

        assert len(original.nested) == len(restored.nested)
        for orig_c, rest_c in zip(original.nested, restored.nested):
            self._assert_templates_equal(orig_c, rest_c)

    def test_round_trip_simple(self):
        original = _make_simple_template()
        data = template_to_dict(original)
        restored, _, _ = dict_to_template(data, model_registry=MODEL_REGISTRY)
        self._assert_templates_equal(original, restored)

    def test_round_trip_nested(self):
        original = _make_nested_template()
        data = template_to_dict(original)
        restored, _, _ = dict_to_template(data, model_registry=MODEL_REGISTRY)
        self._assert_templates_equal(original, restored)

    def test_round_trip_yaml_string(self):
        original = _make_simple_template()
        yaml_str = template_to_yaml_string(original)
        restored, _, _ = template_from_yaml_string(yaml_str, model_registry=MODEL_REGISTRY)
        self._assert_templates_equal(original, restored)

    def test_round_trip_with_metadata(self):
        original = _make_simple_template()
        node_meta = {'Simhyd-RR-ForestHRU': {'position': {'x': 10, 'y': 20}}}
        tpl_meta = {'description': 'Round trip test'}
        data = template_to_dict(original, node_metadata=node_meta, template_metadata=tpl_meta)
        _, restored_node_meta, restored_tpl_meta = dict_to_template(
            data, model_registry=MODEL_REGISTRY
        )
        assert restored_node_meta == node_meta
        assert restored_tpl_meta == tpl_meta

    def test_round_trip_empty(self):
        original = OWTemplate()
        data = template_to_dict(original)
        restored, _, _ = dict_to_template(data, model_registry=MODEL_REGISTRY)
        self._assert_templates_equal(original, restored)


# ---------------------------------------------------------------------------
# File I/O tests
# ---------------------------------------------------------------------------

class TestFileIO:
    def test_write_and_read_file(self):
        original = _make_simple_template()
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False, mode='w') as f:
            path = f.name

        try:
            template_to_yaml(original, path)

            # Verify file is readable YAML text
            with open(path, 'r') as f:
                content = f.read()
            assert 'ow_template' in content
            assert 'Simhyd' in content

            restored, _, _ = template_from_yaml(path, model_registry=MODEL_REGISTRY)
            assert restored.label == 'Simple HRU'
            assert len(restored.nodes) == 2
            assert len(restored.links) == 1
            assert len(restored.inputs) == 1
            assert len(restored.outputs) == 1
        finally:
            os.unlink(path)

    def test_convenience_methods(self):
        original = _make_simple_template()
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False, mode='w') as f:
            path = f.name

        try:
            original.to_yaml(path)
            restored, node_meta, tpl_meta = OWTemplate.from_yaml(
                path, model_registry=MODEL_REGISTRY
            )
            assert restored.label == original.label
            assert len(restored.nodes) == len(original.nodes)
        finally:
            os.unlink(path)

    def test_convenience_dict_methods(self):
        original = _make_simple_template()
        data = original.to_dict()
        assert data['ow_template'] == SCHEMA_VERSION
        restored, _, _ = OWTemplate.from_dict(data, model_registry=MODEL_REGISTRY)
        assert restored.label == original.label

    def test_write_nested_with_file_reference(self):
        """Write a child, then a parent referencing it, and load the parent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            child = _make_simple_template()
            child_path = os.path.join(tmpdir, 'child.yaml')
            template_to_yaml(child, child_path)

            parent = OWTemplate('Parent')
            n = OWNode(MockRouting, name='Router', _process='Routing')
            parent.nodes.append(n)

            # Manually create parent YAML with file reference
            from ruamel.yaml import YAML
            yaml = YAML()
            parent_data = template_to_dict(parent)
            parent_data['nested'] = [{'template': 'child.yaml'}]
            parent_path = os.path.join(tmpdir, 'parent.yaml')
            with open(parent_path, 'w') as f:
                yaml.dump(parent_data, f)

            restored, _, _ = template_from_yaml(
                parent_path, model_registry=MODEL_REGISTRY
            )
            assert restored.label == 'Parent'
            assert len(restored.nested) == 1
            assert restored.nested[0].label == 'Simple HRU'

    def test_nonexistent_file_raises(self):
        with pytest.raises(TemplateLoadError, match="Failed to read"):
            template_from_yaml('/nonexistent/path.yaml')
