"""
YAML persistence for OpenWater templates.

Provides serialization and deserialization of OWTemplate to/from YAML files,
enabling data-driven template definitions and round-trip persistence.

YAML Schema (version 1.0)
--------------------------

A template YAML file has the following top-level structure::

    ow_template: "1.0"           # Required. Schema version string.
    label: "My Template"         # Optional. Human-readable template name.
    metadata: {...}              # Optional. Extensible dict (editor coords, etc.)

    nodes:                       # Optional. List of model nodes.
      - name: "Node-Name"       # Required. Unique within this template.
        model: Simhyd            # Required. OpenWater model type name.
        tags:                    # Optional. Arbitrary key-value tags.
          process: RR            # Serialized without the leading _ used internally.
          hru: ForestHRU
        metadata: {...}          # Optional. Per-node metadata (e.g. position).

    links:                       # Optional. Directed connections between nodes.
      - from: {node: "A", output: runoff}
        to:   {node: "B", input: inflow}

    nested:                      # Optional. Child templates.
      - template: "child.yaml"  # File reference — relative to this YAML's dir.
      - inline:                  # OR inline definition (same schema, recursive).
          ow_template: "1.0"
          label: "Inner"
          nodes: [...]

    inputs:                      # Optional. Template-level input port bindings.
      - alias: rainfall          # Name exposed on the template boundary.
        node: "Node-Name"       # Target node (must exist in nodes list).
        port: rainfall           # Input port name on the target node.
        tags: {}                 # Optional. Matching tags for auto-linking.

    outputs:                     # Optional. Template-level output port bindings.
      - alias: runoff
        node: "Node-Name"
        port: outflow
        tags: {}

Tag serialization rules:

- ``_process`` is written as ``process`` (no leading underscore).
- ``_model`` is omitted (redundant with the ``model`` field on each node).
- ``_run_idx`` and ``_generation`` are omitted (runtime-only, assigned during
  simulation graph construction).
- All other tags are preserved as-is.

Nested template file references are resolved relative to the parent YAML file's
directory. Circular references are detected and raise ``TemplateLoadError``.

See also: ``doc/template-yaml-format.md`` for the full format reference.
"""
import os
from io import StringIO

from ruamel.yaml import YAML

from .template import OWTemplate, OWNode, OWLink, TAG_MODEL, TAG_PROCESS, TAG_RUN_INDEX, TAG_GENERATION

# Tags that are runtime-only and should not be serialized
_RUNTIME_TAGS = {TAG_MODEL, TAG_RUN_INDEX, TAG_GENERATION}

SCHEMA_VERSION = "1.0"


class TemplateLoadError(Exception):
    """Raised when a template YAML file cannot be loaded or validated."""
    pass


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_tags(tags):
    """Convert node tags dict to serializable form.

    - Strips _model (redundant with model field), _run_idx, _generation (runtime)
    - Renames _process -> process
    """
    result = {}
    for k, v in tags.items():
        if k in _RUNTIME_TAGS:
            continue
        if k == TAG_PROCESS:
            result['process'] = v
        else:
            result[k] = v
    return result


def _serialize_node(node, metadata=None):
    """Serialize an OWNode to a dict."""
    d = {
        'name': node.name,
        'model': node.model_name,
    }
    tags = _serialize_tags(node.tags)
    if tags:
        d['tags'] = tags
    if metadata and node.name in metadata:
        d['metadata'] = metadata[node.name]
    return d


def _serialize_link(link):
    """Serialize an OWLink to a dict."""
    return {
        'from': {'node': link.from_node.name, 'output': link.from_output},
        'to': {'node': link.to_node.name, 'input': link.to_input},
    }


def _serialize_port_binding(binding):
    """Serialize an input/output port binding tuple to a dict.

    Bindings are stored as (node, port_name, alias, tags_dict).
    """
    node, port_name, alias, tags = binding
    d = {
        'alias': alias,
        'node': node.name,
        'port': port_name,
    }
    serialized_tags = _serialize_tags(tags)
    if serialized_tags:
        d['tags'] = serialized_tags
    return d


def template_to_dict(template, node_metadata=None, template_metadata=None):
    """Convert an OWTemplate to a plain dict suitable for YAML serialization.

    Parameters
    ----------
    template : OWTemplate
    node_metadata : dict, optional
        Mapping of node name -> metadata dict (e.g. position info for editors)
    template_metadata : dict, optional
        Top-level template metadata dict

    Returns
    -------
    dict
    """
    data = {'ow_template': SCHEMA_VERSION}

    if template.label:
        data['label'] = template.label

    if template_metadata:
        data['metadata'] = template_metadata

    if template.nodes:
        data['nodes'] = [_serialize_node(n, node_metadata) for n in template.nodes]

    if template.links:
        data['links'] = [_serialize_link(l) for l in template.links]

    if template.nested:
        nested_list = []
        for child in template.nested:
            nested_list.append({
                'inline': template_to_dict(child)
            })
        data['nested'] = nested_list

    if template.inputs:
        data['inputs'] = [_serialize_port_binding(b) for b in template.inputs]

    if template.outputs:
        data['outputs'] = [_serialize_port_binding(b) for b in template.outputs]

    return data


def template_to_yaml(template, path, node_metadata=None, template_metadata=None):
    """Write an OWTemplate to a YAML file.

    Parameters
    ----------
    template : OWTemplate
    path : str or Path
        File path to write to
    node_metadata : dict, optional
    template_metadata : dict, optional
    """
    data = template_to_dict(template, node_metadata, template_metadata)
    yaml = YAML()
    yaml.default_flow_style = False
    with open(path, 'w') as f:
        yaml.dump(data, f)


def template_to_yaml_string(template, node_metadata=None, template_metadata=None):
    """Serialize an OWTemplate to a YAML string.

    Parameters
    ----------
    template : OWTemplate
    node_metadata : dict, optional
    template_metadata : dict, optional

    Returns
    -------
    str
    """
    data = template_to_dict(template, node_metadata, template_metadata)
    yaml = YAML()
    yaml.default_flow_style = False
    stream = StringIO()
    yaml.dump(data, stream)
    return stream.getvalue()


# ---------------------------------------------------------------------------
# Link shorthand parsing
# ---------------------------------------------------------------------------

def _parse_shorthand_link(link_str):
    """Parse a compact link string like ``'S1.runoff -> R1.inflow'``.

    Returns a verbose link dict suitable for deserialization.

    Raises ``TemplateLoadError`` on malformed input.
    """
    parts = link_str.split('->')
    if len(parts) != 2:
        raise TemplateLoadError(
            f"Invalid shorthand link (expected 'Node.output -> Node.input'): {link_str!r}"
        )
    left = parts[0].strip()
    right = parts[1].strip()

    left_parts = left.rsplit('.', 1)
    right_parts = right.rsplit('.', 1)
    if len(left_parts) != 2 or len(right_parts) != 2:
        raise TemplateLoadError(
            f"Invalid shorthand link (expected 'Node.output -> Node.input'): {link_str!r}"
        )

    return {
        'from': {'node': left_parts[0], 'output': left_parts[1]},
        'to': {'node': right_parts[0], 'input': right_parts[1]},
    }


def _normalize_links(links_data):
    """Expand compact string links into verbose link dicts.

    Dict entries are passed through unchanged.
    """
    normalized = []
    for entry in (links_data or []):
        if isinstance(entry, str):
            normalized.append(_parse_shorthand_link(entry))
        else:
            normalized.append(entry)
    return normalized


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_template_data(data):
    """Validate template data dict, returning a list of error strings.

    An empty list means the data is valid.
    """
    errors = []

    if 'ow_template' not in data:
        errors.append("Missing required field 'ow_template' (schema version)")

    nodes = data.get('nodes', [])
    node_names = set()
    for i, node in enumerate(nodes):
        name = node.get('name')
        if not name:
            errors.append(f"Node at index {i} missing 'name'")
        elif name in node_names:
            errors.append(f"Duplicate node name: '{name}'")
        else:
            node_names.add(name)

        if 'model' not in node:
            errors.append(f"Node '{name or i}' missing 'model'")

    for i, link in enumerate(data.get('links', [])):
        from_spec = link.get('from', {})
        to_spec = link.get('to', {})
        from_node = from_spec.get('node')
        to_node = to_spec.get('node')
        if from_node and from_node not in node_names:
            errors.append(f"Link {i}: 'from' references unknown node '{from_node}'")
        if to_node and to_node not in node_names:
            errors.append(f"Link {i}: 'to' references unknown node '{to_node}'")
        if 'output' not in from_spec:
            errors.append(f"Link {i}: 'from' missing 'output'")
        if 'input' not in to_spec:
            errors.append(f"Link {i}: 'to' missing 'input'")

    for kind in ('inputs', 'outputs'):
        for i, binding in enumerate(data.get(kind, [])):
            ref_node = binding.get('node')
            if ref_node and ref_node not in node_names:
                errors.append(f"{kind}[{i}]: references unknown node '{ref_node}'")
            if 'alias' not in binding:
                errors.append(f"{kind}[{i}]: missing 'alias'")
            if 'port' not in binding:
                errors.append(f"{kind}[{i}]: missing 'port'")

    return errors


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------

def _deserialize_tags(tags_dict):
    """Convert serialized tags back to internal form.

    Renames process -> _process.
    """
    result = {}
    for k, v in tags_dict.items():
        if k == 'process':
            result[TAG_PROCESS] = v
        else:
            result[k] = v
    return result


def _deserialize_node(node_data, model_registry=None):
    """Create an OWNode from a serialized dict.

    Parameters
    ----------
    node_data : dict
    model_registry : dict, optional
        Mapping of model name -> model type object. If provided, used to
        look up model types without triggering ow-inspect discovery.

    Returns
    -------
    tuple of (OWNode, metadata_or_None)
    """
    name = node_data['name']
    model = node_data['model']
    tags = _deserialize_tags(node_data.get('tags', {}))

    if model_registry and model in model_registry:
        model_type = model_registry[model]
    else:
        model_type = model

    node = OWNode(model_type, name=name, **tags)
    metadata = node_data.get('metadata')
    return node, metadata


def _deserialize_link(link_data, node_lookup):
    """Create an OWLink from a serialized dict.

    Parameters
    ----------
    link_data : dict
    node_lookup : dict
        Mapping of node name -> OWNode
    """
    from_spec = link_data['from']
    to_spec = link_data['to']
    from_node = node_lookup[from_spec['node']]
    to_node = node_lookup[to_spec['node']]
    return OWLink(from_node, from_spec['output'], to_node, to_spec['input'])


def _deserialize_port_binding(binding_data, node_lookup):
    """Create a port binding tuple from serialized dict.

    Returns (node, port_name, alias, tags_dict).
    """
    node = node_lookup[binding_data['node']]
    port = binding_data['port']
    alias = binding_data['alias']
    tags = _deserialize_tags(binding_data.get('tags', {}))
    return (node, port, alias, tags)


def dict_to_template(data, base_dir=None, model_registry=None, _visited_paths=None):
    """Convert a plain dict to an OWTemplate.

    Parameters
    ----------
    data : dict
        Template data (as produced by template_to_dict or loaded from YAML)
    base_dir : str, optional
        Base directory for resolving relative file references in nested templates
    model_registry : dict, optional
        Mapping of model name -> model type object for avoiding ow-inspect
    _visited_paths : set, optional
        Internal: tracks visited file paths for cycle detection

    Returns
    -------
    tuple of (OWTemplate, node_metadata, template_metadata)
        node_metadata: dict mapping node name -> metadata dict
        template_metadata: dict or None
    """
    # Normalize shorthand links before validation
    if 'links' in data:
        data = dict(data, links=_normalize_links(data['links']))

    errors = _validate_template_data(data)
    if errors:
        raise TemplateLoadError("Invalid template data:\n  " + "\n  ".join(errors))

    template = OWTemplate(data.get('label', ''))
    template_metadata = data.get('metadata')
    node_metadata = {}
    node_lookup = {}

    for node_data in data.get('nodes', []):
        node, meta = _deserialize_node(node_data, model_registry)
        template.nodes.append(node)
        node_lookup[node.name] = node
        if meta is not None:
            node_metadata[node.name] = meta

    for link_data in data.get('links', []):
        link = _deserialize_link(link_data, node_lookup)
        template.links.append(link)

    for nested_data in data.get('nested', []):
        if 'template' in nested_data:
            # File reference
            if base_dir is None:
                raise TemplateLoadError(
                    "Cannot resolve file reference without base_dir: "
                    f"'{nested_data['template']}'"
                )
            ref_path = os.path.normpath(os.path.join(base_dir, nested_data['template']))

            if _visited_paths is None:
                _visited_paths = set()
            abs_path = os.path.abspath(ref_path)
            if abs_path in _visited_paths:
                raise TemplateLoadError(f"Cycle detected in nested template references: '{abs_path}'")

            child_tpl, _, _ = template_from_yaml(
                ref_path, model_registry=model_registry,
                _visited_paths=_visited_paths
            )
            template.nested.append(child_tpl)
        elif 'inline' in nested_data:
            child_tpl, _, _ = dict_to_template(
                nested_data['inline'], base_dir=base_dir,
                model_registry=model_registry, _visited_paths=_visited_paths
            )
            template.nested.append(child_tpl)

    for binding_data in data.get('inputs', []):
        binding = _deserialize_port_binding(binding_data, node_lookup)
        template.inputs.append(binding)

    for binding_data in data.get('outputs', []):
        binding = _deserialize_port_binding(binding_data, node_lookup)
        template.outputs.append(binding)

    return template, node_metadata, template_metadata


def template_from_yaml(path, model_registry=None, _visited_paths=None):
    """Load an OWTemplate from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file
    model_registry : dict, optional
        Mapping of model name -> model type object
    _visited_paths : set, optional
        Internal: for cycle detection in nested file references

    Returns
    -------
    tuple of (OWTemplate, node_metadata, template_metadata)

    Raises
    ------
    TemplateLoadError
        If the file cannot be read or the data is invalid
    """
    path = str(path)
    abs_path = os.path.abspath(path)

    if _visited_paths is None:
        _visited_paths = set()
    if abs_path in _visited_paths:
        raise TemplateLoadError(f"Cycle detected in nested template references: '{abs_path}'")
    _visited_paths = _visited_paths | {abs_path}

    try:
        yaml = YAML()
        with open(path, 'r') as f:
            data = yaml.load(f)
    except Exception as e:
        raise TemplateLoadError(f"Failed to read YAML file '{path}': {e}") from e

    base_dir = os.path.dirname(abs_path)
    return dict_to_template(
        data, base_dir=base_dir,
        model_registry=model_registry, _visited_paths=_visited_paths
    )


def template_from_yaml_string(yaml_str, base_dir=None, model_registry=None):
    """Load an OWTemplate from a YAML string.

    Parameters
    ----------
    yaml_str : str
        YAML content
    base_dir : str, optional
        Base directory for resolving file references
    model_registry : dict, optional
        Mapping of model name -> model type object

    Returns
    -------
    tuple of (OWTemplate, node_metadata, template_metadata)
    """
    yaml = YAML()
    data = yaml.load(yaml_str)
    return dict_to_template(data, base_dir=base_dir, model_registry=model_registry)
