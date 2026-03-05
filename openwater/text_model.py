"""
Complete model descriptions for OpenWater — YAML-based.

Extends the structural template format (``persistence.py``) with per-node
configuration (parameters, initial states, timeseries references) and a
time period, producing a model that can be loaded and executed directly in
Python via ``openwater.lib`` (ctypes bindings) without the HDF5/ow-sim
pipeline.  Targeted at small model graphs.

YAML Schema (version 1.0)
--------------------------

A complete-model YAML file uses the top-level key ``ow_model`` (instead of
``ow_template``)::

    ow_model: "1.0"
    label: "Upper Catchment"

    time_period:
      start: "2010-01-01"
      end: "2010-12-31"

    data_sources:
      climate: "climate_data.csv"

    nodes:
      - name: "Simhyd-Forest"
        model: Simhyd
        parameters:
          baseflowCoefficient: 0.3
        initial_states:
          SoilMoistureStore: 100.0
        timeseries:
          rainfall:
            data_source: climate
            column: "Catchment1_Rainfall"

    links:
      - from: {node: "Simhyd-Forest", output: "runoff"}
        to:   {node: "DepthToRate-Scale", input: "input"}

See the module-level docstring in ``persistence.py`` for the structural
(nodes/links/inputs/outputs) portion of the schema.
"""
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx
from ruamel.yaml import YAML

from .persistence import dict_to_template, SCHEMA_VERSION, _parse_shorthand_link
from .template import OWTemplate


class ModelLoadError(Exception):
    """Raised when a complete-model YAML file cannot be loaded or validated."""
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TimeseriesRef:
    """Reference to a column in a named data source."""
    data_source: str
    column: str


@dataclass
class NodeConfig:
    """Per-node configuration: parameters, initial states, timeseries refs."""
    parameters: dict[str, Any] = field(default_factory=dict)
    initial_states: dict[str, float] = field(default_factory=dict)
    timeseries: dict[str, TimeseriesRef] = field(default_factory=dict)


@dataclass
class ModelDefinition:
    """A complete, executable model definition."""
    template: OWTemplate
    node_configs: dict[str, NodeConfig]
    data_source_paths: dict[str, str]
    time_period: tuple[str, str] | None
    base_dir: str | None
    defaults: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResults:
    """Results from executing a model."""
    outputs: dict[str, pd.DataFrame]
    states: dict[str, pd.DataFrame]
    time_period: tuple[str, str] | None


# ---------------------------------------------------------------------------
# Initial state loading
# ---------------------------------------------------------------------------

def load_initial_states(path, model_def):
    """Load initial states from a directory of ``{node}_states.csv`` files.

    For each file found, the **last row** is read as the final state from a
    previous run and applied as the initial states for the corresponding node.
    This enables continuation runs: the output of ``run_model`` (written by
    ``_write_results`` or equivalent) can be fed directly back in.

    Existing initial states in *model_def* are updated (not replaced), so
    values from the CSV files take precedence but any states not present in the
    file are preserved.

    Parameters
    ----------
    path : str or Path
        Directory containing ``{node}_states.csv`` files.
    model_def : ModelDefinition
        Model definition to update in place.
    """
    path = str(path)
    if not os.path.isdir(path):
        raise ModelLoadError(f"Initial states directory does not exist: {path!r}")

    for filename in os.listdir(path):
        if not filename.endswith('_states.csv'):
            continue
        node_name = filename[:-len('_states.csv')]
        filepath = os.path.join(path, filename)
        df = pd.read_csv(filepath, index_col=0)
        if df.empty:
            continue

        last_row = df.iloc[-1]
        if node_name not in model_def.node_configs:
            model_def.node_configs[node_name] = NodeConfig()
        for col in df.columns:
            model_def.node_configs[node_name].initial_states[col] = float(last_row[col])


# ---------------------------------------------------------------------------
# Inline node output extraction
# ---------------------------------------------------------------------------

def _extract_inline_outputs(links_data, nodes_data):
    """Extract inline ``outputs`` sections from node dicts into link dicts.

    Compact string links and verbose dict links in *links_data* are passed
    through as-is (compact strings are normalized later by ``dict_to_template``).

    Returns a combined list of all link entries.
    """
    normalized = list(links_data or [])

    # Inline outputs on nodes
    for node_data in (nodes_data or []):
        node_name = node_data.get('name')
        inline_outputs = node_data.get('outputs')
        if not inline_outputs or not isinstance(inline_outputs, dict):
            continue

        for output_port, destinations in inline_outputs.items():
            # destinations can be a single string or a list of strings
            if isinstance(destinations, str):
                destinations = [destinations]
            for dest in destinations:
                # dest is "NodeName.input_port"
                dest_parts = dest.rsplit('.', 1)
                if len(dest_parts) != 2:
                    raise ModelLoadError(
                        f"Invalid inline output target on node '{node_name}' "
                        f"(expected 'Node.input'): {dest!r}"
                    )
                normalized.append({
                    'from': {'node': node_name, 'output': output_port},
                    'to': {'node': dest_parts[0], 'input': dest_parts[1]},
                })

    return normalized


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(path, model_registry=None, data_sources=None):
    """Load a complete model definition from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.
    model_registry : dict, optional
        Mapping of model name -> model type object (avoids ow-inspect).
    data_sources : dict, optional
        Runtime data source overrides (name -> DataFrame).

    Returns
    -------
    ModelDefinition
    """
    path = str(path)
    try:
        yaml = YAML()
        with open(path, 'r') as f:
            data = yaml.load(f)
    except Exception as e:
        raise ModelLoadError(f"Failed to read YAML file '{path}': {e}") from e

    base_dir = os.path.dirname(os.path.abspath(path))
    return load_model_from_dict(
        data, base_dir=base_dir,
        model_registry=model_registry, data_sources=data_sources,
    )


def load_model_from_dict(data, base_dir=None, model_registry=None, data_sources=None):
    """Load a complete model definition from a parsed dict.

    Parameters
    ----------
    data : dict
        Parsed YAML data with ``ow_model`` key.
    base_dir : str, optional
        Base directory for resolving relative file paths.
    model_registry : dict, optional
        Mapping of model name -> model type object.
    data_sources : dict, optional
        Runtime data source overrides (name -> DataFrame).

    Returns
    -------
    ModelDefinition
    """
    if 'ow_model' not in data:
        raise ModelLoadError("Missing required field 'ow_model' (schema version)")

    # Extract time_period
    time_period = None
    tp_data = data.get('time_period')
    if tp_data:
        start = tp_data.get('start')
        end = tp_data.get('end')
        if not start or not end:
            raise ModelLoadError("time_period must have both 'start' and 'end'")
        time_period = (str(start), str(end))

    # Extract file-level parameter defaults
    defaults = dict(data.get('defaults', {}))

    # Extract data source paths
    data_source_paths = {}
    for name, path_val in data.get('data_sources', {}).items():
        data_source_paths[name] = str(path_val)

    # Strip model-config keys from node dicts before passing to dict_to_template
    node_configs = {}
    nodes_data = data.get('nodes', [])
    stripped_nodes = []
    for node_data in nodes_data:
        node_name = node_data.get('name')
        params = dict(node_data.get('parameters', {}))
        init_states = dict(node_data.get('initial_states', {}))

        ts_refs = {}
        for ts_name, ts_spec in node_data.get('timeseries', {}).items():
            ts_refs[ts_name] = TimeseriesRef(
                data_source=ts_spec['data_source'],
                column=ts_spec['column'],
            )

        node_configs[node_name] = NodeConfig(
            parameters=params,
            initial_states=init_states,
            timeseries=ts_refs,
        )

        # Build a cleaned copy for dict_to_template
        cleaned = {k: v for k, v in node_data.items()
                   if k not in ('parameters', 'initial_states', 'timeseries', 'outputs')}
        stripped_nodes.append(cleaned)

    # Normalize link shorthands (compact strings + inline node outputs)
    all_links = _extract_inline_outputs(data.get('links'), nodes_data)

    # Build template data for dict_to_template
    template_data = {
        'ow_template': data.get('ow_model'),
        'label': data.get('label', ''),
    }
    if stripped_nodes:
        template_data['nodes'] = stripped_nodes
    if all_links:
        template_data['links'] = all_links
    if 'inputs' in data:
        template_data['inputs'] = data['inputs']
    if 'outputs' in data:
        template_data['outputs'] = data['outputs']
    if 'nested' in data:
        template_data['nested'] = data['nested']

    template, _, _ = dict_to_template(
        template_data, base_dir=base_dir, model_registry=model_registry,
    )

    return ModelDefinition(
        template=template,
        node_configs=node_configs,
        data_source_paths=data_source_paths,
        time_period=time_period,
        base_dir=base_dir,
        defaults=defaults,
    )


# ---------------------------------------------------------------------------
# Graph building (custom — avoids instantiate() name mangling)
# ---------------------------------------------------------------------------

def _build_graph(template):
    """Build a NetworkX DiGraph from a template using original node names.

    Edge data includes ``src`` and ``dest`` lists (output/input port names),
    matching the convention in ``template.template_to_graph``.

    Returns
    -------
    nx.DiGraph
    """
    g = nx.DiGraph()
    node_lookup = {n.name: n for n in template.nodes}

    for n in template.nodes:
        g.add_node(n.name, model_name=n.model_name, node_obj=n)

    for link in template.links:
        key = (link.from_node.name, link.to_node.name)
        if key in g.edges:
            g.edges[key]['src'].append(link.from_output)
            g.edges[key]['dest'].append(link.to_input)
        else:
            g.add_edge(key[0], key[1],
                       src=[link.from_output], dest=[link.to_input])

    return g


# ---------------------------------------------------------------------------
# Data source resolution
# ---------------------------------------------------------------------------

def _resolve_data_sources(model_def, runtime_sources=None):
    """Resolve all named data sources to DataFrames.

    Runtime sources (dict of name -> DataFrame) take priority over file paths
    declared in the YAML.

    Returns
    -------
    dict[str, DataFrame]
    """
    runtime_sources = runtime_sources or {}
    resolved = {}

    for name, file_path in model_def.data_source_paths.items():
        if name in runtime_sources:
            resolved[name] = runtime_sources[name]
        else:
            if model_def.base_dir:
                full_path = os.path.join(model_def.base_dir, file_path)
            else:
                full_path = file_path
            resolved[name] = pd.read_csv(full_path)

    # Include any extra runtime sources not in file paths
    for name, df in runtime_sources.items():
        if name not in resolved:
            resolved[name] = df

    return resolved


def _find_linked_input(graph, dest_node, input_name, node_outputs):
    """Find the upstream output array linked to a node's input port.

    Parameters
    ----------
    graph : nx.DiGraph
    dest_node : str
        Name of the destination node.
    input_name : str
        The input port name on the destination node.
    node_outputs : dict
        Mapping of node_name -> {output_name: np.array}.

    Returns
    -------
    np.ndarray or None
    """
    for pred in graph.predecessors(dest_node):
        edge = graph.edges[(pred, dest_node)]
        for src_port, dst_port in zip(edge['src'], edge['dest']):
            if dst_port == input_name:
                pred_outputs = node_outputs.get(pred, {})
                if src_port in pred_outputs:
                    return pred_outputs[src_port]
    return None


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_model(model_def, data_sources=None):
    """Execute a complete model definition.

    Runs each node in topological order using ``openwater.lib`` ctypes
    bindings. Upstream outputs are automatically routed to downstream inputs
    via the link graph.

    Parameters
    ----------
    model_def : ModelDefinition
    data_sources : dict, optional
        Runtime data sources (name -> DataFrame), overriding YAML file paths.

    Returns
    -------
    ModelResults
    """
    import openwater.lib as ow_lib

    resolved_sources = _resolve_data_sources(model_def, data_sources)

    graph = _build_graph(model_def.template)
    exec_order = list(nx.topological_sort(graph))

    # Determine timestep length
    n_timesteps = None
    if model_def.time_period:
        start, end = model_def.time_period
        date_range = pd.date_range(start, end)
        n_timesteps = len(date_range)

    node_outputs = {}   # node_name -> {output_name: np.array}
    all_outputs = {}    # node_name -> DataFrame
    all_states = {}     # node_name -> DataFrame

    for node_name in exec_order:
        node_obj = graph.nodes[node_name]['node_obj']
        description = node_obj.model_type.description
        config = model_def.node_configs.get(node_name, NodeConfig())

        # Build kwargs for the lib function call
        kwargs = {}

        # 1. Gather inputs: linked upstream first, then timeseries, then zeros
        for input_name in description['Inputs']:
            linked = _find_linked_input(graph, node_name, input_name, node_outputs)
            if linked is not None:
                kwargs[input_name] = np.asarray(linked, dtype='d')
            elif input_name in config.timeseries:
                ts_ref = config.timeseries[input_name]
                df = resolved_sources[ts_ref.data_source]
                ts_data = np.asarray(df[ts_ref.column].values, dtype='d')
                kwargs[input_name] = ts_data
                if n_timesteps is None:
                    n_timesteps = len(ts_data)

        # 2. Parameters (node-specific > file defaults > model-type default)
        for param_meta in description['Parameters']:
            pname = param_meta['Name']
            if pname in config.parameters:
                kwargs[pname] = config.parameters[pname]
            elif pname in model_def.defaults:
                kwargs[pname] = model_def.defaults[pname]
            elif 'Default' in param_meta:
                kwargs[pname] = param_meta['Default']

        # 3. Initial states (always pass all states; default to zero)
        for state_name in description['States']:
            value = config.initial_states.get(state_name, 0.0)
            kwargs[state_name] = np.array([value], dtype='d')

        # Call the model function
        model_func = getattr(ow_lib, node_obj.model_name)
        results = model_func(**kwargs)

        # Parse results: outputs come first, then final states
        n_outputs = len(description['Outputs'])
        n_states = len(description['States'])

        output_arrays = results[:n_outputs]
        state_arrays = results[n_outputs:]

        # Store outputs for downstream routing
        node_outputs[node_name] = {}
        output_data = {}
        for i, oname in enumerate(description['Outputs']):
            arr = output_arrays[i]
            if hasattr(arr, 'shape') and len(arr.shape) > 1:
                arr = arr.flatten()
            node_outputs[node_name][oname] = arr
            output_data[oname] = arr

        # Build output DataFrame
        if model_def.time_period and n_timesteps:
            index = pd.date_range(model_def.time_period[0], model_def.time_period[1])
            # Trim index if needed
            first_len = len(next(iter(output_data.values()))) if output_data else 0
            if len(index) != first_len and first_len > 0:
                index = index[:first_len]
            all_outputs[node_name] = pd.DataFrame(output_data, index=index)
        else:
            all_outputs[node_name] = pd.DataFrame(output_data)

        # Build states DataFrame
        state_data = {}
        for i, sname in enumerate(description['States']):
            if i < len(state_arrays):
                arr = state_arrays[i]
                if hasattr(arr, 'shape') and len(arr.shape) > 1:
                    arr = arr.flatten()
                state_data[sname] = arr
        if state_data:
            all_states[node_name] = pd.DataFrame(state_data)
        else:
            all_states[node_name] = pd.DataFrame()

    return ModelResults(
        outputs=all_outputs,
        states=all_states,
        time_period=model_def.time_period,
    )
