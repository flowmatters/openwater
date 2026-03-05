# Complete Model YAML Format

This document describes the YAML file format for defining **complete, executable** OpenWater models. Unlike the structural [template format](template-yaml-format.md), which describes only the graph topology (nodes, links, inputs, outputs), the complete model format also captures per-node **parameters**, **initial states**, **timeseries references**, a **time period**, and **data sources** — everything needed to load and run a model directly from Python.

For conceptual background on graph templates, see [templates.md](templates.md). For the structural-only format, see [template-yaml-format.md](template-yaml-format.md).

## When to use this format

The complete model format is designed for **small model graphs** — a handful of nodes that you want to define, configure, and run without the full HDF5/`ow-sim` pipeline. Typical use cases include:

* Prototyping a model structure with known parameters
* Unit-testing model configurations
* Running small, self-contained simulations from a single file
* Sharing reproducible model definitions with colleagues

For large-scale models (thousands of nodes, batch parameterisation, ensemble runs), the standard HDF5 pipeline via `ModelGraph` and `ow-sim` remains the appropriate approach.

## Schema version

Every complete model file must declare the schema version using the `ow_model` key (as opposed to `ow_template` for structural templates):

```yaml
ow_model: "1.0"
```

This is the only required field. All other sections are optional.

## Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ow_model` | string | **Yes** | Schema version. Currently `"1.0"`. Distinguishes complete models from structural templates (`ow_template`). |
| `label` | string | No | Human-readable name for the model. |
| `time_period` | dict | No | Simulation time period with `start` and `end` date strings. |
| `defaults` | dict | No | File-level parameter defaults, applied by name to any node with a matching parameter. |
| `data_sources` | dict | No | Named data source file paths (resolved relative to the YAML file). |
| `nodes` | list | No | Model nodes, each with optional configuration sections. |
| `links` | list | No | Directed connections between nodes. Same format as templates. |
| `inputs` | list | No | Input port bindings. Same format as templates. |
| `outputs` | list | No | Output port bindings. Same format as templates. |

## Time period

Defines the simulation date range. Both `start` and `end` are required if `time_period` is present.

```yaml
time_period:
  start: "2010-01-01"
  end: "2010-12-31"
```

The time period determines the length of the simulation (number of daily timesteps) and provides a `DatetimeIndex` on the output DataFrames.

## Parameter defaults

Parameters are resolved in priority order: **node-specific > file defaults > model-type default**.

### File-level defaults

The `defaults` section specifies parameter values by name. At run time, any node whose model type declares a parameter with a matching name will receive the default value — unless the node provides its own value. This avoids repeating common values like `DeltaT` across many nodes.

```yaml
defaults:
  DeltaT: 86400
  deadStorage: 0.0
```

Defaults only apply where the parameter name matches a parameter declared by the model type. Unknown names are silently ignored, so it is safe to include defaults that only apply to some node types.

### Model-type defaults

Every model type defines a default value for each of its parameters (e.g. `DeltaT` defaults to `86400` on both `DepthToRate` and `StorageRouting`). These are used automatically when neither the node's `parameters` section nor the file-level `defaults` section provides a value, so parameters with sensible defaults can simply be omitted.

## Data sources

Named references to external data files. Paths are resolved relative to the YAML file's directory. Data sources can be overridden at runtime by passing a `data_sources` dict of DataFrames.

```yaml
data_sources:
  climate: "data/climate_data.csv"
  observations: "data/observed_flow.csv"
```

Currently CSV files are supported. Each CSV is loaded into a pandas DataFrame, and individual columns are referenced from node timeseries sections.

## Nodes

Nodes follow the same base format as the [template format](template-yaml-format.md) (`name`, `model`, `tags`, `metadata`) with three additional optional sections: `parameters`, `initial_states`, and `timeseries`.

```yaml
nodes:
  - name: "Simhyd-Forest"
    model: Simhyd
    tags:
      process: RR
      hru: Forest
    parameters:
      baseflowCoefficient: 0.3
      soilMoistureStoreCapacity: 320.0
    initial_states:
      SoilMoistureStore: 100.0
    timeseries:
      rainfall:
        data_source: climate
        column: "Catchment1_Rainfall"
      pet:
        data_source: climate
        column: "Catchment1_PET"
```

### Parameters

Scalar parameters are specified as key-value pairs, where the key is the parameter name as defined by the model type:

```yaml
parameters:
  baseflowCoefficient: 0.3
  soilMoistureStoreCapacity: 320.0
```

Tabular/array parameters (e.g. storage level-volume-area tables) are specified as lists:

```yaml
parameters:
  nLVA: 5
  levels: [0.0, 5.0, 10.0, 15.0, 20.0]
  volumes: [0.0, 50000.0, 200000.0, 600000.0, 1200000.0]
  areas: [0.0, 1000.0, 3000.0, 7000.0, 12000.0]
```

Any parameters not specified use the model's default values.

### Initial states

State variables to initialise before simulation. Unspecified states default to zero.

```yaml
initial_states:
  SoilMoistureStore: 100.0
  GroundwaterStore: 25.0
```

### Timeseries

Maps model input port names to columns in named data sources:

```yaml
timeseries:
  rainfall:
    data_source: climate
    column: "Station1_Rain"
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `data_source` | string | **Yes** | Name of a data source declared in `data_sources` or provided at runtime. |
| `column` | string | **Yes** | Column name within the data source DataFrame. |

Inputs not covered by timeseries or upstream links receive zero arrays.

## Links

Links can be specified in three styles. All three can be mixed freely within a single file.

### Verbose (dict) format

The same structured format used by the [template format](template-yaml-format.md):

```yaml
links:
  - from: {node: "Simhyd-Forest", output: "runoff"}
    to:   {node: "DepthToRate-Scale", input: "input"}
```

### Compact string format

A single string using `Node.output -> Node.input` syntax:

```yaml
links:
  - Simhyd-Forest.runoff -> DepthToRate-Scale.input
```

### Inline node outputs

When a node output has a single destination, it can be declared directly on the node using an `outputs` section. The value is `DestNode.input_port`:

```yaml
nodes:
  - name: "Simhyd-Forest"
    model: Simhyd
    outputs:
      runoff: DepthToRate-Scale.input
```

For outputs that fan out to multiple destinations, use a list:

```yaml
    outputs:
      outflow:
        - R2.inflow
        - Monitor.input
```

During execution, the output timeseries from the source node is automatically routed to the destination node's input.

## Complete example

```yaml
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
    tags:
      process: RR
    parameters:
      baseflowCoefficient: 0.3
      soilMoistureStoreCapacity: 320.0
    initial_states:
      SoilMoistureStore: 100.0
    timeseries:
      rainfall:
        data_source: climate
        column: "Catchment1_Rainfall"
      pet:
        data_source: climate
        column: "Catchment1_PET"

  - name: "DepthToRate-Scale"
    model: DepthToRate
    parameters:
      area: 150000000.0

links:
  - from: {node: "Simhyd-Forest", output: "runoff"}
    to:   {node: "DepthToRate-Scale", input: "input"}
```

## Python API

### Loading a model

```python
from openwater import load_model

# From a YAML file
model_def = load_model('upper_catchment.yaml')

# From a dict (e.g. already parsed YAML)
from openwater.text_model import load_model_from_dict
model_def = load_model_from_dict(data, base_dir='/path/to/data')
```

### Running a model

```python
from openwater import run_model

results = run_model(model_def)

# Access outputs as DataFrames, keyed by node name
runoff_df = results.outputs['Simhyd-Forest']
print(runoff_df[['runoff', 'baseflow']].head())

# Access final states
states_df = results.states['Simhyd-Forest']
```

### Overriding data sources at runtime

Instead of reading CSV files, you can pass DataFrames directly:

```python
import pandas as pd

climate_df = pd.read_parquet('climate.parquet')  # or any other source
results = run_model(model_def, data_sources={'climate': climate_df})
```

Runtime data sources take priority over file paths declared in the YAML.

### Model registry

When loading models outside of a full OpenWater environment (e.g. in tests), pass a `model_registry` to avoid needing `ow-inspect`:

```python
model_def = load_model(
    'model.yaml',
    model_registry={'Simhyd': my_simhyd_type, 'DepthToRate': my_dtr_type}
)
```

Each registry entry must be an object with a `.name` attribute (string) and a `.description` dict containing `'Inputs'`, `'Outputs'`, `'States'`, and `'Parameters'` lists.

## How it works

The complete model format builds on existing OpenWater infrastructure:

1. **Loading** strips `parameters`, `initial_states`, and `timeseries` from each node dict, then delegates the structural portion (nodes, links, inputs, outputs) to `dict_to_template()` from `persistence.py`. The configuration sections are stored in `NodeConfig` objects keyed by node name.

2. **Graph building** constructs a NetworkX `DiGraph` directly from the template's nodes and links, using original node names as keys. This avoids the name-mangling that occurs when `template.instantiate()` is used in the standard pipeline.

3. **Execution** topologically sorts the graph and runs each node in order. For each node, it gathers inputs (from upstream links, timeseries data sources, or zeros), parameters, and initial states, then calls the model function via `openwater.lib` (ctypes bindings to the Go shared library). Outputs are stored and routed to downstream nodes.

4. **Results** are returned as `ModelResults`, containing `outputs` and `states` dicts mapping node names to DataFrames. If a time period was specified, output DataFrames are indexed by date.

## Relationship to other formats

| Aspect | Template format (`ow_template`) | Complete model format (`ow_model`) |
|--------|------|------|
| Purpose | Structural blueprint | Executable model definition |
| Contains parameters | No | Yes |
| Contains timeseries | No | Yes |
| Time period | No | Yes |
| Execution | Requires HDF5 + `ow-sim` | Runs directly via `openwater.lib` |
| Scale | Any size | Small graphs |
| File key | `ow_template: "1.0"` | `ow_model: "1.0"` |

## Validation

The loader validates:

- Presence of the `ow_model` version field
- If `time_period` is present, both `start` and `end` must be provided
- All structural validation from `dict_to_template()` (unique node names, valid link references, etc.)

Validation errors raise `ModelLoadError` with a descriptive message.
