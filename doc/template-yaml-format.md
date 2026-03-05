# OpenWater Template YAML Format

This document describes the YAML file format for persisting OpenWater graph templates (`OWTemplate`). Templates defined in YAML can be loaded into Python, edited, and saved back — enabling data-driven model construction and future editor integration.

For conceptual background on graph templates, see [templates.md](templates.md).

## Schema version

Every template file must declare the schema version:

```yaml
ow_template: "1.0"
```

This is the only required field. All other sections are optional.

## Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ow_template` | string | **Yes** | Schema version. Currently `"1.0"`. |
| `label` | string | No | Human-readable name for the template. |
| `metadata` | dict | No | Extensible metadata (e.g. description, editor state). |
| `nodes` | list | No | Model nodes in this template. |
| `links` | list | No | Directed connections between nodes. |
| `nested` | list | No | Child templates (inline or file references). |
| `inputs` | list | No | Input port bindings exposed on the template boundary. |
| `outputs` | list | No | Output port bindings exposed on the template boundary. |

## Nodes

Each node represents a single model instance within the template.

```yaml
nodes:
  - name: "Simhyd-RR-ForestHRU"    # Unique within this template
    model: Simhyd                    # OpenWater model type name
    tags:                            # Optional
      process: RR
      hru: ForestHRU
    metadata:                        # Optional
      position: {x: 100, y: 50}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | Unique identifier within this template. Used by links and port bindings to reference this node. |
| `model` | string | **Yes** | OpenWater model type name (e.g. `Simhyd`, `DepthToRate`). Must be a model known to OpenWater at load time — either already discovered or discoverable via `ow-inspect`. A `model_registry` dict can also be passed to the loader to supply model types directly. |
| `tags` | dict | No | Key-value tags for node identification and matching. |
| `metadata` | dict | No | Arbitrary per-node metadata (e.g. editor position). Not used by the simulation engine. |

### Tag serialization rules

Tags are serialized with the following transformations:

| Internal key | YAML key | Notes |
|-------------|----------|-------|
| `_process` | `process` | Leading underscore removed |
| `_model` | *(omitted)* | Redundant with `model` field |
| `_run_idx` | *(omitted)* | Runtime-only, assigned during graph construction |
| `_generation` | *(omitted)* | Runtime-only, assigned during graph construction |
| anything else | unchanged | Preserved as-is |

On deserialization, `process` is mapped back to the internal `_process` key.

## Links

Links define directed data flow between two nodes within the same template.

```yaml
links:
  - from: {node: "Simhyd-RR-ForestHRU", output: runoff}
    to:   {node: "DepthToRate-Scale", input: input}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `from.node` | string | **Yes** | Source node name (must exist in `nodes`). |
| `from.output` | string | **Yes** | Output port name on the source node. |
| `to.node` | string | **Yes** | Destination node name (must exist in `nodes`). |
| `to.input` | string | **Yes** | Input port name on the destination node. |

The referenced output and input ports must be valid for the respective model types.

## Nested templates

Templates can contain child templates, either inline or as file references. Nesting is used to compose larger model structures from reusable building blocks (see [templates.md](templates.md) for the conceptual motivation).

### File reference

```yaml
nested:
  - template: "hru_forest.yaml"
```

The path is resolved relative to the directory of the parent YAML file. Circular references (A includes B which includes A) are detected and raise a `TemplateLoadError`.

### Inline definition

```yaml
nested:
  - inline:
      ow_template: "1.0"
      label: "Inner Template"
      nodes:
        - name: "N1"
          model: Simhyd
```

Inline definitions follow the same schema recursively.

Both forms can be mixed in the same `nested` list.

## Port bindings (inputs & outputs)

Port bindings expose specific node ports on the template boundary, allowing parent templates or the flattening process to connect across template boundaries.

```yaml
inputs:
  - alias: rainfall
    node: "Simhyd-RR-ForestHRU"
    port: rainfall
    tags: {}

outputs:
  - alias: runoff
    node: "DepthToRate-Scale"
    port: outflow
    tags: {}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `alias` | string | **Yes** | Name exposed on the template boundary. Used for matching when auto-linking nested templates. |
| `node` | string | **Yes** | Target node name (must exist in `nodes`). |
| `port` | string | **Yes** | Port name on the target node (`input` for inputs, `output` for outputs). |
| `tags` | dict | No | Additional tags used for matching during auto-linking. Same serialization rules as node tags. |

## Complete example

```yaml
ow_template: "1.0"
label: "Simple HRU"
metadata:
  description: "A basic HRU template with rainfall-runoff and scaling"

nodes:
  - name: "Simhyd-RR-ForestHRU"
    model: Simhyd
    tags:
      process: RR
      hru: ForestHRU
    metadata:
      position: {x: 100, y: 50}

  - name: "DepthToRate-Scale"
    model: DepthToRate
    tags:
      process: Scaling

links:
  - from: {node: "Simhyd-RR-ForestHRU", output: runoff}
    to:   {node: "DepthToRate-Scale", input: input}

inputs:
  - alias: rainfall
    node: "Simhyd-RR-ForestHRU"
    port: rainfall

  - alias: pet
    node: "Simhyd-RR-ForestHRU"
    port: pet

outputs:
  - alias: runoff
    node: "DepthToRate-Scale"
    port: outflow
```

## Python API

### Writing templates

```python
from openwater import OWTemplate

template = OWTemplate('Simple HRU')
# ... build template ...

# Write to file
template.to_yaml('my_template.yaml')

# Or get as dict / YAML string
data = template.to_dict()
yaml_str = template_to_yaml_string(template)
```

### Loading templates

```python
from openwater import OWTemplate

# From file
template, node_metadata, template_metadata = OWTemplate.from_yaml('my_template.yaml')

# From dict
template, node_metadata, template_metadata = OWTemplate.from_dict(data)

# From string
from openwater.persistence import template_from_yaml_string
template, node_metadata, template_metadata = template_from_yaml_string(yaml_str)
```

### Model registry

When loading templates outside of a full OpenWater environment (e.g. in tests or editors), pass a `model_registry` to avoid needing `ow-inspect`:

```python
template, _, _ = OWTemplate.from_yaml(
    'my_template.yaml',
    model_registry={'Simhyd': my_simhyd_type, 'DepthToRate': my_dtr_type}
)
```

Each registry entry must be an object with a `.name` attribute (string) and a `.description` dict containing at least `'Inputs'` and `'Outputs'` lists.

## Validation

The loader validates:

- Presence of the `ow_template` version field
- Node names are unique within each template
- Links reference existing nodes
- Port bindings reference existing nodes
- Required fields (`alias`, `port`) are present on port bindings

Validation errors raise `TemplateLoadError` with a message listing all issues found.

## See also

* [Complete model YAML format](text-model-format.md) — extends this template format with parameters, initial states, timeseries, and a time period for directly executable model definitions.
