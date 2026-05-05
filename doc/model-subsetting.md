# Model Subsetting

OpenWater models are structured as Directed Acyclic Graphs (DAGs) of model functions, stored in HDF5 files. Two properties of this architecture make subsetting straightforward:

1. **Graph structure** allows extracting connected subsets of model elements.
2. **Explicit initial and final states** for all model functions allow splitting simulations temporally and resuming from where they left off.

Three subsetting operations are available, each creating a new, smaller HDF5 model file from an existing one:

| Operation | What it does | Module |
|-----------|-------------|--------|
| **Split** | Divide a model through time into windows that run sequentially | `openwater.split` |
| **Clip** | Extract the upstream subgraph needed to compute specific nodes | `openwater.clip` |
| **Substitute** | Remove nodes and inject prior simulation outputs as fixed inputs | `openwater.substitute` |

## Split — Temporal Decomposition

Split divides a model's input time series into windows. Each window can be simulated independently, with final states from one window becoming initial states for the next. This is useful for:

- Reducing memory requirements for long simulations
- Checkpointing: resume from the last completed window after a failure
- Parallelising post-processing across time windows

### Splitting a model

```python
from openwater.split import split_model

split_model(
    orig_model='full_model.h5',
    structure='structure.h5',       # receives graph structure, dimensions, links
    parameters='parameters.h5',     # optional: separate file for parameters
    init_states='init_states.h5',   # optional: separate file for initial states
    inputs='inputs.h5',             # template path for input files
    input_windows=[365, 730],       # split at these timestep indices
)
# Creates: structure.h5, parameters.h5, init_states.h5,
#          inputs-0.h5, inputs-1.h5, inputs-2.h5
```

When `input_windows` is omitted, use `split_ts=N` to divide evenly into N windows. `split_ts` defaults to 1, so omitting both produces a single window covering the whole series.

### Running a split model

```python
from openwater.split import run_split_model

results = run_split_model(
    structure='structure.h5',
    params='parameters.h5',
    init_states='init_states.h5',
    inputs=['inputs-0.h5', 'inputs-1.h5', 'inputs-2.h5'],
    dests=['results-0.h5', 'results-1.h5', 'results-2.h5'],
    final_states=['states-0.h5', 'states-1.h5', 'states-2.h5'],
)
```

Execution is sequential: final states from window 0 become the initial states for window 1, and so on. The returned `OpenwaterSplitResults` object provides the same interface as `OpenwaterResults`, concatenating time series across windows:

```python
# Time series are seamlessly concatenated across splits
df = results.time_series('Sacramento', 'runoff', columns='catchment')

# Aggregated tables use weighted combination across splits
tbl = results.table('Sacramento', 'runoff', rows='catchment', columns='hru',
                     temporal_aggregator='mean')
```

Aggregations across split windows are weighted by window length, so a `mean` over the full simulation matches the equivalent computation on the unsplit model — not a naive average of per-window means.

### Loading existing split results

```python
from openwater.results import open_split_results

results = open_split_results(
    model_fn='structure.h5',
    results_pattern='results-*.h5',
    input_pattern='inputs-*.h5',  # optional
)
```

## Clip — Spatial Subsetting

Clip extracts the portion of the model graph needed to compute a set of target nodes. Starting from the target nodes, it traverses upstream through the link table and keeps everything required. The result is a self-contained model file that can be simulated independently.

Use cases:

- Focus on a single sub-catchment or outlet
- Reduce model size for faster iteration during calibration
- Extract a test case from a large model

### Clipping by tags (recommended)

The easiest way to clip is by specifying dimension values (tags) that identify the target nodes:

```python
from openwater.template import ModelFile
from openwater.clip import clip_by_tags

model = ModelFile('full_model.h5')

# Keep everything needed to compute StorageRouting at the outlet
clip_by_tags(model, 'clipped.h5',
             model_type='StorageRouting',
             catchment='outlet_catchment')
```

You can target multiple model types or omit `model_type` to search all types:

```python
# All model nodes in a specific catchment, plus their upstream dependencies
clip_by_tags(model, 'clipped.h5', catchment='Murray_River')
```

### Clipping by node indices

For programmatic use, you can specify target nodes directly as `(model_type, run_index)` tuples:

```python
from openwater.clip import clip, resolve_end_nodes

# Resolve tags to indices (useful for inspection before clipping)
end_nodes = resolve_end_nodes(model, model_type='StorageRouting',
                               catchment='outlet_catchment')
# end_nodes = [('StorageRouting', 42)]

clip(model, 'clipped.h5', end_nodes)
```

### What gets preserved

The clipped model file contains:

- All upstream nodes and their parameters, states, and inputs
- Links between kept nodes (renumbered)
- Dimensions subsetted to values used by kept nodes
- Model map tables (renumbered)
- META including time period and version information

## Substitute — Remove and Replace

Substitute creates a reduced model by removing nodes and replacing their outputs with data from a prior simulation. The prior outputs are injected as fixed input time series into the kept nodes that previously received them via links.

This is the most powerful subsetting operation, enabling workflows where you:

- Remove water quantity models (e.g. Sacramento, StorageRouting, Storage) to focus on water quality parameterisation, running the smaller model many times with different parameters
- Remove the upstream portion of a large model to focus on a specific reach, using prior simulation outputs as boundary conditions

### Use case 1: Remove by model type

Remove all nodes of specified model types. The outputs they previously provided to downstream nodes are read from a prior simulation and baked in as inputs.

```python
from openwater.template import ModelFile
from openwater.substitute import substitute

model = ModelFile('full_model.h5')

# First, run the full model to get baseline results
results = model.run(results_fn='baseline_results.h5', overwrite=True)

# Then create a reduced model without water quantity components
substitute(
    model,
    prior_results='baseline_results.h5',
    dest_fn='quality_only.h5',
    model_types_to_remove=['Sacramento', 'StorageRouting', 'Storage'],
)

# The reduced model can now be run repeatedly with different WQ parameters
wq_model = ModelFile('quality_only.h5')
# ... modify parameters ...
wq_results = wq_model.run(results_fn='wq_results.h5', overwrite=True)
```

### Use case 2: Remove above a point

Remove everything upstream of specified nodes. This is the complement of clip: where clip keeps the upstream subgraph, substitute removes it and injects boundary conditions.

```python
from openwater.clip import resolve_end_nodes
from openwater.substitute import substitute

model = ModelFile('full_model.h5')

# Identify the boundary nodes
boundary = resolve_end_nodes(model, model_type='StorageRouting',
                              catchment='focus_reach_inlet')

# Remove everything upstream, inject prior outputs at the boundary
substitute(
    model,
    prior_results='baseline_results.h5',
    dest_fn='downstream_only.h5',
    above_nodes=boundary,
)
```

**The boundary nodes themselves are kept; only their upstream dependencies are removed.** This is the opposite convention from `clip`, where the equivalent `end_nodes` are kept along with everything upstream.

### Fan-in behaviour

When multiple removed nodes feed the same input on a kept node (e.g. two tributaries flowing into a junction), their contributions are **summed**. This is correct for additive flow and load variables, which are the dominant case in hydrological models.

**Caveat:** summation is *not* correct for intensive variables such as concentrations or temperatures. If your boundary links carry non-additive quantities, substitute will produce incorrect inputs at fan-in points.

### What the substituted model contains

- All kept nodes with their original parameters, states, and map tables
- Input arrays with boundary link slots overwritten by prior simulation outputs
- Links between kept nodes only (boundary links are absorbed into inputs)
- Subsetted dimensions and META (including time period)

## Choosing an operation

| I want to... | Use |
|---|---|
| Run a long simulation in stages with checkpointing | **Split** |
| Extract a sub-catchment or upstream area | **Clip** |
| Fix part of the model and iterate on the rest | **Substitute** (by type) |
| Focus on a downstream reach with upstream as boundary conditions | **Substitute** (above nodes) |
| Reduce model size for calibration of specific components | **Substitute** (by type) |

Split operates on the time axis; clip and substitute operate on the graph (spatial) axis. They can be combined: split a substituted model to get both spatial and temporal reduction.
