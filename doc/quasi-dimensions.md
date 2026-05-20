# Quasi-dimensions

A **quasi-dimension** is a named lookup that *behaves* like a model [dimension](dimensions.md) at the parameterisation and reporting APIs, but is not part of the model graph itself. Quasi-dimensions exist to handle groupings that are a pure function of an existing dimension — reporting catchments derived from subcatchments, constituent classes derived from constituents, management zones derived from land use — without inflating the graph with redundant tags.

This page is the user guide. For the rationale behind keeping such groupings out of the graph, see [dimensions.md § When _not_ to add a tag](dimensions.md#when-not-to-add-a-tag).

## What a quasi-dimension is

Three pieces of information:

* **name** — what you reference it by (e.g. `'reporting_catchment'`).
* **keyed_by** — the dimension it is a function of (e.g. `'SC'`). May be a real dimension or another quasi-dimension.
* **mapping** — the actual lookup, one row per value of `keyed_by`.

The mapping must be 1:1 — each `keyed_by` value maps to exactly one quasi-dim value. Many-to-many groupings are genuinely new dimensions, not quasi-dimensions.

## Registering a quasi-dimension

`add_quasi_dim` is the single entry point and accepts the mapping in whatever form is convenient. It is available on three objects:

| Object | When to use it |
|---|---|
| `ModelGraph` | At model-build time, before writing the model file. Supports `persist=True`. |
| `ModelFile` | When working with an existing model file. Supports `persist=True` (writes back to the file). |
| `OpenwaterResults` | When analysing results ad hoc. Session-only — does not mutate the model file. |

All three share the same dispatch, so the call shape is identical regardless of which object you reach for:

```python
# From a CSV file. Column headers default to name/keyed_by; override as needed.
owner.add_quasi_dim('reporting_catchments.csv',
                    key='SC', value='reporting_catchment')

# From a pandas Series. Series name → quasi-dim name; index name → keyed_by.
owner.add_quasi_dim(series, name='reporting_catchment', keyed_by='SC')

# From a dict. Name and keyed_by are required.
owner.add_quasi_dim({1: 'North', 2: 'North', 3: 'South'},
                    name='reporting_catchment', keyed_by='SC')

# From a pre-built QuasiDimension — for sharing across notebooks.
from openwater import quasi_dim
qd = quasi_dim.from_csv('rc.csv', key='SC', value='reporting_catchment')
owner.add_quasi_dim(qd)
```

### Persistence

By default a quasi-dimension lives only in the current Python session. On `ModelGraph` and `ModelFile`, pass `persist=True` to write it into the HDF5 model file alongside the real dimensions:

```python
model.add_quasi_dim(qd, persist=True)         # ModelGraph: written on write_model()
mf.add_quasi_dim(qd, persist=True)            # ModelFile:  flushed to disk immediately
```

This is the right choice when the grouping is canonical for the project — downstream notebooks, the reporting layer, and the next analyst all pick it up automatically when they open the model or results file.

`OpenwaterResults.add_quasi_dim` is **always session-only** — there is no `persist` kwarg. This is deliberate: the results object should not silently mutate the model file behind your back. If you decide an ad-hoc quasi-dim is worth keeping, open the model file with `ModelFile` and re-register it there with `persist=True`.

### Ad-hoc analysis on results

For exploration in a notebook, register a quasi-dim directly on the results object and use it immediately in `time_series` / `table`:

```python
results = model.run_model()
results.add_quasi_dim('reporting_catchments.csv',
                      key='SC', value='reporting_catchment')

results.time_series('Sacramento', 'runoff',
                    'reporting_catchment', 'mean')
```

Quasi-dims that were persisted into the model file are loaded automatically when you open `OpenwaterResults` — adding more at runtime extends that set for this session.

### Management

The same trio of accessors is available on each owner:

```python
owner.quasi_dims()                          # list registered names
owner.quasi_dim('reporting_catchment')      # retrieve the QuasiDimension
owner.remove_quasi_dim('reporting_catchment')   # explicit; required before re-registering
```

Name collisions — with an existing real dimension or an already-registered quasi-dimension — raise on registration. Replacing a registered quasi-dim requires removing it first.

On `OpenwaterResults`, `remove_quasi_dim` is session-only too: removing a persisted-loaded quasi-dim drops it for the current session but the on-disk copy survives. On `ModelFile`, removing a persisted quasi-dim is flushed to disk.

## Using a quasi-dimension in reporting

Wherever a real-dimension name is accepted in `OpenwaterResults.time_series` / `table`, a registered quasi-dim name is accepted too. The same name can be used in two distinct ways:

### Group by the quasi-dimension

```python
results.time_series('Sacramento', 'runoff',
                    'reporting_catchment', aggregator='mean')
```

One column per reporting catchment, each aggregating across the subcatchments that map to it. The `aggregator` argument controls how values within a group combine (`mean` or `sum`).

### Filter by the quasi-dimension, report by a raw dimension

```python
results.time_series('Sacramento', 'runoff',
                    'catchment', aggregator='mean',
                    reporting_catchment='North')
```

One column per subcatchment, restricted to those mapped to `North`. Set-valued filters work too:

```python
results.time_series('Sacramento', 'runoff', 'catchment', 'mean',
                    reporting_catchment=['North', 'South'])
```

The same two patterns apply to `table` via its `rows` and `columns` arguments. Real-dim and quasi-dim filters on the same underlying dimension are intersected — a constraint of `SC=['SC2','SC3']` combined with `reporting_catchment='North'` (which maps to `SC1`, `SC2`) yields just `SC2`.

## Using a quasi-dimension in parameterisation

Two patterns, mirroring the reporting cases.

### Constraint by quasi-dimension

Any parameteriser that accepts a constraint (`DictParameteriser`, `SingleTimeseriesInput`, `DataframeInput`, etc.) accepts a quasi-dim name in that constraint:

```python
DictParameteriser(parameter='dwc',
                  key_format='${SC}',
                  model='EmcDwc',
                  parameters={'1': 0.5, '2': 0.7, ...},
                  constraints={'reporting_catchment': 'North'})
```

The parameteriser sees only the matching real-dim nodes — here, the subcatchments mapped to `North`.

### Broadcast a per-group table to underlying nodes

A CSV (or DataFrame) keyed by a quasi-dim — one row per reporting catchment — is broadcast to all underlying nodes via `ParameterTableAssignment`:

```python
df = pd.read_csv('rc_params.csv')   # columns: reporting_catchment, dwc, emc
ParameterTableAssignment(df, 'EmcDwc')
```

The join condition is widened automatically — any column whose name is a registered quasi-dim is projected onto `nodes_df` (via the underlying real dim) before the merge. A real-dim column and a quasi-dim column can co-exist in the same table; the join uses both.

For the 2-D form (`row_dim` / `column_dim`), either or both may be a quasi-dim:

```python
ParameterTableAssignment(df, model='EmcDwc', parameter='dwc',
                         row_dim='reporting_catchment',
                         column_dim='constituent')
```

## Chaining

A quasi-dim's `keyed_by` may itself be another quasi-dim. The chain is composed and resolved automatically:

```python
model.add_quasi_dim(rc)      # reporting_catchment   keyed_by 'SC'
model.add_quasi_dim(rr)      # reporting_region      keyed_by 'reporting_catchment'

results.time_series('Sacramento', 'runoff',
                    'reporting_region', 'mean')          # SC → RC → RR
results.time_series('Sacramento', 'runoff', 'catchment', 'mean',
                    reporting_region='NorthEast')        # also fine
```

Cycles are rejected at registration. Forward references (registering a chain in any order) are allowed.

## Partial coverage

If a `keyed_by` value is missing from the mapping, resolution raises — loud by design, because silently dropping nodes from a report tends to produce wrong-but-plausible numbers. To bucket unmapped values explicitly, supply a default:

```python
model.add_quasi_dim('reporting_catchments.csv',
                    key='SC', value='reporting_catchment',
                    default='Unassigned')
```

Unmapped `SC` values then group under `'Unassigned'` rather than raising.

## When to reach for a quasi-dimension

A grouping is a quasi-dimension if it is a pure function of an existing dimension — one input value, one output value. Examples:

* `subcatchment → reporting_catchment`
* `constituent → constituent_class` (e.g. `TSS`/`TP`/`TN` → `sediment`/`nutrient`)
* `landuse → landuse_category` (e.g. `Sugarcane`/`Bananas` → `Cropping`)

If the grouping changes the *structure* of the model — distinguishes nodes that would otherwise collide, or governs how nodes are connected — it is a real dimension. See [dimensions.md § When _not_ to add a tag](dimensions.md#when-not-to-add-a-tag) for the full criterion.

## Limits in this release

* Quasi-dimensions are **single-key, 1:1** only. `(SC, CGU) → management_zone` is a planned extension; for now, fold it into a real dimension or compute it in user code.
* Many-to-many groupings (one `SC` belonging to multiple overlapping groups) are out of scope — those are genuinely new dimensions.
* `UniformParameteriser` does not currently accept constraints, so it cannot be filtered by a quasi-dim. Use `DictParameteriser` or `ParameterTableAssignment` for that.
