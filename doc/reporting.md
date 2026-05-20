# Reporting model results

When an Openwater model runs, it writes an HDF5 results file containing a time series for every output (and every input that was computed from upstream nodes) at every node in the model graph. You can read this file with any HDF5 tool, but the `OpenwaterResults` class in the Openwater Python package is the recommended interface.

This document covers:

1. [Loading results](#loading-results)
2. [Discovering what's available](#discovering-whats-available)
3. [Units and conventions](#units-and-conventions)
4. [`time_series` — per-timestep results](#time_series--per-timestep-results)
5. [`table` — scalar results aggregated through time](#table--scalar-results-aggregated-through-time)
6. [Specifying constraints](#specifying-constraints)
7. [Building higher-level reports](#building-higher-level-reports)

All reporting is driven by the model node [tags](dimensions.md) (also called *dimensions*). Tags identify individual nodes and groups of nodes; the reporting functions use them to slice, group, and aggregate results.

## Loading results

`OpenwaterResults` needs both the model file and the results file:

```python
from openwater.results import OpenwaterResults
results = OpenwaterResults('model.h5', 'results.h5')
```

When you call `model.run_model()`, an `OpenwaterResults` is returned for you:

```python
results = model.run_model()
```

## Discovering what's available

Before pulling results you typically want to know what models, variables, and tags exist:

```python
results.models()                  # List of model types in the run (e.g. ['Sacramento', 'EmcDwc', ...])
results.variables_for('Sacramento')   # Inputs and outputs available for that model
results.dims_for_model('Sacramento')  # Tag names used by nodes of that model
results.dims()                    # All tag names across the run
results.dim('catchment')          # Values that the 'catchment' tag takes
```

## Units and conventions

Two conventions trip up new users:

* **Fluxes are per-second.** Even when the model runs on a daily timestep, fluxes such as `runoff` or `totalLoad` are reported in *per-second* units, interpreted as the average over the timestep. Convert to per-day by multiplying by `86400`.
* **Mass is in kilograms.** Multiply by `1e-3` for tons.

These conversions are not applied for you — see [Building higher-level reports](#building-higher-level-reports).

## `time_series` — per-timestep results

```python
results.time_series(model, variable, columns, aggregator=None, **constraints)
```

Returns a `pandas.DataFrame` indexed by time, with one column per distinct value of the `columns` tag.

| Argument      | Type             | Meaning |
|---------------|------------------|---------|
| `model`       | `str`            | Component model type (e.g. `'Sacramento'`). |
| `variable`    | `str`            | Input or output name on that model (e.g. `'runoff'`, `'rainfall'`). For inputs computed from upstream nodes, the computed input is returned. |
| `columns`     | `str` or list    | Tag name to use for the DataFrame columns. Pass a list for a `MultiIndex`. |
| `aggregator`  | `str` (optional) | How to combine multiple nodes that fall into the same column. One of `'mean'` (default), `'sum'`. |
| `**constraints` | `tag=value`    | Restrict the search to nodes matching these tags (see below). |

### Example

```python
ts = results.time_series('Sacramento', 'runoff', 'catchment', 'mean',
                         hru='Grazing Open')
```

This returns a DataFrame whose columns are catchment names (`SC #1`, `SC #2`, …) and whose values are the mean `runoff` across all matching Sacramento nodes in each catchment. The `hru='Grazing Open'` constraint restricts the search to nodes tagged with that HRU.

### Aggregation semantics

A single `columns` tag value can match many nodes — for example, if there are multiple HRUs per catchment, the `catchment` tag alone matches several Sacramento nodes per catchment. The `aggregator` reduces these into one series per column, applied independently at each timestep. If your constraints leave at most one matching node per column, the aggregator has no effect.

If no nodes match, `time_series` raises an exception.

## `table` — scalar results aggregated through time

```python
results.table(model, variable, rows, columns,
              temporal_aggregator='mean', aggregator=None, **constraints)
```

Returns a `pandas.DataFrame` indexed by the `rows` tag, with one column per value of the `columns` tag, and one scalar per cell.

| Argument              | Type             | Meaning |
|-----------------------|------------------|---------|
| `model`               | `str`            | Component model type. |
| `variable`            | `str`            | Input or output name. |
| `rows`                | `str`            | Tag name used for row labels. |
| `columns`             | `str`            | Tag name used for column labels. |
| `temporal_aggregator` | `str`            | How to collapse each time series to a scalar. One of `'mean'` (default), `'sum'`. |
| `aggregator`          | `str` (optional) | How to combine multiple nodes that fall into the same cell. One of `'mean'` (default), `'sum'`. |
| `**constraints`       | `tag=value`      | Restrict the search to nodes matching these tags. |

### Example

```python
results.table('EmcDwc', 'totalLoad', 'cgu', 'constituent',
              temporal_aggregator='sum', aggregator='sum')
```

Each cell aggregates first across nodes (e.g. summing `totalLoad` from every catchment that shares a given `cgu` / `constituent` pair) and then through time (summing the resulting series). The result is a scalar per (`cgu`, `constituent`).

## Specifying constraints

Constraints are passed as keyword arguments and restrict the set of matched nodes. Any tag may be used; multiple constraints combine with AND. If you specify no constraints, every node using the named model matches.

```python
ts = results.time_series('Sacramento', 'runoff', 'catchment', 'mean',
                         hru='Grazing Open')
```

### When the tag name is not a Python identifier

If a tag name contains a space, starts with a digit, contains a hyphen, or is a Python keyword (such as `def`), you can't use it as a keyword argument directly. Pass a dict instead:

```python
ts = results.time_series('Sacramento', 'runoff', 'catchment', 'mean',
                         **{'Functional Unit': 'Grazing Open'})
```

This form is also handy when you want to reuse the same set of constraints across several calls:

```python
constraint = {
    'hru': 'Grazing Open',
    'constituent': 'Sediment - Fine',
}

quick_load = results.time_series('EmcDwc', 'quickLoad', 'catchment', 'mean', **constraint)
slow_load  = results.time_series('EmcDwc', 'slowLoad',  'catchment', 'mean', **constraint)
```

## Building higher-level reports

`time_series` and `table` are deliberately low-level. Real reports usually need:

* **Unit conversions** (per-second → per-day, kg → tons, etc).
* **Normalisation** (e.g. dividing a multi-year total by the number of years to get an annual mean).
* **Composition across multiple model types.** In a heterogeneous catchment model the constituent load for a given `cgu` and `constituent` may be produced by different component models in different parts of the graph. Openwater sees only nodes, tags, and connections — there's no built-in concept of "the load model." Combining such results requires calling `table` (or `time_series`) once per model type and merging.

The recommended pattern is to wrap these conversions in reusable functions:

```python
def mean_annual_loads(results, years):
    # totalLoad is kg/s averaged over each daily timestep.
    # sum through time and across nodes, then convert kg/s/day -> tons/year.
    raw = results.table('EmcDwc', 'totalLoad', 'cgu', 'constituent',
                        temporal_aggregator='sum', aggregator='sum')
    return raw * 86400 * 1e-3 / years
```

Here `years` is whatever span your simulation covers; you can compute it from `results.time_period` if you prefer to derive it automatically.
