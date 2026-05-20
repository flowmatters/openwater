
from typing import List
from . import nodes as node_types
import pandas as pd
import numpy as np
from glob import glob
import logging
logger = logging.getLogger(__name__)

temporal_agg_fns = {
  'sum':lambda a: a.sum(axis=1),
  'mean':lambda a: a.mean(axis=1)
}

agg_fns = {
    'mean':lambda a: a.mean(axis=0),
    'sum':lambda a: a.sum(axis=0)
}

def _open_h5(f):
    if hasattr(f,'filename'):
        return f
    import h5py as h5
    return h5.File(f,'r')


def _is_set_constraint(value):
    '''True if the constraint value is a non-string iterable (set-membership).'''
    if isinstance(value, str):
        return False
    return isinstance(value, (list, tuple, set, frozenset, np.ndarray))


def _allowed_indices(slice_entry, full_size):
    '''Return the list of axis indices that ``slice_entry`` permits.

    Mirrors how _retrieve_data populates the ``slices`` list: ``slice(None)``
    means unrestricted, an int means a single index, a list/array means a
    set of indices.
    '''
    if isinstance(slice_entry, slice):
        return list(range(full_size))
    if isinstance(slice_entry, (list, np.ndarray)):
        return list(slice_entry)
    return [int(slice_entry)]


def _regroup_axis(df, axis, level_projections, output_names, aggregator):
    '''Apply per-level projections to one axis of ``df``, then group+aggregate.

    Parameters
    ----------
    df : DataFrame
    axis : 0 (index) or 1 (columns)
    level_projections : dict[int, QuasiDimension]
        Levels to project, each via a quasi-dim's composed mapping.
    output_names : list[str]
        New name(s) for the axis after projection (in level order).
    aggregator : str
        'sum' or 'mean'. Applied across rows/cols that collide post-projection.
    '''
    cur_axis = df.axes[axis]
    if isinstance(cur_axis, pd.MultiIndex):
        nlevels = cur_axis.nlevels
        new_tuples = []
        for tpl in cur_axis:
            new = list(tpl)
            for lvl, qd in level_projections.items():
                projected = qd.project(pd.Series([tpl[lvl]]))
                new[lvl] = projected.iloc[0]
            new_tuples.append(tuple(new))
        new_axis = pd.MultiIndex.from_tuples(new_tuples, names=output_names)
    else:
        qd = level_projections[0]
        projected = qd.project(pd.Series(list(cur_axis)))
        new_axis = pd.Index(projected.tolist(), name=output_names[0])

    df = df.copy()
    if axis == 0:
        df.index = new_axis
        if isinstance(new_axis, pd.MultiIndex):
            return df.groupby(level=list(range(new_axis.nlevels))).agg(aggregator)
        return df.groupby(level=0).agg(aggregator)
    else:
        df.columns = new_axis
        trans = df.T
        if isinstance(new_axis, pd.MultiIndex):
            grouped = trans.groupby(level=list(range(new_axis.nlevels))).agg(aggregator)
        else:
            grouped = trans.groupby(level=0).agg(aggregator)
        return grouped.T


def _index_run_map(run_map, slices):
    '''Index run_map with a tuple of slice / int / list entries.

    For zero or one advanced (list) index, plain ``run_map[tuple(slices)]``
    gives correct cartesian-style indexing. For two or more list entries,
    NumPy would broadcast them pairwise instead — so we widen all entries
    via ``np.ix_`` to force the cartesian product.
    '''
    n_advanced = sum(1 for s in slices if isinstance(s, (list, np.ndarray)))
    if n_advanced < 2:
        return run_map[tuple(slices)]

    arrays = []
    for dim_size, sl in zip(run_map.shape, slices):
        if isinstance(sl, slice):
            arrays.append(np.arange(dim_size))
        elif isinstance(sl, (list, np.ndarray)):
            arrays.append(np.asarray(sl))
        else:  # int (or numpy scalar)
            arrays.append(np.array([sl]))
    return run_map[np.ix_(*arrays)]

class OpenwaterResults(object):
  def  __init__(self,model,res_file,time_period=None,inputs=None):
    self.model = _open_h5(model)
    self.results = _open_h5(res_file)
    if inputs is None:
      self.inputs = self.model
    else:
      self.inputs = _open_h5(inputs)

    tp = self._read_time_period()
    if tp is not None and time_period is not None:
      logger.warning('Time period found in results metadata (%s), but time_period argument also provided. Using time period from metadata.',res_file)
    self.time_period = tp if tp is not None else time_period
    if self.time_period is None:
      logger.warning('No time period found in results metadata (%s), and no time_period argument provided. Time series results will not have a time index.',res_file)
    self._dimensions={}
    # Rehydrate persisted quasi-dims from the model file (if any). Real dims
    # are the keys under /DIMENSIONS.
    from . import quasi_dim as _qd
    self._quasi_dims = _qd.QuasiDimRegistry(
      real_dim_names=lambda: set(self.dims())
    )
    if 'META' in self.model:
      self._quasi_dims.load_from_h5(self.model['META'])

  def quasi_dims(self):
    return self._quasi_dims.names()

  def quasi_dim(self, name):
    return self._quasi_dims[name]

  @property
  def quasi_dim_registry(self):
    return self._quasi_dims

  def add_quasi_dim(self, source, *, name=None, keyed_by=None,
                    key=None, value=None, default=None):
    '''Register a quasi-dimension for use with this results view.

    Shares dispatch logic with ``ModelGraph.add_quasi_dim``. The new
    quasi-dim is session-only — it is not written to the model file, which
    avoids surprising mutation of the model file via the results object. If
    you want a quasi-dim to persist, register it on the ``ModelGraph`` (at
    model-build time) or via ``ModelFile.add_quasi_dim(..., persist=True)``.
    '''
    from . import quasi_dim as _qd
    return _qd.add_to_registry(self._quasi_dims, source,
                               name=name, keyed_by=keyed_by,
                               key=key, value=value, default=default,
                               persist=False)

  def remove_quasi_dim(self, name):
    '''Remove a registered quasi-dimension from this results view.

    Persisted quasi-dims (those loaded from the model file) can also be
    removed — the removal is session-only and does not write back to disk.
    '''
    self._quasi_dims.remove(name)

  def _resolver(self):
    from .quasi_dim import QuasiDimResolver
    return QuasiDimResolver(
      real_dim_names=lambda: set(self.dims()),
      registry=self._quasi_dims,
    )

  def _read_time_period(self):
    if 'META' in self.model and 'timeperiod' in self.model['META']:
      raw = [d for d in self.model['META']['timeperiod'][...]]
      if isinstance(raw[0], bytes):
        raw = [d.decode() for d in raw]
      return pd.DatetimeIndex([pd.Timestamp.fromisoformat(d) for d in raw])
    return None

  def close(self):
      self.results.close()
      self.model.close()
      if self.model != self.inputs:
        self.inputs.close()


  def dim(self,dim:str)->List:
    if not dim in self._dimensions:
      vals = list(self.model['/DIMENSIONS'][dim][...])
      conv = lambda v: v.decode('utf-8') if hasattr(v,'decode') else v
      vals = [conv(v) for v in vals]
      self._dimensions[dim] = vals

    return self._dimensions[dim]

  def dims(self) -> List[str]:
      vals = list(self.model['/DIMENSIONS'].keys())
      return vals

  def _retrieve_all(self,model,variable):
    desc = getattr(node_types,model)
    is_input = variable in desc.description['Inputs']

    grp_name = '/MODELS/%s'%model
    out_grp = self.results[grp_name]

    if is_input:
      var_idx = desc.description['Inputs'].index(variable)
      if 'inputs' in out_grp:
        dataset = out_grp['inputs']
      else:
        in_grp = self.inputs[grp_name]
        dataset = in_grp['inputs']
    else:
      var_idx = desc.description['Outputs'].index(variable)
      dataset = out_grp['outputs']

    data = dataset[:,var_idx,:]
    assert len(data.shape)==2
    return data

  def _map_runs(self,model):
    map_grp = '/MODELS/%s/map'%model
    dim_names = self.dims_for_model(model)
    dims = {d:self.dim(d) for d in dim_names}

    run_map = self.model[map_grp][...]
    return dim_names, dims, run_map

  def _model_name(self,model):
    if hasattr(model,'name'):
        return model.name
    return model

  def _retrieve_data(self,model,model_variable,**kwargs):
    model = self._model_name(model)
    data = self._retrieve_all(model,model_variable)
    dim_names, dims, run_map = self._map_runs(model)
    slices = [slice(None,None,None) for _ in dim_names]
    for dim_name,dim_value in kwargs.items():
      if not dim_name in dim_names:
        map_grp = '/MODELS/%s/map'%model
        fixed_value = self.model[map_grp].attrs.get(dim_name,None)
        # A fixed (single-valued) dim still has to satisfy the constraint —
        # which means either equality (scalar) or membership (set).
        if _is_set_constraint(dim_value):
          if fixed_value not in dim_value:
            raise Exception('Invalid dimension: %s=%s'%(dim_name,dim_value))
        elif fixed_value != dim_value:
            raise Exception('Invalid dimension: %s=%s'%(dim_name,dim_value))
        continue

      dim_num = dim_names.index(dim_name)
      if _is_set_constraint(dim_value):
        dim_values_list = dims[dim_name]
        try:
          idx_list = [dim_values_list.index(v) for v in dim_value]
        except ValueError as e:
          raise Exception(
            'Invalid value for dimension %s in set constraint: %s'%(dim_name,e))
        # Preserve order (matches user input order); leave duplicates alone —
        # later flattening + non-negative filter handles them.
        slices[dim_num] = idx_list
      else:
        dim_idx = dims[dim_name].index(dim_value)
        slices[dim_num] = dim_idx
    return dim_names, dims, run_map, slices, data

  def time_series(self,model,variable:str,columns,aggregator=None,filter_tags={},**kwargs) -> pd.DataFrame:
    '''
    Return a table (DataFrame) of time series results from the model.

    Parameters:

    * model - the model of interest
    * variable - a variable on the model, either an input or an output
    * columns - a dimension (or list/tuple of dimensions) of the model to use as the columns of the DataFrame.
                When multiple dimensions are provided, the resulting DataFrame will have a MultiIndex on the columns.
    * aggregator - a function name (string) to apply when more than one data series matches a particular column (eg 'mean')
    * **kwargs - used to specify other dimensions to filter by

    For aggregator, see agg_fns.keys()

    For dimensions (row, columns and kwargs), see dims_for_model
    '''
    overlap = set(kwargs).intersection(filter_tags)
    if overlap:
      raise ValueError('Tag(s) %s supplied via both filter_tags and kwargs'%sorted(overlap))
    kwargs.update(filter_tags)

    # Phase 3: expand quasi-dim constraints to real-dim equivalents, and
    # swap quasi-dim entries in `columns` for the real dim they resolve to.
    # Projections are kept so we can re-group the resulting DataFrame's
    # columns back to quasi-dim values.
    resolver = self._resolver()
    kwargs = resolver.resolve_constraints(kwargs)

    if isinstance(columns, str):
      columns = [columns]
    original_columns = list(columns)
    quasi_levels = {}
    real_columns = []
    for i, c in enumerate(original_columns):
      if resolver.is_quasi(c):
        composed = resolver.project_index(c)
        real_columns.append(composed.keyed_by)
        quasi_levels[i] = composed
      else:
        real_columns.append(c)
    columns = real_columns

    dim_names, dims, run_map, slices, data = self._retrieve_data(model,variable,**kwargs)

    multi = len(columns) > 1

    report_dims = [dim_names.index(c) for c in columns]
    # Restrict the report-dim iteration to indices permitted by any constraint
    # on that dim — otherwise the report loop's per-iteration assignment to
    # current_slices[rd] would clobber a constraint on the same dim.
    report_pairs = []
    for c, rd in zip(columns, report_dims):
      idxs = _allowed_indices(slices[rd], run_map.shape[rd])
      vals = [dims[c][i] for i in idxs]
      report_pairs.append(list(zip(idxs, vals)))

    import itertools
    all_sequences = {}
    found_match=False
    for combo in itertools.product(*report_pairs):
      indices = [idx for idx, _ in combo]
      names = tuple(name for _, name in combo)

      current_slices = slices[:]
      for rd, idx in zip(report_dims, indices):
        current_slices[rd] = idx
      run_indices = _index_run_map(run_map, current_slices)
      run_indices = run_indices.flatten()
      run_indices = run_indices[run_indices>=0]
      col_data = data[run_indices,:]
      col_key = names if multi else names[0]
      if col_data.shape[0]==1:
        all_sequences[col_key] = col_data[0,:]
        found_match=True
      elif col_data.shape[0]>1:
        all_sequences[col_key] = agg_fns[aggregator or 'mean'](col_data)
        found_match=True

    if not found_match:
      raise Exception(f'No matching model nodes for model {model}, with column tag {columns} and constraint tags {kwargs}.')

    result = pd.DataFrame(all_sequences,index=self.time_period)
    if multi:
      result.columns = pd.MultiIndex.from_tuples(result.columns, names=columns)

    if quasi_levels:
      result = _regroup_axis(result, axis=1,
                             level_projections=quasi_levels,
                             output_names=original_columns,
                             aggregator=aggregator or 'mean')
    return result

  def all_time_series(self,model,model_variable,**kwargs) -> pd.DataFrame:
    '''
    Return a table (DataFrame) of time series results from the model with multi-level columns representing all tags for the model
    '''
    dim_names, dims, run_map, slices, data = self._retrieve_data(model,model_variable,**kwargs)
    r = {}
    for run_map_coords in (zip(*np.where(run_map>-1))):
        tags = tuple([dims[dn][ix] for dn,ix in zip(dim_names,run_map_coords)])
        wanted = True
        for k,v in kwargs.items():
          present = tags[dim_names.index(k)]
          if _is_set_constraint(v):
            if present not in v:
              wanted = False
              break
          elif present != v:
            wanted = False
            break
        if not wanted:
          continue

        run_index = run_map[run_map_coords]
        r[tags] = data[run_index,:]
    r = pd.DataFrame(r,index=self.time_period)
    r.columns.set_names(dim_names,inplace=True)
    return r

  def table(self,model,variable:str,rows:str,columns:str,temporal_aggregator:str='mean',aggregator:str=None,**kwargs) -> pd.DataFrame:
    '''
    Return a table (DataFrame) of aggregated model results from the model.

    Parameters:

    * model - the model of interest
    * variable - a variable on the model, either an input or an output
    * row - a dimension of the model to use as the rows of the DataFrame
    * columns - a dimension of the model to use as the columns of the DataFrame
    * temporal_aggregator - a function name (string) to reduce the timeseries results to a single value (default='mean')
    * aggregator - a function name (string) to apply when more than one data series matches a particular row/column (eg 'mean')
    * **kwargs - used to specify other dimensions to filter by

    For temporal_aggregator, see temporal_agg_fns.keys()

    For aggregator, see agg_fns.keys()

    For dimensions (row, columns and kwargs), see dims_for_model
    '''
    # Phase 3: expand quasi-dim constraints + swap quasi rows/columns for
    # their real-dim equivalents, then re-group the resulting DataFrame.
    resolver = self._resolver()
    kwargs = resolver.resolve_constraints(kwargs)
    original_rows, original_columns = rows, columns
    rows_qd = resolver.project_index(rows) if resolver.is_quasi(rows) else None
    cols_qd = resolver.project_index(columns) if resolver.is_quasi(columns) else None
    if rows_qd is not None:
      rows = rows_qd.keyed_by
    if cols_qd is not None:
      columns = cols_qd.keyed_by
    if rows == columns:
      raise ValueError(
        f"rows and columns resolve to the same real dimension {rows!r}; "
        "this is not supported"
      )

    dim_names, dims, run_map, slices, data = self._retrieve_data(model,variable,**kwargs)
    data = temporal_agg_fns[temporal_aggregator](data)

    col_dim = dim_names.index(columns)
    row_dim = dim_names.index(rows)

    # Restrict iteration to indices allowed by any constraint on rows/cols.
    col_idxs = _allowed_indices(slices[col_dim], run_map.shape[col_dim])
    row_idxs = _allowed_indices(slices[row_dim], run_map.shape[row_dim])
    column_names = [dims[columns][i] for i in col_idxs]
    row_names = [dims[rows][j] for j in row_idxs]

    table_data = {}
    for col_pos, (i, col_name) in enumerate(zip(col_idxs, column_names)):
      col_data = []
      for j in row_idxs:
        current_slices = slices[:]
        current_slices[col_dim] = i
        current_slices[row_dim] = j
        run_indices = _index_run_map(run_map, current_slices).flatten()
        cell_data = data[run_indices]
        if cell_data.shape[0]==1:
          col_data.append(cell_data[0])
        else:
          col_data.append(agg_fns[aggregator or 'mean'](cell_data))
      table_data[col_name] = col_data
    df = pd.DataFrame(table_data,index=row_names)
    if rows_qd is not None:
      df = _regroup_axis(df, axis=0,
                         level_projections={0: rows_qd},
                         output_names=[original_rows],
                         aggregator=aggregator or 'mean')
    if cols_qd is not None:
      df = _regroup_axis(df, axis=1,
                         level_projections={0: cols_qd},
                         output_names=[original_columns],
                         aggregator=aggregator or 'mean')
    return df

  def models(self) -> List[str]:
    return list(self.model['/MODELS'].keys())

  def variables_for(self,model) -> List[str]:
    if hasattr(model,'name'):
        desc = model
    else:
        desc = getattr(node_types,model)
    return desc.description['Inputs'] + desc.description['Outputs']

  def dims_for_model(self,model) -> List[str]:
    model = self._model_name(model)
    map_grp = '/MODELS/%s/map'%model
    if not map_grp in self.model:
        raise Exception('Missing model type: %s'%model)
    return [d.decode('utf-8') for d in self.model[map_grp].attrs['DIMS']]

class OpenwaterSplitResults(object):
  def  __init__(self,splits,time_period=None):
    assert len(splits)
    if isinstance(splits[0],tuple):
      self._results = [OpenwaterResults(model,res) for (model,res) in splits]
    else:
      self._results = splits
    self.time_period = self._results[0].time_period or time_period

  def close(self):
    for split in self._results:
      split.close()

  def dim(self,dim:str)->List:
    return self._results[0].dim(dim)

  def dims(self) -> List[str]:
    return self._results[0].dims()

  def time_series(self,model,variable:str,columns:str,aggregator=None,**kwargs) -> pd.DataFrame:
    '''
    Return a table (DataFrame) of time series results from the model.

    Parameters:

    * model - the model of interest
    * variable - a variable on the model, either an input or an output
    * columns - a dimension of the model to use as the columns of the DataFrame
    * aggregator - a function name (string) to apply when more than one data series matches a particular column (eg 'mean')
    * **kwargs - used to specify other dimensions to filter by

    For aggregator, see agg_fns.keys()

    For dimensions (row, columns and kwargs), see dims_for_model
    '''
    all_dfs = [split.time_series(model,variable,columns,aggregator,**kwargs) for split in self._results]
    concat = pd.concat(all_dfs)
    result = concat.set_index(self.time_period)
    return result

  def table(self,model,variable:str,rows:str,columns:str,temporal_aggregator:str='mean',aggregator:str=None,**kwargs) -> pd.DataFrame:
    '''
    Return a table (DataFrame) of aggregated model results from the model,
    combining results across all time splits.

    For temporal splits that partition the time axis:
    - 'sum' aggregator: tables from each split are summed (sums are additive)
    - 'mean' aggregator: tables are combined as a weighted mean by timestep count

    Parameters:

    * model - the model of interest
    * variable - a variable on the model, either an input or an output
    * rows - a dimension of the model to use as the rows of the DataFrame
    * columns - a dimension of the model to use as the columns of the DataFrame
    * temporal_aggregator - a function name (string) to reduce the timeseries results to a single value (default='mean')
    * aggregator - a function name (string) to apply when more than one data series matches a particular row/column (eg 'mean')
    * **kwargs - used to specify other dimensions to filter by

    For temporal_aggregator, see temporal_agg_fns.keys()

    For aggregator, see agg_fns.keys()

    For dimensions (row, columns and kwargs), see dims_for_model
    '''
    split_tables = [s.table(model, variable, rows, columns, temporal_aggregator, aggregator, **kwargs)
                    for s in self._results]
    if temporal_aggregator == 'sum':
      return sum(split_tables)
    elif temporal_aggregator == 'mean':
      weights = [len(s.time_period) for s in self._results]
      total = sum(weights)
      return sum(t * w for t, w in zip(split_tables, weights)) / total
    else:
      raise ValueError(f'Unsupported temporal_aggregator for split results: {temporal_aggregator}')

  def models(self) -> List[str]:
    return self._results[0].models()

  def variables_for(self,model) -> List[str]:
    return self._results[0].variables_for(model)

  def dims_for_model(self,model) -> List[str]:
    return self._results[0].dims_for_model(model)

def open_split_results(model_fn,results_pattern,input_pattern=None,time_period=None):
  results_filenames = list(sorted(glob(results_pattern)))
  if input_pattern is not None:
    inputs_filenames = list(sorted(glob(input_pattern)))
    assert len(inputs_filenames) == len(results_filenames)
  else:
    inputs_filenames = None

  individual_result_objects = [OpenwaterResults(model_fn,res_file,inputs=None if inputs_filenames is None else inputs_filenames[ix])\
                               for ix,res_file in enumerate(results_filenames)]
  return OpenwaterSplitResults(individual_result_objects,time_period=time_period)
