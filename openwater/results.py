
from typing import List
from . import nodes as node_types
from . import stamps
import pandas as pd
import numpy as np
import re
from glob import glob
import logging
logger = logging.getLogger(__name__)

temporal_agg_fns = {
  'sum':lambda a: a.sum(axis=1),
  'mean':lambda a: a.mean(axis=1),
  'min':lambda a: a.min(axis=1),
  'max':lambda a: a.max(axis=1)
}

PERCENTILE_PATTERN = re.compile(r'^p(\d{1,3}(?:\.\d+)?)$')

def resolve_temporal_agg(name):
  '''
  Resolve a temporal aggregator name to a function over a [runs, time] array.

  Supports the named aggregators in temporal_agg_fns plus percentiles
  expressed as 'p<q>' (eg 'p10', 'p99.5'). Raises KeyError for unknown names.
  '''
  if name in temporal_agg_fns:
    return temporal_agg_fns[name]
  m = PERCENTILE_PATTERN.match(name or '')
  if m:
    q = float(m.group(1))
    if q > 100:
      raise KeyError('Unknown temporal aggregator: %s'%name)
    return lambda a: np.percentile(a, q, axis=1)
  raise KeyError('Unknown temporal aggregator: %s'%name)

TEMPORAL_GROUPINGS = ('year','month','water_year','month_of_year')

def _grouping_keys(index,temporal_grouping,water_year_start=7):
  '''
  Map a DatetimeIndex to group labels for a temporal grouping.

  Water years are labelled by their starting calendar year (water year 2020
  with water_year_start=7 covers 2020-07-01 to 2021-06-30).
  '''
  if temporal_grouping == 'year':
    return index.year
  if temporal_grouping == 'month':
    return index.to_period('M').astype(str)
  if temporal_grouping == 'water_year':
    return np.where(index.month >= water_year_start,index.year,index.year-1)
  if temporal_grouping == 'month_of_year':
    return index.month
  raise ValueError('Unknown temporal grouping: %s'%temporal_grouping)

def _df_temporal_agg(df_or_grouped,name):
  '''Apply a temporal aggregator by name to a DataFrame or GroupBy (time on axis 0).'''
  if name in ('sum','mean','min','max'):
    return getattr(df_or_grouped,name)()
  m = PERCENTILE_PATTERN.match(name or '')
  if m:
    q = float(m.group(1))
    if q > 100:
      raise KeyError('Unknown temporal aggregator: %s'%name)
    return df_or_grouped.quantile(q/100.0)
  raise KeyError('Unknown temporal aggregator: %s'%name)

def _grouped_table_from_timeseries(ts,temporal_grouping,temporal_aggregator,water_year_start=7):
  keys = _grouping_keys(ts.index,temporal_grouping,water_year_start)
  return _df_temporal_agg(ts.groupby(keys),temporal_aggregator)

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

# Model reporting. Model kernels emit a report code plus up to three numbers,
# with no idea which subcatchment or date they represent; the translation back
# to places and dates happens here, using the model file's metadata. See
# openwater-core/docs/MODEL_REPORTING.md.

REPORTS_GROUP = '/REPORTS'
MAX_REPORT_ARGS = 3

SEVERITY_LABELS = {0: 'none', 1: 'warning', 2: 'error'}
SEVERITY_VALUES = {label: value for value, label in SEVERITY_LABELS.items()}

REPORT_COLUMNS = ['date', 'timestep', 'model', 'severity', 'code', 'report', 'message']
REPORT_COUNT_COLUMNS = ['model', 'severity', 'code', 'report', 'occurrences']


def _decode(value):
  return value.decode('utf-8') if hasattr(value, 'decode') else value


def _severity_threshold(severity):
  '''Lowest severity value to include. None or "all" means everything.'''
  if severity is None or severity == 'all':
    return 0
  if severity in SEVERITY_VALUES:
    return SEVERITY_VALUES[severity]
  raise ValueError('Unknown severity %s. Expected one of %s'
                   % (severity, sorted(SEVERITY_VALUES) + ['all']))


def _render_report(template, args, n_args):
  '''Substitute {0}, {1} and {2} in a report template.

  Numbers are formatted the way the engine formats them, so a message reads the
  same whether it came from ow-single or from a results file.
  '''
  message = template
  for i in range(min(int(n_args), MAX_REPORT_ARGS)):
    message = message.replace('{%d}' % i, '%.6g' % args[i])
  return message


class VariableMismatchError(Exception):
  '''A results file's variables can't be matched to the loaded model descriptions.'''


# (results signature, library signature) pairs already warned about, so that
# opening the many files of a split run says it once.
_warned_signatures = set()


class OpenwaterResults(object):
  def  __init__(self,model,res_file,time_period=None,inputs=None,allow_variable_mismatch=False):
    '''
    Open the results of a model run.

    allow_variable_mismatch: Results files written before variable names were
      recorded are read by position, against the model descriptions of the
      loaded openwater-core. When a file has a different number of variables
      than the loaded description, positions can't be trusted and reading
      raises VariableMismatchError. Pass True to read by position anyway.
    '''
    self.model = _open_h5(model)
    self.results = _open_h5(res_file)
    if inputs is None:
      self.inputs = self.model
    else:
      self.inputs = _open_h5(inputs)
    self.allow_variable_mismatch = allow_variable_mismatch
    self._check_stamp()

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

  def _time_selection(self,time_period=None,months=None):
    '''
    Return (index, mask) selecting a subset of the time axis.

    * time_period - optional (start, end) tuple; either end may be None (open)
    * months - optional list of month numbers (1-12) to keep

    Returns (self.time_period, None) when no subsetting is requested.
    '''
    if time_period is None and months is None:
      return self.time_period, None
    if self.time_period is None:
      raise ValueError('Cannot subset results that have no time index')
    idx = self.time_period
    mask = np.ones(len(idx),dtype=bool)
    if time_period is not None:
      start,end = time_period
      if start is not None:
        mask &= (idx >= pd.Timestamp(start))
      if end is not None:
        mask &= (idx <= pd.Timestamp(end))
    if months is not None:
      mask &= np.isin(idx.month,list(months))
    return idx[mask], mask

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

  def version(self):
    '''The version stamp ow-sim wrote to the results file, as a dict.

    Keys are ``openwater_version``, ``signature_hash``, ``created_by`` and
    ``created_timestamp`` for the build that produced the results, and
    ``model_openwater_version`` and ``model_signature_hash`` for the model
    file they were produced from. Empty for results written before stamping.
    '''
    return stamps.read_stamp(self.results)

  def _check_stamp(self):
    '''Warn, once, when the results were written by a build with different model interfaces.

    Results that record their variable names are still read correctly. Older
    ones are read by position against the loaded library's descriptions, and
    the stamp is the only sign that those positions may have moved.
    '''
    stamp = stamps.read_stamp(self.results)
    file_signature = stamp.get(stamps.SIGNATURE_ATTR)
    if file_signature in (None, 'unknown'):
      return

    try:
      from . import lib
      current_signature = lib.get_core_signature_hash()
      current_version = lib.get_core_version()
    except Exception as e:
      logger.debug('Could not check results version: %s', e)
      return

    if current_signature in (None, 'unknown') or current_signature == file_signature:
      return

    key = (file_signature, current_signature)
    if key in _warned_signatures:
      return
    _warned_signatures.add(key)

    filename = getattr(self.results, 'filename', 'the results file')
    logger.warning(
      'Results signature mismatch!\n'
      '  Results written by: %s\n'
      '  Current version: %s\n'
      '  Model variables may have been added, removed or reordered between the two. '
      'Variables are read by name where the results file records them, and by position '
      'against the current version otherwise. To read with the matching release, use '
      "openwater.releases.use_for_model('%s', install=True).",
      stamp.get(stamps.VERSION_ATTR, file_signature), current_version, filename)

  def _variable_datasets(self,model):
    '''The datasets a model's variables can come from, as {kind: dataset}.

    Inputs are read from the results file when ow-sim was asked to write them
    there, and from the inputs (or model) file otherwise.
    '''
    grp_name = '/MODELS/%s'%model
    out_grp = self.results[grp_name]

    datasets = {}
    if 'outputs' in out_grp:
      datasets['outputs'] = out_grp['outputs']
    if 'inputs' in out_grp:
      datasets['inputs'] = out_grp['inputs']
    elif grp_name in self.inputs and 'inputs' in self.inputs[grp_name]:
      datasets['inputs'] = self.inputs[grp_name]['inputs']
    return datasets

  def _described_variables(self,model):
    '''Variable names from the loaded openwater-core, as {kind: names}, or {} if it doesn't know the model.'''
    from .nodes import _flux_names
    desc = getattr(node_types,model,None)
    if desc is None:
      return {}
    return {
      'inputs': _flux_names(desc.description['Inputs']),
      'outputs': _flux_names(desc.description['Outputs']),
    }

  def _locate_variable(self,model,variable):
    '''Find the dataset and position holding a variable.

    Names recorded in the file win: they say what is actually stored where,
    whichever build wrote it. A dataset without names is read by position
    against the loaded library's description, which is only safe when that
    description still matches the file.
    '''
    datasets = self._variable_datasets(model)
    recorded = {kind: stamps.variable_names(ds,kind) for kind,ds in datasets.items()}

    for kind in ['inputs','outputs']:
      names = recorded.get(kind)
      if names is not None and variable in names:
        return datasets[kind], names.index(variable)

    described = self._described_variables(model)
    for kind in ['inputs','outputs']:
      if variable not in described.get(kind,[]):
        continue
      if kind not in datasets:
        raise Exception('No %s recorded for %s'%(kind,model))
      if recorded.get(kind) is not None:
        # The file says what it holds, and this variable isn't in it.
        continue

      names = described[kind]
      dataset = datasets[kind]
      if dataset.shape[1] != len(names):
        message = ('%s %s in %s has %d variables, but the model description has %d (%s). '
                   'The file was probably written by a different openwater-core build, so %s '
                   'would be read from the wrong position.'%(
                     model, kind, getattr(dataset.file,'filename','the file'), dataset.shape[1],
                     len(names), ', '.join(names), variable))
        if not self.allow_variable_mismatch:
          raise VariableMismatchError(
            message + ' Activate the release that wrote the file '
            '(openwater.releases.use_for_model), or open the results with '
            'allow_variable_mismatch=True to read by position anyway.')
        logger.warning(message)
      return dataset, names.index(variable)

    available = []
    for kind in ['inputs','outputs']:
      names = recorded.get(kind)
      available += names if names is not None else described.get(kind,[])
    raise Exception('Unknown variable %s for %s. Available: %s'%(variable,model,', '.join(available)))

  def _retrieve_all(self,model,variable):
    dataset, var_idx = self._locate_variable(model,variable)
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

  def time_series(self,model,variable:str=None,columns=None,aggregator=None,filter_tags={},time_period=None,months=None,**kwargs) -> pd.DataFrame:
    '''
    Return a table (DataFrame) of time series results from the model.

    Parameters:

    * model - the model of interest
    * variable - a variable on the model, either an input or an output
               - if None, other query parameters should be set to return a single time series,
               - and this function will then return all variables in a single dataframe.
    * columns - a dimension (or list/tuple of dimensions) of the model to use as the columns of the DataFrame.
                When multiple dimensions are provided, the resulting DataFrame will have a MultiIndex on the columns.
    * aggregator - a function name (string) to apply when more than one data series matches a particular column (eg 'mean')
    * time_period - optional (start, end) tuple to subset the time axis (either end may be None)
    * months - optional list of month numbers (1-12) to keep
    * **kwargs - used to specify other dimensions to filter by

    For aggregator, see agg_fns.keys()

    For dimensions (row, columns and kwargs), see dims_for_model
    '''
    overlap = set(kwargs).intersection(filter_tags)
    if overlap:
      raise ValueError('Tag(s) %s supplied via both filter_tags and kwargs'%sorted(overlap))
    kwargs.update(filter_tags)

    if variable is None:
      variables = self.variables_for(model)
      if len(kwargs):
        col = list(kwargs.keys())[0]
      else:
        col = None
      all_dfs = {v:self.time_series(model,v,col,aggregator,time_period=time_period,months=months,**kwargs).rename(columns=lambda _:v) for v in variables}
      return pd.concat(all_dfs.values(),axis=1)

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
    index, mask = self._time_selection(time_period,months)
    if mask is not None:
      data = data[:,mask]

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

    result = pd.DataFrame(all_sequences,index=index)
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
    columns = pd.MultiIndex.from_tuples(list(r.keys()),names=dim_names) if r else \
              pd.MultiIndex.from_arrays([[]]*len(dim_names),names=dim_names)
    return pd.DataFrame({k:r[k] for k in r},index=self.time_period,columns=columns)

  def table(self,model,variable:str,rows:str,columns:str,temporal_aggregator:str='mean',aggregator:str=None,time_period=None,months=None,**kwargs) -> pd.DataFrame:
    '''
    Return a table (DataFrame) of aggregated model results from the model.

    Parameters:

    * model - the model of interest
    * variable - a variable on the model, either an input or an output
    * row - a dimension of the model to use as the rows of the DataFrame
    * columns - a dimension of the model to use as the columns of the DataFrame
    * temporal_aggregator - a function name (string) to reduce the timeseries results to a single value (default='mean').
                            Accepts named aggregators ('mean', 'sum', 'min', 'max') and percentiles expressed as 'p<q>' (eg 'p50', 'p99.5').
    * aggregator - a function name (string) to apply when more than one data series matches a particular row/column (eg 'mean')
    * time_period - optional (start, end) tuple to subset the time axis (either end may be None)
    * months - optional list of month numbers (1-12) to keep
    * **kwargs - used to specify other dimensions to filter by

    For named temporal aggregators, see temporal_agg_fns.keys(); percentiles use 'p<q>' notation.

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
    _, mask = self._time_selection(time_period,months)
    if mask is not None:
      data = data[:,mask]
    data = resolve_temporal_agg(temporal_aggregator)(data)

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

  def grouped_table(self,model,variable:str,columns,temporal_grouping:str,temporal_aggregator:str='mean',aggregator:str=None,time_period=None,months=None,water_year_start:int=7,**kwargs) -> pd.DataFrame:
    '''
    Return a table (DataFrame) of model results aggregated to temporal groups.

    Rows are temporal groups (one per year/month/water year/month-of-year),
    columns are the values of the `columns` dimension.

    Parameters:

    * model, variable, columns, aggregator, **kwargs - as for time_series
    * temporal_grouping - one of TEMPORAL_GROUPINGS
    * temporal_aggregator - reduction within each group ('mean','sum','min','max','p<q>')
    * time_period, months - optional subsetting applied before grouping
    * water_year_start - first month of the water year (default 7);
                         water years are labelled by their starting year
    '''
    ts = self.time_series(model,variable,columns,aggregator,time_period=time_period,months=months,**kwargs)
    return _grouped_table_from_timeseries(ts,temporal_grouping,temporal_aggregator,water_year_start)

  def models(self) -> List[str]:
    return list(self.model['/MODELS'].keys())

  def variables_for(self,model) -> List[str]:
    '''The input and output variables available for a model.

    Taken from the names recorded in the files where there are any, and from
    the loaded openwater-core's description of the model otherwise.
    '''
    model = self._model_name(model)
    described = self._described_variables(model)
    datasets = self._variable_datasets(model) if '/MODELS/%s'%model in self.results else {}

    result = []
    for kind in ['inputs','outputs']:
      names = stamps.variable_names(datasets[kind],kind) if kind in datasets else None
      result += names if names is not None else described.get(kind,[])
    return result

  def dims_for_model(self,model) -> List[str]:
    model = self._model_name(model)
    map_grp = '/MODELS/%s/map'%model
    if not map_grp in self.model:
        raise Exception('Missing model type: %s'%model)
    return [d.decode('utf-8') for d in self.model[map_grp].attrs['DIMS']]

  # --- model reports -------------------------------------------------------

  def _reporting_models(self) -> List[str]:
    if REPORTS_GROUP not in self.results:
      return []
    group = self.results[REPORTS_GROUP]
    return sorted(name for name in group if hasattr(group[name], 'keys'))

  def _check_report_signature(self):
    '''Reports are only meaningful against the model file that produced them.

    Run indices and timesteps are bare integers: ``clip`` renumbers run indices
    and ``ModelFile.retime`` rewrites ``META/timeperiod``, either of which would
    silently re-locate or re-date every report. A wrong catchment name is worse
    than no catchment name, so a mismatch raises.
    '''
    stamped = stamps.read_stamp(self.results).get(stamps.MODEL_SIGNATURE_ATTR)
    if stamped is None:
      # Unstamped results, or produced from an unstamped model file: there is
      # nothing to compare.
      return

    file_signature = self.model.attrs.get(stamps.SIGNATURE_ATTR)
    if file_signature is None:
      logger.warning('Results carry a model signature (%s) but the model file records none; '
                     'cannot confirm the reports belong to this model.', stamped)
      return

    file_signature = _decode(file_signature)
    if file_signature != stamped:
      raise Exception(
        'Model reports were produced from a model file with signature %s, but this model file has %s. '
        'Run indices and timesteps would be mapped to the wrong places and dates. '
        'Pass check_signature=False to read them anyway.'%(stamped, file_signature))

  def _report_catalog(self, model):
    '''The model's report catalogue, read from the results file.

    Deliberately not read from the installed library: an old results file stays
    readable against a newer build, which is why ow-sim writes it alongside the
    reports.
    '''
    grp = self.results['%s/%s'%(REPORTS_GROUP, model)]
    names = [_decode(n) for n in grp['names'][...]]
    messages = [_decode(m) for m in grp['messages'][...]]
    if 'code_severity' in grp:
      severities = [int(s) for s in grp['code_severity'][...]]
    else:
      severities = [SEVERITY_VALUES['warning']] * len(names)
    return names, messages, severities

  def _inverse_run_map(self, model):
    '''Map each global run index back to its position in every dimension.

    Returns (dim_names, dim_labels, inverse) where inverse[run] is the tuple of
    dimension indices for that run, or (None, None, None) when the model file
    cannot say -- reports are still usable then, just without place names.
    '''
    try:
      dim_names, dims, run_map = self._map_runs(model)
    except Exception as e:
      logger.warning('No run map for %s (%s); reports will not carry dimension labels.', model, e)
      return None, None, None

    run_map = np.asarray(run_map)
    if run_map.size == 0:
      return dim_names, dims, None

    flat = run_map.ravel()
    coords = np.stack(np.unravel_index(np.arange(flat.size), run_map.shape), axis=1)
    inverse = np.full((int(flat.max()) + 1, run_map.ndim), -1, dtype=int)
    valid = flat >= 0
    inverse[flat[valid]] = coords[valid]
    return dim_names, dims, inverse

  def _dimension_columns(self, model, runs):
    '''Dimension label columns for a set of run indices.'''
    dim_names, dims, inverse = self._inverse_run_map(model)
    if inverse is None or dim_names is None:
      return {}

    columns = {}
    for axis, dim_name in enumerate(dim_names):
      labels = dims[dim_name]
      values = []
      for run in runs:
        if 0 <= run < len(inverse):
          position = inverse[run][axis]
          values.append(labels[position] if 0 <= position < len(labels) else None)
        else:
          values.append(None)
      columns[dim_name] = values
    return columns

  def _dates_for(self, timesteps):
    '''Dates for report timesteps. -1 means a configuration fault, which has
    no timestep and must not be given one.'''
    if self.time_period is None:
      logger.warning('No time period available; reports will not carry dates.')
      return [pd.NaT] * len(timesteps)

    n = len(self.time_period)
    return [self.time_period[t] if 0 <= t < n else pd.NaT for t in timesteps]

  def _filter_by_dimensions(self, df, constraints):
    for name, value in constraints.items():
      if name not in df.columns:
        # A model without this dimension cannot satisfy a constraint on it.
        return df.iloc[0:0]
      if _is_set_constraint(value):
        df = df[df[name].isin(list(value))]
      else:
        df = df[df[name] == value]
    return df

  def reports(self, model=None, severity='warning', check_signature=True, **kwargs) -> pd.DataFrame:
    '''Return the reports raised by models during the run, one row per record.

    Model kernels report a code and a few numbers against a cell and a timestep;
    this turns those into dates, place names and messages using the model file's
    metadata.

    Parameters:

    * model - restrict to one model (name or node type); default all models
    * severity - minimum severity to include: 'warning' (default), 'error', or
      'all'
    * check_signature - verify the reports were produced against this model
      file. Turning this off can silently mis-attribute reports to the wrong
      places and dates.
    * **kwargs - filter by dimension, eg catchment='sc12'

    Detail records are capped per model, cell and report code during the run, so
    this shows *examples* of each condition. Use report_counts() for how often
    each one actually occurred.
    '''
    threshold = _severity_threshold(severity)
    if check_signature:
      self._check_report_signature()

    wanted = self._model_name(model) if model is not None else None
    frames = []

    for model_name in self._reporting_models():
      if wanted is not None and model_name != wanted:
        continue

      grp = self.results['%s/%s'%(REPORTS_GROUP, model_name)]
      if 'run' not in grp:
        continue  # counts only: the model ran and reported nothing

      names, messages, severities = self._report_catalog(model_name)
      runs = [int(r) for r in grp['run'][...]]
      timesteps = [int(t) for t in grp['timestep'][...]]
      codes = [int(c) for c in grp['code'][...]]
      record_severities = [int(s) for s in grp['severity'][...]]
      args = np.asarray(grp['args'][...])

      rendered = []
      for i, code in enumerate(codes):
        template = messages[code] if 0 <= code < len(messages) else ''
        rendered.append(_render_report(template, args[i], MAX_REPORT_ARGS))

      frame = pd.DataFrame({
        'date': self._dates_for(timesteps),
        'timestep': timesteps,
        'model': model_name,
        'severity': [SEVERITY_LABELS.get(s, str(s)) for s in record_severities],
        'code': codes,
        'report': [names[c] if 0 <= c < len(names) else str(c) for c in codes],
        'message': rendered,
        'run': runs,
      })
      frame = frame[[SEVERITY_VALUES.get(s, 0) >= threshold for s in frame['severity']]]

      for dim_name, values in self._dimension_columns(model_name, list(frame['run'])).items():
        frame[dim_name] = values

      frames.append(frame.drop(columns=['run']))

    return self._finalise_report_frame(frames, REPORT_COLUMNS, kwargs)

  def report_counts(self, model=None, severity='warning', check_signature=True, **kwargs) -> pd.DataFrame:
    '''Return how often each report was raised, per model, cell and code.

    Unlike reports(), these counts are complete: they cover every occurrence,
    including those whose detail record was dropped by the per-code cap. Rows
    with no occurrences are omitted.
    '''
    threshold = _severity_threshold(severity)
    if check_signature:
      self._check_report_signature()

    wanted = self._model_name(model) if model is not None else None
    frames = []

    for model_name in self._reporting_models():
      if wanted is not None and model_name != wanted:
        continue

      grp = self.results['%s/%s'%(REPORTS_GROUP, model_name)]
      if 'counts' not in grp:
        continue

      names, _, severities = self._report_catalog(model_name)
      counts = np.asarray(grp['counts'][...])
      if counts.size == 0:
        continue

      runs, codes = np.nonzero(counts)
      if len(runs) == 0:
        continue

      frame = pd.DataFrame({
        'model': model_name,
        'severity': [SEVERITY_LABELS.get(severities[c], str(severities[c])) for c in codes],
        'code': [int(c) for c in codes],
        'report': [names[c] if c < len(names) else str(c) for c in codes],
        'occurrences': [int(counts[r, c]) for r, c in zip(runs, codes)],
        'run': [int(r) for r in runs],
      })
      frame = frame[[SEVERITY_VALUES.get(s, 0) >= threshold for s in frame['severity']]]

      for dim_name, values in self._dimension_columns(model_name, list(frame['run'])).items():
        frame[dim_name] = values

      frames.append(frame.drop(columns=['run']))

    return self._finalise_report_frame(frames, REPORT_COUNT_COLUMNS, kwargs)

  def _finalise_report_frame(self, frames, base_columns, constraints):
    if not frames:
      return pd.DataFrame(columns=base_columns)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = self._filter_by_dimensions(combined, constraints)

    ordered = base_columns + [c for c in combined.columns if c not in base_columns]
    return combined[ordered].reset_index(drop=True)

class OpenwaterSplitResults(object):
  def  __init__(self,splits,time_period=None):
    assert len(splits)
    if isinstance(splits[0],tuple):
      self._results = [OpenwaterResults(model,res) for (model,res) in splits]
    else:
      self._results = splits
    first_tp = self._results[0].time_period
    self.time_period = time_period if time_period is not None else first_tp

  def close(self):
    for split in self._results:
      split.close()

  def dim(self,dim:str)->List:
    return self._results[0].dim(dim)

  def dims(self) -> List[str]:
    return self._results[0].dims()

  def time_series(self,model,variable:str,columns,aggregator=None,time_period=None,months=None,**kwargs) -> pd.DataFrame:
    '''
    Return a table (DataFrame) of time series results from the model.

    Parameters:

    * model - the model of interest
    * variable - a variable on the model, either an input or an output
    * columns - a dimension of the model to use as the columns of the DataFrame
    * aggregator - a function name (string) to apply when more than one data series matches a particular column (eg 'mean')
    * time_period - optional (start, end) tuple to subset the time axis (either end may be None)
    * months - optional list of month numbers (1–12) to filter
    * **kwargs - used to specify other dimensions to filter by

    For aggregator, see agg_fns.keys()

    For dimensions (row, columns and kwargs), see dims_for_model
    '''
    all_dfs = [split.time_series(model,variable,columns,aggregator,**kwargs) for split in self._results]
    concat = pd.concat(all_dfs)
    result = concat.set_index(self.time_period)
    if time_period is not None:
      start,end = time_period
      result = result.loc[start:end]
    if months is not None:
      result = result[np.isin(result.index.month,list(months))]
    return result

  def table(self,model,variable:str,rows:str,columns:str,temporal_aggregator:str='mean',aggregator:str=None,time_period=None,months=None,**kwargs) -> pd.DataFrame:
    '''
    Return a table (DataFrame) of aggregated model results from the model,
    combining results across all time splits.

    For temporal splits that partition the time axis without subsetting:
    - 'sum' aggregator: tables from each split are summed (sums are additive)
    - 'mean' aggregator: tables are combined as a weighted mean by timestep count

    For subset or percentile aggregators, the full concatenated timeseries is
    used (general path via time_series + _df_temporal_agg).

    Parameters:

    * model - the model of interest
    * variable - a variable on the model, either an input or an output
    * rows - a dimension of the model to use as the rows of the DataFrame
    * columns - a dimension of the model to use as the columns of the DataFrame
    * temporal_aggregator - a function name (string) to reduce the timeseries results to a single value (default='mean')
    * aggregator - a function name (string) to apply when more than one data series matches a particular row/column (eg 'mean')
    * time_period - optional (start, end) tuple to subset the time axis (either end may be None)
    * months - optional list of month numbers (1–12) to filter
    * **kwargs - used to specify other dimensions to filter by

    For temporal_aggregator, see temporal_agg_fns.keys()

    For aggregator, see agg_fns.keys()

    For dimensions (row, columns and kwargs), see dims_for_model
    '''
    if time_period is None and months is None and temporal_aggregator in ('sum','mean'):
      # Fast path: recombine per-split tables without touching timeseries
      split_tables = [s.table(model, variable, rows, columns, temporal_aggregator, aggregator, **kwargs)
                      for s in self._results]
      if temporal_aggregator == 'sum':
        return sum(split_tables)
      weights = [len(s.time_period) for s in self._results]
      total = sum(weights)
      return sum(t * w for t, w in zip(split_tables, weights)) / total

    # General path (subsets, percentiles): aggregate the concatenated timeseries
    ts = self.time_series(model,variable,(rows,columns),aggregator,time_period=time_period,months=months,**kwargs)
    reduced = _df_temporal_agg(ts,temporal_aggregator)
    return reduced.unstack(columns)

  def grouped_table(self,model,variable:str,columns,temporal_grouping:str,temporal_aggregator:str='mean',aggregator:str=None,time_period=None,months=None,water_year_start:int=7,**kwargs) -> pd.DataFrame:
    '''
    As OpenwaterResults.grouped_table, computed over the concatenated splits.
    '''
    ts = self.time_series(model,variable,columns,aggregator,time_period=time_period,months=months,**kwargs)
    return _grouped_table_from_timeseries(ts,temporal_grouping,temporal_aggregator,water_year_start)

  def models(self) -> List[str]:
    return self._results[0].models()

  def variables_for(self,model) -> List[str]:
    return self._results[0].variables_for(model)

  def dims_for_model(self,model) -> List[str]:
    return self._results[0].dims_for_model(model)

  def reports(self,model=None,severity='warning',check_signature=True,**kwargs) -> pd.DataFrame:
    '''Return the reports raised across every split. See OpenwaterResults.reports.'''
    frames = [s.reports(model,severity,check_signature,**kwargs) for s in self._results]
    frames = [f for f in frames if len(f)]
    if not frames:
      return pd.DataFrame(columns=REPORT_COLUMNS)
    return pd.concat(frames,ignore_index=True,sort=False)

  def report_counts(self,model=None,severity='warning',check_signature=True,**kwargs) -> pd.DataFrame:
    '''Return report counts across every split. See OpenwaterResults.report_counts.

    Splits partition the time axis, so a condition that spans a boundary is
    counted in each split it occurs in; the counts are summed per model, code
    and location.
    '''
    frames = [s.report_counts(model,severity,check_signature,**kwargs) for s in self._results]
    frames = [f for f in frames if len(f)]
    if not frames:
      return pd.DataFrame(columns=REPORT_COUNT_COLUMNS)

    combined = pd.concat(frames,ignore_index=True,sort=False)
    grouping = [c for c in combined.columns if c != 'occurrences']
    return combined.groupby(grouping,dropna=False,as_index=False)['occurrences'].sum()

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

# TODO:
# * Implement indexable results
#   eg results.time_series.StorageRouting.storage.by.catchment()
