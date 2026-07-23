from string import Template
import numpy as np
import pandas as pd
import string
from .array_params import get_parameter_locations, param_starts
from .nodes import create_indexed_parameter_table
import logging
logger = logging.getLogger(__name__)


def _model_key(model):
    '''Normalise a model reference (class, node-type object or string) to its name.'''
    if model is None:
        return None
    if hasattr(model,'name'):
        return model.name
    return str(model)


# Normalised fill-rule representations:
#   ('value', float)  -- fill uncovered timesteps with a constant
#   ('ffill',)        -- nearest-edge hold (carry last value into trailing gaps,
#                        first value into leading gaps)
_FFILL_ALIASES = {'ffill','forward','forward_fill','forward-fill','edge','hold','nearest'}


def _normalise_fill(rule):
    if isinstance(rule,tuple) and len(rule) and rule[0] in ('value','ffill'):
        return rule
    if isinstance(rule,str):
        key = rule.strip().lower()
        if key in _FFILL_ALIASES:
            return ('ffill',)
        if key == 'zero':
            return ('value',0.0)
        raise ValueError(f'Unknown fill rule: {rule!r}')
    if isinstance(rule,(int,float,np.integer,np.floating)):
        return ('value',float(rule))
    raise ValueError(f'Unknown fill rule: {rule!r}')


class FillRules(object):
    '''Rules for filling new/uncovered timesteps when resizing a model's period.

    Rules resolve most-specific-first: (model, variable) -> variable -> model
    -> default. A rule is a specific numeric value, ``'zero'`` (== 0.0) or
    ``'ffill'`` (nearest-edge hold: carry the last known value into trailing
    gaps and the first known value into leading gaps).

    Example
    -------
    >>> rules = (FillRules(default=0.0)
    ...          .set('ffill', variable='rainfall')
    ...          .set(5.0, variable='pet', model='GR4J'))
    '''
    def __init__(self,default=0.0):
        self._default = _normalise_fill(default)
        self._by_model_var = {}
        self._by_var = {}
        self._by_model = {}

    def set(self,rule,variable=None,model=None):
        '''Register a fill rule.

        Provide ``variable`` and/or ``model`` to scope the rule; omit both to
        change the default. Returns self so calls can be chained.
        '''
        rule = _normalise_fill(rule)
        model = _model_key(model)
        if variable is not None and model is not None:
            self._by_model_var[(model,variable)] = rule
        elif variable is not None:
            self._by_var[variable] = rule
        elif model is not None:
            self._by_model[model] = rule
        else:
            self._default = rule
        return self

    def rule_for(self,model,variable):
        model = _model_key(model)
        if (model,variable) in self._by_model_var:
            return self._by_model_var[(model,variable)]
        if variable in self._by_var:
            return self._by_var[variable]
        if model in self._by_model:
            return self._by_model[model]
        return self._default


def align_source_indices(old_period,new_period):
    '''For each timestep in ``new_period`` return the position of the matching
    timestep in ``old_period`` (by exact timestamp), or -1 where there is no
    match. Alignment is by timestamp, not position, so periods at different
    frequencies only overlap where timestamps coincide.
    '''
    old_period = pd.DatetimeIndex(old_period)
    new_period = pd.DatetimeIndex(new_period)
    return np.asarray(old_period.get_indexer(new_period))


def _edge_hold(plane,covered):
    '''Nearest-edge hold along the time axis for one input.

    ``plane`` is (n_cells, n_timesteps); ``covered`` is a boolean mask over the
    time axis marking timesteps carried over from the old array. Uncovered
    timesteps take the last covered value before them, or the first covered
    value after them if none precedes. Returns the filled plane.
    '''
    tmp = plane.T.astype(float).copy()          # (n_timesteps, n_cells)
    tmp[~covered,:] = np.nan
    filled = pd.DataFrame(tmp).ffill(axis=0).bfill(axis=0).to_numpy()
    filled = np.nan_to_num(filled,nan=0.0)      # no covered timesteps at all
    return filled.T


def build_resized_input_array(old_arr,src_for_new,var_names,model,fill_rules):
    '''Build a resized (n_cells, n_inputs, T_new) input array from ``old_arr``.

    ``src_for_new`` (see align_source_indices) maps each new timestep to an old
    timestep index or -1. Carried-over timesteps are copied; the remainder is
    filled per ``fill_rules`` (keyed by model + input variable name).
    '''
    n_cells,n_inputs,_ = old_arr.shape
    src_for_new = np.asarray(src_for_new)
    T_new = len(src_for_new)
    new_arr = np.zeros((n_cells,n_inputs,T_new),dtype=old_arr.dtype)
    covered = src_for_new >= 0
    if covered.any():
        new_arr[:,:,covered] = old_arr[:,:,src_for_new[covered]]
    if covered.all():
        return new_arr

    gap = ~covered
    for input_num,var in enumerate(var_names):
        kind = fill_rules.rule_for(model,var)
        if kind[0] == 'value':
            if kind[1] != 0.0:                  # gaps already zero
                new_arr[:,input_num,gap] = kind[1]
        elif kind[0] == 'ffill':
            new_arr[:,input_num,:] = _edge_hold(new_arr[:,input_num,:],covered)
    return new_arr


def _read_model_period(grp):
    '''Read the model time period (as a DatetimeIndex) from META/timeperiod on
    the file containing ``grp``, or None if not present.'''
    f = grp.file
    if 'META' not in f or 'timeperiod' not in f['META']:
        return None
    raw = f['META']['timeperiod'][...]
    vals = [d.decode() if isinstance(d,bytes) else d for d in raw]
    return pd.DatetimeIndex([pd.Timestamp.fromisoformat(v) for v in vals])


def _models_match(configured,trial):
  if configured is None:
    return True

  if configured == trial:
    return True

  if hasattr(configured,'name'):
    return configured.name == trial.name
  return configured == trial.name

def _locate_parameter_in_description(model_desc,parameter):
    matching_params = [i for i,p in enumerate(model_desc.description['Parameters']) if p['Name']==parameter]
    if len(matching_params):
        return 'parameters',matching_params[0],slice(None)

    if parameter in model_desc.description['States']:
        state_index = model_desc.description['States'].index(parameter)
        #[i for i,s in enumerate(model_desc.description['States']) if s['Name']==parameter]
        return 'states',slice(None),state_index

    raise Exception('Unknown parameter or state: %s'%parameter)

def _constraint_matches(present_value, constraint_value):
    '''Check a single tag value against a constraint.

    A constraint value that is a non-string iterable (list, tuple, set,
    numpy array, etc.) is treated as a set-membership test. Any other value
    (including strings) is treated as a scalar equality test.
    '''
    if isinstance(constraint_value, str):
        return present_value == constraint_value
    if isinstance(constraint_value, (list, tuple, set, frozenset, np.ndarray)):
        return present_value in constraint_value
    return present_value == constraint_value


def _matches_constraints(constraint_tags,present_tags,resolver=None):
    '''Match a dict of tag constraints against a node's tags.

    If ``resolver`` is provided, any quasi-dim keys in ``constraint_tags`` are
    first expanded to real-dim equivalents (with set-membership semantics)
    via the resolver, so callers can transparently constrain by quasi-dims.
    '''
    if resolver is not None and constraint_tags:
        # Only call into the resolver if at least one key is a quasi-dim;
        # avoids constructing a new dict in the common case.
        if any(resolver.is_quasi(k) for k in constraint_tags):
            constraint_tags = resolver.resolve_constraints(dict(constraint_tags))
    for k,v in (constraint_tags or {}).items():
        if k not in present_tags:
            return False
        if not _constraint_matches(present_tags[k], v):
            return False
    return True

def initialise_model_inputs(model,model_grp,n_cells,n_inputs,n_timesteps):
    if not 'inputs' in model_grp:
        logger.info('Initialising model inputs: %s (%d x %d x %d',model,n_cells,n_inputs,n_timesteps)
        model_grp.create_dataset('inputs',shape=(n_cells,n_inputs,n_timesteps),dtype=np.float64,fillvalue=0)

class Parameteriser(object):
    def __init__(self):
        self._parameterisers = []

    def append(self,parameteriser):
        if parameteriser is None:
          return

        self._parameterisers.append(parameteriser)

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df,resolver=None):
        for p in self._parameterisers:
            p.parameterise(model_desc,grp,instances,dims,nodes,nodes_df,resolver=resolver)


VALID_ALIGN_MODES = ('position','dates')


class DataframeInput(object):
    def __init__(self,dataframe,column_format,model,constraint_tags,align='position'):
        if align not in VALID_ALIGN_MODES:
            raise ValueError("align must be one of %s, got %r"%(VALID_ALIGN_MODES,align))
        self.df = dataframe
        self.model = model
        self.constraint_tags = constraint_tags
        self.align_dates = (align == 'dates')

        if isinstance(column_format,str):
            column_format = Template(column_format)
        self.column_format = column_format

    def applies(self,model):
        return _models_match(self.model,model)

    def _column_for(self,resolver=None,**kwargs):
        if not _matches_constraints(self.constraint_tags,kwargs,resolver=resolver):
            return None
        col_name = self.column_format.substitute(**kwargs)
        if col_name in self.df.columns:
            return col_name
        return None

    def get_series(self,resolver=None,**kwargs):
        col_name = self._column_for(resolver=resolver,**kwargs)
        if col_name is None:
            return None
        return np.array(self.df[col_name])

    def get_dated_series(self,resolver=None,**kwargs):
        '''Return the matching column as a pandas Series (retaining its date
        index) for date-aligned application, or None if no column matches.'''
        col_name = self._column_for(resolver=resolver,**kwargs)
        if col_name is None:
            return None
        return self.df[col_name]

class DataframeInputs(object):
    def __init__(self,allow_nans=False):
        self._inputs = {}
        self._allow_nans = allow_nans
    
    def inputter(self,df,input_name,col_format,model=None,align='position',**kwargs):
        '''
        Map timeseries inputs for a given input name, using the supplied dataframe (df).

        Parameters
        ----------
        df : pandas.DataFrame with timeseries data (ie a date time index)
             With align='position' (default) the dataframe must match the
             temporal dimensions of the model (same number of rows, applied
             positionally over the whole time axis). With align='dates' the
             dataframe's DatetimeIndex is matched against the model's time
             period and only the overlapping timesteps are written, leaving the
             rest of the existing series untouched.
        input_name : str
              The name of the timeseries input to map
        col_format : str
              A string template that can be used to identify the column name for a given node using the node's tags.
              Use Python string template syntax, e.g. 'Catchment ${catchment} - ${constituent}'
        model : str, optional
              The model to which this input applies. If not provided, the input will be applied to all models with the input.
        align : {'position','dates'}, optional
              'position' (default) replaces the whole time axis positionally and
              requires the series length to match the model. 'dates' aligns by
              the dataframe's date index and applies only the overlapping
              sub-period in place (the model must already have a time period).
        kwargs : dict, optional
              Tags that must be matched for the input to be applied to a given node.
        '''
        assert df is not None
        if align not in VALID_ALIGN_MODES:
            raise ValueError("align must be one of %s, got %r"%(VALID_ALIGN_MODES,align))
        if not len(df) or not len(df.columns):
            logger.warning('Empty dataframe provided for input %s with column format %s',input_name,col_format)
            return

        if not input_name in self._inputs:
            self._inputs[input_name] = []
        self._inputs[input_name].append(DataframeInput(df,col_format,model,kwargs,align=align))

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df,resolver=None):
        description = model_desc.description
        inputs = description['Inputs']
        if not len(set(inputs).intersection(set(self._inputs.keys()))):
            return

        logger.info('==== Called for %s ====',model_desc.name)
        logger.info('Timeseries for %s',self._inputs.keys())
        logger.debug(inputs)
        logger.debug(grp)
        logger.debug(list(nodes.items())[0])
        logger.debug(instances.shape)
        model_period = None  # read lazily, only when a date-aligned inputter needs it
        i = 0
        applied = 0
        for node_name,node in nodes.items():
            run_idx = node['_run_idx']
            for input_num,input_name in enumerate(inputs):
                if not input_name in self._inputs:
                    continue

                inputters = self._inputs[input_name]
                positional = [inputter for inputter in inputters if not inputter.align_dates]
                lengths = set([len(inputter.df) for inputter in positional])
                if len(lengths) > 1:
                    logger.error(f'Differing length inputs for {input_name}: {lengths}')

                for inputter in inputters:
                    if not inputter.applies(model_desc):
                        continue

                    if inputter.align_dates:
                        if 'inputs' not in grp:
                            raise Exception(f'Cannot apply date-aligned input {input_name} to {model_desc.name}: '
                                            'the model has no existing input series. Apply the full period first '
                                            "(align='position') or retime the model.")
                        if model_period is None:
                            model_period = _read_model_period(grp)
                            if model_period is None:
                                raise Exception('Cannot apply date-aligned input: model file has no META/timeperiod')
                        series = inputter.get_dated_series(resolver=resolver,**node)
                        if series is None:
                            continue
                        if self._apply_dated_series(grp,run_idx,input_num,series,model_period,input_name,node_name):
                            applied += 1
                        continue

                    initialise_model_inputs(model_desc.name,grp,len(nodes_df),len(inputs),len(positional[0].df))

                    data = inputter.get_series(resolver=resolver,**node)
                    if data is None:
                        continue
                    applied += 1
                    actual_len = len(data)
                    expected_len = grp['inputs'].shape[2]
                    if actual_len != expected_len:
                        raise Exception(f'Timeseries mismatch on {input_name} to {model_desc.name}. Expecting {expected_len} timesteps, but provided with {actual_len}')

                    self._check_nans(data,input_name,node_name)
                    grp['inputs'][run_idx,input_num,:] = data

            if (i%100 == 0) and (applied>0):
                logger.info('Processing %s. Applied %d inputs ()',node_name,applied)
            i += 1
        logger.info('Applied %d timeseries inputs to %s',applied,model_desc.name)

    def _check_nans(self,data,input_name,node_name):
        if np.isnan(data).any():
            if self._allow_nans:
                logger.warning(f'NaN values found in input data for {input_name} on node {node_name}, but _allow_nans is True. Data: {data}')
            else:
                logger.error(f'NaN values found in input data for {input_name} on node {node_name}. Data: {data}')
                raise Exception('NaN values found in input data')

    def _apply_dated_series(self,grp,run_idx,input_num,series,model_period,input_name,node_name):
        '''Write a partial, date-aligned series into an existing input array,
        leaving timesteps outside the series' date range untouched. Returns True
        if any values were written.'''
        idx = pd.DatetimeIndex(series.index)
        pos = np.asarray(model_period.get_indexer(idx))
        valid = pos >= 0
        if not valid.any():
            logger.warning('No overlapping dates for %s on %s; skipping',input_name,node_name)
            return False
        if not valid.all():
            logger.warning('%d/%d timesteps for %s on %s fall outside the model period; ignoring those',
                           int((~valid).sum()),len(valid),input_name,node_name)
        values = np.asarray(series.values,dtype=np.float64)[valid]
        self._check_nans(values,input_name,node_name)
        existing = grp['inputs'][run_idx,input_num,:]
        existing[pos[valid]] = values
        grp['inputs'][run_idx,input_num,:] = existing
        return True

class SingleTimeseriesInput(object):
    def __init__(self,series,the_input,model=None,**tags):
        self.series = series
        self.model = model
        self.the_input = the_input
        self.tags = tags

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df,resolver=None):
        if not _models_match(self.model,model_desc):
            return

        description = model_desc.description
        inputs = description['Inputs']
        matching_inputs = [i for i,nm in enumerate(inputs) if nm==self.the_input]
        if len(matching_inputs)==0:
            return

        logger.info('==== SingleTimeseriesInput(%s) called for %s ===='%(self.the_input,model_desc.name))
        input_num = matching_inputs[0]
        data = np.array(self.series)

        i = 0
        for node_name,node in nodes.items():
            if not _matches_constraints(self.tags, node, resolver=resolver):
                continue

            run_idx = node['_run_idx']
            grp['inputs'][run_idx,input_num,:] = data

            if i%100 == 0:
                logger.debug('Processing %s'%node_name)
            i += 1

class ParameterTableAssignment(object):
    '''
    Parameterise OpenWater models from a DataFrame.


    '''
    def __init__(self,df,model,parameter=None,column_dim=None,row_dim=None,dim_columns=None,complete=True,skip_na=False):
        self.df = df
        self.column_dim = column_dim
        self.row_dim = row_dim
        self.model = model
        self.parameter = parameter
        self.dim_columns = dim_columns
        self.complete = complete
        self.skip_na = skip_na

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df,resolver=None):
        if not _models_match(self.model,model_desc):
            return

        logger.info('Applying parameter table to %s'%model_desc.name)

        if None in [self.column_dim,self.row_dim,self.parameter is None]:
           self._parameterise_nd(model_desc,grp,instances,dims,nodes,nodes_df,resolver=resolver)
        else:
            self._parameterise_2d(model_desc,grp,instances,dims,nodes,resolver=resolver)

    def _parameterise_nd(self,model_desc,grp,instances,dims,nodes,nodes_df,resolver=None):
        names = [p['Name'] for p in model_desc.description['Parameters']] + model_desc.description['States']
        current_data = {}
        for p in names:
            if not p in self.df.columns:
                continue
            dest_grp,dest_idx0,dest_idx1 = self.locate(model_desc,p)
            existing = grp[dest_grp][dest_idx0,dest_idx1]
            try:
                assert len(existing.shape)==1
                assert existing.shape[0]>=len(nodes)
            except:
                logger.error(f"{dest_grp}, {dest_idx0}, {dest_idx1}, {p,existing.shape,len(nodes)}")
                raise
            current_data[p] = existing

        # param_data = {p:np.zeros(instances.size,dtype='float64') for i,p in enumerate(param_names) if p in self.df.columns}
        ignored = []
        # print('model',model_desc)
        # print('nodes_df',len(nodes_df),nodes_df.columns)
        # print('df',len(self.df),self.df.columns)
        real_join = set(dims.keys()).intersection(set(self.df.columns))
        # Quasi-dim columns in the table can also be joined, after projecting
        # the underlying real dim onto nodes_df via the resolver.
        quasi_join = set()
        if resolver is not None:
            quasi_join = {c for c in self.df.columns
                          if c not in dims and resolver.is_quasi(c)}
        join_keys = real_join | quasi_join
        if not len(join_keys):
            raise Exception(f'Table has no columns matching model dimensions. Dims: {dims.keys()}. Columns: {self.df.columns}')

        if quasi_join:
            nodes_df = resolver.extend_nodes_df(nodes_df, quasi_join)

        joined = pd.merge(nodes_df,self.df,how='inner',on=list(join_keys))
        # print('joined',len(joined),joined.columns)

        indices = np.array(joined._run_idx)
        for p,arr in current_data.items():
            srs = joined[p]
            if self.skip_na:
                srs.fillna(0.0,inplace=True)
            arr[indices] = np.array(srs)

        # if True:
        #     raise Exception('BOO')

        # for _,node in nodes.items():
        #     # TODO: very slow. could we have a dataframe of all the nodes with tags and join them?
        #     subset = self.df
        #     for dim in dims.keys():
        #         if not dim in subset.columns:
        #             if not dim in ignored:
        #                 print('%s not specified in table, ignoring'%dim)
        #             ignored.append(dim)
        #             continue
        #         subset = subset[subset[dim]==node[dim]]

        #     if len(subset)==0 and not self.complete:
        #         continue

        #     if len(subset)>1:
        #         for dc in self.dim_columns:
        #             if not dc in subset.columns:
        #                 continue
        #             subset = subset[subset[dc]==node[dc]]

        #     if not len(subset)==1:
        #         print('=== Model: %s ==='%model_desc.name)
        #         print('=== Dims ===')
        #         print(list(dims.keys()))
        #         print('=== Node ===')
        #         print(node)
        #         print('=== Subset ===')
        #         print(subset)
        #         # print('=== Original ===')
        #         # print(self.df)
        #         assert len(subset)==1

        #     run_idx = node['_run_idx']
        #     for p,arr in current_data.items():
        #         val = subset[p]
        #         if self.skip_na:
        #             val.fillna(0.0,inplace=True)
        #         arr[run_idx] = val

        for p,vals in current_data.items():
            # if not p in param_data:
            #     print('--> No parameters for %s'%p)
            #     continue
            dest_grp, dest_idx0, dest_idx1 = self.locate(model_desc,p)
            logger.info('Applying %s for %s'%(dest_grp,p))
            grp[dest_grp][dest_idx0,dest_idx1]=vals

    def _parameterise_2d(self,model_desc,grp,instances,dims,nodes,resolver=None):
        dest_grp, dest_idx0, dest_idx1 = self.locate(model_desc,self.parameter)
        logger.debug(f'{dest_grp}, {dest_idx0}, {dest_idx1}')

        # If column_dim / row_dim names a quasi-dim, project the node's
        # underlying real-dim value through the chain at lookup time.
        col_proj = None
        row_proj = None
        if resolver is not None:
            if resolver.is_quasi(self.column_dim):
                col_proj = resolver.project_index(self.column_dim)
            if resolver.is_quasi(self.row_dim):
                row_proj = resolver.project_index(self.row_dim)

        param_data = np.zeros(len(nodes),dtype='float64')
        for _,node in nodes.items():
            run_idx = node['_run_idx']
            if col_proj is not None:
                col = col_proj.project(pd.Series([node[col_proj.keyed_by]])).iloc[0]
            else:
                col = node[self.column_dim]
            if row_proj is not None:
                row = row_proj.project(pd.Series([node[row_proj.keyed_by]])).iloc[0]
            else:
                row = node[self.row_dim]

            if col not in self.df.columns and self.skip_na:
                continue

            series = self.df[col]

            if row not in series.index and self.skip_na:
                continue

            param = series[row]
            if np.isnan(param) and self.skip_na:
                continue

            param_data[run_idx] = param

        logger.debug(param_data)
        grp[dest_grp][dest_idx0,dest_idx1]=param_data

    def locate(self,model_desc,parameter):
        return _locate_parameter_in_description(model_desc,parameter)

class DefaultParameteriser(object):
    def __init__(self,model_name=None,**kwargs):
        self._model = model_name
        self._params = kwargs

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df,resolver=None):
        if not _models_match(self._model,model_desc):
            return

        logger.info('Applying default parameters: %s'%model_desc.name)
        for param_num, param in enumerate(model_desc.description['Parameters']):
            pname = param['Name']
            pdefault = param['Default']
            grp['parameters'][param_num,:] = self._params.get(pname,pdefault)

class UniformParameteriser(object):
    def __init__(self,model_name=None,**kwargs):
        self._model = model_name
        self._params = kwargs

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df,resolver=None):
        if not _models_match(self._model,model_desc):
            return

        logger.info('Applying uniform parameters: %s'%model_desc.name)
        for param_num, param in enumerate(model_desc.description['Parameters']):
            pname = param['Name']
            if not pname in self._params:
                continue
            grp['parameters'][param_num,:] = self._params[pname]

class UniformInput(object):
    def __init__(self,input_name,val,length):
        self.input_name = input_name
        self.value = val
        self._length = length

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df,resolver=None):
      inputs = model_desc.description['Inputs']
      for input_num,input_name in enumerate(inputs):
          if input_name!=self.input_name:
            continue
          initialise_model_inputs(model_desc.name,grp,len(nodes_df),len(inputs),self._length)
          logger.info('Uniform %s = %f'%(self.input_name,self.value))
          for cell in range(len(nodes_df)):
            if hasattr(self.value,'__call__'):
              grp['inputs'][cell,input_num,:] = self.value(cell)
            else:
              grp['inputs'][cell,input_num,:] = self.value

class DictParameteriser(object):
    def __init__(self,parameter,key_format,model=None,parameters={},constraints={},**kwargs):
        self.parameter = parameter
        self.key_format = string.Template(key_format)
        self.model = model
        self.constraints = constraints
        self.parameters = parameters
        self.parameters.update(**kwargs)

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df,resolver=None):
        if not _models_match(self.model,model_desc):
            return

        dest_grp,dest_idx0,dest_idx1 = _locate_parameter_in_description(model_desc,self.parameter)
        for ix, row in nodes_df.iterrows():
            if not _matches_constraints(self.constraints,row,resolver=resolver):
                continue

            if dest_grp=='parameters':
                dest_idx1 = row._run_idx
            else:
                dest_idx0 = row._run_idx
            grp[dest_grp][dest_idx0,dest_idx1] = self.parameters[self.key_format.substitute(row)]

class DimensionParameterSizer(object):
    def __init__(self):
        pass

    def applies(self,desc):
        if not 'Dimensions' in desc:
            return False

        if not len(desc['Dimensions']):
            return False

        return True

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df,resolver=None):
        desc = model_desc.description
        if not self.applies(desc):
            return

        new_param_locs = get_parameter_locations(desc,grp['parameters'][...])
        len_params = new_param_locs[-1][1]

        grp.create_dataset('new_params',shape=(len_params,len(nodes_df)),dtype=np.float64,fillvalue=0)#,compression='gzip')
        grp['new_params'][0:grp['parameters'].shape[0],:] = grp['parameters'][:,:]
        grp.pop('parameters')
        grp.move('new_params','parameters')


def populate_table_parameters(existing,param_start,tables,key_format,column_lookup={}):
    '''
    '''

    #existing = model.indexed_parameters(model_type)
    key_format = string.Template(key_format)
    arr = np.array(existing)

#     tables = {fn:pd.read_csv(fn,index_col=0) for fn in glob(pattern)}
    #{(fn.split('/')[-1].split('.')[0].replace('storage_lva_','')):pd.read_csv(fn,index_col=0) for fn in glob(os.path.join(SRC_FILES,'storage_lva*csv*'))}
    tags = existing.index.names
    missed = False
    for i,ix_vals in enumerate(existing.index):
        vals = {}
        if len(tags)==1:
            vals[tags[0]]=ix_vals
        else:
            logger.error(ix_vals)
            raise Exception('not supported')
        key = key_format.substitute(**vals)
        if key not in tables:
            logger.error(f'==== NO TABLE WITH KEY {key}, format={key_format.template}, vals={vals}, tags={tags} ====')
            missed = True
            continue
        tbl = tables[key]
#             print('No file %s for tags %s'%(fn,str(row)))
        
        # Extract tags from fn
        for col in tbl.columns:
            if col in param_start:
                column_lookup[col]=col

        logger.debug(f'{i}, {ix_vals}')
        for param,col in column_lookup.items():
            vals = tbl[col]
            param_idx = param_start[param]
            logger.debug(f'{param}, {i},{param_idx}, {len(vals)}')
            arr[i,param_idx:(param_idx+len(vals))] = np.array(vals)
    result = pd.DataFrame(arr,index=existing.index,columns=existing.columns)
    assert not missed
    return result

def _raw_parameters(model_map,vals):
    df = pd.DataFrame(model_map)
    logger.debug(df)
    dim_cols = [col for col in df.columns if (not col.startswith('_') and not col=='node')]
    df = df.set_index(list(dim_cols))

    param_df = pd.DataFrame(vals).transpose().reindex(index=df['_run_idx'])

    result = param_df.set_index(df.index)
#     for k,v in tags.items():
#         result = result[result[k]==v]

    return result

class LoadArraysParameters(object):
    def __init__(self,table_lookup,key_format,len_parameter,column_lookup=None,model=None):
        self.table_lookup = table_lookup
        self.key_format = key_format

        if column_lookup is None:
            column_lookup = {}
        self.column_lookup = column_lookup
        self.len_parameter = len_parameter
        self.model = model

        self.lengths = {k:len(tbl) for k,tbl in table_lookup.items()}
        self.nested = [
            DictParameteriser(len_parameter,key_format,model,self.lengths),
            DimensionParameterSizer()
        ]

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df,resolver=None):
        if not _models_match(self.model,model_desc):
            return
        logger.info('Running LoadArrayParameters for %s.',model_desc.name)
        logger.info('self.model=%s',self.model)

        for p in self.nested:
            p.parameterise(model_desc,grp,instances,dims,nodes,nodes_df,resolver=resolver)

        raw = _raw_parameters(nodes_df,grp['parameters'][...])
        indexed = create_indexed_parameter_table(model_desc.description,raw)
        starts = param_starts(model_desc.description,indexed.transpose())
        populated = populate_table_parameters(indexed,starts,self.table_lookup,self.key_format,self.column_lookup)
        final = np.array(populated).transpose()
        for ix, row in nodes_df.iterrows():
            run_index = row._run_idx
            
            grp['parameters'][:,run_index] = final[:,ix]

class NestedParameteriser(object):
    def __init__(self,nested=[]):
        self.nested = nested[:]

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df,resolver=None):
        for np in self.nested:
            if np is None: continue

            np.parameterise(model_desc,grp,instances,dims,nodes,nodes_df,resolver=resolver)

class CustomParameteriser(object):
    def __init__(self,fn,model=None,filter=None):
        self.model = model
        self.fn = fn
        self.filter = None

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df,resolver=None):
        if not _models_match(self.model,model_desc):
            return

        self.fn(model_desc,grp,instances,dims,nodes,nodes_df)
