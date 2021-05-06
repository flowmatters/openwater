
from typing import List
from . import nodes as node_types
import pandas as pd

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

class OpenwaterResults(object):
  def  __init__(self,model,res_file,time_period=None,inputs=None):
    self.model = _open_h5(model)
    self.results = _open_h5(res_file)
    if inputs is None:
      self.inputs = self.model
    else:
      self.inputs = _open_h5(inputs)

    self.time_period = time_period
    self._dimensions={}

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
    model = self._model_name(model)
    data = self._retrieve_all(model,variable)
    dim_names, dims, run_map = self._map_runs(model)

    slices = [slice(None,None,None) for _ in dim_names]
    for dim_name,dim_value in kwargs.items():
      if not dim_name in dim_names:
        map_grp = '/MODELS/%s/map'%model
        fixed_value = self.model[map_grp].attrs.get(dim_name,None)
        if fixed_value != dim_value:
            raise Exception('Invalid dimension: %s=%s'%(dim_name,dim_value))
        continue

      dim_num = dim_names.index(dim_name)
      dim_idx = dims[dim_name].index(dim_value)
      slices[dim_num] = dim_idx
    
    report_dim = dim_names.index(columns)

    all_sequences = {}
    for i,col_name in enumerate(dims[columns]):
      if i == run_map.shape[report_dim]:
        # Looks like a dummy tag value not present here
        # HACK. Could be other things (eg incomplete set of tags)
        continue

      current_slices = slices[:]
      current_slices[report_dim] = i
      run_indices = run_map[tuple(current_slices)]
      run_indices = run_indices.flatten()
      run_indices = run_indices[run_indices>=0]
      col_data = data[run_indices,:]
      if col_data.shape[0]==1:
        all_sequences[col_name] = col_data[0,:]
      else:
        all_sequences[col_name] = agg_fns[aggregator or 'mean'](col_data)
    return pd.DataFrame(all_sequences,index=self.time_period)

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
    model = self._model_name(model)
    data = self._retrieve_all(model,variable)
    dim_names, dims, run_map = self._map_runs(model)
    slices = [slice(None,None,None) for _ in dim_names]

    data = temporal_agg_fns[temporal_aggregator](data)
    column_names = dims[columns]
    row_names = dims[rows]

    col_dim = dim_names.index(columns)
    row_dim = dim_names.index(rows)

    table_data = {}
    for i,col_name in enumerate(column_names):
      col_data = []
      for j,_ in enumerate(row_names):
        current_slices = slices[:]
        current_slices[col_dim] = i
        current_slices[row_dim] = j
        run_indices = run_map[tuple(current_slices)].flatten()
        cell_data = data[run_indices]
        if cell_data.shape[0]==1:
          col_data.append(cell_data[0])
        else:
          col_data.append(agg_fns[aggregator or 'mean'](cell_data))
      table_data[col_name] = col_data
    return pd.DataFrame(table_data,index=row_names)

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
      self._results = [OpenwaterResults(model,res,None) for (model,res) in splits]
    else:
      self._results = splits
    self.time_period = time_period

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
    raise Exception('Not implemented')

  def models(self) -> List[str]:
    return self._results[0].models()

  def variables_for(self,model) -> List[str]:
    return self._results[0].variables_for(model)

  def dims_for_model(self,model) -> List[str]:
    return self._results[0].dims_for_model(model)

