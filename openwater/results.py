
from . import nodes as node_types
import pandas as pd

agg_fns = {
    'mean':lambda a: a.mean(axis=0)
}
class OpenwaterResults(object):
  def  __init__(self,model,res_file,time_period=None):
    self.model = model
    self.results = res_file
    self.time_period = time_period
    self._dimensions={}

  def _time_series(self,model,dataset,var_idx,columns,aggregator=None,**kwargs):
    data = dataset[:,var_idx,:]
    assert len(data.shape)==2

    map_grp = '/MODELS/%s/map'%model
    dim_names = self.dims_for_model(model)
    dims = {d:self.dim(d) for d in dim_names}

    run_map = self.model[map_grp][...]
    slices = [slice(None,None,None) for _ in dim_names]
    for dim_name,dim_value in kwargs.items():
      dim_num = dim_names.index(dim_name)
      dim_idx = dims[dim_name].index(dim_value)
      slices[dim_num] = dim_idx
    
    report_dim = dim_names.index(columns)

    all_sequences = {}
    for i,col_name in enumerate(dims[columns]):
      current_slices = slices[:]
      current_slices[report_dim] = i
      run_indices = run_map[tuple(current_slices)]
      run_indices = run_indices.flatten()
      col_data = data[run_indices,:]
      if col_data.shape[0]==1:
        all_sequences[col_name] = col_data[0,:]
      else:
        all_sequences[col_name] = agg_fns[aggregator or 'mean'](col_data)
    return pd.DataFrame(all_sequences,index=self.time_period)

  def dim(self,dim):
    if not dim in self._dimensions:
      vals = list(self.model['/DIMENSIONS'][dim][...])
      conv = lambda v: v.decode('utf-8') if hasattr(v,'decode') else v
      vals = [conv(v) for v in vals]
      self._dimensions[dim] = vals

    return self._dimensions[dim]

  def time_series(self,model,variable,columns,aggregator=None,**kwargs):
    desc = getattr(node_types,model)
    is_input = variable in desc.description['Inputs']

    grp_name = '/MODELS/%s'%model
    out_grp = self.results[grp_name]
    in_grp = self.model[grp_name]
    
    if is_input:
      var_idx = desc.description['Inputs'].index(variable)
      if 'inputs' in out_grp:
        dataset = out_grp['inputs']
      else:
        dataset = in_grp['inputs']
    else:
      var_idx = desc.description['Outputs'].index(variable)
      dataset = out_grp['outputs']

    return self._time_series(model,dataset,var_idx,columns,aggregator,**kwargs)

  def dims_for_model(self,model):
    map_grp = '/MODELS/%s/map'%model
    return [d.decode('utf-8') for d in self.model[map_grp].attrs['DIMS']]