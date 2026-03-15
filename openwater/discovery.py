
from . import single, ensemble, lib, nodes
import os
import numpy as np
from functools import reduce

OW_BIN=os.environ.get('OW_BIN',os.path.join(os.path.expanduser('~'),'bin'))

def set_exe_path(p):
  global OW_BIN
  OW_BIN=p

def _exe_path(prog,family='ow'):
  import sys
  import os
  result = os.path.join(OW_BIN,'%s-%s'%(family,prog))
  if sys.platform=='win32':
    result += '.exe'
  return result

def _lib_path():
  import sys
  import os
  result = os.path.join(OW_BIN,'libopenwater')
  if sys.platform=='win32':
    result += '.dll'
  elif sys.platform=='darwin':
    result += '.dylib'
  else:
    result += '.so'
  return result

class VariableList(list):
  '''A list of variable names that also carries detailed metadata (units, etc).

  Behaves like a regular list of strings for backward compatibility,
  but provides a .details attribute with the full variable descriptions.
  '''
  def __init__(self, variables):
    if variables and isinstance(variables[0], dict):
      super().__init__([v['Name'] for v in variables])
      self.details = variables
    else:
      super().__init__(variables)
      self.details = [{'Name': v, 'Units': ''} for v in variables]

  def units(self, name):
    '''Get units string for a variable by name.'''
    for v in self.details:
      if v['Name'] == name:
        return v.get('Units', '')
    raise KeyError(name)

  def parsed_unit(self, name):
    '''Get parsed pint Unit for a variable by name, or None if unspecified.'''
    from .units import parse_unit
    return parse_unit(self.units(name))

  def is_compatible(self, name, other_unit_str):
    '''Check if a variable's units are compatible with another unit string.'''
    from .units import are_compatible
    return are_compatible(self.units(name), other_unit_str)

def _variable_units_str(name, details):
  '''Format a variable name with units for documentation.'''
  for v in details:
    if v['Name'] == name:
      units = v.get('Units', '')
      if units:
        return '%s (%s)' % (name, units)
  return name

_DOC_SEP='\n  * '
_DOC_TEMPLATE='''
Function parameters:
  * %s
  * %s
  * %s

Returns
  * %s
'''

def _param_default(param_meta):
  dims = param_meta.get('Dimensions',[])
  if len(dims)==0:
    return '%f'%param_meta['Default']
  return 'Empty %d-D array'%len(dims)

def _param_doc(param_meta):
  name = param_meta['Name']
  default = _param_default(param_meta)
  parts = ['%s: Model parameter (default: %s)' % (name, default)]
  units = param_meta.get('Units', '')
  if units:
    parts[0] = '%s (%s): Model parameter (default: %s)' % (name, units, default)
  desc = param_meta.get('Description', '')
  if desc:
    parts.append(desc)
  range_vals = param_meta.get('Range', [0, 0])
  if range_vals and (range_vals[0] != 0 or range_vals[1] != 0):
    parts.append('range: [%s, %s]' % (range_vals[0], range_vals[1]))
  return '. '.join(parts)

def _make_model_doc(func,description,return_states=None):
  input_details = description['Inputs'].details if isinstance(description['Inputs'], VariableList) else []
  output_details = description['Outputs'].details if isinstance(description['Outputs'], VariableList) else []
  inputs_doc = _DOC_SEP.join([
    '%s: Input timeseries (default: zero length)' % _variable_units_str(i, input_details)
    for i in description['Inputs']])
  params_doc = _DOC_SEP.join([_param_doc(p) for p in description['Parameters']])
  states_doc = _DOC_SEP.join(['%s: Initial state variable (default: 0.0)'%s for s in description['States']])
  returns = [
    '%s : Output timeseries' % _variable_units_str(o, output_details)
    for o in description['Outputs']]
  if return_states:
    returns += ['%s: %s'%(s,return_states) for s in description['States']]

  outputs_doc = _DOC_SEP.join(returns)
  func.__doc__ = _DOC_TEMPLATE%(inputs_doc,params_doc,states_doc,outputs_doc)
  func.__output_names__ = list(description['Outputs'])

def _flatten_params(desc,params):
    model_params = desc['Parameters']
    dims = set(sum([p['Dimensions'] for p in model_params],[]))
    if not len(dims):
        return params

    dim_sizes = {d:0 for d in dims}
    for meta,param in zip(model_params,params):
        if meta['Name'] in dim_sizes:
            dim_sizes[meta['Name']] = int(max(dim_sizes[meta['Name']],param))
            continue
        if not len(meta['Dimensions']):
            continue

        if hasattr(param,'shape'):
            shape = param.shape
        elif hasattr(param,'__len__'):
            shape = (len(param),)
        else:
            shape = tuple([1])

        for ix,d in enumerate(meta['Dimensions']):
            dim_sizes[d] = int(max(dim_sizes[d],shape[ix]))

    result = []
    for meta,param in zip(model_params,params):
        if meta['Name'] in dim_sizes:
            result += [dim_sizes[meta['Name']]]
        elif len(meta['Dimensions'])==0:
            result += [param]
        else:
            target_shape = tuple([dim_sizes[d] for d in meta['Dimensions']])
            target_length = reduce(lambda x,y: x*y,target_shape,1)
            if hasattr(param,'shape'):
                if param.shape==target_shape:
                    result += list(param.reshape((target_length,)))
                else:
                    raise Exception('reszing arrays not supported')
            elif hasattr(param,'__len__'):
                if len(param)==target_length:
                    result += param
                else:
                  raise Exception('reizing lists not supported')
            else:
              result += [param]*target_length

    # resize to max
    # set dimension parameter (eg nLVA to max)
    return result

def _collect_arguments(description,args,kwargs):
  param_names = [p['Name'] for p in description['Parameters']]

  inputs=[np.empty(0) for _ in description['Inputs']]
  params=[p['Default'] for p in description['Parameters']]
  states=[np.empty(0) for _ in description['States']]
  outputs=[np.empty(0) for _ in description['Outputs']]

  n_in = len(description['Inputs'])
  n_pa = len(description['Parameters'])
  for i,arg in enumerate(args):
    if i < n_in:
      inputs[i] = arg
    elif i < (n_in+n_pa):
      params[i-n_in] = arg
    else:
      states[i-n_in-n_pa] = arg

  for p,v in kwargs.items():
    if p in description['Inputs']:
      input_i = description['Inputs'].index(p)
      inputs[input_i] = v
    elif p in param_names:
      param_i = param_names.index(p) # TODO
      params[param_i] = v
    else:
      state_i = description['States'].index(p)
      states[state_i] = v

  return inputs, _flatten_params(description,params), states, outputs

def _transform_metadata(model_meta):
  '''Transform raw JSON metadata to use VariableList for Inputs and Outputs.'''
  model_meta['Inputs'] = VariableList(model_meta.get('Inputs', []))
  model_meta['Outputs'] = VariableList(model_meta.get('Outputs', []))
  return model_meta

def discover(*args):
  import os
  import json
  import subprocess
  metadata = json.loads(subprocess.check_output([_exe_path('inspect')]).decode('utf-8'))

  if len(args):
    metadata = {k:v for k,v in metadata.items() if k in args}

  for model_name,model_meta in metadata.items():
    _transform_metadata(model_meta)
    single._create_model_func(model_name,model_meta)
    ensemble._create_model_func(model_name,model_meta)
    lib._create_model_func(model_name,model_meta)
    nodes._create_model_type(model_name,model_meta)
  return list(metadata.keys())

