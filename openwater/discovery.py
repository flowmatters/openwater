
from . import single, ensemble, lib, nodes
import os
OW_BIN=os.path.join(os.path.expanduser('~'),'src/projects/openwater')

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
  else:
    result += '.so'
  return result

_DOC_SEP='\n  * '
_DOC_TEMPLATE='''
Function parameters:
  * %s
  * %s

Returns
  * %s
'''

def _make_model_doc(func,description):
  inputs_doc = _DOC_SEP.join(['%s: Input timeseries (default: zero length)'%i for i in description['Inputs']])
  params_doc = _DOC_SEP.join(['%s: Mode; parameter (default: %f)'%(p['Name'],p['Default']) for p in description['Parameters']])
  outputs_doc = _DOC_SEP.join(['%s : Output timeseries'%o for o in description['Outputs']])
  func.__doc__ = _DOC_TEMPLATE%(inputs_doc,params_doc,outputs_doc)


def _collect_arguments(description,args,kwargs):
  param_names = [p['Name'] for p in description['Parameters']]

  inputs=[[] for _ in description['Inputs']]
  params=[p['Default'] for p in description['Parameters']]
  states=[[] for _ in description['States']]
  outputs=[[] for _ in description['Outputs']]

  n_in = len(description['Inputs'])
  for i,arg in enumerate(args):
    if i < n_in:
      inputs[i] = list(arg)
    else:
      params[i-n_in] = arg

  for p,v in kwargs.items():
    if p in description['Inputs']:
      input_i = description['Inputs'].index(p)
      inputs[input_i] = v
    else:
      param_i = param_names.index(p) # TODO
      params[param_i] = v

  return inputs, params, states, outputs

def discover():
  import os
  import json
  import subprocess
  metadata = json.loads(subprocess.check_output([_exe_path('inspect')]))

  for model_name,model_meta in metadata.items():
    single._create_model_func(model_name,model_meta)
    ensemble._create_model_func(model_name,model_meta)
    lib._create_model_func(model_name,model_meta)
    nodes._create_model_type(model_name,model_meta)
  return list(metadata.keys())


