import logging
logger = logging.getLogger(__name__)


_DOC_SEP='\n  * '
_DOC_TEMPLATE='''
Function parameters:
  * %s
  * %s

Returns
  * %s
'''




def _create_model_func(name,description):
  import sys
  import subprocess
  import json
  from pandas import DataFrame
  import numpy as np
  from .discovery import _exe_path
  thismodule = sys.modules[__name__]
  executable_path = _exe_path('ensemble','ows')

  def model_func(*args,**kwargs):
    param_names = [p['Name'] for p in description['Parameters']]

    inputs=[[] for _ in description['Inputs']]
    params=[p['Default'] for p in description['Parameters']]

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

    config = {
      'Name':name,
      'Inputs':[],
      'Parameters':[]
    }

    for i in range(len(description['Inputs'])):
      config['Inputs'].append({
        'Name':description['Inputs'][i],
        'Values':inputs[i]
      })

    for i in range(len(description['Parameters'])):
      config['Parameters'].append({
        'Name':description['Parameters'][i]['Name'],
        'Value':params[i]
      })

    msg = json.dumps(config).encode('utf-8')
    #print(msg)

    p = subprocess.Popen([executable_path,name],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         stdin=subprocess.PIPE)
    out, err = p.communicate(msg)
    out = out.decode('utf-8')
    err = err.decode('utf-8')
    try:
      out = json.loads(out)
      arr = np.array(out['RunResults']['Outputs']).transpose()
      out['RunResults']['Outputs'] = DataFrame(arr,columns=description['Outputs'])
    except Exception as e: 
      logger.error(e)
      pass
    return out,err

  model_func.__name__ = name

  inputs_doc = _DOC_SEP.join(['%s: Input timeseries (default: zero length)'%i for i in description['Inputs']])
  params_doc = _DOC_SEP.join(['%s: Mode; parameter (default: %f)'%(p['Name'],p['Default']) for p in description['Parameters']])
  outputs_doc = _DOC_SEP.join(['%s : Output timeseries'%o for o in description['Outputs']])
  model_func.__doc__ = _DOC_TEMPLATE%(inputs_doc,params_doc,outputs_doc)

  setattr(thismodule,name,model_func)