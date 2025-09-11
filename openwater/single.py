import logging
logger = logging.getLogger(__name__)


def _create_model_func(name,description):
  import sys
  import subprocess
  import json
  from pandas import DataFrame
  import numpy as np
  from .discovery import _exe_path, _collect_arguments, _make_model_doc
  thismodule = sys.modules[__name__]
  executable_path = _exe_path('single')

  def model_func(*args,**kwargs):
    inputs, params, _, _ = _collect_arguments(description,args,kwargs)
    config = {
      'Name':name,
      'Inputs':[],
      'Parameters':[]
    }

    for i in range(len(description['Inputs'])):
      config['Inputs'].append({
        'Name':description['Inputs'][i],
        'Values':list(inputs[i])
      })

    for i in range(len(description['Parameters'])):
      config['Parameters'].append({
        'Name':description['Parameters'][i]['Name'],
        'Value':params[i]
      })

    msg = json.dumps(config,indent=2)
    #print(msg)
    f = open('tmp-%s.json'%name,'w')
    f.write(str(msg))
    f.close()
    msg = msg.encode('utf-8')

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
  _make_model_doc(model_func,description)

  setattr(thismodule,name,model_func)

