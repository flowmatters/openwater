from .array_params import get_parameter_locations
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)


MODELS={}

class ReadOnlyObjectDict(object):
  def __init__(self,d):
    self._d = d
  def __dir__(self):
    return list(self._d.keys()) + dir(super())
  def __getattr__(self,k):
    try:
      return self._d[k]
    except KeyError:
      raise AttributeError(k)
  def __repr__(self):
    return repr(self._d.keys())
  def __str__(self):
    return str(self._d.keys())
  def __iter__(self):
    return iter(self._d)
  def __len__(self):
    return len(self._d)
  def __contains__(self,k):
    return k in self._d

def _create_model_type(name,description):
  import sys
  thismodule = sys.modules[__name__]
  desc = ModelDescription(name,description)
  MODELS[name] = description
  setattr(thismodule,name,desc)

class ModelDescription(object):
  def __init__(self,name,description):
    self.name = name
    self.description = description
    self.inputs = ReadOnlyObjectDict({i:i for i in description['Inputs']})
    self.outputs = ReadOnlyObjectDict({o:o for o in description['Outputs']})
    self.states = ReadOnlyObjectDict({s:s for s in description['States']})
    self.parameters = ReadOnlyObjectDict({p['Name']:p['Name'] for p in description['Parameters']})

  def __str__(self):
    return 'Openwater Model Description: %s'%self.name
  
  def __repr__(self):
    return self.__str__()

def create_indexed_parameter_table(desc,raw):
  param_locs = get_parameter_locations(desc,np.array(raw).transpose())
  # param_starts = {}
  idx = {
    'parameter':[],
    'index':[]
  }
  current_idx=0
  for ix,param in enumerate(desc['Parameters']):
    p_start = param_locs[ix][0]
    p_end = param_locs[ix][1]
    # param_starts[param['Name']]=p_start
    idx['parameter'] += [param['Name']] * (p_end-p_start)
    idx['index'] += list(range(p_end-p_start))

  index_names = [n for n in raw.index.names if not n.startswith('_') and n!='node']
  indexed_params = pd.concat([raw.transpose(),pd.DataFrame(idx)],axis=1).set_index(['parameter','index'])
  indexed_params = indexed_params.transpose()
  logger.debug(f'{indexed_params.index.names}, {index_names}')
  indexed_params.index.names = index_names
  return indexed_params

def models_with_term(term,term_type=None):
  '''
  Find all model types that have a timeseries flux matching 'term'

  Parameters
  ----------
  term : str
    The term to search for
  term_type : str, optional
    The type of term to search for (e.g. 'Input','Output')

  Returns
  -------
  list
    A list of model types that have the term
  '''
  if term_type is None:
    return list(set(models_with_term(term,'Input') + models_with_term(term,'Output')))

  if not term_type.endswith('s'):
    term_type = term_type+'s'

  result = [k for k,desc in MODELS.items() if term in desc[term_type]]
  return result
