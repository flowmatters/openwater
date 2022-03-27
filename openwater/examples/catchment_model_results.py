from openwater.results import OpenwaterResults, OpenwaterSplitResults
from veneer.general import _extend_network
import pandas as pd
import json
import os

class OpenwaterCatchmentModelResults(object):
  def __init__(self,base_name):
    if base_name.endswith('.h5'):
      base_name = base_name[:-3]
    model_fn = base_name +'.h5'
    model_results_fn = base_name + '-outputs.h5'
    if os.path.exists(model_results_fn):
      self.results = OpenwaterResults(model_fn,model_results_fn)
    elif os.path.exists(base_name + '-outputs-0.h5'):
      from glob import glob
      outputs = list(sorted(glob(base_name+'-outputs-?.h5')))
      inputs = list(sorted(glob(base_name+'-inputs-?.h5')))
      if not len(inputs):
        inputs = [None for _ in outputs]
      splits = [OpenwaterResults(model_fn,output_fn,inputs=input_fn) \
                        for output_fn,input_fn in zip(outputs,inputs)]
      self.results = OpenwaterSplitResults(splits=splits)
    else:
      raise Exception(f'Could not find results for model basename {base_name}')

    with open(f'{base_name}.network.json','r') as fp:
      self.network = _extend_network(json.load(fp))

    self.catchments = self.results.dim('catchment')

    with open(f'{base_name}.meta.json','r') as fp:
      self.meta = json.load(fp)
    self.meta['fus'] = self.results.dim('cgu')

    self.results.time_period = pd.date_range(self.meta['start'],
                                             self.meta['end'],
                                             freq=self.meta.get('timestep',None))
  def generation_model(self,c,fu):
    return 'EmcDwc','totalLoad'

  def transport_model(self,c):
    return 'LumpedConstituentRouting','outflowLoad'

  def regulated_links(self):
    from veneer.extensions import _feature_id
    network = self.network
    outlet_nodes = network.outlet_nodes()
    outlets = [n['properties']['name'] for n in outlet_nodes]
    network.partition(outlets,'outlet')
    storages = network['features'].find_by_icon('/resources/StorageNodeModel')
    extractions = network['features'].find_by_icon('/resources/ExtractionNodeModel')

    impacted_by_storage = []
    for s in storages._list+extractions._list:
        outlet = s['properties']['outlet']
        outlet_id = _feature_id(network['features'].find_by_name(outlet)[0])
        impacted_by_storage += network.path_between(s,outlet_id)

    ids = set([_feature_id(f) for f in impacted_by_storage])
    network_df = network.as_dataframe()
    impacted_by_storage = network_df[network_df['id'].isin(ids)]
    links_downstream_storage = [l.replace('link for catchment ','') for l in impacted_by_storage[impacted_by_storage.feature_type=='link'].name]
    return links_downstream_storage
