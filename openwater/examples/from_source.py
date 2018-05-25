'''
Functionality for importing a Source model structure to Openwater.

Only workable in limited circumstances:

* Unregulated models
* Equivalent component models implemented in openwater
* No 'functions' etc
'''
import os
import numpy as np
from openwater.catchments import SemiLumpedCatchment
import openwater.template as templating
from openwater.config import Parameteriser, ParameterTableAssignment

MODEL_LOOKUP = {}
STANDARD_SOURCE_COLUMN_TRANSLATIONS = {
    'Catchment':'catchment',
    'Constituent':'constituent'   
}

MODEL_PARAMETER_TRANSLATIONS = {
    'EmcDwc':{
        'dryMeanConcentration':'DWC',
        'eventMeanConcentration':'EMC',
        'Functional Unit':'cgu'
    }
}

def init_lookup():
  import openwater.nodes as node_types
  MODEL_LOOKUP.update({
    'EmcDwcCGModel':node_types.EmcDwc,
    'AWBM':node_types.RunoffCoefficient,
    'SimHydCs':node_types.Simhyd
  })


def build_model_lookup(source_source,default=None,simplify_names=True):
    lookup_table = list(zip(source_source.enumerate_names(),source_source.get_models()))
    if simplify_names:
        tags_to_keep = [i for i in range(len(lookup_table[0][0])) if len(set([t[0][i] for t in lookup_table]))>1]
        lookup_table = [(tuple([t[0][i] for i in tags_to_keep]),t[1]) for t in lookup_table]
    def lookup_fn(*args):
        for keys,model in lookup_table:
            if len(keys) != len(args):
                raise Exception('Mismatch keys: Source set looks like: %s. Openwater providing %s'%(str(keys),str(args)))
            match = True
            arg_copy = list(args[:])
            for key in keys:
                if not key in arg_copy:
                    match = False
                    break
                arg_copy.remove(key)
            if match:
                return MODEL_LOOKUP[model.split('.')[-1]]
        if default is None:
            raise Exception('Cannot find model for %s and no default provided'%(str(args)))
        return default
    return lookup_fn

def build_template(v):
  catchment_template = SemiLumpedCatchment()
  fus = set(v.model.catchment.get_functional_unit_types())
  constituents = v.model.get_constituents()
  
  catchment_template.hrus = fus
  catchment_template.cgus = fus
  catchment_template.cgu_hrus = {fu:fu for fu in fus}
  catchment_template.constituents = constituents

  catchment_template.rr = build_model_lookup(v.model.catchment.runoff)
  catchment_template.cg = build_model_lookup(v.model.catchment.generation)
  return catchment_template

def build_catchment_graph(model_structure,network):
  catchments = network['features'].find_by_feature_type('catchment')
  links = network['features'].find_by_feature_type('link')
  nodes = network['features'].find_by_feature_type('node')

  g = None

  for catchment in catchments:
      tpl = model_structure.get_template(catchment=catchment['properties']['name'])
      g = templating.template_to_graph(g,tpl)

  for catchment in catchments:
      src = catchment['properties']['name']
      link = links.find_one_by_id(catchment['properties']['link'])
      ds_node = link['properties']['to_node']
      try:
          ds_link = links.find_one_by_from_node(ds_node)
          ds_catchment = catchments.find_one_by_link(ds_link['id'])
      except:
          ds_catchment = None

      if ds_catchment is None:
          print('%s is end of system'%src)
          continue
      dest = ds_catchment['properties']['name']

      print('%s -> %s'%(src,dest))
      model_structure.link_catchments(g,src,dest)

  return templating.ModelGraph(g)

def build_parameter_lookups(target):
    result = {}
    for model_type, df in target.tabulate_parameters().items():
        ow_type = MODEL_LOOKUP[model_type.split('.')[-1]].name
        if ow_type in MODEL_PARAMETER_TRANSLATIONS:
            translations = {**MODEL_PARAMETER_TRANSLATIONS[ow_type],
                            **STANDARD_SOURCE_COLUMN_TRANSLATIONS}
        else:
            translations = STANDARD_SOURCE_COLUMN_TRANSLATIONS
        result[ow_type] = df.rename(columns=translations)

    return result

def build_input_lookups(target,v):
    length = 0
    result = {}
    name_columns = target.name_columns

    for model_type, df in target.tabulate_inputs().items():
        if length==0:
            for col in df.columns:
                if col in name_columns:
                    continue
                vals = [val for val in df[col] if val!='']
                if len(vals)==0:
                    continue
                data = v.data_source_item(vals[0])
                length = len(data)
        ow_type = MODEL_LOOKUP[model_type.split('.')[-1]].name
        translations = STANDARD_SOURCE_COLUMN_TRANSLATIONS
        result[ow_type] = df.rename(columns=translations)       

    return result,length

def translate(src_veneer,dest_fn):
    if os.path.exists(dest_fn):
        raise Exception('File exists')

    v = src_veneer
    init_lookup()
    catchment_template = build_template(v)
    net = v.network()
    model = build_catchment_graph(catchment_template,net)

    climate_inputs,simulation_length = build_input_lookups(v.model.catchment.runoff,v)
    runoff_parameters = build_parameter_lookups(v.model.catchment.runoff)
    cg_parameters = build_parameter_lookups(v.model.catchment.generation)

    p = Parameteriser()

    for model_type, parameters in {**runoff_parameters,**cg_parameters}.items():
        parameteriser = ParameterTableAssignment(parameters,model_type)
        p.append(parameteriser)

    for model_type, inputs in climate_inputs.items():
        inputter = SourceTimeSeriesTranslator(inputs,model_type,v)
        p.append(inputter)

    model._parameteriser = p

    model.write_model(dest_fn,simulation_length)
    return  model

class SourceTimeSeriesTranslator(object):

    def __init__(self,df,model_type,v):
        self.df=df
        self.model_type=model_type
        self.v=v
        self.data_source_cache = {}

    def get_time_series(self,path):
        if not path in self.data_source_cache:
            self.data_source_cache[path] = np.array(self.v.data_source_item(path))[:,0]
        return self.data_source_cache[path]

    def parameterise(self,model_desc,grp,instances,dims,nodes):
        if model_desc.name != self.model_type:
            return
        print('Input time series for %s'%self.model_type)

        dataset = grp['inputs']
        input_names = model_desc.description['Inputs']

        for _,node in nodes.items():
            subset = self.df
            for dim in dims.keys():
                subset = subset[subset[dim]==node[dim]]
            assert len(subset)==1
            subset = subset.iloc[0]

            run_idx = node['_run_idx']

            for i,name in enumerate(input_names):
                path = subset[name]
                if path == '': continue
                data = self.get_time_series(path)
                dataset[run_idx,i,:] = data
