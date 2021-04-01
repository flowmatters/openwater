'''
Functionality for importing a Source model structure to Openwater.

Only workable in limited circumstances:

* Unregulated models
* Equivalent component models implemented in openwater
* No 'functions' etc
'''
import os
import numpy as np
import pandas as pd
from openwater.catchments import SemiLumpedCatchment
import openwater.template as templating
from openwater.config import Parameteriser, ParameterTableAssignment
from openwater.timing import init_timer, report_time, close_timer

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

def link_catchment_lookup(network):
  res = {}

  catchments = network[network.feature_type=='catchment']
  links = network[network.feature_type=='link']

  for _,catchment in catchments.iterrows():
    cname = catchment['name']
    link = links[links.id==catchment['link']].iloc[0]
    res[link['name']] = cname

  return res

def filter_nodes(nodes):
    IGNORE_NODES=['Confluence','Gauge','WaterUser']
    for ignore in IGNORE_NODES:
        nodes = nodes[~nodes.icon.str.contains(ignore)]
    return nodes

def build_catchment_graph(model_structure,network,progress=print,custom_processing=None):
  init_timer('Sort features')
  if hasattr(network,'as_dataframe'):
    network = network.as_dataframe()

  catchments = network[network.feature_type=='catchment']
  links = network[network.feature_type=='link']
  nodes = network[network.feature_type=='node']

  g = None
  report_time('Build catchments')
  for _,catchment in catchments.iterrows():
      tpl = model_structure.get_template(catchment=catchment['name']).flatten()
      g = templating.template_to_graph(g,tpl)

  interesting_nodes = filter_nodes(nodes)
  print(interesting_nodes[['id','name','icon']])
  # TODO! Re-enable
#   for _,node in interesting_nodes.iterrows():
#       node_type = node['icon'].split('/')[-1].replace('NodeModel','')
#       tpl = model_structure.get_node_template(node=node['name'],node_type=node_type).flatten()
#       g = templating.template_to_graph(g,tpl)

  report_time('Link catchments')
  c2l = pd.merge(catchments[['name','link']],links[['id','from_node','to_node']],left_on='link',right_on='id',how='inner')
  c2l['from_name'] = c2l['name']
  c2l['to_name'] = c2l['name']
  join_table = pd.merge(c2l[['from_name','link','to_node']],c2l[['to_name','from_node']],left_on='to_node',right_on='from_node',how='inner')
  froms = list(join_table.from_name)
  tos = list(join_table.to_name)
  for from_c,to_c in zip(froms,tos):
      model_structure.link_catchments(g,from_c,to_c)

  # Detect where we have a node of interest and link to it... (and from it to the next link)


#   for _,feature in join_table.iterrows():
#       model_structure.link_catchments(g,feature['from_name'],feature['to_name'])
    #   src = feature['name']
    #   #link = links[links.id==catchment['link']].iloc[0]
    #   ds_node = feature['to_node']
    #   ds_links = links[links.from_node==ds_node]['id']
    #   ds_catchments = catchments[catchments.link.isin(ds_links)]
    #   if len(ds_catchments):
    #       ds_catchment = ds_catchments.iloc[0]
    #   else:
    #       progress('%s is end of system'%src)
    #       continue

    #   dest = ds_catchment['name']

    #   progress('%s -> %s'%(src,dest))
    #   model_structure.link_catchments(g,src,dest)

#   for _,catchment in catchments.iterrows():
#       src = catchment['name']
#       link = links[links.id==catchment['link']].iloc[0]
#       ds_node = link['to_node']
#       ds_links = links[links.from_node==ds_node]['id']
#       ds_catchments = catchments[catchments.link.isin(ds_links)]
#       if len(ds_catchments):
#           ds_catchment = ds_catchments.iloc[0]
#       else:
#           progress('%s is end of system'%src)
#           continue

#       dest = ds_catchment['name']

#       progress('%s -> %s'%(src,dest))
#       model_structure.link_catchments(g,src,dest)

  if custom_processing is not None:
      report_time('Custom graph processing')
      g = custom_processing(g)

  report_time('Initialise ModelGraph') # About 3/4 of time (mostly in compute simulation order)
  res = templating.ModelGraph(g)
  close_timer()

  return res

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
