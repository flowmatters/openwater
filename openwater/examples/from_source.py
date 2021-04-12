'''
Functionality for importing a Source model structure to Openwater.

Only workable in limited circumstances:

* Unregulated models
* Equivalent component models implemented in openwater
* No 'functions' etc
'''
import os
from glob import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import openwater.nodes as n
from openwater.catchments import SemiLumpedCatchment, \
    DOWNSTREAM_FLOW_FLUX, DOWNSTREAM_LOAD_FLUX, \
    UPSTREAM_FLOW_FLUX, UPSTREAM_LOAD_FLUX
import openwater.template as templating
from openwater.template import OWLink
from openwater.config import Parameteriser, ParameterTableAssignment, \
    NestedParameteriser, DataframeInputs, UniformParameteriser
from openwater.timing import init_timer, report_time, close_timer
import json
from veneer.general import _extend_network

EXPECTED_LINK_PREFIX='link for catchment '

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
    },
    'Surm':{
        'fieldCapacityFraction':'fcFrac',
        's1max':'smax',
        'Functional Unit':'hru'
    }
}

LOOKUP_COLUMN_ORDER=['Constituent','Functional Unit','Catchment','Constituent Source']

def init_lookup():
  import openwater.nodes as node_types
  MODEL_LOOKUP.update({
    'EmcDwcCGModel':node_types.EmcDwc,
    'AWBM':node_types.RunoffCoefficient,
    'SimHydCs':node_types.Simhyd,
    'MusicRR':node_types.Surm,
    'NullRainfallModel':None,
    'StraightThroughRouting':node_types.Lag
  })

def df_model_lookup(df,default=None):
    if df is None or not len(df):
        return default

    key_columns = [c for c in LOOKUP_COLUMN_ORDER if c in df.columns]
    #key_columns = [c for c in df.columns if c != 'model']
    print('key_columns',key_columns)
    keys = df[key_columns].to_dict(orient='records')
    keys = [tuple([k[kc] for kc in key_columns]) for k in keys]
    vals = [m.split('.')[-1] for m in list(df.model)]
    lookup_table = dict(zip(keys,vals))
    print('table of %d rows'%len(lookup_table))
    return build_model_lookup(lookup_table=lookup_table,default=default)


def build_model_lookup(source_source=None,lookup_table=None,default=None,simplify_names=True):
    if lookup_table is None:
        lookup_table = list(zip(source_source.enumerate_names(),source_source.get_models()))
        if simplify_names:
            tags_to_keep = [i for i in range(len(lookup_table[0][0])) if len(set([t[0][i] for t in lookup_table]))>1]
            lookup_table = [(tuple([t[0][i] for i in tags_to_keep]),t[1]) for t in lookup_table]
            lookup_table = dict(lookup_table)
    ##### TODO: This thing is accepting ordered arguments.
    ##### That seems daft... Should be keywords rather than relying on the same order every time
    def lookup_fn(*args):
        model = lookup_table.get(tuple(args))
        if model is not None:
            return MODEL_LOOKUP[model.split('.')[-1]]
        # for keys,model in lookup_table:
        #     match = True
        #     arg_copy = list(args[:])
        #     for key in keys:
        #         if not key in arg_copy:
        #             match = False
        #             break
        #         arg_copy.remove(key)
        #     if match:
        #         return MODEL_LOOKUP[model.split('.')[-1]]
        # print('No match',args,lookup_table[0])

        eg_key = list(lookup_table.keys())[0]
        if len(args) != len(eg_key):
            raise Exception('Mismatch keys: Source set looks like: %s. Openwater providing %s'%(str(eg_key),str(args)))

        if default is None:
            raise Exception('Cannot find model for %s and no default provided'%(str(args)))
        return default
    return lookup_fn


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

def make_node_linkages(row,nodes):
    linkages = []
    assert row.from_node == row.to_node
    from_c = row.from_name
    to_c = row.to_name
    direct_link_catchments = True #not(pd.isna(from_c) or pd.isna(to_c))

    if pd.isna(from_c):
      std_source = {
        'link_name':row.link_name
      }
    else:
      std_source = {
        'catchment':from_c
      }

    if pd.isna(to_c):
      std_dest = {
        'link_name':row.to_link_name
      }
    else:
      std_dest = {
        'catchment':to_c
      }

    if row.to_node in nodes.id:
      node_name = list(nodes[nodes.id==row.to_node].name)[0]
      linkages.append((
        {
          'node_name':node_name
        },
        std_dest
      ))

      # IF node takes upstream input
      # direct_link_catchments = False

    if direct_link_catchments:
        linkages.append((
          std_source,
          std_dest
        ))
    return linkages

def find_upstream_connectibles(link,connections,links,nodes):
  connectibles = []
  link_details = links[links.id==invalid_row.link].iloc[0]
  us_node = link_details.from_node
  us_linkages = connections[connections.to_node==us_node]
  for ix, us_row in us_linkages.iterrows():
    if pd.isna(us_row.from_name) and us_row.to_node not in nodes.id:
      connectibles += find_upstream_connectibles(us_row.link,connections,links,nodes)
      continue
    connectibles.append({
      'from_name':'',
      'link':'',
      'to_node':'',
      'to_name':'',
      'from_node':''
    })
  return connectibles

def make_network_topology(catchments,nodes,links):
  links['link_name'] = links['name']
  c2l = pd.merge(catchments[['name','link']],
                 links[['id','link_name','from_node','to_node']],
                 left_on='link',right_on='id',how='right')
  c2l['from_name'] = c2l['name']
  c2l['to_name'] = c2l['name']
  c2l['to_link_name'] = c2l['link_name']

  c2l['link'] = c2l['id']
  node_connection_table = pd.merge(c2l[['from_name','link','link_name','to_node']],
                        c2l[['to_name','to_link_name','from_node']],
                        left_on='to_node',right_on='from_node',how='inner')

  # node_connection_table = collapse_links(node_connection_table,nodes,links)

  linkages = []
  for ix, row in node_connection_table.iterrows():
    node_linkages = make_node_linkages(row,nodes)
    if not len(node_linkages):
      raise Exception('No linkages made for %s'%str(row))
    linkages += node_linkages

  return linkages

def build_catchment_graph(model_structure,network,progress=print,custom_processing=None):
  init_timer('Sort features')
  if hasattr(network,'as_dataframe'):
    network = network.as_dataframe()

  catchments = network[network.feature_type=='catchment']
  links = network[network.feature_type=='link']
  nodes = network[network.feature_type=='node']

  templates = []
#   g = None
  report_time('Build catchments')
  for _,catchment in catchments.iterrows():
      tpl = model_structure.get_template(catchment=catchment['name']).flatten()
      templates.append(tpl)
    #   g = templating.template_to_graph(g,tpl)

  for _,link in links.iterrows():
    # print(link.id)
    # print(sorted(list(catchments.link)))
    if catchments.link.isin([link.id]).any():
      continue

    print(link.name,link.id,'has no catchment, getting link template')

    tpl = model_structure.get_link_template(link_name=link['name']).flatten()
    templates.append(tpl)

  interesting_nodes = filter_nodes(nodes)
  print(interesting_nodes[['id','name','icon']])
  # TODO! Re-enable
  for ix,node in interesting_nodes.iterrows():
      print(ix,node)
      node_type = node['icon'].split('/')[-1].replace('NodeModel','')
      tpl = model_structure.get_node_template(node_name=node['name'],node_type=node_type).flatten()
      templates.append(tpl)
    #   g = templating.template_to_graph(g,tpl)

  report_time('Link catchments')

  # TODO - Interesting nodes. If to_node / from_node between links,
  #        then process... somehow...

  master_template = templating.OWTemplate()

#   g = None
  for tpl in templates:
      master_template.nest(tpl)
    #   g = templating.template_to_graph(g,tpl)
  master_template = master_template.flatten()

#   g = None
#   g = templating.template_to_graph(g,master_template)

#   froms = list(join_table.from_name)
#   tos = list(join_table.to_name)

#   for from_c,to_c in zip(froms,tos):

  linkages = make_network_topology(catchments,interesting_nodes,links)
  for ix, (from_link,to_link) in enumerate(linkages):
    flow_link = master_template.make_link(
            DOWNSTREAM_FLOW_FLUX,
            UPSTREAM_FLOW_FLUX,
            from_tags=from_link,
            to_tags=to_link,
            from_exclude_tags=['constituent'],
            to_exclude_tags=['constituent'])
    master_template.add_link(flow_link)

    for con in model_structure.constituents:
      from_link.update(constituent=con)
      to_link.update(constituent=con)
      con_link = master_template.make_link(
        DOWNSTREAM_LOAD_FLUX,
        UPSTREAM_LOAD_FLUX,
        from_tags=from_link,
        to_tags=to_link)
      master_template.add_link(con_link)

      # if node has upstream connectors, link from from_c to node
      # else link from from_c to to_c

  g = None
  g = templating.template_to_graph(g,master_template)

#   assert False
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

def get_default_node_template(node_type,constituents,**kwargs):
    template = templating.OWTemplate(node_type)

    if node_type not in DEFAULT_NODE_TEMPLATES:
        raise Exception(f'Unsupported node: {node_type} at {kwargs.get("node","unnamed node")}')

    DEFAULT_NODE_TEMPLATES[node_type](template,constituents,**kwargs)
    return template

def build_parameter_lookups(target):
    result = {}
    if hasattr(target,'tabulate_parameters'):
        target = target.tabulate_parameters()
    for model_type, df in target.items():
        ow_model_type = MODEL_LOOKUP[model_type.split('.')[-1]]
        if ow_model_type is None:
            continue
        ow_type = ow_model_type.name
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

class VeneerModelConfigurationProvider(object):
    def __init__(self,v):
        self.v = v

class FileBasedModelConfigurationProvider(object):
    def __init__(self,path,climate_patterns):
        self.data_path = path
        self.climate_patterns = climate_patterns

    def _find_files(self,pattern,ignoring=[]):
        files = [os.path.basename(fn) for fn in \
            glob(os.path.join(self.data_path,pattern))]
        def in_ignore_list(fn):
            return len([pattern for pattern in ignoring if pattern in fn])

        files = [fn for fn in files if not in_ignore_list(fn)]
        return files

    def _load_json(self,f):
        return json.load(open(os.path.join(self.data_path, f + '.json')))

    def _load_csv(self,f):
        fn = os.path.join(self.data_path, f)
        if not fn.endswith('csv'):
            fn = fn + '.csv'

        if not os.path.exists(fn):
            fn = fn + '.gz'
            if not os.path.exists(fn):
                return None

        data = pd.read_csv(fn, index_col=0, parse_dates=True)
        if 'NetworkElement' in data.columns:
            data['Catchment'] = data['NetworkElement'].str.replace(EXPECTED_LINK_PREFIX,'',case=False)

        return data

    def get_functional_unit_types(self):
        return self._load_json('fus')

    def get_constituents(self):
        return self._load_json('constituents')

    def get_fu_areas(self):
        return self._load_csv('fu_areas')

    def runoff_models(self):
        return df_model_lookup(self._load_csv('runoff_models'))

    def routing_models(self):
        return df_model_lookup(self._load_csv('routing_models'))

    def generation_models(self):
        import openwater.nodes as node_types
        return df_model_lookup(self._load_csv('cgmodels'),node_types.EmcDwc)

    def transport_models(self):
        import openwater.nodes as node_types
        return df_model_lookup(self._load_csv('transportmodels'),node_types.Lag)

    def network(self):
        network = gpd.read_file(os.path.join(self.data_path, 'network.json'))
        network_veneer = _extend_network(self._load_json('network'))
        return network_veneer

    def _load_parameters(self,fn_prefix,model_table):
        models = set(model_table.model)
        return {m:self._load_csv(f'{fn_prefix}{m}') for m in models}

    def runoff_parameters(self):
        return self._load_parameters('rr-',self._load_csv('runoff_models'))

    def generation_parameters(self):
        return self._load_parameters('cg-',self._load_csv('cgmodels'))

    def routing_parameters(self):
        return self._load_parameters('fr-',self._load_csv('routing_models'))

    def climate_data(self):
        data = self._load_csv('climate')
        data = data[:(365*24)] # 1 year

        time_steps = data.index
        delta_t = (time_steps[1]-time_steps[0]).total_seconds()

        inputter = DataframeInputs()
        for variable,column in self.climate_patterns.items():
            inputter.inputter(data,variable,column)
        return inputter, data.index, delta_t

    def inflows(self,time_period):
        #### TODO: Units!

        files = self._find_files('timeseries-inflow-*.csv',[
            'timeseries-inflow-concentration',
            'timeseries-inflow-load'
        ])
        if not len(files):
            return None

        def make_col_name(fn):
            return fn.replace('timeseries-inflow-','').replace('.csv','')
        all_data = [self._load_csv(fn) for fn in files]
        all_data = pd.DataFrame({make_col_name(fn):df[df.columns[0]] \
            for fn,df in zip(files,all_data)})
        all_data = all_data.reindex(time_period)
        return all_data

    def inflow_loads(self,inflows):
        conc_files = self._find_files('timeseries-inflow-concentration-*.csv')

        if not len(conc_files):
            return None

        def make_conc_col_name(fn):
            fn = fn.replace('timeseries-inflow-concentration-','').replace('.csv','')
            components = fn.split('-')
            constituent = components[0]
            node_name = '-'.join(components[1:])
            return f'{node_name}:{constituent}'

        all_conc_data = [self._load_csv(fn) for fn in conc_files]
        all_conc_data = pd.DataFrame({make_conc_col_name(fn):df[df.columns[0]] \
            for fn,df in zip(conc_files,all_conc_data)})
        all_conc_data = all_conc_data.reindex(inflows.index)

        loads = all_conc_data.copy()
        for col in loads.columns:
            node_name = col.split(':')[0]
            inflow_ts = inflows[node_name]
            ### TODO: UNITS!
            loads[col] *= inflow_ts
        return loads


class SourceOpenwaterModelBuilder(object):
    def __init__(self,source_provider):
        self.provider = source_provider

    def build_catchment_template(self):
        catchment_template = SemiLumpedCatchment()
        catchment_template.node_template = get_default_node_template
        fus = set(self.provider.get_functional_unit_types())
        constituents = self.provider.get_constituents()

        catchment_template.hrus = fus
        catchment_template.cgus = fus
        catchment_template.cgu_hrus = {fu:fu for fu in fus} # Allow overriding!
        catchment_template.constituents = constituents

        catchment_template.rr = self.provider.runoff_models()
        catchment_template.cg = self.provider.generation_models()
        catchment_template.routing = self.provider.routing_models()
        catchment_template.transport = self.provider.transport_models()
        # TODO Constituent transport 

        return catchment_template

    def _fu_areas_parameteriser(self):
        res = NestedParameteriser()
        fu_areas = self.provider.get_fu_areas()
        area_models = [
            ('DepthToRate','area')
        ]
        for model,prop in area_models:
            res.nested.append(ParameterTableAssignment(fu_areas,model,prop,'cgu','catchment'))

        return res

    def build_ow(self,dest_fn):
        if os.path.exists(dest_fn):
            raise Exception('File exists')

        init_lookup()

        print('Building parameters')
        # TEMPORARILY DISABLE TO TEST LINKS
        runoff_parameters = build_parameter_lookups(self.provider.runoff_parameters())
        cg_parameters = build_parameter_lookups(self.provider.generation_parameters())
        routing_params = build_parameter_lookups(self.provider.routing_parameters())
        # /TEMPORARILY DISABLE TO TEST LINKS

        print('Building model structure')
        catchment_template = self.build_catchment_template()
        net = self.provider.network()
        model = build_catchment_graph(catchment_template,net)
        print('Got graph, configuring parameters')


        p = Parameteriser()
        # TEMPORARILY DISABLE TO TEST LINKS
        p.append(self._fu_areas_parameteriser())

        for model_type, parameters in {**runoff_parameters,**cg_parameters}.items():
            print('Building parameteriser for %s'%model_type)
            parameteriser = ParameterTableAssignment(parameters,model_type)
            p.append(parameteriser)

        print('Configuring climate data')
        climate_inputs,time_period, delta_t = self.provider.climate_data()
        simulation_length = len(time_period)
        p.append(climate_inputs)
        # /TEMPORARILY DISABLE TO TEST LINKS
        # simulation_length = 10

        print('Configuring inflow timeseries')
        inflows = self.provider.inflows(time_period)
        if inflows is not None:
            inflow_inputs = DataframeInputs()
            p.append(inflow_inputs)

            inflow_inputs.inputter(inflows,'input','${node_name}',model='Input')

            inflow_loads = self.provider.inflow_loads(inflows)
            inflow_inputs.inputter(inflow_loads,'inputLoad','${node_name}:${constituent}',model='PassLoadIfFlow')

        delta_t_parameteriser = UniformParameteriser('DepthToRate',DeltaT=delta_t)
        p.append(delta_t_parameteriser)


        # Not needed
        # for model_type, inputs in climate_inputs.items():
        #     inputter = SourceTimeSeriesTranslator(inputs,model_type,v)
        #     p.append(inputter)

        model._parameteriser = p

        print('Writing model')
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

def build_extraction_node_template(template,constituents,**kwargs):
    demand = template.add_node(n.PartitionDemand,process='demand',**kwargs)
    prop = template.add_node(n.ComputeProportion,process='demand_proportion',**kwargs)
    template.add_link(OWLink(demand,'extraction',prop,'numerator'))
    for con in constituents:
        con_ext = template.add_node(n.VariablePartition,process='constituent_extraction',constituent=con,**kwargs)
        template.add_link(OWLink(prop,'proportion',con_ext,'fraction'))

def build_inflow_node_template(template:templating.OWTemplate,constituents: list,**kwargs):
    flow_node = template.add_node(n.Input,process='inflow',**kwargs)
    template.define_output(flow_node,'output',DOWNSTREAM_FLOW_FLUX,**kwargs)

    for con in constituents:
        load_input = template.add_node(n.PassLoadIfFlow,process='inflow_load',constituent=con,**kwargs)
        template.add_link(OWLink(flow_node,'output',load_input,'flow'))
        template.define_output(load_input,'outputLoad',DOWNSTREAM_LOAD_FLUX,constituent=con,**kwargs)

def build_storage_node_template(template,constituents,**kwargs):
    raise Exception(f'Storages not supported at {kwargs.get("node","unnamed node")}')
    #     storage = template.add_node(n.Storage,process='storage',**kwargs)
    #     for con in self.constituents:
    #         con_ext = template.add_node(n.???,process='extraction',constituent=con,**kwargs)
    #         template.add_link(storage,'proportion',con_ext,'fraction')

DEFAULT_NODE_TEMPLATES={
    'Extraction':build_extraction_node_template,
    'InjectedFlow':build_inflow_node_template,
    'Storage':build_storage_node_template
}
