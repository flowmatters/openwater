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
from openwater.discovery import set_exe_path,discover
import pandas as pd
import geopandas as gpd
import openwater.nodes as n
from openwater.catchments import SemiLumpedCatchment, \
    DOWNSTREAM_FLOW_FLUX, DOWNSTREAM_LOAD_FLUX, \
    UPSTREAM_FLOW_FLUX, UPSTREAM_LOAD_FLUX, \
    get_model_for_provider
import openwater.template as templating
from openwater.template import OWLink,ModelFile
from openwater.config import Parameteriser, ParameterTableAssignment, \
    NestedParameteriser, DataframeInputs, UniformParameteriser, \
    LoadArraysParameters
from openwater.timing import init_timer, report_time, close_timer
import json
from functools import reduce
from veneer.general import _extend_network
from veneer.utils import split_network
from .const import *
import logging
logger = logging.getLogger(__name__)

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

DEFAULT_AREAL_MODELS = [
    ('DepthToRate','area')
]

def _create_surm_initial_states(raw_params):
  raw_params['SoilMoistureStore'] = raw_params['initialSoilMoistureFraction'] * raw_params['smax']
  raw_params['Groundwater'] = raw_params['initialGroundwaterLevel']
  return raw_params[['catchment','hru','SoilMoistureStore','Groundwater']]

MODELS_WITH_INITIAL_STATES_AS_PARAMETERS = {
  'Surm':_create_surm_initial_states
}

def init_lookup():
  import openwater.nodes as node_types
  MODEL_LOOKUP.update({
    'EmcDwcCGModel':node_types.EmcDwc,
    'AWBM':node_types.RunoffCoefficient,
    'SimHydCs':node_types.Simhyd,
    'Sacramento':node_types.Sacramento,
    'MusicRR':node_types.Surm,
    'NullRainfallModel':None,
    'StraightThroughRouting':node_types.Lag,
    'StorageRouting':node_types.StorageRouting,
    'NullLinkInstreamModel':node_types.LumpedConstituentRouting
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
    eg_key = list(lookup_table.keys())[0]
    if len(args) < len(eg_key):
      raise Exception('Mismatch keys: Source set looks like: %s. Openwater providing %s'%(str(eg_key),str(args)))

    model = lookup_table.get(tuple(args[:len(eg_key)]))
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


    if default is None:
      raise Exception('Cannot find model for %s and no default provided'%(str(args)))
    return default
  return lookup_fn

def merge_storage_tables(directory,fsvs,fsls):
    def read_csv(fn):
        fn = os.path.join(directory,f'{fn}.csv')
        if not os.path.exists(fn):
            fn = fn + '.gz'
        return pd.read_csv(fn,index_col=0)
    meta_fn = os.path.join(directory,'storage_meta.json')
    if not os.path.exists(meta_fn):
      return None

    storage_meta = json.load(open(meta_fn))

    outlet_links = storage_meta['outlet_links']
    storages = list(outlet_links.keys())
    outlets = storage_meta['outlets']
    outlets = {n:outlets[l[0]] for n,l in outlet_links.items()}

    result = {}
    for node,node_outlets in outlets.items():
        lva = read_csv(f'storage_lva_{node}')
        lva = lva.drop_duplicates(subset='volume',keep='last').reset_index()
        if min(lva.volume) > 0:
            logger.warning('No zero row in storage LVA: %s'%node)
            lva = lva.append({'volume':0,'area':0,'level':0},ignore_index=True)
        if min(lva.area) > 0:
            logger.warning('No zero minimum area in storage LVA: %s'%node)
            lva.loc[0,'area'] = 0

        lva = lva.set_index('volume')
        if fsvs[node] not in lva.index:
            lva.loc[fsvs[node],'level'] = round(fsls[node],3)
        lva = lva.sort_index()
        lva = lva.interpolate()
        lva = lva.reset_index().set_index('level')

        release_curves = []
        levels = set(lva.index)
        for outlet in node_outlets:
            release_curve = read_csv(f'storage_release_{node}_{outlet}')
            levels = levels.union(set(release_curve.level))
            release_curve = release_curve.set_index('level')
            release_curves.append(release_curve)
        levels = sorted(levels)
        release_curves = [tbl.reindex(levels).interpolate() for tbl in release_curves]
        lva = lva.reindex(levels)
        lva = lva.interpolate()

        node_table = reduce(lambda a,b: a+b,release_curves)
        node_table['volumes'] = lva.volume
        node_table['areas'] = lva.area
        node_table = node_table.reset_index().rename(columns={
            'minimum':'minRelease',
            'maximum':'maxRelease',
            'level':'levels'
        })
        result[node] = node_table
    return result

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
    IGNORE_NODES=['Confluence','Gauge','WaterUser','MinimumFlow']
    for ignore in IGNORE_NODES:
        nodes = nodes[~nodes.icon.str.contains(ignore)]
    return nodes

def _catchment_or_link(c,l):
    if pd.isna(c):
        if pd.isna(l):
            return None

        return {
            'link_name':l
        }
    return {
        'catchment':c
    }

def make_node_linkages(rows,nodes):
    assert len(set(rows.from_node))==1

    the_node = {
        'node_name':nodes[nodes.id==rows.from_node.iloc[0]].name.iloc[0]
    }

    linkages = []
    to_linkages = set()
    from_linkages = set()

    def unmake_link(c,l):
        the_link = _catchment_or_link(c,l)
        if the_link is None:
            return None
        return (list(the_link.keys())[0],list(the_link.values())[0])
    def remake_link(lnk):
        return {
            lnk[0]:lnk[1]
        }

    for _,row in rows.iterrows():
        to_linkages = to_linkages.union({unmake_link(row.from_name,row.from_link_name)})
        from_linkages = from_linkages.union({unmake_link(row.to_name,row.to_link_name)})

    to_linkages = to_linkages - {None}
    for to_link in to_linkages:
        linkages.append((remake_link(to_link),the_node))

    from_linkages = from_linkages - {None}
    for from_link in from_linkages:
        linkages.append((the_node,remake_link(from_link)))
    return linkages

    # from_c = row.from_name
    # to_c = row.to_name
    # direct_link_catchments = not pd.isna(row.to_node) #not(pd.isna(from_c) or pd.isna(to_c))

    # std_source = _catchment_or_link(from_c,row.link_name)
    # std_dest = _catchment_or_link(to_c,row.to_link_name)

    # link_from_node = None
    # if (nodes is not None) and (nodes.id.isin([row.from_node]).any() or nodes.id.isin([row.to_node]).any()):
    #     if pd.isna(row.to_node):
    #         relevant_node = row.from_node
    #     else:
    #         relevant_node = row.to_node

    #     node_details = nodes[nodes.id==relevant_node]
    #     node_name = node_details.name.iloc[0]
    #     node_linkage = {
    #         'node_name':node_name
    #     }
    #     linkages.append((
    #         node_linkage,
    #         std_dest
    #     ))

    #     if node_details.accepts_inflow.iloc[0]:
    #         direct_link_catchments = False
    #         linkages.append((
    #             std_source,
    #             node_linkage
    #         ))

    # if direct_link_catchments:
    #     linkages.append((
    #       std_source,
    #       std_dest
    #     ))
    # return linkages

def make_confluence(row):
    return [
        (
            _catchment_or_link(row.from_name,row.from_link_name),
            _catchment_or_link(row.to_name,row.to_link_name)
        )
    ]

def make_network_topology(catchments,nodes,links):
  links['link_name'] = links['name']
  catchments['catchment_name'] = catchments['name']
  c2l = pd.merge(catchments[['catchment_name','link']],
                 links[['id','link_name','from_node','to_node']],
                 left_on='link',right_on='id',how='right')
  assert len(c2l) == len(links)
  c2l['from_name'] = c2l['catchment_name']
  c2l['to_name'] = c2l['catchment_name']

  c2l['from_link_name'] = c2l['link_name']
  c2l['to_link_name'] = c2l['link_name']

  c2l['link'] = c2l['id']
  node_connection_table = pd.merge(c2l[['from_name','link','from_link_name','to_node']],
                        c2l[['to_name','to_link_name','from_node']],
                        left_on='to_node',right_on='from_node',how='right')

  node_connection_table.loc[~node_connection_table.from_node.isin(nodes.id),'from_node'] = np.nan
  node_connection_table.loc[~node_connection_table.to_node.isin(nodes.id),'to_node'] = np.nan

  confluences = node_connection_table[~pd.isna(node_connection_table.from_link_name) & \
                                      ~pd.isna(node_connection_table.to_link_name) & \
                                      pd.isna(node_connection_table.from_node)]
  linkages = []
  for joining_node in nodes.id:
    node_linkages = make_node_linkages(node_connection_table[node_connection_table.from_node==joining_node],nodes)
    linkages += node_linkages

  for _, row in confluences.iterrows():
    linkages += make_confluence(row)

  return linkages

def build_catchment_graph(model_structure,network,progress=print,custom_processing=None):
  init_timer('Sort features')
  if hasattr(network,'as_dataframe'):
    network = network.as_dataframe()

  catchments = network[network.feature_type=='catchment']
  links = network[network.feature_type=='link']
  nodes = network[network.feature_type=='node']

  templates = []
  report_time('Build catchments')
  for _,catchment in catchments.iterrows():
      tpl = model_structure.get_template(catchment=catchment['name']).flatten()
      templates.append(tpl)

  for _,link in links.iterrows():
    if catchments.link.isin([link.id]).any():
      continue

    tpl = model_structure.get_link_template(link_name=link['name']).flatten()
    templates.append(tpl)

  interesting_nodes = filter_nodes(nodes)
  interesting_nodes['accepts_inflow'] = False
  for ix,node in interesting_nodes.iterrows():
      node_type = node['icon'].split('/')[-1].replace('NodeModel','')
      tpl = model_structure.get_node_template(node_name=node['name'],node_type=node_type).flatten()
      interesting_nodes.loc[ix,'accepts_inflow']=tpl.has_input(UPSTREAM_FLOW_FLUX)
      templates.append(tpl)
  print(interesting_nodes[['id','name','icon','accepts_inflow']])

  report_time('Link catchments')

  master_template = templating.OWTemplate()

  for tpl in templates:
      master_template.nest(tpl)
  master_template = master_template.flatten()

  linkages = make_network_topology(catchments,interesting_nodes,links)
  for ix, (from_link,to_link) in enumerate(linkages):
    flow_links = master_template.make_link(
            DOWNSTREAM_FLOW_FLUX,
            UPSTREAM_FLOW_FLUX,
            from_tags=from_link,
            to_tags=to_link,
            from_exclude_tags=['constituent'],
            to_exclude_tags=[])#['constituent'])
    for l in flow_links:
      master_template.add_link(l)

    for con in model_structure.constituents:
      con_from_link = dict(**from_link,constituent=con)
      con_to_link = dict(**to_link,constituent=con)
      con_links = master_template.make_link(
        DOWNSTREAM_LOAD_FLUX,
        UPSTREAM_LOAD_FLUX,
        from_tags=con_from_link,
        to_tags=con_to_link)

      for l in con_links:
        master_template.add_link(l)

  g = None
  g = templating.template_to_graph(g,master_template)

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

def get_default_node_template(node_type,constituents,templates=None,**kwargs):
    if templates is None:
        templates = DEFAULT_NODE_TEMPLATES

    template = templating.OWTemplate(node_type)

    if node_type not in templates:
        raise Exception(f'Unsupported node: {node_type} at {kwargs.get("node_name","unnamed node")}')

    templates[node_type](template,constituents,**kwargs)
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

def fu_areas_parameteriser(fu_areas,area_models=DEFAULT_AREAL_MODELS):
    res = NestedParameteriser()

    for model,prop in area_models:
        res.nested.append(ParameterTableAssignment(fu_areas,model,prop,'cgu','catchment',skip_na=True))

    return res

def storage_parameteriser(builder):
    p = NestedParameteriser()

    static_storage_parameters = builder._load_csv('storage_params')
    if static_storage_parameters is None:
        return p

    static_storage_parameters = static_storage_parameters.rename(columns={
        'NetworkElement':'node_name',
        'InitialStorage':'currentVolume'
    })
    fsvs = dict(zip(static_storage_parameters['node_name'],static_storage_parameters['FullSupplyVolume']))
    fsls = dict(zip(static_storage_parameters['node_name'],static_storage_parameters['FullSupplyLevel']))

    storage_tables = merge_storage_tables(builder.data_path,fsvs=fsvs,fsls=fsls)
    if storage_tables is None:
        return p

    storage_parameters = LoadArraysParameters(storage_tables,'${node_name}','nLVA',model='Storage')
    p.nested.append(storage_parameters)

    demands_at_storages = builder._load_time_series_csv('Results/regulated_release_volume')
    if demands_at_storages is not None:
        demands_at_storages = demands_at_storages * PER_DAY_TO_PER_SECOND

        storage_demand_inputs = DataframeInputs()
        storage_demand_inputs.inputter(demands_at_storages, 'demand', '${node_name}',model='Storage')
        p.nested.append(storage_demand_inputs)

    p.nested.append(ParameterTableAssignment(static_storage_parameters,
                                                'Storage',
                                                dim_columns=['node_name'],
                                                complete=True))

    storage_climate = builder._load_time_series_csv('storage_climate')
    if not len(storage_climate.dropna()):
        raise Exception('NO storage climate')

    storage_climate = storage_climate.rename(columns=_rename_storage_variable)

    storage_climate_inputs = DataframeInputs()
    storage_climate_inputs.inputter(storage_climate,'rainfall','${node_name} Rainfall',model='Storage')
    storage_climate_inputs.inputter(storage_climate,'pet','${node_name} Evaporation',model='Storage')
    p.nested.append(storage_climate_inputs)

    storage_fsl_inputs = DataframeInputs()
    storage_target_cap = pd.DataFrame()


    for k,tbl in storage_tables.items():
        cap = tbl['volumes'].max() - fsvs[k]# m3
        storage_target_cap[k] = np.ones((len(storage_climate,)))*cap
    assert len(storage_target_cap)==len(storage_climate)
    storage_fsl_inputs.inputter(storage_target_cap,'targetMinimumCapacity','${node_name}',model='Storage')

    p.nested.append(storage_fsl_inputs)

    return p

def demand_parameteriser(builder):
    network = builder.network()
    nodes = network['features'].find_by_feature_type('node')
    water_users = nodes.find_by_icon('/resources/WaterUserNodeModel')

    extraction_params = builder._load_csv('extraction_point_params')
    demands = pd.DataFrame()
    for wu in water_users:
        wu_name = wu['properties']['name']
        us_links = network.upstream_links(wu['properties']['id'])
        assert len(us_links)==1
        extraction_node_id = us_links[0]['properties']['from_node']
        extraction_node = nodes.find_by_id(extraction_node_id)[0]
        extraction_node_name = extraction_node['properties']['name']
        is_extractive = \
          extraction_params[extraction_params.NetworkElement==extraction_node_name].IsExtractive.iloc[0]
        if not is_extractive:
          print(f'Skipping non-extractive node: {extraction_node_name}')
          continue
        demand = builder._load_csv(f'timeseries-demand-{wu_name}')

        if demand is None:
            pattern_demand = builder._load_csv(f'monthly-pattern-demand-{wu_name}')
            if pattern_demand is None:
                builder.warn('No demand for %s. Possibly function editor (unsupported)',wu_name)
                continue

            demand = np.array(pattern_demand.volume)
            month_ts = builder.time_period.month-1
            monthly_demand_m3 = demand[month_ts]
            monthly_demand_m3 = pd.Series(monthly_demand_m3,index=builder.time_period)
            daily_demand_m3 = monthly_demand_m3 / monthly_demand_m3.index.days_in_month
            daily_demand_m3s = daily_demand_m3 * PER_DAY_TO_PER_SECOND
            demand = pd.DataFrame({'TSO':daily_demand_m3s})
        else:
            demand = demand.reindex(builder.time_period)
            if 'TSO' not in demand.columns:
                renames = {demand.columns[0]:'TSO'}
                print('No TSO column. Renaming',renames)
                demand = demand.rename(columns=renames)

        print(demand.columns)
        demands[extraction_node_name] = demand['TSO']

    demands = demands
    i = DataframeInputs()
    i.inputter(demands, 'demand', '${node_name}', model='PartitionDemand')

    return i

def inflow_parameteriser(builder):
    p = NestedParameteriser()
    logger.info('Configuring inflow timeseries')
    inflows = builder.inflows(builder.time_period)

    if inflows is not None:
        inflow_inputs = DataframeInputs()
        p.nested.append(inflow_inputs)

        inflow_inputs.inputter(inflows,'input','${node_name}',model=n.Input)

        inflow_loads = builder.inflow_loads(inflows)
        if inflow_loads is not None:
            inflow_inputs.inputter(inflow_loads,'inputLoad','${node_name}:${constituent}',model='PassLoadIfFlow')
    return p

def loss_parameteriser(builder):
    losses = builder._load_all_csvs('loss-table-')
    if not len(losses):
        return None

    logger.info('Configuring losses')
    print(losses.keys())

    adjusted_losses = {}
    should_fail = False
    for k,tbl in losses.items():
        if tbl is None:
            print(f'No loss table for {k}')
            should_fail = True
            continue

        tbl = tbl.rename(columns={'inflow':'inputAmount'})
        if len(tbl[tbl.inputAmount==0])==0:
            logger.warning(f'No 0 entry in loss table for {k}')
            tbl= tbl.append({'inputAmount':0,'loss':0},ignore_index=True).sort_values('inputAmount')
        if max(tbl.inputAmount) < 1e6:
            logger.warning(f'Loss table for {k} lacks upper cap on inflow. Extending to 1e6')
            tbl = tbl.append({'inputAmount':1e6,'loss':max(tbl.loss)},ignore_index=True)
        tbl['proportion'] = tbl['loss']/tbl['inputAmount']
        tbl = tbl.fillna(0)
        adjusted_losses[k] = tbl[['inputAmount','proportion']]

    assert not should_fail

    loss_tables = LoadArraysParameters(adjusted_losses,'${node_name}','nPts',model='RatingCurvePartition')

    return loss_tables

def node_model_parameteriser(builder):
    p = NestedParameteriser()

    p.nested.append(storage_parameteriser(builder))
    p.nested.append(demand_parameteriser(builder))
    p.nested.append(inflow_parameteriser(builder))
    p.nested.append(loss_parameteriser(builder))
    return p

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

# class VeneerModelConfigurationProvider(object):
#     def __init__(self,v):
#         self.v = v

class FileBasedModelConfigurationProvider(object):
    def __init__(self,path,climate_patterns,time_period=None):
        self.data_path = path
        self.climate_patterns = climate_patterns
        self.time_period = time_period

        self.inflow_fill = 0.0
        nw = self.network().as_dataframe()
        catchments = nw[nw.feature_type=='catchment']
        links = nw[nw.feature_type=='link']
        lookup_df = pd.merge(catchments[['name','link']],links[['id','name']],
                               left_on='link',right_on='id',how='inner').rename(
                                   columns={'name_x':'catchment','link':'link_id','name_y':'link'})
        self.catchment_link_lookup = dict(zip(lookup_df['link'],lookup_df['catchment']))
        self.warnings = []

    def warn(self,msg,*args):
        self.warnings.append(msg.format(*args))
        logger.warn(msg,*args)

    def _find_files(self,pattern,ignoring=[]):
        patterns = [pattern, pattern+'.gz']
        files = sum([list(glob(os.path.join(self.data_path,p))) for p in patterns],[])
        files = [os.path.basename(fn) for fn in files]

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
                logger.warn('No such file: %s',fn)
                return None

        data = pd.read_csv(fn, index_col=0, parse_dates=True)
        if 'NetworkElement' in data.columns:
            # data['Catchment'] = data['NetworkElement'].str.replace(EXPECTED_LINK_PREFIX,'',case=False)
            data['Catchment'] = data['NetworkElement'].map(self.catchment_link_lookup)
            data['Catchment'].fillna(data['NetworkElement'],inplace=True)
            data['link_name'] = data['NetworkElement']

        return data

    def _load_all_csvs(self,prefix):
        files = [os.path.basename(fn) for fn in self._find_files(f'{prefix}*.csv*')]
        data = {}
        for f in files:
            fn = f.replace('.csv','').replace('.gz','')
            tbl = self._load_csv(fn)
            if tbl is None:
                raise Exception(f'Could not load expected file: {fn} ({f})')
            data[fn.replace(prefix,'')] = tbl
        return data

    def _load_time_series_csv(self,f):
        df = self._load_csv(f)
        if df is None:
            return None
        return df.reindex(self.time_period)

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
        if model_table is None or ('model' not in model_table.columns):
          return {}
        models = set(model_table.model)
        return {m:self._load_csv(f'{fn_prefix}{m}') for m in models}

    def runoff_parameters(self):
        return self._load_parameters('rr-',self._load_csv('runoff_models'))

    def generation_parameters(self):
        return self._load_parameters('cg-',self._load_csv('cgmodels'))

    def routing_parameters(self):
        return self._load_parameters('fr-',self._load_csv('routing_models'))

    def climate_data(self):
        data = self._load_time_series_csv('climate')
        # if self.time_period is not None:
        #   data = data.reindex(self.time_period)

        time_steps = data.index
        delta_t = (time_steps[1]-time_steps[0]).total_seconds()

        inputter = DataframeInputs()
        for variable,column in self.climate_patterns.items():
            inputter.inputter(data,'input',column,n.Input,variable=variable)

        return inputter, data.index, delta_t

    def inflows(self,time_period):
        #### TODO: Units!

        files = self._find_files('timeseries-inflow-*.csv',[
            'timeseries-inflow-concentration',
            'timeseries-inflow-load'
        ])
        if not len(files):
            return None

        files = [fn.replace('.csv','').replace('.gz','') for fn in files]
        def make_col_name(fn):
            return fn.replace('timeseries-inflow-','')
        all_data = [self._load_csv(fn) for fn in files]
        all_data = pd.DataFrame({make_col_name(fn):df[df.columns[0]] \
            for fn,df in zip(files,all_data)})
        all_data = all_data.reindex(time_period)
        if self.inflow_fill is not None:
          all_data = all_data.fillna(self.inflow_fill)
        return all_data

    def inflow_loads(self,inflows):
        conc_files = self._find_files('timeseries-inflow-concentration-*.csv')

        if not len(conc_files):
            return None

        def clean_fn(fn):
          return fn.replace('.csv','').replace('.gz','')
        def make_conc_col_name(fn):
            fn = clean_fn(fn)
            fn = fn.replace('timeseries-inflow-concentration-','')
            components = fn.split('-')
            constituent = components[0]
            node_name = '-'.join(components[1:])
            return f'{node_name}:{constituent}'

        all_conc_data = [self._load_csv(clean_fn(fn)) for fn in conc_files]
        all_conc_data = pd.DataFrame({make_conc_col_name(fn):df[df.columns[0]] \
            for fn,df in zip(conc_files,all_conc_data)})
        all_conc_data = all_conc_data.reindex(inflows.index)

        loads = all_conc_data.copy()
        for col in loads.columns:
            node_name = col.split(':')[0]
            inflow_ts = inflows[node_name]
            ### TODO: UNITS!
            loads[col] *= inflow_ts
        if self.inflow_fill is not None:
          loads = loads.fillna(self.inflow_fill)

        return loads


class SourceOpenwaterModelBuilder(object):
    def __init__(self,source_provider,ignore_fus=[]):
        self.provider = source_provider
        self.ignore_fus=ignore_fus

    def build_catchment_template(self):
        catchment_template = SemiLumpedCatchment()
        catchment_template.climate_inputs = ['rainfall','pet']
        catchment_template.node_template = get_default_node_template
        fus = set(self.provider.get_functional_unit_types()) - set(self.ignore_fus)
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

    def remove_ignored_rows(self,param_df):
        if 'Functional Unit' in param_df.columns:
            print(len(param_df))
            param_df = param_df[~param_df['Functional Unit'].isin(self.ignore_fus)]
            print(len(param_df))
            print()

        return param_df

    def build_parameter_lookups(self,params):
        params = {k:self.remove_ignored_rows(df) for k,df in params.items()}
        params = {k:df for k,df in params.items() if len(df)}
        return build_parameter_lookups(params)

    def build_ow(self,dest_fn):
        if os.path.exists(dest_fn):
            raise Exception('File exists')

        init_lookup()

        print('Building model structure')
        catchment_template = self.build_catchment_template()
        net = self.provider.network()
        model = build_catchment_graph(catchment_template,net)
        model.time_period=self.provider.time_period
        print('Got graph, configuring parameters')

        p = Parameteriser()
        p.append(fu_areas_parameteriser(self.provider.get_fu_areas()))

        p.append(node_model_parameteriser(self.provider))

        print('Building parameters')
        runoff_parameters = self.build_parameter_lookups(self.provider.runoff_parameters())
        cg_parameters = self.build_parameter_lookups(self.provider.generation_parameters())
        routing_params = self.build_parameter_lookups(self.provider.routing_parameters())
        for model_type, parameters in {**runoff_parameters,**cg_parameters,**routing_params}.items():
            print('Building parameteriser for %s'%model_type)
            parameteriser = ParameterTableAssignment(parameters,model_type)
            p.append(parameteriser)

            if model_type in MODELS_WITH_INITIAL_STATES_AS_PARAMETERS:
              initial_states = MODELS_WITH_INITIAL_STATES_AS_PARAMETERS[model_type](parameters)
              states_parameteriser = ParameterTableAssignment(initial_states,model_type)
              p.append(states_parameteriser)

        print('Configuring climate data')
        climate_inputs,time_period, delta_t = self.provider.climate_data()
        simulation_length = len(time_period)
        p.append(climate_inputs)

        for dt_model in ['DepthToRate','StorageRouting','LumpedConstituentRouting']:
          p.append(UniformParameteriser(dt_model,DeltaT=delta_t))

        p.append(UniformParameteriser('PassLoadIfFlow',scalingFactor=1.0))
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
    template.define_output(demand,'outflow',DOWNSTREAM_FLOW_FLUX,**kwargs)

    prop = template.add_node(n.ComputeProportion,process='demand_proportion',**kwargs)
    template.add_link(OWLink(demand,'extraction',prop,'numerator'))
    template.define_input(connections=[(demand,'input'),(prop,'denominator')],alias=UPSTREAM_FLOW_FLUX,**kwargs)
    # denominator is??? the demand (ie an input timeseries?)

    for con in constituents:
        con_ext = template.add_node(n.VariablePartition,process='constituent_extraction',constituent=con,**kwargs)
        template.add_link(OWLink(prop,'proportion',con_ext,'fraction'))
        template.define_input(con_ext,'input',UPSTREAM_LOAD_FLUX,constituent=con,**kwargs)
        template.define_output(con_ext,'output2',DOWNSTREAM_LOAD_FLUX,constituent=con,**kwargs)
        #TODO Or output2?

def build_inflow_node_template(template:templating.OWTemplate,constituents: list,**kwargs):
    flow_node = template.add_node(n.Input,process='inflow',variable='inflow',**kwargs)
    template.define_output(flow_node,'output',DOWNSTREAM_FLOW_FLUX,**kwargs)

    for con in constituents:
        load_input = template.add_node(n.PassLoadIfFlow,process='inflow_load',constituent=con,**kwargs)
        template.add_link(OWLink(flow_node,'output',load_input,'flow'))
        template.define_output(load_input,'outputLoad',DOWNSTREAM_LOAD_FLUX,constituent=con,**kwargs)

def build_loss_node_template(template:templating.OWTemplate,constituents: list,**kwargs):
    loss = template.add_node(n.RatingCurvePartition,process='loss',**kwargs)

    prop = template.add_node(n.ComputeProportion,process='loss_proportion',**kwargs)
    template.add_link(OWLink(loss,'output1',prop,'numerator'))

    template.define_input(connections=[(loss,'input'),(prop,'denominator')],alias=UPSTREAM_FLOW_FLUX,**kwargs)
    template.define_output(loss,'output2',DOWNSTREAM_FLOW_FLUX,**kwargs)

    # template.add_link(OWLink(loss,'output2',prop,'numerator'))
    # denominator is? inflow - ie an overloaded input? 
    for con in constituents:
        con_ext = template.add_node(n.VariablePartition,process='constituent_loss',constituent=con,**kwargs)
        template.add_link(OWLink(prop,'proportion',con_ext,'fraction'))
        template.define_input(con_ext,'input',UPSTREAM_LOAD_FLUX,constituent=con,**kwargs)
        template.define_output(con_ext,'output2',DOWNSTREAM_LOAD_FLUX,constituent=con,**kwargs)
        #TODO Or output2?

def storage_template_builder(constituent_model_map=None):
  def build_storage_node_template(template,constituents,**kwargs):
    nonlocal constituent_model_map
    if constituent_model_map is None:
      constituent_model_map = n.LumpedConstituentRouting
    tag_values = list(kwargs.values())

    storage = template.add_node(n.Storage,process='storage',**kwargs)
    template.define_output(storage,'outflow',DOWNSTREAM_FLOW_FLUX,**kwargs)
    inflow_connections = [
      (storage,'inflow')
    ]

    for con in constituents:
      constituent_model = get_model_for_provider(
        constituent_model_map,con,*tag_values)
      constituent_node = template.add_node(constituent_model,
                                           process='storage-constituents',
                                           constituent=con,
                                           **kwargs)
      if has_input(constituent_model,'inflow'):
        inflow_connections.append((constituent_node,'inflow'))

      template.define_input(constituent_node,'inflowLoad',UPSTREAM_LOAD_FLUX,constituent=con,**kwargs)
      template.define_output(constituent_node,'outflowLoad',DOWNSTREAM_LOAD_FLUX,constituent=con,**kwargs)

      template.add_link(OWLink(storage,'volume',constituent_node,'storage'))
      template.add_link(OWLink(storage,'outflow',constituent_node,'outflow'))

    template.define_input(connections=inflow_connections,alias=UPSTREAM_FLOW_FLUX,**kwargs)

  return build_storage_node_template

def has_input(model,the_input):
  if isinstance(model,str):
    model = getattr(n,model)
  return the_input in model.description['Inputs']

def _rename_storage_variable(col):
    SUFFIXES=['InMetresPerSecond']
    for sfx in SUFFIXES:
        col = col.replace(sfx,'')
    return col

DEFAULT_NODE_TEMPLATES={
    'Extraction':build_extraction_node_template,
    'InjectedFlow':build_inflow_node_template,
    'Storage':storage_template_builder(), # Default to lumped constituent routing
    'Loss':build_loss_node_template
}

def node_lookups(network_fn):
  import veneer

  network_raw = json.load(open(network_fn,'r'))
  network = _extend_network(network_raw)
  network_df = network.as_dataframe()
  catchments = network_df[network_df.feature_type=='catchment'][['name','link']]
  links = network_df[network_df.feature_type=='link'][['id','from_node','name']].rename(columns={'name':'name_link'})
  nodes = network_df[network_df.feature_type=='node'][['id','name']]
  upstream_nodes = pd.merge(pd.merge(catchments,links,how='right',left_on='link',right_on='id'),nodes,left_on='from_node',right_on='id',suffixes=['_catchment','_node'])[['name_catchment','name_node','name_link']]
  upstream_nodes = upstream_nodes.rename(columns=lambda c:c[5:])
  c2n= upstream_nodes.set_index('catchment')['node'].to_dict()
  if 'dummy-catchment' in c2n:
    c2n.pop('dummy-catchment')

  l2usn= upstream_nodes.set_index('link')['node'].to_dict()

  outlets = network.outlet_nodes()
  outlet_ids = [n['properties']['id'] for n in outlets]
  outlet_names = [n['properties']['name'] for n in outlets]
  outlet_lookup = dict(zip(outlet_ids,outlet_names))

  upstream_links = network_df[network_df.to_node.isin(outlet_ids)]
  upstream_links = upstream_links.drop_duplicates(subset=['to_node'],keep=False)
  upstream_catchments = network_df[network_df.link.isin(upstream_links.id)]

  us_catchment_to_node = upstream_links.set_index('id').loc[upstream_catchments.link]['to_node']
  c2outlet = dict(zip(upstream_catchments.name,[outlet_lookup[n] for n in us_catchment_to_node]))
  if 'dummy-catchment' in c2outlet:
    c2outlet.pop('dummy-catchment')

  upstream_links = upstream_links[~upstream_links.id.isin(upstream_catchments.link)]
  l2outlet = dict(zip(upstream_links.name,[outlet_lookup[n] for n in upstream_links.to_node]))

  return c2n,l2usn,c2outlet,l2outlet

def _arg_parser():
  import argparse
  def parse_time_period(txt):
    result = txt.split('-')
    if len(result)!=2:
      raise argparse.ArgumentTypeError(f'{txt} is not a valid time period')
    return result

  from veneer import extract_config as ec
  parser = ec._base_arg_parser()
  parser.add_argument('-o','--openwater',help='Path to Openwater binaries',default=None)
  parser.add_argument('--timeperiod',help='Time period for converted model (format yyyy/mm/dd-yyyy/mm/dd)',type=parse_time_period,default=None)#'2000/01/01-2000/12/31')
  parser.add_argument('--timestep',help='Timestep for converted model',default='1d')
  parser.add_argument('--destination',help='Destination directory for converted model',default='.')
  parser.add_argument('--existing', help='Use existing model structure and only convert parameters', action='store_true')
  parser.add_argument('--split',type=int,help='Split the model into multiple files (structure, parameters, initial states and input timeseries) where number is number of input timeseries files',default=0)
  parser.add_argument('--run',help='Run the converted model',action='store_true',default=False)
  return parser

def write_model_and_metadata(model_fn,model_obj,meta,network):
  '''
  Write a newly built catchment model to disk with associated metadata.

  model_fn: str, with file extension (.h5)
  model_obj: ModelFile or ModelGraph
  meta: dict, with metadata
  network: GeoDataFrame, with network topology

  model_obj, meta and network typically from a catchment model builder
  '''

  if isinstance(model_obj,ModelFile):
    model_obj.write()
  else:
    model_obj.write_model(model_fn)

  json.dump(meta,open(model_fn.replace('.h5','.meta.json'),'w'),indent=2,default=str)

  links,nodes,catchments = split_network(network)
  links.to_file(model_fn.replace('.h5','.links.json'),driver='GeoJSON')
  nodes.to_file(model_fn.replace('.h5','.nodes.json'),driver='GeoJSON')
  catchments.to_file(model_fn.replace('.h5','.catchments.json'),driver='GeoJSON')

def build_main(builder,model,timeperiod,openwater=None,existing=False,run=False,**kwargs):
  print('Build')
  if openwater is not None:
    set_exe_path(openwater)
  discover()

  dest = os.path.abspath(kwargs.get('destination','.'))
  if not os.path.exists(dest):
    os.makedirs(dest)

  model_fn = os.path.join(dest,model+'.h5')
  if existing:
      model_file = ModelFile(model_fn)
  else:
      model_file = None

  source = os.path.abspath(os.path.join(kwargs.get('extractedfiles','.'),model))

  time_period = pd.date_range(timeperiod[0],timeperiod[1],freq=kwargs.get('timestep','1d'))
  model_obj, meta, network = builder(source,existing=model_file)

  write_model_and_metadata(model_fn,model_obj,meta,network)

  if run:
    model_obj.run(time_period,overwrite=True,verbose=True)

if __name__=='__main__':
  import veneer.extract_config as ec
  args = ec._parsed_args(_arg_parser())
  build_main(None,**args)
