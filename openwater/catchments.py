'''
Configuration node-link network catchment models for Openwater.

Includes:

* Support for delineation based on a DEM, using TauDEM
* Spatial parameterisation (python-rasterstats)
* Spatial input generation using climate-utils
'''
from openwater import OWTemplate, OWLink, OWNode
import openwater.nodes as n
import openwater.template as templating

DOWNSTREAM_FLOW_FLUX='flow_downstream'
DOWNSTREAM_LOAD_FLUX='load_downstream'
UPSTREAM_FLOW_FLUX='flow_upstream'
UPSTREAM_LOAD_FLUX='load_upstream'

# Need:
# * Routines for delineation
# * 

def get_model_for_provider(provider,*args):
  if provider is None:
      return None
  if hasattr(provider,'__call__'):
    return provider(*args)
  if hasattr(provider,'__getitem__'):
    return provider[args[0]]
  return provider

class SemiLumpedCatchment(object):
  def __init__(self):
    self.climate_inputs = []
    self.hrus = ['HRU']
    self.cgus = ['CGU']
    self.cgu_hrus = {}
    self.constituents = ['Con']

    self.rr = n.Simhyd
    self.cg = n.EmcDwc
    self.routing = n.Muskingum
    self.transport = n.LumpedConstituentRouting
    self.node_template = None

  def model_for(self,provider,*args):
    if provider is None:
        return None
    if hasattr(provider,'__call__'):
      return provider(*args)
    if hasattr(provider,'__getitem__'):
      return provider[args[0]]
    return provider

  def get_template(self,**kwargs) -> OWTemplate:
    tag_values = list(kwargs.values())
    template, routing_node, transport_nodes = self._get_link_template_with_nodes(**kwargs)

    climate_nodes = {cvar: template.add_node(n.Input,process='input',variable=cvar,**kwargs) for cvar in self.climate_inputs}

    runoff = {}
    for hru in self.hrus:
      runoff_model = self.model_for(self.rr,hru,*tag_values)
      if runoff_model is None: continue

      runoff_node = template.add_node(runoff_model,process='RR',hru=hru,**kwargs)
      runoff[hru] = runoff_node

      for clim_var, clim_node in climate_nodes.items():
        template.add_link(OWLink(clim_node,'output',runoff_node,clim_var))

    for cgu in self.cgus:
      runoff_node = runoff.get(self.cgu_hrus.get(cgu,cgu))
      if runoff_node is None:
          continue

      runoff_scale_node = template.add_node(n.DepthToRate,process='ArealScale',cgu=cgu,component='Runoff',**kwargs)
      quickflow_scale_node = template.add_node(n.DepthToRate,process='ArealScale',cgu=cgu,component='Quickflow',**kwargs)
      baseflow_scale_node = template.add_node(n.DepthToRate,process='ArealScale',cgu=cgu,component='Baseflow',**kwargs)

      template.add_link(OWLink(runoff_node,'runoff',runoff_scale_node,'input'))
      template.add_link(OWLink(runoff_node,'quickflow',quickflow_scale_node,'input'))
      template.add_link(OWLink(runoff_node,'baseflow',baseflow_scale_node,'input'))

      if routing_node.has_input('lateral'):
        catchment_inflow_flux = 'lateral'
      else:
        catchment_inflow_flux = 'inflow'

      template.add_link(OWLink(runoff_scale_node,'outflow',routing_node,catchment_inflow_flux))

      for con in self.constituents:
        gen_node = template.add_node(self.model_for(self.cg,con,cgu,*tag_values),process='ConstituentGeneration',constituent=con,cgu=cgu,**kwargs)
        template.add_link(OWLink(quickflow_scale_node,'outflow',gen_node,'quickflow'))
        template.add_link(OWLink(baseflow_scale_node,'outflow',gen_node,'baseflow'))

        transport_node = transport_nodes[con]
        if transport_node.has_input('lateralLoad'):
          template.add_link(OWLink(gen_node,'totalLoad',transport_node,'lateralLoad'))
          template.add_link(OWLink(runoff_scale_node,'outflow',transport_node,'inflow'))
        else:
          template.add_link(OWLink(gen_node,'totalLoad',transport_node,'inflow'))

        # if (self.routing is not None) and transport_node.has_input('outflow'):
        #     template.add_link(OWLink(routing_node,'outflow',transport_node,'outflow'))
        #     template.add_link(OWLink(routing_node,'storage',transport_node,'reachVolume'))

    return template

  def _get_link_template_with_nodes(self,**kwargs) -> (OWTemplate, OWNode, dict):
    tag_values = list(kwargs.values())
    template = OWTemplate()
    routing_model = self.model_for(self.routing,*tag_values)
    routing_node = template.add_node(routing_model,process='FlowRouting',**kwargs)
    template.define_input(routing_node,'inflow',UPSTREAM_FLOW_FLUX,**kwargs)
    template.define_output(routing_node,'outflow',DOWNSTREAM_FLOW_FLUX,**kwargs)

    transport = {}
    for con in self.constituents:
      # transport_node = 'Transport-%s'%(con)
      transport_model = self.model_for(self.transport,con,*tag_values)
      if transport_model is None:
          transport[con]=None
          continue

      transport_node = template.add_node(transport_model,process='ConstituentRouting',constituent=con,**kwargs)
      if transport_node.has_input('outflow'):
          template.add_link(OWLink(routing_node,'outflow',transport_node,'outflow'))

      if transport_node.has_input('inflowLoad'):
        load_in_flux = 'inflowLoad'
      else:
        load_in_flux = 'inflow'
      template.define_input(transport_node,load_in_flux,UPSTREAM_LOAD_FLUX,**kwargs)

      if transport_node.has_output('outflowLoad'):
        load_out_flux = 'outflowLoad'
      else:
        load_out_flux='outflow'
      template.define_output(transport_node,load_out_flux,DOWNSTREAM_LOAD_FLUX,**kwargs)

      transport[con]=transport_node

    return template, routing_node, transport

  def get_link_template(self,**kwargs) -> OWTemplate:
    tpl, _, _ = self._get_link_template_with_nodes(**kwargs)
    return tpl

  def get_node_template(self,node_type,**kwargs) -> templating.OWTemplate:
      return self.node_template(node_type,self.constituents,**kwargs)

  def link_catchments(self,graph,upstream,downstream):
    linkages = [(self.routing,'%s-FlowRouting (%s)',[])] + \
               [(self.transport,('%%s-ConstituentRouting-%s (%%s)')%c,[c]) for c in self.constituents]
    for (model_lookup,lt,lookup_args) in linkages:
        us_model = self.model_for(model_lookup,upstream,*lookup_args)
        ds_model = self.model_for(model_lookup,downstream,*lookup_args)

        src_node = lt%(str(upstream),us_model.name)
        dest_node = lt%(str(downstream),ds_model.name)#'%d/%s'%(to_cat,lt)

        if graph_node_has_flux(graph,src_node,'outflowLoad','Output'):
            src = 'outflowLoad'
        else:
            src = 'outflow'

        if graph_node_has_flux(graph,dest_node,'inflowLoad','Input'):
            dest = 'inflowLoad'
        else:
            dest = 'inflow'

        # n = graph.nodes[src_node]

        check_graph_node_has_flux(graph,src_node,src,'Output')
        check_graph_node_has_flux(graph,dest_node,dest,'Input')

        graph.add_edge(src_node,dest_node,src=[src],dest=[dest])

def graph_node_has_flux(graph,node_name,flux_name,flux_type):
    node = graph.node[node_name]
    model_type_name = node['_model']
    model_type = getattr(n,model_type_name)
    model_description = model_type.description
    return flux_name in model_description[f'{flux_type}s']

def check_graph_node_has_flux(graph,node_name,flux_name,flux_type):
    if not graph_node_has_flux(graph,node_name,flux_name,flux_type):
        raise templating.InvalidFluxException(node_name,flux_name,flux_type)

def delineate(dem,threshold,fill_pits=False):
  '''
  Generate catchment boundaries and stream network for use in Openwater, using TauDEM.

  Returns:

  * tree, waterhseds (polygons), streams, coords, order
  '''
  import taudem as td
  import rasterio as rio
  import numpy as np
  import pandas as pd

  transform = rio.open(dem).transform
  if fill_pits:
    filled = td.pitremove(dem)
  else:
    filled = dem
  d8p, d8s = td.d8flowdir(filled)
  d8ua = td.aread8(d8p,nc=True,geotransform=transform)
  d8streams = d8ua > threshold
  d8streams = d8streams.astype('i')
  tree, watersheds, streams, coords, order = td.streamnet(d8p,filled,d8ua,d8streams,geotransform=transform)
  watersheds_poly = td.utils.to_polygons(watersheds,transform=transform)
  watersheds_poly = watersheds_poly.dissolve('GRIDCODE').reset_index()

  pairs = list(zip(*np.unique(watersheds,return_counts=True)))
  pairs = pd.DataFrame(pairs,columns=['WSNO','CellCount'])
  pairs = pairs[pairs.WSNO>=0]
  pairs = pairs.set_index('WSNO')
  CELL_SIZE = 2
  CELL_AREA = CELL_SIZE**2
  pairs['CellCountArea'] = pairs['CellCount'] * CELL_AREA   

  streams['LocalArea'] = streams['DSContArea'] -  streams['USContArea']
  streams['LocalAreaKm'] = streams['LocalArea'] * 1e-6
  streams = streams.join(pairs,on='WSNO',how='inner')

  return tree, watersheds_poly, streams, coords, order

def build_catchment_graph(model_structure,catchments):
  g = None
  tpl =  model_structure.get_template()
  for wsno in list(catchments.WSNO):
    g = templating.template_to_graph(g,tpl,catchment=wsno)

  for i,row in catchments.iterrows():
    src = row.WSNO
    dest = row.DSLINKNO
    if dest < 0: continue
    model_structure.link_catchments(g,src,dest)

  return templating.ModelGraph(g)
