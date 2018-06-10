'''
Configuration node-link network catchment models for Openwater.

Includes:

* Support for delineation based on a DEM, using TauDEM
* Spatial parameterisation (python-rasterstats)
* Spatial input generation using climate-utils
'''
from openwater import OWTemplate, OWLink
import openwater.nodes as n


# Need:
# * Routines for delineation
# * 

class SemiLumpedCatchment(object):
  def __init__(self):
    self.hrus = ['HRU']
    self.cgus = ['CGU']
    self.cgu_hrus = {'CGU':'HRU'}
    self.constituents = ['Con']

    self.rr = n.Simhyd
    self.cg = n.EmcDwc
    self.routing = n.Muskingum
    self.transport = n.LumpedConstituentRouting

  def model_for(self,provider,*args):
    if hasattr(provider,'__call__'):
      return provider(*args)
    if hasattr(provider,'__getitem__'):
      return provider[args[0]]
    return provider

  def get_template(self):
    template = OWTemplate()

    routing_node = template.add_node(self.routing,process='FlowRouting')
    transport = {}
    for con in self.constituents:
      # transport_node = 'Transport-%s'%(con)
      transport_node = template.add_node(self.model_for(self.transport,con),process='ConstituentRouting',constituent=con)
      template.add_link(OWLink(routing_node,'outflow',transport_node,'outflow'))
      transport[con]=transport_node

    runoff = {}
    for hru in self.hrus:
      runoff_node = template.add_node(self.model_for(self.rr,hru),process='RR',hru=hru)
      runoff[hru] = runoff_node

    for cgu in self.cgus:
      runoff_node = runoff[self.cgu_hrus[cgu]]

      runoff_scale_node = template.add_node(n.DepthToRate,process='ArealScale',cgu=cgu,component='Runoff')
      quickflow_scale_node = template.add_node(n.DepthToRate,process='ArealScale',cgu=cgu,component='Quickflow')
      baseflow_scale_node = template.add_node(n.DepthToRate,process='ArealScale',cgu=cgu,component='Baseflow')

      template.add_link(OWLink(runoff_node,'runoff',runoff_scale_node,'input'))      
      template.add_link(OWLink(runoff_node,'quickflow',quickflow_scale_node,'input'))      
      template.add_link(OWLink(runoff_node,'baseflow',baseflow_scale_node,'input'))      

      template.add_link(OWLink(runoff_scale_node,'outflow',routing_node,'lateral'))

      for con in self.constituents:
        gen_node = template.add_node(self.model_for(self.cg,con,cgu),process='ConstituentGeneration',constituent=con,cgu=cgu)
        template.add_link(OWLink(quickflow_scale_node,'outflow',gen_node,'quickflow'))
        template.add_link(OWLink(baseflow_scale_node,'outflow',gen_node,'baseflow'))

        transport_node = transport[con]
        template.add_link(OWLink(gen_node,'totalLoad',transport_node,'lateralLoad'))
        template.add_link(OWLink(runoff_scale_node,'outflow',transport_node,'inflow'))

    return template
  
  def link_catchments(self,graph,upstream,downstream):
    linkages = [('%s-FlowRouting ('+self.routing.name+')','outflow','inflow')] + \
               [(('%%s-ConstituentRouting-%s ('+self.transport.name+')')%c,'outflowLoad','inflowLoad') for c in self.constituents]
    for (lt,src,dest) in linkages:
        src_node = lt%(str(upstream))
        dest_node = lt%(str(downstream))#'%d/%s'%(to_cat,lt)
        graph.add_edge(src_node,dest_node,src=[src],dest=[dest])


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
