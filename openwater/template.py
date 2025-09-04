import os
import sys
import h5py
import shutil
from subprocess import Popen, PIPE
from queue import Queue, Empty  # python 3.x
from threading  import Thread
from time import sleep
import numpy as np
import pandas as pd
import networkx as nx
from functools import reduce
from . import nodes as node_types
from .timing import init_timer, report_time, close_timer
from itertools import chain
from .array_params import get_parameter_locations
from .nodes import create_indexed_parameter_table
from .file import _tabulate_model_scalars_from_file
import logging
logger = logging.getLogger(__name__)
import time

# Non blocking IO solution from http://stackoverflow.com/a/4896288
ON_POSIX = 'posix' in sys.builtin_module_names
TAG_PROCESS='_process'
TAG_MODEL='_model'
TAG_RUN_INDEX='_run_idx'
TAG_GENERATION='_generation'
META_TAGS=[TAG_PROCESS,TAG_MODEL,TAG_RUN_INDEX,TAG_GENERATION]
DEFAULT_TIMESTEPS=365
LINK_TABLE_COLUMNS = ['%s_%s'%(n,c) for n in ['src','dest'] for c in ['generation','model','node','gen_node','var']]

def connections_match(o,i):
  o_node,_,o_alias,o_tags = o
  i_node,_,i_alias,i_tags = i

  if o_alias != i_alias:
#    print('aliases do not match')
    return False
  o_tags = o_tags.copy()
  o_tags.update(o_node.tags)
  i_tags = i_tags.copy()
  i_tags.update(i_node.tags)
  return tags_match(o_tags, i_tags)

def tags_match(o_tags,i_tags):
  common_keys = list(set(o_tags.keys()).intersection(i_tags.keys()))
  for ck in common_keys:
    if ck in ['_process','_model']: continue
    if o_tags[ck] != i_tags[ck]:
#      print('common key (%s) does not match (%s vs %s)'%(ck,o_node.tags[ck],i_node.tags[ck]))
      return False

  return True

def model_name(model_type_or_name):
  if hasattr(model_type_or_name,'name'):
    return model_type_or_name.name
  return model_type_or_name

class OWTemplate(object):
  def __init__(self,lbl=''):
    self.label=lbl
    self.nodes = []
    self.links = []
    self.nested = []
    self.inputs = []
    self.outputs = []

  def define_input(self,node=None,name=None,alias=None,connections=None,**kwargs):
    if connections is None:
      connections = [
        (node,name)
      ]

    if alias is None:
      alias = connections[0][1]

    for con_node, con_name in connections:
      if con_name is None:
        # Would be nice to have a short hand to define every input of this
        # node as an input to the graph (and similarly every output of a node
        # as an output of the graph
        # But the node currently stores the model name)
        raise InvalidFluxException(node,'(no name provided)','input')

      if not con_node.has_input(con_name):
          raise InvalidFluxException(con_node,con_name,'input')

      self.inputs.append((con_node,con_name,alias,kwargs))

  def define_output(self,node,name=None,alias=None,**kwargs):
    if not node.has_output(name):
        raise InvalidFluxException(node,name,'output')

    if alias is None:
      alias = name

    self.outputs.append((node,name,alias,kwargs))

  def _has_flux(self,alias,fluxes):
    for fl in fluxes:
      if fl[2]==alias:
        return True
    return False

  def has_input(self,alias):
    return self._has_flux(alias,self.inputs)

  def add_node(self,model_type=None,name=None,process=None,**tags):
    '''
    Add a node to the graph.

    Parameters
    ----------
    model_type : str
      The type of model to add.
    name : str
      The name of the node.
    process : str
      The process to which the node belongs.
    tags : dict
      A dictionary of tags to add to the node.

    Returns
    -------
    node : Node
      The node that was added.
      This node object is used when creating links with `OWLink`
    '''
    # if hasattr(node_or_name,'model_type'):
    #   self.nodes.append(node_or_name)
    # else:
    if process and not TAG_PROCESS in tags:
        tags[TAG_PROCESS]=process

    new_node = OWNode(model_type,name,**tags)
    self.nodes.append(new_node)
    return new_node

  def add_link(self,link):
    '''
    Add a link to the graph.

    Parameters
    ----------
    link : OWLink
      The link to add.
    '''
    self.links.append(link)

  def add_conditional_link(self,from_node,from_output,to_node,possibly_inputs,model):
    for pi in possibly_inputs:
      if pi in model.description['Inputs']:
        self.add_link(OWLink(from_node,from_output,to_node,pi))
        return True
    return False

  def match_labelled_flux(self,fluxes,flux_name,flux_tags,exclude_tags,return_all=False):
      result = []
      required_tags = set(flux_tags.keys())
      for node,name,alias,stored_tags in fluxes:
        if flux_name != alias:
          continue
        stored_tags = dict(**stored_tags)
        stored_tags.update(**node.tags)
        if len(required_tags.difference(stored_tags.keys())):
          continue
        skip = False
        for et in exclude_tags:
            if et in stored_tags:
                skip = True
                break
        if skip: continue

        if tags_match(flux_tags,stored_tags):
          if return_all:
            result.append((node,name))
          else:
            return node, name
      if return_all:
        return result
      return None,None

  def make_link(self,output_name,input_name,
                from_node=None,from_tags={},from_exclude_tags=[],
                to_node=None,to_tags={},to_exclude_tags=[]):
    link_txt = f'{output_name}({from_node or str(from_tags)}) -> {input_name}({to_node or str(to_tags)})'
    if from_node is None:
        from_node, new_output_name = self.match_labelled_flux(
            self.outputs,output_name,from_tags,from_exclude_tags)
        if from_node is None or new_output_name is None:
            n_outputs = len(self.outputs)
            raise Exception('%s: No matching output for %s, with tags %s. Have %d outputs'%(link_txt,new_output_name or output_name,str(from_tags),n_outputs))
        output_name = new_output_name
    if to_node is None:
        destinations = self.match_labelled_flux(
            self.inputs,input_name,to_tags,to_exclude_tags,return_all=True)
        # to_node, new_input_name
        if not len(destinations): #to_node is None or new_input_name is None:
            raise Exception('%s: No matching input for %s, with tags %s'%(link_txt,input_name,str(to_tags)))
        # input_name = new_input_name
    else:
        destinations = [(to_node,input_name)]

    return [OWLink(from_node,output_name,dest_node,dest_input) for dest_node, dest_input in destinations]

  def flatten(self):
    '''
    Generate a single, flat template containing all nested
    templates, instantiating links between nested templates based
    on input and output descriptions.

    When instantiating links, the order of the nested templates matters,
    with links only instantiated from outputs of earlier nested templates to
    inputs of later nested templates.
    '''
    result = OWTemplate(self.label)
    result.nodes += self.nodes
    result.links += self.links

    result.inputs += self.inputs
    result.outputs += self.outputs

    flattened = [t.flatten() for t in self.nested]
    available_outputs = []
    used_outputs = []
    for child in flattened:
      result.nodes += child.nodes
      result.links += child.links

      for child_input in child.inputs:
        input_linked = False
        for previous_output in available_outputs:
          if connections_match(previous_output,child_input):
#            print('linking',previous_output,child_input)
            result.add_link(OWLink(previous_output[0],previous_output[1],child_input[0],child_input[1]))
            used_outputs.append(previous_output)
            input_linked = True
        if not input_linked:
          result.inputs.append(child_input)

      available_outputs += child.outputs
    #unused_outputs = set(available_outputs).difference(set(used_outputs))
    unused_outputs = [o for o in available_outputs if not o in used_outputs]
    result.outputs+= list(unused_outputs)
    return result

  def nest(self,other):
    '''
    Add all nodes and links from other to this template,
    connecting all outputs from this template to inputs in other AND

    '''
    self.nested.append(other)

  def instantiate(self,**instance_tags):
    res = OWTemplate()
    node_map = {}
    for n in self.nodes:
      new_node = res.add_node(n.model_type,**n.tags,**instance_tags)
      node_map[n.name] = new_node

    for l in self.links:
      new_from = node_map[l.from_node.name]
      new_to = node_map[l.to_node.name]
      new_link = res.add_link(OWLink(new_from,l.from_output,new_to,l.to_input))

    return res

  def matching_nodes(self,nested=False,**kwargs):
    result =[n for n in self.nodes if n.matches(**kwargs)]
    if nested:
      for nested_tpl in self.nested:
        result += nested_tpl.matching_nodes(nested,**kwargs)

    return result

  def get_node(self,nested=False,**kwargs):
    nodes = self.matching_nodes(nested=nested,**kwargs)
    if len(nodes) != 1:
      raise Exception('Expected 1 node matching %s, found %d'%(kwargs,len(nodes)))
    return nodes[0]

class OWNode(object):
  def __init__(self,model_type,name=None,**tags):
    self.tags = tags
    self.set_model_type(model_type)

    if name:
      self.name = name
    else:
      self.name = self.make_name()

  def set_model_type(self,model_type):
    self.model_type = model_type
    if hasattr(model_type,'name'):
      self.model_name = model_type.name
      self.model_type = model_type
    else:
      self.model_name = model_type
      import openwater.nodes as node_types
      if not hasattr(node_types,self.model_name):
        from openwater.discovery import discover
        discover(self.model_name)
      self.model_type = getattr(node_types,self.model_name)

    self.tags[TAG_MODEL] = self.model_name

  def make_name(self):
    std_names = ['catchment','model',TAG_PROCESS,'constituent','hru','lu']
    for k in sorted(self.tags.keys()):
      if k.startswith('_'):continue

      if not k in std_names:
        std_names.append(k)
    return '-'.join([str(self.tags.get(k,None)) for k in std_names if k in self.tags])

  def __str__(self):
    return '%s (%s)'%(self.name,self.model_name)

  def has_output(self,name):
    return name in self.model_type.description['Outputs']

  def has_input(self,name):
    return name in self.model_type.description['Inputs']

  def matches(self,**kwargs):
    for tag,val in kwargs.items():
      if tag not in self.tags:
        return False
      if self.tags[tag] != val:
        return False
    return True

class OWLink(object):
  '''
  A directional link between to model graph nodes
  '''
  def __init__(self,from_node,from_output,to_node,to_input):
    assert from_node is not None
    assert from_output is not None
    assert to_node is not None
    assert to_input is not None

    if not from_node.has_output(from_output):
        raise InvalidFluxException(from_node,from_output,'output')
    if not to_node.has_input(to_input):
        raise InvalidFluxException(to_node,to_input,'input')

    self.from_node = from_node
    self.from_output = from_output
    self.to_node = to_node
    self.to_input = to_input


# class OWSystem(object):
#   def __init__(self):
#     self.nodes = []
#     self.links = []

#   def add_node(self,name,)

def template_to_graph(g:nx.DiGraph,tpl:OWTemplate,allow_duplicates=False,**tags) -> nx.DiGraph:
  """
  Add all the nodes and links in an Openwater Template to a graph

  Parameters
  ----------
  g: nx.DiGraph
    an existing graph object. If None, a new graph object will be created
  tpl: OWTemplate
    Openwater Template to add to the graph
  allow_duplicates:
    Whether to allow duplicate links (ie between the same two nodes and variables) or whether to throw an exception
    Defaults to False (ie raise exception on duplicate)
  tags:
    Additional tags to assign to all nodes in the template when adding to the graph

  Returns
  -------
  nx.DiGraph
    The graph object passed in as g, or the new graph object created

  Raises
  ------
  Exception
    when duplicate link encountered (unless allow_duplicates=True)
  """
  if not g:
    g = nx.DiGraph()
  nodes = {}
  nw = tpl.instantiate(**tags)
  for n in nw.nodes:
      g.add_node(str(n),**n.tags)
      nodes[str(n)] = n

  for l in nw.links:
      key = (str(nodes[str(l.from_node)]),str(nodes[str(l.to_node)]))
      if key in g.edges:
          existing = g.edges[key]
          if not allow_duplicates and \
             (l.from_output in existing['src']) and \
             (l.to_input in existing['dest']):
              raise Exception(f'Duplicate link along {key}, between {l.from_output} and {l.to_input}')
          existing['src'].append(l.from_output)
          existing['dest'].append(l.to_input)
          continue

      g.add_edge(key[0],key[1],
                 src=[l.from_output],dest=[l.to_input])

  return g

def node_matches(n,**kwargs):
    for k,v in kwargs.items():
        if not k in n:
            return False
        if n[k]!=v:
            return False
    return True

def match_nodes(g,**kwargs):
    return [name for name,node in g.nodes.items() if node_matches(node,**kwargs)]

def model_type(n):
    return str(n).split('(')[1][:-1]

def ancestors_by_node(g):
    ancestors_by_node = {}
    # by_node_type_gen = {}
    # node_gen = {}
    for n in list(g.nodes):
        ancestors = nx.ancestors(g,n, backend="cugraph")
        ancestors_by_node[n] = ancestors

        # mt = model_type(n)

        # if not mt in by_node_type_gen:
        #     by_node_type_gen[mt] = {}

        # n_ancestors = len(ancestors)
        # if not n_ancestors in by_node_type_gen[mt]:
        #     by_node_type_gen[mt][n_ancestors] = []
        # by_node_type_gen[mt][n_ancestors].append(n)
        # node_gen[n]=n_ancestors
    return ancestors_by_node #,by_node_type_gen,node_gen

def assign_stages(order,node_gen,by_node_type_gen):
    done = {}
    stages = []
    i = 0
    for n in order:
        if n in done: continue

        gen = node_gen[n]
        mt = model_type(n)
        others = by_node_type_gen[mt][gen]
        while len(stages) <= gen:
            stages.append([])

        stages[gen] += others
        for o in others: done[o]=i
        i += 1
    return stages

def map_stages(stages):
    result = {}
    for i,s in enumerate(stages):
        for n in s:
            result[n] = i
    return result

THRESHOLD_N_JOBS=500
def find_first_small_stage(stages):
    for i,s in enumerate(stages):
        if len(s)<THRESHOLD_N_JOBS:
            return i

    return -1

def flatten(l_of_l):
    return [item for sublist in l_of_l for item in sublist]

class SimulationSorter(object):
  def __init__(self,graph):
    self.graph = graph
    self.descendants_by_node = {}
    self.ancestors_by_node = {}
    self.node_ancestry_df = None
    self.node_descendent_df = None
    self.imm_descendants_by_node={}

  def descendants_cached(self,g,n):
      if not n in self.descendants_by_node:
          self.descendants_by_node[n] = nx.descendants(g,n)
      return self.descendants_by_node[n]

  def immediate_descendants_cached(self,g,n):
      if not n in self.imm_descendants_by_node:
          self.imm_descendants_by_node[n] = set(g.successors(n))
      return self.imm_descendants_by_node[n]

  def bring_forward(self,g,stages):
      init_timer('bring_forward')
      first_small_stage = find_first_small_stage(stages)
      for i in range(first_small_stage,len(stages)-1):
          si = stages[i]
          if not len(si):
              continue

          all_descendents = set(flatten([self.descendants_cached(g,n) for n in si]))

          si1 = stages[i+1]
          candidates = [n for n in si1 if not n in all_descendents]
          if len(candidates):
              stages[i] += candidates
              stages[i+1] = list(set(stages[i+1]) - set(candidates)) # [n for n in stages[i+1] if not n in candidates] # set difference?
      stages = [s for s in stages if len(s)]
      close_timer()
      return stages

  def latest_possible(self,g,n,n_stages,node_stages):
      current = node_stages[n]
      next_stage = current+1
      descendants = self.immediate_descendants_cached(g,n)
      # descendent_stages = np.array([node_stages[d] for d in descendants])
      # earliest_descendent = descendent_stages.min()
      # return earliest_descendent - 1

      lowest = n_stages
      for d in descendants:
          descendent_stage = node_stages[d]
          # if descendent_stage == 0:
          #     print(n,d,n_stages,current,descendent_stage,lowest)
          if descendent_stage >= lowest:
              continue
          if descendent_stage == next_stage:
              return current
          lowest = descendent_stage
          # if descendent_stage < lowest:
          #     lowest = descendent_stage
      if not lowest:
          print(n,current,n_stages,d,descendent_stage)
          raise Exception('Boo')
      return lowest-1

  # @profile
  def push_back_ss(self,g,stages):
    init_timer('push_back_ss')
    to_remove = {n:[] for n in range(len(stages))}
    to_add = {n:[] for n in range(len(stages))}

  #   first_small_stage = find_first_small_stage(stages)

    visited = {}
  #   init_timer('map node stages')
    node_stages = map_stages(stages)
  #   close_timer()

    count = 0
    nodes_downstream = 0

    for i in range(len(stages)-1,-1,-1):
      # init_timer('stage %d'%i)
      stage_nodes = list(set(stages[i]).union(set(to_add[i])) - set(to_remove[i]))
      stages[i] = stage_nodes
      nodes_downstream += len(stage_nodes)
      for n in stage_nodes:
        already_visited = n in visited
        if already_visited and visited[n]==i:
          # Node last visited as an ancestor and not moved, so no reason to look further at ancestors
          continue
        ancestors = self.ancestors_by_node[n]
        for a in ancestors:
          current_stage = node_stages[a]
          visited[a]=current_stage
          if current_stage == (i-1):
              continue # Already as late as possible
          new_stage = self.latest_possible(g,a,len(stages),node_stages)
          if new_stage==current_stage:
              continue
          to_add[new_stage].append(a)
          to_remove[current_stage].append(a)
          node_stages[a] = new_stage
    stages = [s for s in stages if len(s)]
    close_timer()
    return stages

  def compute_simulation_order(self):
    print("COMPUTING SEQUENTIAL ORDER " + time.strftime("%Y-%m-%d %H:%M:%S.", time.localtime()) + f"{int(time.time() * 1000) % 1000:03d}")
    g = self.graph
    # self.descendants_by_node = {}
    stages = list(nx.topological_generations(g))
    import pickle
    with open("stages.pickle", 'wb') as handle:
       pickle.dump(stages, handle)
    # self.ancestors_by_node = ancestors_by_node(g)
    if len(stages)==1:
      return stages
    print("FOUND SEQUENTIAL ORDER " + time.strftime("%Y-%m-%d %H:%M:%S.", time.localtime()) + f"{int(time.time() * 1000) % 1000:03d}")
    
    return stages

def tag_set(nodes):
    return reduce(lambda a,b: a.union(b),[set(n.keys()) for n in nodes])

def proc_model(node):
    return '%s/%s'%(node[TAG_PROCESS],node[TAG_MODEL])

def match_model_name(node_name):
    return g.nodes[node_name][TAG_MODEL]
    #    return np.bytes_(re.match(re.compile('.*\(([\w\d]+)\)'),node_name)[1])

def sort_nodes(nodes):
    '''
    Sort a group of nodes by relevant criteria

    (Currently just name - but ultimately by tags in some way!)
    '''
    return sorted(nodes)


class ModelGraph(object):
    def __init__(self,graph,initialise=True,time_period=None):
        self._graph = graph
        self._parameteriser = None
        self._last_write = None
        self.time_period = time_period
        if initialise:
            self.initialise()

    def initialise(self):
        init_timer('Compute simulation order')
        sorter = SimulationSorter(self._graph)
        self.order = sorter.compute_simulation_order()
        report_time('Tag nodes')
        for i,gen in enumerate(self.order):
            for node in gen:
                self._graph.nodes[node][TAG_GENERATION]=i
        self.sequence = flatten(self.order)

        nodes = self._graph.nodes
        self.model_names = list({n[TAG_MODEL] for n in self._graph.nodes.values()} )

        proc_models = {proc_model(nodes[n]) for n in nodes}

        node_names_by_process_and_model = {pm:[n for n in nodes if proc_model(nodes[n])==pm] for pm in proc_models}
        nodes_by_process_and_model = {pm:[nodes[n] for n in proc_nodes] for pm,proc_nodes in node_names_by_process_and_model.items()}

        tags_by_process_and_model = {p:list(tag_set(nodes)-set(META_TAGS)) for p,nodes in nodes_by_process_and_model.items()}
        self.all_tags = set().union(*tags_by_process_and_model.values())
        self.distinct_values = {t:sorted(set([nodes[n][t] for n in nodes if t in nodes[n]])) for t in self.all_tags}
        report_time('Assign runtime metadata')
        for pm in proc_models:
            node = nodes_by_process_and_model[pm][0]
            self.assign_run_indices(node[TAG_PROCESS],node[TAG_MODEL])
        close_timer()

    def assign_run_indices(self,proc,model_type):
        '''
        Assign run indices to each model run within a given process, p (eg 'rainfall runoff')
        '''
        i = 0
        nodes = self._graph.nodes
        for gen in self.order:
            relevant_gen = [n for n in gen if nodes[n][TAG_MODEL]==model_type] # and nodes[n][TAG_PROCESS]==proc] ### BAD ASSUMPTION!
            relevant_gen = sort_nodes(relevant_gen)

            for n in relevant_gen:
                node = nodes[n]
                node[TAG_RUN_INDEX] = i
                i += 1

    def nodes_matching(self,model,**tags):
        model = model_name(model)
        if not '_model' in tags:
            tags['_model'] = model

        def tags_match(node):
            for k,v in tags.items():
                if not k in node or node[k] != v:
                    return False
            return True
        return  {n:self._graph.nodes[n] for n in self._graph.nodes if tags_match(self._graph.nodes[n])}

    def write_model(self,f,timesteps=DEFAULT_TIMESTEPS):
        init_timer('Write model file')
        init_timer('Write meta and dimensions')
        close = False
        if hasattr(f,'upper'):
            h5f = h5py.File(f,'w')
            close = True
        else:
            h5f = f

        try:
            self._write_meta(h5f)
            report_time('Write model groups')
            self._write_model_groups(h5f,timesteps)
            report_time('Write links')
            self._write_links(h5f)
            report_time('Write dimensions')
            self._write_dimensions(h5f)
        finally:
            if close: h5f.close()
        self._last_write=f
        close_timer()
        close_timer()

    def run(self,time_period,model_fn=None,results_fn=None,**kwargs):
        '''

        kwargs: Arguments and fflags to pass directly to ow-sim, including:

        * overwrite (boolean): Overwrite existing output file if it exists
        * verbose (boolean): Show verbose logging during simulation
        '''
        if model_fn:
            self.write_model(model_fn,len(time_period))
        model_fn = self._last_write

        if not model_fn:
            raise Exception('model_fn not provided and model not previously saved')

        return _run(time_period,model_fn,results_fn,**kwargs)

    def _flux_number(self,node_name,flux_type,flux_name):
        node = self._graph.nodes[node_name]

        model_type = node[TAG_MODEL]
        desc = getattr(node_types,model_type).description
        flux_type = flux_type.capitalize()
        if not flux_type.endswith('s'):
            flux_type += 's'
        if not flux_name in desc[flux_type]:
            raise Exception('Unknown %s flux %s on %s'%(flux_type[:-1],flux_name,model_type))
            #return -1
        return desc[flux_type].index(flux_name)

    def _write_meta(self,h5f):
        meta = h5f.create_group('META')
        meta.create_dataset('models',
                            data=[np.bytes_(n) for n in self.model_names],
                            dtype='S%d'%max([len(mn) for mn in self.model_names]))
        if self.time_period is not None:
          dates = np.array([ts.isoformat() for ts in self.time_period],dtype=h5py.special_dtype(vlen=str))
          meta.create_dataset('timeperiod',data=dates)

    def _write_dimensions(self,f):
        dimensions = f.create_group('DIMENSIONS')
        for t in self.all_tags:
            vals = self.distinct_values[t]
            if hasattr(vals[0],'__len__'):
                vals = [np.bytes_(v) for v in vals]
            dimensions.create_dataset(t,data=vals)

    def _map_process(self,node_set):
        '''
        For a given model (eg 'GR4J'), organise all model runs
        by the parameterisation dimensions (eg catchment x hru) and assign indices
        '''

        nodes = self._graph.nodes
        def dim_tags(node_name):
            node = nodes[node_name]
            keys = node.keys()
            return set(keys) - set(META_TAGS)

        dimsets = {frozenset(dim_tags(n)) for n in node_set}
        all_dims = set(chain.from_iterable(dimsets))
        # assert len(dimsets)==1 # don't support one process having different dimensions
        # Should at least support attributes (tags that only ever have one value)
        dimensions = all_dims # list(dimsets)[0]

#        dim_values = {d:sorted({nodes[n][d] for n in node_set}) for d in dimensions}
        if len(dimsets) > 1:
            print('Populating nodes with dummy dimension values')
            for dim in dimensions:
                added_dummy = False
                dummy_val = f'dummy-{dim}'
                for node in node_set:
                    if not dim in nodes[node]:
                        nodes[node][dim] = dummy_val
                        if not added_dummy and (dummy_val not in self.distinct_values[dim]):
                            self.distinct_values[dim].append(dummy_val)
                        added_dummy = True

        dim_values = {d:sorted({nodes[n][d] for n in node_set}) for d in dimensions}
        attributes = {d:vals[0] for d,vals in dim_values.items() if (len(vals)==1) and (len(node_set)>1)}
        dimension_values = {d:vals for d,vals in dim_values.items() if (len(vals)>1) or (len(node_set)==1)}
        dimensions = [d for d in dimensions if not d in attributes]

        if not len(dimensions):
            print(attributes)
            print(len(node_set))
            n = self._graph.nodes[node_set[0]]
            raise Exception(f'No dimensions for model {n[TAG_MODEL]} in process {n[TAG_PROCESS]}')

        if len([d for d in dimensions if len(dimension_values[d])==0]):
            print('Dimension(s) with 0 length:',dimension_values)
            raise Exception('Dimension(s) with 0 length')
        # dims = tags_by_process[p]
        # dimensions = [distinct_values[d] for d in dims]
        shape = tuple([len(self.distinct_values[d]) for d in dimensions])

        model_instances = np.ones(shape=shape,dtype=np.int32) * -1
        for node_name in node_set:
            node = nodes[node_name]

            loc = tuple([self.distinct_values[d].index(node[d]) for d in dimensions])
            if len(loc) < len(shape):
                print(loc,node)
            model_instances[loc] = node[TAG_RUN_INDEX]

        return dimension_values, attributes,model_instances

    def _write_model_groups(self,f,n_timesteps):
        models_grp = f.create_group('MODELS')
        nodes = self._graph.nodes

        models = {nodes[n][TAG_MODEL] for n in nodes}
        self.model_batches = {}
        for mx,m in enumerate(models):
            model_msg ='Writing model %s (%d/%d)'%(m,mx+1,len(models))
            init_timer(model_msg)
            print(model_msg)
            model_grp = models_grp.create_group(m)

            model_nodes = [n for n in nodes if nodes[n][TAG_MODEL]==m]
            processes_for_model = {nodes[n][TAG_PROCESS] for n in model_nodes}
            # assert(len(processes_for_model)==1) # not necessary?

            dims,attributes,instances = self._map_process(model_nodes)
            ds = model_grp.create_dataset('map',dtype=instances.dtype,data=instances,fillvalue=-1)

            # write out model index
            ds.attrs['PROCESSES']=[np.bytes_(s) for s in list(processes_for_model)]
            ds.attrs['DIMS']=[np.bytes_(d) for d in dims]
            for attr,val in attributes.items():
                ds.attrs[attr]=val

            self.model_batches[m] = np.cumsum([len([n for n in gen if nodes[n][TAG_MODEL]==m]) for gen in self.order])

            model_meta = getattr(node_types,m)
            if hasattr(model_meta,'description'):
                desc = model_meta.description
                n_states = len(desc['States']) # Compute, based on parameters...
                n_params = len(desc['Parameters'])
                n_inputs = len(desc['Inputs'])
            else:
                print('No description for %s'%m)
                desc = None
                n_states = 3
                n_params = 4
                n_inputs = 2

#            batch_counts = [len(mc.get(m,[])) for mc in model_counts_by_generation]

            model_grp.create_dataset('batches',shape=(len(self.order),),dtype=np.uint32,data=self.model_batches[m],fillvalue=-1)

            n_cells = len(model_nodes) # instances.size
            # Init states....
            model_grp.create_dataset('states',shape=(n_cells,n_states),dtype=np.float64,fillvalue=0)

            model_grp.create_dataset('parameters',shape=(n_params,n_cells),dtype=np.float64,fillvalue=0)

            # model_grp.create_dataset('inputs',shape=(n_cells,n_inputs,n_timesteps),dtype=np.float64,fillvalue=0)

            if (self._parameteriser is not None) and (desc is not None):
                node_dict = {n:nodes[n] for n in model_nodes}
                nodes_df = pd.DataFrame([nodes[n] for n in model_nodes])
                for k,v in attributes.items():
                    nodes_df[k] = v
                full_dims = dict(**dims,**attributes)
                init_timer('Parameterisation')
                self._parameteriser.parameterise(model_meta,model_grp,instances,full_dims,node_dict,nodes_df)
                close_timer()
            close_timer()

    def gen_index(self,node):
        global_idx = node['_run_idx']
        model_name = node['_model']
        gen = node['_generation']
        if gen:
            start_of_gen = self.model_batches[model_name][gen-1]
        else:
            start_of_gen = 0
        return global_idx - start_of_gen

    def link_table(self):
        model_lookup = dict([(m,i) for i,m in enumerate(self.model_names)])
        link_table = []

        for l_from,l_to in self._graph.edges:
            link_data = self._graph.edges[(l_from,l_to)]
            for src_var,dest_var in zip(link_data['src'],link_data['dest']):
                link = {}
                f_node = self._graph.nodes[l_from]
                t_node = self._graph.nodes[l_to]

                link['src_generation'] = f_node['_generation']
                link['src_model'] = model_lookup[f_node['_model']]
                link['src_node'] = f_node['_run_idx']
                link['src_gen_node'] = self.gen_index(f_node)
                link['src_var'] = self._flux_number(l_from,'output',src_var)

                link['dest_generation'] = t_node['_generation']
                link['dest_model'] = model_lookup[t_node['_model']]
                link['dest_node'] = t_node['_run_idx']
                link['dest_gen_node'] = self.gen_index(t_node)
                link['dest_var'] = self._flux_number(l_to,'input',dest_var)
                link_table.append(link)
        link_table = pd.DataFrame(link_table)
        col_order = LINK_TABLE_COLUMNS
        if not len(link_table):
          return pd.DataFrame(columns=col_order)

        link_table = link_table[col_order]
        sort_order = ['src_generation','src_model','src_gen_node','dest_generation','dest_model','dest_gen_node']
        return link_table.sort_values(sort_order)

    def _write_links(self,f):
        table = np.array(self.link_table())
        if len(table):
          f.create_dataset('LINKS',dtype=np.uint32,data=table)
        else:
          f.create_dataset('LINKS',dtype=np.uint32,shape=[0,len(LINK_TABLE_COLUMNS)])

def dim_val(v):
    if hasattr(v,'decode'):
        return v.decode()
    return v

class ModelFile(object):
    def __init__(self,fn):
        self.filename = fn
        import h5py
        self._h5f = h5py.File(self.filename,'r')
        self._dimensions = {k:[dim_val(d) for d in self._h5f['DIMENSIONS'][k][...]] for k in self._h5f['DIMENSIONS']}
 #       print(self._dimensions)
        self._links = pd.DataFrame(self._h5f['LINKS'][...],columns=LINK_TABLE_COLUMNS)
        self._models = self._h5f['META']['models'][...]
        if 'timeperiod' in self._h5f['META']:
          timesteps = [d for d in self._h5f['META']['timeperiod'][...]]
          if isinstance(timesteps[0],bytes):
            timesteps = [d.decode() for d in timesteps]

          self.time_period = pd.DatetimeIndex([pd.Timestamp.fromisoformat(d) for d in timesteps])
        self._parameteriser = None

    def __enter__(self):
      return self

    def __exit__(self, type, value, traceback):
      self.close()

    def _matches(self,model,**tags):
        model_dims = [d.decode() for d in self._h5f['MODELS'][model]['map'].attrs['DIMS']]
        # print(model_dims)
        lookup = {}
        for tag,value in tags.items():
            if not tag in model_dims:
                return False
            if not value in self._dimensions[tag]:
                return False
            lookup[tag] = self._dimensions[tag].index(value)

        idx = [lookup.get(d,slice(None,None)) for d in model_dims]
        # print(model,list(zip(model_dims,idx)))
        return np.any(self._h5f['MODELS'][model]['map'][tuple(idx)] >= 0)

    def models_matching(self,**tags):
        result = []
        for k in self._h5f['MODELS']:
            # print(k)
            if self._matches(k,**tags):
                result.append(k)
        return result

    def _map_model_dims(self,model):
        model = model_name(model)
        model_map = self._h5f['MODELS'][model]['map'][...]
        m_dims = [dim_val(d) for d in self._h5f['MODELS'][model]['map'].attrs['DIMS']]
        dims = {d:self._h5f['DIMENSIONS'][d][...] for d in m_dims}
        dim_indices = list(zip(*np.where(model_map>=0)))#np.logical_not(np.isnan(model_map)))))
        def translate_dims(tpl):
            return [dim_val(dims[d][ix]) for d,ix in zip(m_dims,tpl)]

        dim_columns = [translate_dims(di)+[model_map[di]] for ix,di in enumerate(dim_indices) if model_map[di]>=0]

        return {d:[di[i] for di in dim_columns] for i,d in enumerate(m_dims+['_run_idx'])}

    def _raw_parameters(self,model,**tags):
        vals = self._h5f['MODELS'][model]['parameters'][...]

        model_map = self._map_model_dims(model)
        df = pd.DataFrame(model_map)
        dim_cols = set(df.columns) - {'_run_idx'}
        df = df.set_index(list(dim_cols))

        param_df = pd.DataFrame(vals).transpose().reindex(index=df['_run_idx'])

        result = param_df.set_index(df.index)
        return result

    def dims_for_model(self,model):
      return [d.decode() for d in self._h5f['MODELS'][model]['map'].attrs['DIMS']]

    def parameters(self,model,**tags):
        return _tabulate_model_scalars_from_file(self._h5f,
                                                 model_name(model),
                                                 self._map_model_dims(model),
                                                 'parameters',
                                                 **tags)

    def initial_states(self,model,**tags):
        return _tabulate_model_scalars_from_file(self._h5f,
                                                 model_name(model),
                                                 self._map_model_dims(model),
                                                 'states',
                                                 **tags)

    def indexed_parameters(self,model,tabular=False,**tags):
        '''
        Return a table of parameters for a model that includes dimensioned (tabular) parameters

        Parameters
        ----------
        model : string
          Model name
        tabular : boolean
          If True, return a tabular data frame (ie with one column per parameter and one row per index)
          Only works for a single model node
        tags : dict
          Tags to filter the model nodes by
        '''
        raw = self._raw_parameters(model,**tags)
        desc = getattr(node_types,model).description
        indexed = create_indexed_parameter_table(desc,raw)
        index_names = indexed.index.names
        indexed = indexed.reset_index()
        for k,v in tags.items():
            indexed = indexed[indexed[k]==v]
        indexed = indexed.set_index(index_names)

        if tabular:
           if len(indexed)>1:
              raise Exception('Cannot return tabular data for more than one model node')
           idx = indexed.index[0]
           return indexed.transpose().reset_index().pivot(index='index',columns='parameter',values=idx)
        return indexed

    def nodes_matching(self,model,**tags):
        if hasattr(model,'name'):
            model = model.name
        nodes = pd.DataFrame(self._map_model_dims(model))
        for tag,tag_val in tags.items():
            nodes = nodes[nodes[tag]==tag_val]
        return nodes

    def link_table(self):
        """
        Return the table of links between model nodes as a Data Frame.
        """
        linkages = pd.DataFrame(self._h5f['LINKS'][...],columns=LINK_TABLE_COLUMNS)
        all_models = np.array([m.decode() for m in list(self._h5f['META']['models'][...])])

        descriptions = {mod:getattr(node_types,mod).description for mod in all_models}

        linkages.src_model = all_models[linkages.src_model]
        linkages.dest_model = all_models[linkages.dest_model]

        linkages.src_var = [descriptions[m]['Outputs'][v] for m,v in zip(linkages.src_model,linkages.src_var)]
        linkages.dest_var = [descriptions[m]['Inputs'][v] for m,v in zip(linkages.dest_model,linkages.dest_var)]
        return linkages

    def links_between(self,dest_mod=None,dest_var=None,src_mod=None,src_var=None,src_tags={},dest_tags={},annotate=True,**kwargs):
        """
        Identify the links between particular model graph nodes.

        Optionally (and by default), label src node and destination nodes by tags

        All parameters are optional. By default returns all links, with all tags.

        Parameters
        ----------
        dest_mod : string
          Destination model type (eg EmcDwc) and only show links to this model type
        dest_var : string
          Destination variable (eg inflow) and only show links to this variable
        src_mod: string
          Source model type (eg EmcDwc) and only show links from this model type
        src_var: string
          Source variable (eg outflow) and only show links from this variable
        src_tags: dict
          Only show links from graph nodes with all these tags
        dest_tags: dict
          Only show links to graph nodes with all these tags
        annotate: boolean
          Add source node and destination node tags as columns to the data frame
        kwargs:
          Only show links between graph nodes with all of these tags
        """
        linkages = self.link_table()
        if dest_mod:
            dest_mod = model_name(dest_mod)
            linkages = linkages[linkages.dest_model==dest_mod]
            nodes = self.nodes_matching(dest_mod,**dest_tags,**kwargs)
            linkages = linkages[linkages.dest_node.isin(nodes._run_idx)]

        if dest_var:
            linkages = linkages[linkages.dest_var==dest_var]

        if src_mod:
            src_mod = model_name(src_mod)
            linkages = linkages[linkages.src_model==src_mod]
            nodes = self.nodes_matching(src_mod,**src_tags,**kwargs)
            linkages = linkages[linkages.src_node.isin(nodes._run_idx)]

        if src_var:
            linkages = linkages[linkages.src_var==src_var]

        if annotate:
            model_maps = {m:pd.DataFrame(self._map_model_dims(m)) for m in set(linkages.src_model).union(linkages.dest_model)}

            def annotate_tbl(prefix):
                tag_names = set([c for m in set(linkages[f'{prefix}_model']) for c in model_maps[m].columns])-{'_run_idx'}
                for tag_name in tag_names:
                    col = f'{prefix}_{tag_name}'
                    rows = [model_maps[m][model_maps[m]._run_idx==n] for m,n in zip(linkages[f'{prefix}_model'],linkages[f'{prefix}_node'])]
                    linkages[col] = [row[tag_name].iloc[0] if tag_name in row else '-' for row in rows]

            annotate_tbl('src')
            annotate_tbl('dest')
        return linkages

    def close(self):
        self._h5f.close()
        self._h5f = None

    def copy(self,dest_fn):
        shutil.copy(self.filename,dest_fn)
        return ModelFile(dest_fn)

    def write(self,clear_inputs=False):
        try:
            self.close()
            import h5py
            self._h5f = h5py.File(self.filename,'r+')
            if self._parameteriser is None:
                print('Nothing to do')
                return

            models_grp = self._h5f['MODELS']
            models = list(models_grp.keys())
            for m in models:
                print('Parameterising %s'%str(m))
                model_grp = models_grp[m]

                instances = model_grp['map'][...]
                dims = [dim_val(d) for d in model_grp['map'].attrs['DIMS']]

                dim_map = self._map_model_dims(m)

                nodes = ['%s-%d'%(m,ix) for ix in range(len(dim_map[dims[0]]))]

                # dims,attributes,instances = self._map_process(model_nodes)

                model_meta = getattr(node_types,m)

                # for k,v in attributes.items():
                #     nodes_df[k] = v
                # full_dims = dict(**dims,**attributes)

                node_dict = {n:{d:vals[ix] for d,vals in dim_map.items()} for ix,n in enumerate(nodes)}
                nodes_df = pd.DataFrame({'node':nodes})
                for d, vals in dim_map.items():
                    nodes_df[d] = vals

                if clear_inputs and 'inputs' in model_grp:
                    del model_grp['inputs']

                # initialise parameters and states if they don't exist!

                self._parameteriser.parameterise(model_meta,model_grp,instances,dim_map,node_dict,nodes_df)
        finally:
            self.close()
            self._h5f = h5py.File(self.filename,'r')

    def run(self,time_period,results_fn=None,**kwargs):
        '''

        kwargs: Arguments and fflags to pass directly to ow-sim, including:

        * overwrite (boolean): Overwrite existing output file if it exists
        * verbose (boolean): Show verbose logging during simulation
        '''
        return _run(time_period,self.filename,results_fn,**kwargs)

def _run(time_period,model_fn=None,results_fn=None,**kwargs):
    '''

    kwargs: Arguments and fflags to pass directly to ow-sim, including:

    * overwrite (boolean): Overwrite existing output file if it exists
    * verbose (boolean): Show verbose logging during simulation
    '''
    from openwater.discovery import _exe_path
    from openwater.results import OpenwaterResults

    if not results_fn:
        base,ext = os.path.splitext(model_fn)
        results_fn = '%s_outputs%s'%(base,ext)
        print('INFO: No output filename provided. Writing to %s'%results_fn)

    cmd_line = [_exe_path('sim')]
    for k,v in kwargs.items():
      cmd_line += ow_sim_flag_text(k,v)
    cmd_line.append(model_fn),
    cmd_line.append(results_fn)
    # "%s %s %s %s"%(_exe_path('sim'),flags,model_fn,results_fn)

    logger.debug('Running with command line: %s',cmd_line)
    proc = Popen(cmd_line,stdout=PIPE,stderr=PIPE,bufsize=1, close_fds=ON_POSIX)
    std_out_queue,std_out_thread = configure_non_blocking_io(proc,'stdout')
    std_err_queue,std_err_thread = configure_non_blocking_io(proc,'stderr')

    err = []
    out = []
    finished = False
    while not finished:
        if proc.poll() is not  None:
            finished = True

        end_stream=False
        while not end_stream:
            try:
                line = std_err_queue.get_nowait().decode('utf-8')
                err.append(line)
                print('ERROR %s'%(line,))
                sys.stdout.flush()
            except Empty:
                end_stream = True

        end_stream = False
        while not end_stream:
            try:
                line = std_out_queue.get_nowait().decode('utf-8')
                out.append(line)
                print(line)
                sys.stdout.flush()
            except Empty:
                end_stream = True
                sleep(0.05)

    assert proc.returncode==0
    return OpenwaterResults(model_fn,results_fn,time_period)

def run_simulation(model,output='model_outputs.h5',overwrite=False):
    import openwater.discovery
    cmd = '%s/ow-sim'%openwater.discovery.OW_BIN
    if overwrite:
        cmd += ' -overwrite'
    cmd = '%s %s %s'%(cmd,model,output)
    res = os.system(cmd)
    return res

def _enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

def configure_non_blocking_io(proc,stream):
    queue = Queue()
    thread = Thread(target=_enqueue_output,args=(getattr(proc,stream),queue))
    thread.daemon = True
    thread.start()
    return queue,thread

def ow_sim_flag_text(k,v):
  k = k.replace('_','-')
  k = '-%s'%k
  if v == False:
      return []
  if v == True:
      return [k]

  if hasattr(v,'__len__') and not isinstance(v,str):
    if hasattr(v,'items'):
      v = ','.join([f'{v_key}:{v_val}' for v_key,v_val in v.items()])
    else:
      v = ','.join(v)
  return [k,str(v)]

class InvalidFluxException(Exception):
    def __init__(self,node,flux_name,flux_type):
        super(InvalidFluxException,self).__init__(f'Invalid flux: Node ({node}) has no {flux_type} named {flux_name}')
        self.node = node
        self.flux_type = flux_type
        self.flux_name = flux_name

