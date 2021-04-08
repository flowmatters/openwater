import os
import sys
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

class OWTemplate(object):
  def __init__(self,lbl=''):
    self.label=lbl
    self.nodes = []
    self.links = []
    self.nested = []
    self.inputs = []
    self.outputs = []

  def define_input(self,node,name=None,alias=None,**kwargs):
    if name is None:
      # Would be nice to have a short hand to define every input of this
      # node as an input to the graph (and similarly every output of a node
      # as an output of the graph
      # But the node currently stores the model name)
      pass

    if not node.has_input(name):
        raise InvalidFluxException(node,name,'input')

    if alias is None:
      alias = name
    self.inputs.append((node,name,alias,kwargs))

  def define_output(self,node,name=None,alias=None,**kwargs):
    if alias is None:
      alias = name

    if not node.has_output(name):
        raise InvalidFluxException(node,name,'output')

    self.outputs.append((node,name,alias,kwargs))

  def add_node(self,model_type=None,name=None,process=None,**tags):
    # if hasattr(node_or_name,'model_type'):
    #   self.nodes.append(node_or_name)
    # else:
    if process and not TAG_PROCESS in tags:
        tags[TAG_PROCESS]=process

    new_node = OWNode(model_type,name,**tags)
    self.nodes.append(new_node)
    return new_node

  def add_link(self,link):
    self.links.append(link)

  def add_conditional_link(self,from_node,from_output,to_node,possibly_inputs,model):
    for pi in possibly_inputs:
      if pi in model.description['Inputs']:
        self.add_link(OWLink(from_node,from_output,to_node,pi))
        return True
    return False

  def match_labelled_flux(self,fluxes,flux_name,flux_tags,exclude_tags):
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
          return node, name
      return None,None

  def make_link(self,output_name,input_name,
                from_node=None,from_tags={},from_exclude_tags=[],
                to_node=None,to_tags={},to_exclude_tags=[]):
    if from_node is None:
        from_node, new_output_name = self.match_labelled_flux(
            self.outputs,output_name,from_tags,from_exclude_tags)
        if from_node is None or new_output_name is None:
            n_outputs = len(self.outputs)
            raise Exception('No matching output for %s, with tags %s. Have %d outputs'%(new_output_name,str(from_tags),n_outputs))
        output_name = new_output_name
    if to_node is None:
        to_node, new_input_name = self.match_labelled_flux(
            self.inputs,input_name,to_tags,to_exclude_tags)

        if to_node is None or new_input_name is None:
            raise Exception('No matching input for %s, with tags %s'%(new_input_name,str(to_tags)))
        input_name = new_input_name

    return OWLink(from_node,output_name,to_node,input_name)

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

class OWNode(object):
  def __init__(self,model_type,name=None,**tags):
    self.model_type = model_type
    if hasattr(model_type,'name'):
      self.model_name = model_type.name
      self.model_type = model_type
    else:
      self.model_name = model_type
      import openwater.nodes as node_types
      from openwater.discovery import discover
      discover()
      self.model_type = getattr(node_types,self.model_name)

    self.tags = tags
    self.tags[TAG_MODEL] = self.model_name

    if name:
      self.name = name
    else:
      self.name = self.make_name()

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

class OWLink(object):
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

def template_to_graph(g,tpl,**tags) -> nx.DiGraph:
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

def group_run_order(g):
    ancestors_by_node = {}
    by_node_type_gen = {}
    node_gen = {}
    for n in list(g.nodes):
        ancestors = nx.ancestors(g,n)
        ancestors_by_node[n] = ancestors

        mt = model_type(n)
    
        if not mt in by_node_type_gen:
            by_node_type_gen[mt] = {}
    
        n_ancestors = len(ancestors)
        if not n_ancestors in by_node_type_gen[mt]:
            by_node_type_gen[mt][n_ancestors] = []
        by_node_type_gen[mt][n_ancestors].append(n)
        node_gen[n]=n_ancestors
    return ancestors_by_node,by_node_type_gen,node_gen

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

descendants_by_node={}
imm_descendants_by_node={}
# cache_queries = 0
# cache_misses = 0
def descendants_cached(g,n):
    # global cache_queries, cache_misses
    # cache_queries += 1

    if not n in descendants_by_node:
        # cache_misses += 1
        descendants_by_node[n] = nx.descendants(g,n)
    return descendants_by_node[n]

    # return list(node_descendent_df[node_descendent_df[n]].index)

def immediate_descendants_cached(g,n):
    if not n in imm_descendants_by_node:
        # cache_misses += 1
        imm_descendants_by_node[n] = set(g.successors(n))
    return imm_descendants_by_node[n]

def bring_forward(g,stages):
    init_timer('bring_forward')
    first_small_stage = find_first_small_stage(stages)
    for i in range(first_small_stage,len(stages)-1):
        si = stages[i]
        if not len(si):
            continue

        all_descendents = set(flatten([descendants_cached(g,n) for n in si]))

        si1 = stages[i+1]
        candidates = [n for n in si1 if not n in all_descendents]
        if len(candidates):
            stages[i] += candidates
            stages[i+1] = list(set(stages[i+1]) - set(candidates)) # [n for n in stages[i+1] if not n in candidates] # set difference?
    stages = [s for s in stages if len(s)]
    close_timer()
    return stages

def latest_possible(g,n,n_stages,node_stages):
    current = node_stages[n]
    next_stage = current+1
    descendants = immediate_descendants_cached(g,n)
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

#shifts = 0

# def push_back_orig(g,stages):
#     first_small_stage = find_first_small_stage(stages)
# #     visited = {}
#     global shifts
#     node_stages = map_stages(stages)
#     count = 0
#     nodes_downstream = 0
#     for i in range(len(stages)-1,-1,-1):
#         stage_nodes = stages[i]
#         nodes_downstream += len(stage_nodes)
#         print(i)
#         for n in stage_nodes:
# #             if (n in visited) and visited[n]==i:
# #                 # Node visited as an ancestor and not moved, so no reason to look further at ancestors
# #                 continue
#             ancestors = ancestors_by_node[n]
#             for a in ancestors:
#                 current_stage = node_stages[a]
# #                 visited[a]=current_stage
#                 if current_stage == (i-1):
#                     continue # Already as late as possible
#                 new_stage = latest_possible(g,a,len(stages),node_stages)
#                 if new_stage==current_stage:
#                     continue

#                 shifts += 1
#                 stages[new_stage].append(a)
#                 stages[current_stage].remove(a)
#                 node_stages[a] = new_stage
#                 #print(i,n,a,current_stage,new_stage)
#                 #count += 1
#                 #assert(count<10)
#     stages = [s for s in stages if len(s)]
#     return stages

# @profile
def push_back_ss(g,stages):
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
#  global shifts
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
      ancestors = ancestors_by_node[n]
      for a in ancestors:
        current_stage = node_stages[a]
        visited[a]=current_stage
        if current_stage == (i-1):
            continue # Already as late as possible
        new_stage = latest_possible(g,a,len(stages),node_stages)
        if new_stage==current_stage:
            continue
#        shifts += 1
        to_add[new_stage].append(a)
        to_remove[current_stage].append(a)
        #stages[new_stage].append(a)
        #stages[current_stage].remove(a)
        node_stages[a] = new_stage
        #print(i,n,a,current_stage,new_stage)
        #count += 1
        #assert(count<10)
    # close_timer()
  stages = [s for s in stages if len(s)]
  close_timer()
  return stages

def compute_simulation_order(graph):
  init_timer('compute_simulation_order')
  init_timer('Get basic order')
  global descendants_by_node
  descendants_by_node = {}
  g = graph
  sequential_order = list(nx.topological_sort(g))
  global ancestors_by_node #, node_ancestry_df, node_descendent_df
  ancestors_by_node,by_node_type_gen,node_gen = group_run_order(g)
  stages = assign_stages(sequential_order,node_gen,by_node_type_gen)
  stages = [s for s in stages if len(s)]

#   report_time('create node ancestry dataframe for %d nodes'%len(ancestors_by_node))
#   node_ancestry_df = pd.DataFrame(data=False,index=list(g.nodes),columns=list(g.nodes))
#   for k,ancestors in ancestors_by_node.items():
#       node_ancestry_df[k][ancestors] = True
#   node_descendent_df = node_ancestry_df.transpose()
  report_time('Grouping model stages/generations')

  n_stages = len(stages)
  new_n_stages = 0
  iteration = 1
  while new_n_stages<n_stages:
    init_timer('Iteration %d'%iteration)
    n_stages = len(stages)
    stages = bring_forward(g,stages)
    stages = push_back_ss(g,stages)
    new_n_stages = len(stages)
    iteration += 1
    close_timer()
  close_timer()
  close_timer()
  return stages

def tag_set(nodes):
    return reduce(lambda a,b: a.union(b),[set(n.keys()) for n in nodes])

def proc_model(node):
    return '%s/%s'%(node[TAG_PROCESS],node[TAG_MODEL])

def match_model_name(node_name):
    return g.nodes[node_name][TAG_MODEL]
    #    return np.string_(re.match(re.compile('.*\(([\w\d]+)\)'),node_name)[1])

def sort_nodes(nodes):
    '''
    Sort a group of nodes by relevant criteria
    
    (Currently just name - but ultimately by tags in some way!)
    '''
    return sorted(nodes)

   
class ModelGraph(object):
    def __init__(self,graph,initialise=True):
        self._graph = graph
        self._parameteriser = None
        self._last_write = None
        if initialise:
            self.initialise()

    def initialise(self):
        init_timer('Compute simulation order')
        self.order = compute_simulation_order(self._graph)
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
        if hasattr(model,'name'):
            model = model.name
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
            import h5py
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
        meta.create_dataset('models',data=[n.encode('utf8') for n in self.model_names])

    def _write_dimensions(self,f):
        dimensions = f.create_group('DIMENSIONS')
        for t in self.all_tags:
            vals = self.distinct_values[t]
            if hasattr(vals[0],'__len__'):
                vals = [np.string_(v) for v in vals]
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
                        if not added_dummy:
                            self.distinct_values[dim].append(dummy_val)
                            added_dummy = True

        dim_values = {d:sorted({nodes[n][d] for n in node_set}) for d in dimensions}
        attributes = {d:vals[0] for d,vals in dim_values.items() if (len(vals)==1) and (len(node_set)>1)}
        dimension_values = {d:vals for d,vals in dim_values.items() if (len(vals)>1) or (len(node_set)==1)}
        dimensions = [d for d in dimensions if not d in attributes]

        if not len(dimensions):
            print(attributes)
            print(len(node_set))
            raise 'No dimensions'

        if len([d for d in dimensions if len(dimension_values[d])==0]):
            print('Dimension(s) with 0 length:',dimension_values)
            raise Exception('Dimension(s) with 0 length')
        # dims = tags_by_process[p]
        # dimensions = [distinct_values[d] for d in dims]
        shape = tuple([len(self.distinct_values[d]) for d in dimensions])

        model_instances = np.ones(shape=shape,dtype=np.uint32) * -1
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
            ds.attrs['PROCESSES']=[np.string_(s) for s in list(processes_for_model)]
            ds.attrs['DIMS']=[np.string_(d) for d in dims]
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

            n_cells = instances.size
            # Init states....
            model_grp.create_dataset('states',shape=(n_cells,n_states),dtype=np.float64,fillvalue=0)

            model_grp.create_dataset('parameters',shape=(n_params,n_cells),dtype=np.float64,fillvalue=0)

            model_grp.create_dataset('inputs',shape=(n_cells,n_inputs,n_timesteps),dtype=np.float64,fillvalue=0)

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
        link_table = link_table[col_order]
        sort_order = ['src_generation','src_model','src_gen_node','dest_generation','dest_model','dest_gen_node']
        return link_table.sort_values(sort_order)

    def _write_links(self,f):
        table = np.array(self.link_table())
        f.create_dataset('LINKS',dtype=np.uint32,data=table)

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
        self._parameteriser = None

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
        return np.any(self._h5f['MODELS'][model]['map'][tuple(idx)] > 0)

    def models_matching(self,**tags):
        result = []
        for k in self._h5f['MODELS']:
            # print(k)
            if self._matches(k,**tags):
                result.append(k)
        return result

    def _map_model_dims(self,model):
        model_map = self._h5f['MODELS'][model]['map'][...]
        m_dims = [dim_val(d) for d in self._h5f['MODELS'][model]['map'].attrs['DIMS']]
        dims = {d:self._h5f['DIMENSIONS'][d][...] for d in m_dims}
        dim_indices = list(zip(*np.where(model_map>=0)))#np.logical_not(np.isnan(model_map)))))
        def translate_dims(tpl):
            return [dim_val(dims[d][ix]) for d,ix in zip(m_dims,tpl)]

        dim_columns = [translate_dims(di)+[model_map[di]] for ix,di in enumerate(dim_indices) if model_map[di]>=0]

        return {d:[di[i] for di in dim_columns] for i,d in enumerate(m_dims+['_run_idx'])}

    def parameters(self,model,**tags):
        vals = self._h5f['MODELS'][model]['parameters'][...]
        desc = getattr(node_types,model)
        names = [p['Name'] for p in desc.description['Parameters']]

        result = pd.DataFrame(self._map_model_dims(model))
        print(vals.shape)
        order = list(result['_run_idx'])
        for ix,name in enumerate(names):
            result[name] = vals[ix,:][order]

        # result = pd.DataFrame(vals.transpose(),columns=names)
        # for k,vals in dims.items():
        #     result = result[:len(vals)]
        #     result[k] = vals

        for k,v in tags.items():
            result = result[result[k]==v]

        return result

    def write(self):
        try:
            self._h5f.close()
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

                self._parameteriser.parameterise(model_meta,model_grp,instances,dim_map,node_dict,nodes_df)
        finally:
            self._h5f.close()
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

    # flags = ' '.join([ow_sim_flag_text(k,v) 
    cmd_line = [_exe_path('sim')]
    for k,v in kwargs.items():
        flag_text = ow_sim_flag_text(k,v)
        if len(flag_text):
            cmd_line.append(flag_text)
    cmd_line.append(model_fn),
    cmd_line.append(results_fn)
    # "%s %s %s %s"%(_exe_path('sim'),flags,model_fn,results_fn)

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
    if v == False:
        return ''
    if v == True:
        return '-%s'%k
    return '-%s %s'%(k,str(v))

class InvalidFluxException(Exception):
    def __init__(self,node,flux_name,flux_type):
        super(InvalidFluxException,self).__init__(f'Invalid flux: Node ({node}) has no {flux_type} named {flux_name}')
        self.node = node
        self.flux_type = flux_type
        self.flux_name = flux_name

