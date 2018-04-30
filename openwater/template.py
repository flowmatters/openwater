import numpy as np
import pandas as pd
import networkx as nx
from functools import reduce
from . import nodes as node_types

TAG_PROCESS='_process'
TAG_MODEL='_model'
TAG_RUN_INDEX='_run_idx'
TAG_GENERATION='_generation'
META_TAGS=[TAG_PROCESS,TAG_MODEL,TAG_RUN_INDEX,TAG_GENERATION]
DEFAULT_TIMESTEPS=365

class OWTemplate(object):
  def __init__(self):
    self.nodes = []
    self.links = []
  
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
    else:
      self.model_name = model_type
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

class OWLink(object):
  def __init__(self,from_node,from_output,to_node,to_input):
    self.from_node = from_node
    self.from_output = from_output
    self.to_node = to_node
    self.to_input = to_input

# class OWSystem(object):
#   def __init__(self):
#     self.nodes = []
#     self.links = []

#   def add_node(self,name,)

def template_to_graph(g,tpl,**tags):
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
    
        n_ancestors_of_type = len([a for a in ancestors if model_type(a)==mt])
        if not n_ancestors_of_type in by_node_type_gen[mt]:
            by_node_type_gen[mt][n_ancestors_of_type] = []
        by_node_type_gen[mt][n_ancestors_of_type].append(n)
        node_gen[n]=n_ancestors_of_type
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
        stages.append(others)
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
# cache_queries = 0
# cache_misses = 0
def descendants_cached(g,n):
    # global cache_queries, cache_misses
    # cache_queries += 1
    if not n in descendants_by_node:
        # cache_misses += 1
        descendants_by_node[n] = nx.descendants(g,n)
    return descendants_by_node[n]

def bring_forward(g,stages):
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
            stages[i+1] = [n for n in stages[i+1] if not n in candidates]
    stages = [s for s in stages if len(s)]
    return stages

def latest_possible(g,n,n_stages,node_stages):
    current = node_stages[n]
    lowest = n_stages

    descendants = descendants_cached(g,n)
    for d in descendants:
        descendent_stage = node_stages[d]
        if descendent_stage == 0:
            print(n,d,n_stages,current,descendent_stage,lowest)
        if descendent_stage == (current+1):
            return current
        if descendent_stage < lowest:
            lowest = descendent_stage
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


def push_back_ss(g,stages):
  to_remove = {n:[] for n in range(len(stages))}
  to_add = {n:[] for n in range(len(stages))}
  
  first_small_stage = find_first_small_stage(stages)

  visited = {}
  node_stages = map_stages(stages)
  count = 0
  nodes_downstream = 0
#  global shifts
  for i in range(len(stages)-1,-1,-1):
    stage_nodes = list(set(stages[i]).union(set(to_add[i])) - set(to_remove[i]))
    stages[i] = stage_nodes
    nodes_downstream += len(stage_nodes)
    for n in stage_nodes:
      if (n in visited) and visited[n]==i:
        # Node visited as an ancestor and not moved, so no reason to look further at ancestors
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
  stages = [s for s in stages if len(s)]
  return stages

def compute_simulation_order(graph):
  global descendants_by_node
  descendants_by_node = {}
  g = graph
  sequential_order = list(nx.topological_sort(g))
  global ancestors_by_node
  ancestors_by_node,by_node_type_gen,node_gen = group_run_order(g)
  stages = assign_stages(sequential_order,node_gen,by_node_type_gen)
  n_stages = len(stages)
  new_n_stages = 0
  while new_n_stages<n_stages:
    n_stages = len(stages)
    stages = bring_forward(g,stages)
    stages = push_back_ss(g,stages)
    new_n_stages = len(stages)

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
        if initialise:
            self.initialise()

    def initialise(self):
        self.order = compute_simulation_order(self._graph)
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
        print('tags_by_process_and_model',tags_by_process_and_model)

        self.all_tags = set().union(*tags_by_process_and_model.values())
        print('all_tags',self.all_tags)

        self.distinct_values = {t:sorted(set([nodes[n][t] for n in nodes if t in nodes[n]])) for t in self.all_tags}
        print(self.distinct_values['constituent'])

        for pm in proc_models:
            node = nodes_by_process_and_model[pm][0]
            self.assign_run_indices(node[TAG_PROCESS],node[TAG_MODEL])

    def assign_run_indices(self,proc,model_type):
        '''
        Assign run indices to each model run within a given process, p (eg 'rainfall runoff')
        '''
        i = 0
        nodes = self._graph.nodes
        for gen in self.order:
            relevant_gen = [n for n in gen if nodes[n][TAG_MODEL]==model_type and nodes[n][TAG_PROCESS]==proc] ### BAD ASSUMPTION!
            relevant_gen = sort_nodes(relevant_gen)

            for n in relevant_gen:
                node = nodes[n]
                node[TAG_RUN_INDEX] = i
                i += 1

    def write_model(self,f,timesteps=DEFAULT_TIMESTEPS):
        close = False
        if hasattr(f,'upper'):
            import h5py
            h5f = h5py.File(f,'w')
            close = True
        else:
            h5f = f

        try:
            self._write_meta(h5f)
            self._write_dimensions(h5f)
            self._write_model_groups(h5f,timesteps)
            self._write_links(h5f)
        finally:
            if close: h5f.close()

    def _flux_number(self,node_name,flux_type,flux_name):
        node = self._graph.nodes[node_name]
        
        model_type = node[TAG_MODEL]
        desc = getattr(node_types,model_type).description
        flux_type = flux_type.capitalize()
        if not flux_type.endswith('s'):
            flux_type += 's'
        if not flux_name in desc[flux_type]:
            return -1
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
        assert len(dimsets)==1 # don't support one process having different dimensions
        # Should at least support attributes (tags that only ever have one value)
        dimensions = list(dimsets)[0]

        dim_values = {d:sorted({nodes[n][d] for n in node_set}) for d in dimensions}
        attributes = {d:vals[0] for d,vals in dim_values.items() if len(vals)==1}
        dimension_values = {d:vals for d,vals in dim_values.items() if len(vals)>1}
        dimensions = [d for d in dimensions if not d in attributes]

        # dims = tags_by_process[p]
        # dimensions = [distinct_values[d] for d in dims]
        shape = tuple([len(dimension_values[d]) for d in dimensions])

        model_instances = np.ones(shape=shape,dtype=np.uint32) * -1
        for node_name in node_set:
            node = nodes[node_name]

            loc = tuple([dimension_values[d].index(node[d]) for d in dimensions])
            if len(loc) < len(shape):
                print(loc,node)
            model_instances[loc] = node[TAG_RUN_INDEX]

        return dimension_values, attributes,model_instances

    def _write_model_groups(self,f,n_timesteps):
        models_grp = f.create_group('MODELS')
        nodes = self._graph.nodes

        models = {nodes[n][TAG_MODEL] for n in nodes}
        self.model_batches = {}
        for m in models:
            model_grp = models_grp.create_group(m)

            model_nodes = [n for n in nodes if nodes[n][TAG_MODEL]==m]
            processes_for_model = {nodes[n][TAG_PROCESS] for n in model_nodes}
            assert(len(processes_for_model)==1) # not necessary?

            dims,attributes,instances = self._map_process(model_nodes)
            ds = model_grp.create_dataset('map',dtype=np.uint32,data=instances,fillvalue=-1)

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
                self._parameteriser.parameterise(model_meta,model_grp,instances,dims,node_dict)

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
        col_order = ['%s_%s'%(n,c) for n in ['src','dest'] for c in ['generation','model','node','gen_node','var']]
        link_table = link_table[col_order]
        sort_order = ['src_generation','src_model','src_gen_node','dest_generation','dest_model','dest_gen_node']
        return link_table.sort_values(sort_order)

    def _write_links(self,f):
        table = np.array(self.link_table())
        f.create_dataset('LINKS',dtype=np.uint32,data=table)