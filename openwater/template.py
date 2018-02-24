import networkx as nx


class OWTemplate(object):
  def __init__(self):
    self.nodes = []
    self.links = []
  
  def add_node(self,model_type=None,name=None,**tags):
    # if hasattr(node_or_name,'model_type'):
    #   self.nodes.append(node_or_name)
    # else:
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
    if name:
      self.name = name
    else:
      self.name = self.make_name()

  def make_name(self):
    std_names = ['catchment','model','process','constituent','hru','lu']
    for k in sorted(self.tags.keys()):
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

#    print(current)
    descendants = descendants_cached(g,n)
    for d in descendants:
        descendent_stage = node_stages[d]
        if descendent_stage == 0:
            print(n,d,n_stages,current,descendent_stage,lowest)
#        print(d,descendent_stage)
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
    print(i)
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
  stages = bring_forward(g,stages)
  stages = push_back_ss(g,stages)
  return stages