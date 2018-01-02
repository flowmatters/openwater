import graphviz
from graphviz import Digraph
def graph_template(template,prefix='',dot=None):
  if not dot:
    dot = Digraph()

  for n in template.nodes:
    dot.node(prefix+n.name, label='%s\n(%s)'%(prefix+n.name,n.model_type))

  for l in template.links:
    dot.edge(prefix+l.from_node.name,prefix+l.to_node.name,label='%s --> %s'%(l.from_output,l.to_input))

  return dot
