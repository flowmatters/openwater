# Implementing model graphs

`ow-sim` is the main simulation program in Openwater and is used to run graphs of model nodes, in a pre-determined order, transferring relevant outputs of nodes to inputs of other nodes, as required.

The graphs are directed acyclic graphs, created using the openwater Python library and stored in a HDF5 file. In addition to being used by `ow-sim`, the HDF5-stored graphs are used in querying and reporting on the model.

# Components of the Graph

The model graph is made up of
* **Nodes (vertices):** which are attributed with a component model type, which defined the algorithm used at runtime, and
* **Links (edges):** which define connections between nodes.

Each node has a set of named inputs and named outputs, which are determined by the component model type. Each link connects a single, named, output of one node to a single, named, input of another node.

The graph defines a topology, which can be used to define a sequence for model execution. In order to support parallel computation, Openwater groups nodes into generations, where each node in a generation can run in parallel.

Nodes are further attributed with multiple, user defined tags, which support model setup and post processing. See [dimensions](dimensions.md) for  more details.

# Creating the graph

Model graphs are built using the `OWTemplate` class. Once configured with individual nodes and links, and optional nested templates, an `OWTemplate` can be converted to an instance `ModelGraph`, which can be populated with parameters, initial states and input timeseries and subsequently saved to disk. A saved model can be subsequently reloaded as a `ModelFile`, supporting modification of inputs, states and parameters, but not structural change.

![Building an openwater model](ModelLifecycle.png)

```python
from openwater.template import OWTemplate

template = OWTemplate()
```

Graphs can be built up by adding nodes (`add_node`) and links (`add_link`), or by nesting other graphs, with custom tags.

## Nodes

A node is specified with a component model type (eg `EmcDwc`) and a set of user defined tags:

```python
runoff = template.add_node('Sacamento')
depth_to_rate = template.add_node('DepthToRate',component='Quickflow')
emc_dwc = template.add_node('EmcDwc',constituent='Sediment')
```

## Links

The `add_link` method takes an instance of `OWLink`, which is constructed from node references and the names of input and output fluxes:

```pythonfrom openwater.template import OWLink

quickflow_to_vol = OWLink(runoff,'surfaceRunoff',depth_to_rate,'input')
template.add_link(runoff_to_vol)

quickflow_to_sediment = OWLink(depth_to_rate,'output',emc_dwc,'quickflow')
template.add_link(quickflow_to_sediment)
```

## Templates

The `OWTemplate` class is so-named because it is typically used to build up a graph using instances of nested templates.

```python
catchment = OWTemplate()

for hru in HYDROLOGIC_RESPONSE_UNITS:
  hru_template = make_hru_template(hru)
  catchment.nest(hru_template)
```

![Nesting templates](NestedTemplates.png)


