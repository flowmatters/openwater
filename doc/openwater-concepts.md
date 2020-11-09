# Graph concepts

This document elaborates on the graph structure of Openwater models, including how the graph is translated to an execution plan, and how useful metadata is captured to help with model parameterisation and reporting.

## Openwater Graphs

Openwater models are represented as directed acyclic graphs.

Nodes represent model processes, and these are implemented by a particular algorithm, eg Simhyd for Rainfall Runoff.

Links/Edges transfer information between nodes without performing any additional process. (So, as an important difference between some node-link network based hydrological models, streamflow routing processes happen in **nodes** in Openwater)

### Node metadata

Nodes include information about the model type (algorithm) to be used, as well as user-defined metadata that can capture the structure of the model.

This metadata takes the form of tag names and associated values, where the tag names are used for all models of a particular model type (eg all nodes using Simhyd might have the `Catchment` and `Hydrological Response Unit` tags) and the values identify a particular node.


Collectively, the graph defines the model structure:

* The nodes and links define the depedencies in the simulation, from which a simulation plan can be derived,
* The metadata captures the structure of the model, as seen by the user, and supports model parameterisation and results retrieval

## Simulation Plans

The model graph can be used to identify a simulation _order_, whereby all individual components are placed in a sequence, whereby component simulations take place before the simulation of dependent components.

Typically, there are many simulation orders that satisfy the overall constraints, and are functionally equivalent.

However, when the simulation is undertaken concurrently, with nodes running in parallel, some simulation orders may be more efficient than others.

Open Water groups the nodes into 'generations', where all nodes in a generation can be run together. Importantly no two nodes in a singe generation can depend upon the outputs of the other, and no node in a generation can depend on the outputs of a node in a later generation.

When assigning nodes to generations, Open Water seeks to:

* Minimise the overall number of generations, and
* Keep the size of each generation as similar as possible in terms of the number of nodes.




```python

```

## Model Dimensions

The Openwater model is, ultimately, a single graph of components and linkages. While, conceptually, the model might be made up of response units, grouped into catchments, which exist in a topology, the graph represents the finest scale model components. That is, the graph doesn't retain any information about the user's conceptualisation of the model.

For this reason, we capture metadata in the graph that allows us to reconstruct the conceptual structure as required. The metadata is captured as tags and corresponding values, which become **dimensions**. Each type of model is represented as one or more dimensions.

For example, in the semi-lumped catchment recipe, each of the _n_ sub-catchments has m Hydrological Response Units, with using a rainfall runoff models - so rainfall runoff has two dimensions - Subcatchment and Hydrological Response Unit - and is represented as a two dimensional, n x m structure.

Similarly, each of the sub-catchments has p Constituents Generation Units, each of which generations each of the q Constituents using a constituent generation model, so constituent generation has the three dimensions - Subcatchment, Constituent Generation Unit and Constituent.


```python

```


```python

```

## Helpers, Parameterisers and Recipes


```python

```


```python

```
