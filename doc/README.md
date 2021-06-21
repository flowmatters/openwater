# Openwater: Introduction and Concepts

Openwater is a catchment modelling system built with the following objectives:

1. Flexibility:
  * **Modular** - Component algorithms with defined 
  * **Algorithmic** - Ability to add new model algorithms
  * **Model Structure and Methodology** - Not tied to any particular spatial structure.
  * **High level, scripted interface** - Models constructed in Python scripts. Layers of modules and recipes for capturing common approaches
2. Performance:
  * **Optimise workloads** - Schedules models based on dependencies and workload
  * **Group like models** - To allow for data-parallel approaches
  * **High level languages** - Model kernels implemented in native languages
3. Free and Open Source:
  * **Free as in speech** - License allows you to use, share and modify the software
  * **Permissive License** - Means you can create a proprietary derivative if you wish
  * **Community** - Hopefully people see value in working together to advance the common good

## Using Openwater

Openwater is developed as software libraries, intended to be accessed from a scripting environment or incorporated into other programs.

The underlying model algortihms are implemented in Golang, with higher level model interaction, such as setup, parameterisation and reporting, supported through a Python library. This Python library is intended to be used from interactive scripting environments, such as Jupyter notebooks, or to serve the basis for developing higher level, application specific programs.

**TODO: Figure demonstrating layered approach.

Software requirements are described [here](environment.md)

## Openwater concepts

Openwater allows users to construct models of a particular system (eg a model of a particular river catchment), comprising a _graph_ of component models, each typically representing some process, occurring at some location, within the overall system. For example, a component model might represent the _rainfall runoff process_, within a single subcatchment of a broader catchment. The graph provides the overall structure, linking outputs of individual component models to inputs of other component models.

As these graphs can grow to thousands of individual nodes, Openwater provides, and encourages, a number of mechanisms for organising these graphs, including templates, for repeatedly defining similar graphs, such as using the same graph for each elementary spatial unit, and dimensions, as a way of attributing and identifying individual nodes within the graph.

These concepts are elaborated in [openwater-concepts](openwater-concepts.md), and demonstrated in the  following examples.

###  Example 1 - Semi Lumped Catchment Model

Model characteristics

* Catchment structure derived from DEM
* Catchment response units (within subcatchments) defined by land use
* Gridded climate data used to derive catchment average precip and PET

In this example, we'll use an Openwater model _'recipe'_ - a predefined method for generating a model structure from input data. As we'll see, we provide a few key details (data paths, model parameters) and then the model setup and simulation process proceeds automatically. Later exmamples will dive into this logic to see how customised approaches can be used.

**_See [Example-1-1-SimpleCatchmentModel.ipynb](Example-1-1-SimpleCatchmentModel.ipynb)_**


### Example 2 - Custom Model Template

The first example demonstrates how a model recipe can be used to setup a model based on a predefined model structure.

Here, we will look at the building blocks of the previous recipe, to see how model structures are defined, and how new model structures can be created.

As we'll see later, Openwater models are collections of individual model components, connected together in a graph showing how the outputs of one model become inputs of other models. Ultimately is just this _one, single graph_, that shows how everything in the overall model fits togehter.

However, it is often useful to conceptualise the model as two (or potentially more) groups of nested graphs. For example:

1. **A subcatchment level, _template_ graph**, that sets out how different components fit together within a single sub-catchment, and
2. **A basin graph**, that connects different subcatchments together.

Or, perhaps:

1. **A response unit template graph**, that shows how a single land type is modelled,
2. **A sub-catchment template**, that connects together the different land types represented in a subcatchment,
3. **A basin graph**, connecting subcatchments.

The semi-lumped catchment example, above, is built using a variation of this 3-tiered approach. In the example, the subcatchment template is made up of overlapping Hydrological Response Units (for runoff) and Constituent Generation Units (for water quality), with 1 HRU driving 0 to many CGUs. The various CGUs are summed together to form the output of subcatchment. Subcatchments are defined using the DEM and the TauDEM package, with the resulting catchment topology used to link the subcatchment templates together.

**_See [Example-1-2-CustomTemplate.ipynb](Example-1-2-CustomTemplate.ipynb)_**

## Developing component models

Individual component models implement discrete algorithms, such as rainfall runoff, or river transport processes.

These component models are implemented in Golang, although they could be implemented in other natively compiled languages, such as Fortran or C.

Information regarding component model development and interacting with the Openwater framework is described [elsewhere](model-kernels.md).
