# Model parameterisation

Each individual node, within an Openwater model graph, can be configured independently, and generic tools are available to apply data sets, to nodes, using the [tags](dimensions.md) applied to the nodes.

## Values to configure

There are three key types of values required by model nodes:

1. Model parameters, which are often, but not necessariy scalars or tables of values,
2. Initial state values, which are also scalars or tables, and
3. Timeseries inputs, which are provided as arrays with one value per model timestep

Each component model type defines its own set of parameters, initial states and input timeseries, but the process of assigning values is consistent across model types.

### Parameters

Model parameters are fixed through a simulation, with the same values used for every model timestep.

Examples include constituent concentrations, runoff coefficients, soil store capacities and scaling factors.

Parameters are not modified by the model itself.

### Initial states

State variables are values that are computed by the model itself and change throughout a simulation. All component models in Openwater expose their state variables, providing a way for users to provide initial values. Furthermore, at the end of a simulation, it is possible to retrieve the final value of each state variable, which can be useful in certain simulation situations, such as hot starting a model with a known state.

### Input timeseries

Input timeseries can vary throughout the simulation and are provided, by the user, on a per-timestep basis.

Input timseries are also the data that are exchanged between model nodes along links. So, in many situations, a model node's input timeseries will *not* be configured directly by the user, but, rather, will be set during the simulation after predecessor nodes are executed.


## Identifying nodes for configuration

When configuring model nodes, with parameters, initial states and input time series, it is necessary to identify the particular model nodes that are to receive particular values.

The node tags are used for this process, and node tags can be either specified completely, to identify a single node, or incompletely, to identify a group of nodes with a matching set of tags.

## Applying parameters

There are four main ways to apply model parameters:

1. Assign the default parameter values for each parameter in a model, typically for *all* model nodes using the corresponding model type,
2. Specify a single value for particular model parameters and apply the value to matching model nodes,
3. Specify a table of values for a *single* model parameter, where the column headers for the table identify the value of one tag type and the index (row headers) of the table identify the value of a second tag type, and
4. Specify a table containing one or more model parameters, in labelled columns, where other labelled columns specify the values of tag types used to match model nodes.

### Default parameters


### Constant parameters


### Tables of single parameter

![Table of land use areas](figures/Parameterise-Table-2D.png)


![Table of land use areas](figures/Parameterise-Table-2D-Code.png)

### Tables of multiple parameters

![Table of land use areas](figures/Parameterise-Table-ND.png)


## Applying input time series

![Table of land use areas](figures/Parameterise-Timeseries.png)


![Table of land use areas](figures/Parameterise-Timeseries-Code.png)


## Custom parameterisation logic

While the above parameterisation functionality covers a wide range of model setup situations, it is possible to write bespoke model parameterisation in Python. See for example:

<todo>

