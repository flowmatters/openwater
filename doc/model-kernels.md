# Model Kernels

This page outlines some of the implementation concepts for component model kernels in Openwater.

## Native code

While Openwater is access through Python, the underlying model engine is written in Go and compiled to native code. Similarly the model kernels representing particular algorithms are written in a high level language and compiled.

At this stage, the model kernels are written in Go, but it is anticipated that we will also support C and Fortran model kernels.

## Scope and scale of model kernels

Model kernels implement an algorithm that can be applied at a single node of an Openwater model graph. The same algorithm may be used multiple, or many, times in a single graph, typically to represent the same process at a different location, and having input timeseries and parameters vary between those locations.

## Model kernel 'contract'

In order to work with Openwater, a component model must be implemented in a way that is consistent with the framework, using agreed approaches for accepting inputs and parameters and for returning model outputs to the rest of the framework.

There are two versions of this component model 'contract':

* The _generic contract_, which gives the software developer the most flexibility in terms of implementation approaches, including potentially implementing data-parallel model kernels and/or implementing in languages other than Go, and
* The _simplified contract_, in which the software developer implements a function, representing the execution of the algorithm at a single 'node', and accompanying structured documentation allowing an Openwater utility program to generate the more generic contract.

Presently, most of the component models, within the core Openwater library, are implemented using the simplified contract. 

The following sections describe the requirements of the contract, with notes identifying differences between generic contract and the simplified contract.

## Self describing

Each component model kernel requires structured documentation that lists the following:

* Model inputs, which are provided to the model, with as a timeseries (ie vary through time, with a value per model timestep)
* Model parameters, which are provided to the model as a single value for a simulation (ie static through time)
* Internal model states, which change through time and are maintained by the model
* Model outputs, which are reported by the model as timeseries

For each of these variables, (ie for each model parameter, model state, etc), the structured documentation must contain the name that will be used to identify the variable (eg rainfall). The structured documentation can optionally contain the units of the variable, as well as minimum, maximum and default values.

A component model kernel can describe a flexible number of variables, such as a changing number of parameters or states, based on the value of some other parameter. For example, a component model kernel might describe a parameter `n`, the value of which might determine the number of entries in a lookup table, represented as parameters `level` and `volume`, where each of `level` and `volume` would consequently be arrays of length `n`.

In the simplified contract, this documentation is provided as a comment, in the Go code, starting with the marker `/*OW-SPEC`, and describing a the inputs, states, parameters and outputers. See [example](https://github.com/flowmatters/openwater-core/blob/master/models/rr/simhyd.go#:~:text=/*OW-SPEC,*/). This description is used to generate code describing the model in a way that is queryable at runtime, including from calling Python code.

## Full or partial time periods

Component model kernels are expected to accept, as a single invocation, an arbitrary number of timesteps, ranging from a single time step, to all timesteps in the simulation or somewhere in between. 

This means that:

1. Component model kernels are responsible for their own, internal timestepping, informed by the timesteps in the input timeseries,
2. All model states must be externalised and desribed in the structured documentation, with the component model kernel accepting initial values for all states and subsequently returning final values of all states.

Furthermore, the Openwater system has no concept of global simulation 'time' (ie current day in the simulation), and all input timeseries are provided as plain arrays. Individual component model kernels may compute and track time based on model parameters, or, more typically, the model graph may include a single `DateGenerator` model node, which can be used to pass date components to invidual model components as input timeseries.

## Data Parallel

Component model kernels are expected to accept, as a single invocation an arbitrary number of independent graph nodes, using the same algorithm.

Thus, a single invocation of a model kernel may be for any of the following combinations of timesteps and graph nodes:

* A single timestep of a single graph node,
* Multiple timesteps of a single graph node,
* A single timestep of multiple graph nodes,
* Multiple timesteps of multiple graph nodes.

By requiring model kernels to support multiple, independent, graph nodes, in a single invocation, the framework is able to support component model kernels implemented in data parallel approaches.

This is one area where the _simplified contract_ varies considerably from the _generic contract_. In the simplified contract, the component model algorithm is implemented as a function, representing an arbitrary number of _timesteps_, but only a single _model graph node_. The code generation utility is then used to fulfill the generic contract, with generated code that accepts multiple graph nodes in a single invocation.

This leads to pairs of files, with one representing the model, implemented according to the simplified contract (see [example](https://github.com/flowmatters/openwater-core/blob/master/models/rr/simhyd.go)) and a second file, output from the code generation, representing the generic contract (see [matching example](https://github.com/flowmatters/openwater-core/blob/master/models/rr/generated_Simhyd.go)). Importantly, software developers should never directly edit these generated files, and any changes should be the result of modifying the documentation, contained in the `/*OW-SPEC` comment within the simplified contract file.

## Simplified contract implementation

Component model kernels following the _simplified contract_ require two key items:

1. A function (currently a Go function) implementing the algorithm a single model graph node, for an arbitrary number of timesteps, and
2. Structured documentation, in YAML format, describing the inputs, parameters, states and ouputs, along with information identifying the function to be called.

The YAML documentation should provide a model name (which does not need to relate to the function name). So, in the following example, the YAML is describing a model called `EmcDwc`.

The documentation then includes sections for `inputs`, `states`, `parameters` and `outputs`, each of which must be present even if empty. For example, the `states` section is empty in the following example. These four sections each list variables by name (such as `quickflow`, `EMC` in the example), and, potentially, characteristics of those variables, such as units, ranges (for parameters, described in brackets: `[0.1,1000.0]`) and a longer description (`Event Mean Concentration`).

The `implementation` section includes details of the function to be called, including the name of the function (`emcDWC` in the example), as well as tags for whether the function is implemented as a `scalar` function (representing a single graph node), the language, and whether the model outputs are passed as empty arrays into the function or whether they are returned from the function. At present, the following values are the only options supported:

* `type: scalar`
* `lang: go`
* `outputs: params`

The `init` section describes the default behaviour for initialising state variables where none are provided. The tag `zero: true` indicates that all state variables should be initialised to 0.0 (although there are no state variables in the example below). In cases where a model has a variable number of state variables, it is also possible to specify custom functions specifically for sizing and initialising the state variables (based on model parameters). This involves using the following items in the `init` section:

* `function: initGR4J`
* `type: scalar`
* `lang: go`

```yaml
/*OW-SPEC
EmcDwc:
    inputs:
		quickflow: m^3.s^-1
		baseflow: m^3.s^-1
    states:
    parameters:
		EMC: '[0.1,10000]mg.L^-1 Event Mean Concentration'
		DWC: '[0.1,10000]mg.L^-1 Dry Weather Concentration'
	outputs:
		quickLoad: kg.s^-1
		slowLoad: kg.s^-1
		totalLoad: kg.s^-1
	implementation:
		function: emcDWC
		type: scalar
		lang: go
		outputs: params
	init:
		zero: true
	tags:
		constituent generation
*/
```

The information in the model documentation determines the calling signature that Openwater will assume for the implementation function, with the assumption being that the function will accept parameters for inputs, states, parameters _and outputs_, in that order, with each group being ordered according to the order in the documentation.

So, in the above example, Openwater assumes that the `emcDWC` function has the following signature:

```go
func emcDWC(
    quickflow, slowflow data.ND1Float64, // Inputs
    // Initial States - none present, would be as float64
    emc, dwc float64, // Parameters
    quickLoad, slowLoad, totalLoad data.ND1Float64 // Outputs
    ) // Would return final states here, eg (state1, state2 float64)
    {
        // Implementation
```

Inputs are provided as 1D arrays of 64 bit floating point numbers, using an Openwater specific array type (`data.ND1Float64`). Outputs are similarly implemented as 1D arrays, and these are passed, as parameters, to the model function, to be populated within the implementation.

## Registering Component Model Kernels with the framework

Once implemented, a model needs to be registered with the Openwater library, in order to be available for use in model graph nodes. This is handled automatically when a component model kernel is implemented, using the _simplified contract_ and stored within one of the existing `model` subdirectories of the [`openwater-core`](https://github.com/flowmatters/openwater-core/tree/master/models) package.

When implementing a component model kernel in a directory outside of the existing `models` subdirectory, it is necessary to ensure that the relevant directory is `import`ed, somewhere in the initialisation process of the relevant openwater-core executable program (eg `ow-sim`). This can happen through adding an entry to the `import` statement in [`init.go`](https://github.com/flowmatters/openwater-core/blob/master/models/init.go).

When implementing the generic contract, it is necessary to import the `sim` package (`"github.com/flowmatters/openwater-core/sim"`) and then register a factory function for the new model type with the `sim.Catalog` instance:

```go
sim.Catalog["EmcDwc"] = buildEmcDwc
```

# Model meta types

Openwater has no concept of model 'meta types'.

Some other modelling frameworks have some concept of component model 'meta types', where a group of models, each representing the same basic proces, are grouped, within the framework, having some similarities. For example, different rainfall runoff algorithms might be grouped together in such a way as to guarantee certain properties are implemented consistently, such as all rainfall runoff models accepting a `rainfall` input and producing a `runoff` output, with corresponding specifications of each, such as units.

This concept is _not_ present in Openwater, where the model graph approach allows arbitrary connections between model nodes and between different variables of models. In practice, higher level modelling software, built on top of the core Openwater system, can implement such concepts as needed. This may be desirable in order to simplify model setup and reporting in those situations.

