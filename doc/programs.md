# Standalone Openwater programs

Openwater is intended to be used from Python, either in an interactive environment, such as Jupyter notebooks, or from other scripts and higher level programs.

However the core Openwater models, written in Go, are available from a number of standalone, executable programs, available on each supported  platform.

## ow-sim

`ow-sim` is the core Openwater simulation engine, built for executing model graphs, described in HDF5 files, and, typically, writing outputs to HDF5 files.

`ow-sim` is called by the Openwater Python libraries when executing a model graph, but can also be called independently. This might be useful when, for example, the model graph is created on one system and then executed on another. Alternatively, it may be useful to run `ow-sim` independently when there are many combinations of model inputs and model parameter sets to run through a single model graph.

The simplest use of `ow-sim` is to specify a single input file (HDF5) describing the model graph, including graph topology, model inputs and parameters, and a single output file (HDF5) for storing outputs:

```
ow-sim my-model.h5 my-model-outputs.h5
```

`ow-sim` will run the model graph, in the order described in the input HDF5 file, and write all outputs to disk in the single HDF5 output file.

`ow-sim` supports a number of additional options, including 

* Reading different aspects of the model graph from different files:

```
  -initial-states string
        specify file for initial states
  -input-timeseries string
        specify file for input timeseries
  -parameters string
        specify file for model parameters
```

* Writing final states to a different file to output timeseries:

```
  -final-states string
        specify file for final states
```

* Controlling the writing of outputs, by specifying which component model kernels to write output timeseries for, specifying which component model kernels to **not** write output timeseries for, and using different files for different component model kernels:

```
  -no-outputs-for string
        do not write model outputs for specified models. Specify as command separated list of model names
  -outputs string
        split output files by model type. Specify as <model>:<fn>,<model>:<fn>,...
  -outputs-for string
        only write model outputs for specified models. Specify as command separated list of model names
```

* Likewise, controlling the writing of 'final inputs' for models:

```
  -inputs-for string
        only write final model inputs for specified models. Specify as command separated list of model names
  -no-inputs-for string
        do not write final model inputs for specified models. Specify as command separated list of model names
```

* Miscellaneous options:

```
  -cpuprofile string
        write cpu profile to file
  -overwrite
        overwrite existing output files
  -run-only string
        specify a subset of models to run. Note: Will still process links and write inputs for models that receive links (subject to input writing options).
  -v    show progress of simulation generations (shorthand)
  -verbose
        show progress of simulation generations
  -writer
        operate as an output writer for another simulation process
```

**Note:** Final inputs are the inputs received by model graph nodes that have come, entirely, or partially, from other model graph nodes. By default, these are treated the same as model outputs and written to disk. Final inputs are availble from the [reporting](reporting.md) functions as if they were outputs. This is to aid in reporting model function in situations where multiple model graph node outputs link to a single input of another model graph node. For example, in cases where two or more upstream 'reaches' flow into a confluence. For efficiency, `ow-sim` does not write final inputs of component model kernels that do not, in the current model graph, receive any inputs from other models. So, for example, if a rainfall runoff model receives rainfall and potential evapotranspiration as input timeseries, and, in the current model graph, these inputs are always provided by outside data, then `ow-sim` will not write these inputs out to disk as 'final inputs'.

## ow-single

`ow-single` runs a single component model kernel, for a single model graph node. This can be useful for testing component model kernels, or in situations where a graph is not required.

`ow-single` is used by the `openwater.single` module in Python, which provides Python function wrappers for calling model kernels directly.

`ow-single` can also be used as a standalone program, where it will read the model configuration on standard input in JSON format, with the following structure:

```
{
    "Name":"model name, eg EmcDwc",
    "Inputs":[
        {
            "Name":"quickflow",
            "Values":[
                // Array of floats
            ]
        },
        // ...
    ],
    "Parameters":[
        {
            "Name":"EMC",
            "Value": 100.0
        },
        // ...
    ]
}
```

## ow-inspect

`ow-inspect` reports all model metadata, relating to known component model kernels, in JSON format.

`ow-inspect` is used to initialise the Openwater Python library (through the `openwater.discovery` module), and can also be used to provide Openwater metadata to other programs.

`ow-inspect` returns the component model metadata in JSON format.





