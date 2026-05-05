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
  -no-outputs
        do not write model outputs for any models
  -no-outputs-for string
        do not write model outputs for specified models. Specify as comma separated list of model names
  -outputs string
        split output files by model type. Specify as <model>:<fn>,<model>:<fn>,...
  -outputs-for string
        only write model outputs for specified models. Specify as comma separated list of model names
  -only-outputs-for string
        only write model outputs for the specified models (strict whitelist). Specify as comma separated list of model names
```

* Likewise, controlling the writing of 'final inputs' for models:

```
  -inputs-for string
        only write final model inputs for specified models. Specify as comma separated list of model names
  -no-inputs
        do not write final model inputs for any models
  -no-inputs-for string
        do not write final model inputs for specified models. Specify as comma separated list of model names
  -only-inputs-for string
        only write final model inputs for the specified models (strict whitelist). Specify as comma separated list of model names
```

* Performance and output tuning:

```
  -compress-outputs int
        compress output datasets with deflate (gzip) at the given level (1=fastest, 9=best, 0=off). Reduces output file size at some CPU cost
  -link-workers int
        number of worker goroutines for parallel link processing; 0 = auto (min(NumCPU, 8))
  -max-write-ahead int
        max generations the simulation can run ahead of the output writer. Lower values reduce peak memory; higher values allow more overlap. 0 = unlimited (default 4)
```

* Miscellaneous options:

```
  -cpuprofile string
        write cpu profile to file
  -overwrite
        overwrite existing output files
  -q, -quiet
        suppress non-error log output
  -v, -verbose
        show progress of simulation generations
  -version
        display version information
  -writer
        operate as an output writer for another simulation process
```

### Calling ow-sim from Python

The Python library invokes `ow-sim` for you when you call `ModelFile.run(...)`, `openwater.split.run_split_model(...)`, or related entry points. Any keyword argument you don't recognise as a Python-side parameter is forwarded through to `ow-sim` as a flag by `ow_sim_flag_text` (in `openwater/template.py`), with this translation:

* Underscores in the kwarg name become dashes, and a leading `-` is added: `final_states` → `-final-states`, `no_outputs_for` → `-no-outputs-for`.
* Boolean `True` emits the bare flag (e.g. `overwrite=True` → `-overwrite`); boolean `False` omits it entirely.
* Any other value is passed as the flag's argument (e.g. `compress_outputs=6` → `-compress-outputs 6`).
* Positional arguments are filled in by the Python wrapper: the model file is the first positional argument, and the results file (`results_fn`, or the per-window `dests` entry for split runs) is the second.

So the ow-sim invocation

```
ow-sim -overwrite -final-states states.h5 -compress-outputs 6 -outputs-for Sacramento,StorageRouting model.h5 results.h5
```

is equivalent to

```python
model = ModelFile('model.h5')
model.run(
    results_fn='results.h5',
    overwrite=True,
    final_states='states.h5',
    compress_outputs=6,
    outputs_for='Sacramento,StorageRouting',
)
```

The same kwargs pass through `run_split_model(..., **kwargs)`, so flags such as `overwrite`, `verbose`, `compress_outputs`, and the various `*_for` selectors apply to each window's `ow-sim` call.

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





