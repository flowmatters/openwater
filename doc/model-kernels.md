# Model Kernels

This notebook outlines some of the implementation concepts for the model kernels available in OpenWater.


## Native code

While Open Water is access through Python, the underlying model engine is written in Go and compiled to native code. Similarly the model kernels representing particular algorithms are written in a high level language and compile.

At this stage, the model kernels are written in Go, but it is anticipated that we will also support C and Fortran model kernels.



```python

```

## Data Parallel

Model kernels are expected to accept, as a single invocation, n, independent cells, user defined.

How that is implemented is model dependent, but allows for data-parallel solutions


```python

```

## Full or partial time periods

Models kernels are expected to accept, as a single invocation, m timesteps, user defined

That could be 1 time step, all the timesteps in the simulation or some portion. 

States must be externalised - initial states should be accepted by the kernel and final states returned.




```python

```

## Model descriptions

model kernels should be self describing in terms of inputs, outputs, parameters and states


```python

```


```python

```

## Code generation

YAML description

wrapper generated...

Currently suports...




```python

```


```python

```
