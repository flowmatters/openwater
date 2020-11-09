# Reporting model results

When Openwater models are run, a results file is produced. Presently, the results file contains timeseries outputs for each model node within the model graph. 

The results file also contains _input_ timeseries for model nodes that receive inputs from predecessor model nodes. These timeseries are recorded as results as they may be the sum of outputs from multiple predecessor nodes and hence they may not be easily reconstructed.

While the results files, and, indeed the model input files, are HDF5 files, it is recommended to use the Openwater Python package routines for retrieving results. Presently, there are two main ways to retrieve results:

1. Retrieve timeseries from the model, either individually or aggregated across a group of nodes sharing _one_ common tag, and
2. Retrieving tables of results, aggregated from timeseries to a scalar, summarised across a group of nodes sharing _two_ common tags.

In both cases, the functionality is based on using the model node [tags](dimensions.md) to identify individual model nodes or groups of model nodes with common tags.

The following examples illustrate the two main options

## Retrieving time series

![Retrieving time series results](figures/Results-Timeseries.png)

## Retrieving tables


![Retrieving tables of results](figures/Results-Table.png)


## Higher level

Model / parameter


Units


