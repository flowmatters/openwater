# Installation / Environment

Using Openwater requires a number of software packages, with most of the dependencies being commonly installed, by default, in scientific Python distributions.

At a minimum, you will need:

* Python 3,
* Openwater Python package (this repository),
* Openwater Core binaries for your OS/architecture. These are downloaded and managed for you with the `ow-releases` command &mdash; see [Getting set up](installation.md) &mdash; or you can compile your own from the [openwater-core](https://github.com/flowmatters/openwater-core) repository,
* numpy and pandas
* HDF5 library and h5py Python package
* networkx Python package
* requests Python package (used to query releases on GitHub)

Installing the Openwater Core binaries, activating a particular version, and matching a version to a model file you've been given, are all described in [Getting set up](installation.md).

It is common for Openwater workflows to involve other packages, particularly including packages for processing spatial data. Indeed some of the example code in this repository relies on other packages, including

* graphviz - for visualisation model graphs
* veneer-py - for converting models from eWater Source to Openwater
* geopandas - for processing vector data
* rasterio - for gridded data
* taudem - for generating subcatchments and topologies from DEMs

