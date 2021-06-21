# Installation / Environment

Using Openwater requires a number of software packages, with most of the dependencies being commonly installed, by default, in scientific Python distributions.

At a minimum, you will need:

* Python 3,
* Openwater Python package (this repository),
* Openwater binaries for your OS/architecture. You can compile your own from the [openwater-core](https://github.com/flowmatters/openwater-core) repository,
* numpy and pandas
* HDF5 library and h5py Python package
* networkx Python package

It is common for Openwater workflows to involve other packages, particularly including packages for processing spatial data. Indeed some of the example code in this repository relies on other packages, including

* graphviz - for visualisation model graphs
* veneer-py - for converting models from eWater Source to Openwater
* geopandas - for processing vector data
* rasterio - for gridded data
* taudem - for generating subcatchments and topologies from DEMs

