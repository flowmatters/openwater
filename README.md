# openwater

Open source hydrological modelling system.

This package is the main Python library. Relies on openwater-core.

## Installation / Upgrade

openwater (python) can be installed using `pip` from a terminal:

```
pip install https://github.com/flowmatters/openwater/archive/master.zip
```

At this stage we haven't tagged releases so you just install from the latest version.

To upgrade, uninstall the one you've got, then install again

```
pip uninstall -y openwater
pip install https://github.com/flowmatters/openwater/archive/master.zip
```

## Openwater Core

This package needs the Openwater Core binaries to run models. Install and activate the latest release with:

```
ow-releases install
eval $(ow-releases use-latest)
```

If you've been given a model file built with a particular version, install and activate that version with:

```
eval $(ow-releases use-for-model their-model.h5 --install)
```

See [doc/installation.md](doc/installation.md) for the full story, including version compatibility and using your own build.
