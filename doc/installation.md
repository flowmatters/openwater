# Getting set up: installing and activating OpenWater Core

An OpenWater installation has two parts:

1. **The Python package** (`openwater`) — model building, parameterisation and reporting. Installed once, with `pip`.
2. **OpenWater Core** — the Go binaries (`ow-sim`, `ow-inspect`, `ow-single`, `ows-ensemble`) and the shared library `libopenwater`, which actually run the models. You can have **several versions installed side by side**, and you choose one per session.

The second part is the one that needs explaining. This document covers installing core releases, activating one, and — the common case when someone hands you a model file — working out which version that file needs.

**Just been handed a model file to run?** Install the Python package, then skip to [Running a model file from someone else](#running-a-model-file-from-someone-else); it installs the core version you need along the way.

Contents:

* [Install the Python package](#install-the-python-package)
* [Install a core release](#install-a-core-release)
* [Activate a release](#activate-a-release)
* [Running a model file from someone else](#running-a-model-file-from-someone-else)
* [Why versions matter](#why-versions-matter)
* [Using your own build](#using-your-own-build)
* [Troubleshooting](#troubleshooting)
* [Command and function reference](#command-and-function-reference)

## Install the Python package

```
pip install https://github.com/flowmatters/openwater/archive/master.zip
```

See [environment.md](environment.md) for the other Python packages you are likely to need (numpy, pandas, h5py, networkx, and so on).

This gives you the `openwater` package plus two command line programs: `ow-releases` (this document), and `ow-run`, for running [text-model YAML files](text-model-format.md) — not the HDF5 model files discussed here.

## Install a core release

Core releases are published on GitHub and downloaded for your platform (Linux, macOS or Windows) automatically.

```
# What's available?
ow-releases list

# Install the most recent release
ow-releases install

# ...or a specific one
ow-releases install --version 1.0.0+5a422e6.2b6d69b9

# What do I have locally?
ow-releases installed
```

Releases install to `~/.openwater/installations/<version>`, one directory per version. Nothing is overwritten when you install another version, and nothing is activated: installing and activating are separate steps.

The same thing from Python:

```python
from openwater import releases

releases.install_latest()
releases.install_version('1.0.0+5a422e6.2b6d69b9')
releases.list_installed()
```

## Activate a release

Activating means pointing OpenWater at one installed version. There are two ways to do it, and they cover different situations.

### From Python (for the current session)

```python
from openwater import releases

releases.use_latest()                          # newest installed version
releases.use('1.0.0+5a422e6.2b6d69b9')         # a specific version
releases.use('2b6d69b9')                       # any build with this signature
releases.use_latest(install=True)              # install the newest published release, then activate it
```

This sets the search path and runs `openwater.discovery.discover()`, which interrogates `ow-inspect` and populates `openwater.nodes`, `openwater.single` and friends with the model types that release provides. **Do this before you build or load a model**, because the available node types come from the activated release.

### From the shell (for the whole session, including the CLI tools)

The `ow-releases use*` commands don't change your environment directly — a child process can't — so they *print* the statements that do, for you to evaluate:

```bash
# Linux / macOS (bash, zsh)
eval $(ow-releases use-latest)
eval $(ow-releases use 1.0.0+5a422e6.2b6d69b9)
```

```powershell
# Windows PowerShell
ow-releases use-latest --shell powershell | Invoke-Expression
```

This sets `OW_BIN` and prepends the installation directory to `PATH`, so both the Python library and the command line tools (`ow-sim`, `ow-inspect`, ...) use that version. Python still needs `discovery.discover()` to read the model metadata, but it will find the right binaries without you naming a version:

```python
import openwater.discovery
openwater.discovery.discover()
```

Use the shell form when you want one version for a whole terminal session or notebook server; use the Python form when a script should pick its own version. If you run `ow-releases use-latest` without the `eval`, you'll just see the export statement printed, along with a reminder on stderr.

## Running a model file from someone else

A model file (HDF5) records the core version it was built with, so you don't have to ask. If the matching version isn't installed, `--install` fetches it:

```bash
eval $(ow-releases use-for-model their-model.h5 --install)
```

```
their-model.h5 was built with OpenWater Core 1.0.0+5a422e6.2b6d69b9
```

That activates the right core for the shell, so you can now run the model with `ow-sim`:

```bash
ow-sim their-model.h5 their-model-outputs.h5
```

See [programs.md](programs.md) for `ow-sim`'s options (`-overwrite`, `-final-states`, `-outputs-for`, and so on).

Or do the whole thing from Python:

```python
from openwater import releases
from openwater.template import ModelFile

releases.use_for_model('their-model.h5', install=True)

model = ModelFile('their-model.h5')
model.run(results_fn='their-model-outputs.h5')
```

`use_for_model` looks for the exact version the file was built with, then for any installed build with the same *model signature* (see below) — those are interchangeable for running the file. Without `install=True` it raises `ValueError` rather than downloading anything.

To look before you leap:

```python
>>> releases.model_file_version('their-model.h5')
ModelFileVersion(version='1.0.0+5a422e6.2b6d69b9', signature_hash='2b6d69b9',
                 created_by='openwater-py 0.1',
                 created_timestamp='2026-01-15T04:21:00.123456')

>>> releases.find_installed_for_model('their-model.h5')   # what could run it now
[]
```

Very old model files may not carry this metadata, in which case `model_file_version` reports `None` for both `version` and `signature_hash`, and `find_installed_for_model`, `use_for_model` and `install_for_model` raise `ValueError` rather than returning an empty result. You'll need to ask whoever sent you the file which version to use, then activate it by hand — everything else works as normal.

## Why versions matter

Core versions look like this:

```
MAJOR.MINOR.PATCH+BUILD_SHA[-BRANCH].SIGNATURE_HASH
1.0.0+5a422e6.2b6d69b9
```

The last component is a **signature hash** — a hash of every model interface (input, output, state and parameter names) in that build. Model files are laid out according to those interfaces, so:

* **Same signature hash ⇒ the model file will run.** Bug fixes, performance work and refactoring inside the core all keep the signature, so `1.0.1+d9e1f2.2b6d69b9` runs anything `1.0.0+5a422e6.2b6d69b9` built.
* **Different signature hash ⇒ the file may not run correctly**, because some model's inputs, outputs, states or parameters have changed.

This is why version matching is by signature rather than by exact build, and why the tools will happily activate a different build that shares one.

If you run a model with a mismatched core, `ModelFile.run(...)` logs a warning telling you which version the file wants:

```
Model file signature mismatch!
  File version: 1.0.0+5a422e6.2b6d69b9
  Current version: 1.1.0+b7c3e11.9d4f2c1a
  This may cause errors. Activate the matching release with
  openwater.releases.use_for_model('their-model.h5', install=True), rebuild the
  model file with the current version, or use skip_version_check=True to override.
```

It is a warning, not an error — the run proceeds, and `skip_version_check=True` silences it — but treat results from a mismatched run with suspicion.

The versioning scheme is described in full in [VERSIONING.md](https://github.com/flowmatters/openwater-core/blob/master/VERSIONING.md) in the openwater-core repository.

## Using your own build

If you build openwater-core yourself, point OpenWater at the build directory instead of an installed release:

```bash
eval $(ow-releases use-custom /path/to/openwater-core/bin)
```

```python
import openwater.discovery
openwater.discovery.set_exe_path('/path/to/openwater-core/bin')
openwater.discovery.discover()
```

Either way you can also just set `OW_BIN` yourself before starting Python; it defaults to `~/bin` when unset. Note that a local build's version, and therefore its signature, comes from its own git state, so `ow-releases` won't recognise it as an installed release and `use_for_model` won't find it.

## Troubleshooting

**`NoActiveReleaseError: No OpenWater release active; cannot access 'openwater.nodes.<X>'`**

Nothing has been activated yet in this Python session. Call `releases.use_latest()`, `releases.use_for_model(...)`, or `openwater.discovery.discover()` if `OW_BIN` is already set.

**`FileNotFoundError` for `ow-inspect` or `ow-sim`**

`OW_BIN` points somewhere without the binaries — often because `ow-releases use...` was run without `eval`. Check with `ow-releases installed` and `echo $OW_BIN`.

**`ValueError: ... is not installed in ~/.openwater/installations`**

The version the model file wants isn't installed locally. Add `--install` (CLI) or `install=True` (Python).

**A model file has no version metadata**

It predates version stamping. Activate a release explicitly and check the results carefully.

**`AssertionError` from `model.run(...)`**

`ow-sim` exited non-zero; the reason is in the log output above the traceback. A common cause is an output file that already exists — pass `overwrite=True` to `model.run(...)`, or `-overwrite` to `ow-sim`.

**Disk usage**

Each release is a full set of binaries. `ow-releases installed` lists what you have, with paths; delete a directory under `~/.openwater/installations` to remove that version.

## Command and function reference

### `ow-releases`

| Command | Purpose |
|---|---|
| `list` | Available releases on GitHub |
| `latest` | Details of the most recent release |
| `install [--version V] [--force]` | Download and install a release |
| `installed` | Locally installed releases |
| `use V` | Print activation for version `V` |
| `use-latest [--install]` | Print activation for the newest installed release |
| `use-for-model FILE [--install]` | Print activation for the version `FILE` was built with |
| `use-custom PATH` | Print activation for a directory of binaries |

`install` also takes `--dest` &mdash; there it names the installation directory itself, rather than the directory installations are placed under, so a release installed that way won't be found by `installed` or the `use*` commands.

All four `use*` commands take `--shell posix|powershell|cmd` (auto-detected); `use`, `use-latest`, `use-for-model` and `installed` also take `--dest`, for a non-default installations directory. Wrap the `use*` commands in `eval $(...)` (or `| Invoke-Expression`) to take effect.

### `openwater.releases`

| Function | Purpose |
|---|---|
| `get_releases()`, `latest_release()`, `list_available_versions()` | Query releases on GitHub |
| `install_latest()`, `install_version(v)` | Download and install |
| `list_installed()`, `find_installed(v)` | Query local installations |
| `use(v)`, `use_latest(install=False)` | Activate an installed release |
| `model_file_version(fn)` | The version and signature recorded in a model file |
| `find_installed_for_model(fn)` | Local installations that can run a model file, best match first |
| `use_for_model(fn, install=False)` | Activate the version a model file needs |
| `install_for_model(fn)` | Install the version a model file needs, without activating |
| `find_release_for_signature(h)` | The newest remote release with a given signature hash |

`install_latest` and `install_version` take `dest` as the installation directory itself; every other `dest` above (and every `--dest` on `installed` and the `use*` commands) is the *base* directory that per-version installations live under.

`find_installed`, `use`, `use_for_model`, `install_for_model` and the CLI's `use` all accept a full version, a partial version (`1.0.0`, `1.0.0+5a422e6`) or a bare signature hash (`2b6d69b9`); where several installations match, the most recently published wins.
