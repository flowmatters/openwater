# OpenWater Python Library Copilot Instructions

## Project Overview
The OpenWater Python library provides a high-level, user-friendly API for hydrological modeling. It wraps the openwater-core Go library and provides utilities for model setup, execution, and analysis.

## Technology Stack
- **Language**: Python 3.x
- **Build System**: setuptools with pyproject.toml
- **Key Dependencies**: Listed in requirements.txt
- **Native Integration**: Calls openwater-core via shared library (libopenwater.so)

## Project Structure
- `openwater/`: Main package directory
  - `lib.py`: Interface to native openwater-core library
  - `nodes.py`: Node/model definitions
  - `catchments.py`: Catchment modeling utilities
  - `ensemble.py`: Ensemble simulation support
  - `results.py`: Results handling and analysis
  - `config.py`: Configuration management
  - `examples/`: Example scripts and models
- `doc/`: Documentation
- `setup.py`, `pyproject.toml`: Package configuration

## Code Style
- Follow PEP 8 conventions
- Use type hints where appropriate
- Document functions with docstrings (numpy/scipy style)
- Keep functions focused and composable
- Use meaningful variable names

## Testing
- Write tests using pytest (when present)
- Test both success and error cases
- Mock native library calls when appropriate
- Include integration tests with real models

## Installation
- Users install via pip from GitHub
- Depends on openwater-core being built and available
- Shared library must be findable by Python ctypes

## Initialisation

- Library initialised at runtime for a particulare version of openwater-core
- Model versions managed by library in user's home directory
- Metadata from openwater-core version (ow-inspect) used to determine available model types, signatures and hence compatibility with model files

## Key Patterns
- High-level abstractions over core library
- Pythonic API design (use iterators, context managers, etc.)
- NumPy integration for array data
- Pandas integration for time series (where used)

## Integration with openwater-core
- Loads libopenwater.so via ctypes/cffi
- Must maintain compatibility with C API
- Handle marshaling between Python and C types
- Manage memory carefully across language boundary

## Development Workflow
- openwater-core must be built first
- Changes to core API may require updates here
- Test with actual models to verify integration
- Update examples when adding new features

## Documentation
- Keep README.md up to date
- Document public APIs thoroughly
- Include examples in docstrings
- Update doc/ when adding major features
