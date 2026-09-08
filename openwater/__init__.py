from importlib.metadata import PackageNotFoundError, version as _distribution_version

try:
    __version__ = _distribution_version('openwater')
except PackageNotFoundError:
    # Running from a source checkout that hasn't been installed.
    # Keep in step with the version in pyproject.toml.
    __version__ = '0.1'

from .template import OWTemplate, OWNode, OWLink
from .persistence import (
    template_to_yaml,
    template_from_yaml,
    template_to_dict,
    dict_to_template,
    TemplateLoadError,
)
from .text_model import (
    ModelDefinition,
    ModelResults,
    ModelLoadError,
    load_model,
    load_initial_states,
    run_model,
)