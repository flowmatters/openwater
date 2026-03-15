"""
Visual style configuration for OpenWater model categories.

Colors and symbols used for graph visualisation, consistent with
the Periodic Table of OpenWater Models.

Model groups are discovered at runtime from ow-inspect metadata
(the 'Group' field corresponds to the Go package name).
"""

# Model group colours (hex) – keyed by Go package name.
GROUP_COLORS = {
    "rr":          "#3B82F6",
    "routing":     "#14B8A6",
    "storage":     "#8B5CF6",
    "generation":  "#F97316",
    "conversion":  "#F59E0B",
    "functions":   "#22C55E",
    "climate":     "#22C55E",
}

# Default colour for models whose group is unknown or unmapped.
_DEFAULT_COLOR = "#6B7280"


def color_for_model(model_name):
    """Return the hex colour string for a given model name."""
    group = group_for_model(model_name)
    if group is None:
        return _DEFAULT_COLOR
    return GROUP_COLORS.get(group, _DEFAULT_COLOR)


def group_for_model(model_name):
    """Return the group (Go package name) for a model, or None if unknown."""
    from . import nodes
    desc = nodes.MODELS.get(model_name)
    if desc is not None:
        return desc.get('Group', None)
    return None
