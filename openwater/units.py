'''
Unit parsing, validation, and conversion for OpenWater model metadata.

Uses pint for dimensional analysis. Only lightweight Unit objects are used;
simulation data stays as plain numpy arrays with no Quantity wrapping.
'''

import pint

_ureg = None

def get_registry():
    '''Get the shared pint UnitRegistry (created once, cached).'''
    global _ureg
    if _ureg is None:
        _ureg = pint.UnitRegistry(cache_folder=':auto:')
    return _ureg

def parse_unit(unit_str):
    '''Parse a unit string into a pint Unit, or None if empty/unparseable.

    Returns None for empty strings or dimensionless markers that
    aren't real units (e.g. "dayOfYear").
    '''
    if not unit_str or not unit_str.strip():
        return None
    ureg = get_registry()
    try:
        return ureg.Unit(unit_str)
    except (pint.UndefinedUnitError, pint.errors.UndefinedUnitError):
        return None

def are_compatible(unit_str_a, unit_str_b):
    '''Check if two unit strings are dimensionally compatible.

    Returns True if both parse and are compatible (e.g. mm and m),
    or if either is empty/unparseable (treated as dimensionless/unspecified).
    Returns False only if both parse but are incompatible.
    '''
    ua = parse_unit(unit_str_a)
    ub = parse_unit(unit_str_b)
    if ua is None or ub is None:
        return True
    return ua.is_compatible_with(ub)

def are_equal(unit_str_a, unit_str_b):
    '''Check if two unit strings represent the same unit.

    Returns True if both parse to the same unit, or if both are empty.
    '''
    ua = parse_unit(unit_str_a)
    ub = parse_unit(unit_str_b)
    if ua is None and ub is None:
        return True
    if ua is None or ub is None:
        return False
    return ua == ub

def conversion_factor(from_unit_str, to_unit_str):
    '''Get the multiplicative conversion factor between two compatible units.

    Returns the factor such that value_in_to_units = value_in_from_units * factor.
    Raises pint.DimensionalityError if units are incompatible.
    Returns 1.0 if either unit string is empty/unparseable.
    '''
    ua = parse_unit(from_unit_str)
    ub = parse_unit(to_unit_str)
    if ua is None or ub is None:
        return 1.0
    ureg = get_registry()
    return ureg.convert(1.0, ua, ub)

def check_connection(output_unit_str, input_unit_str):
    '''Validate a model connection's unit compatibility.

    Returns:
        'exact_match' - units are identical
        'compatible' - units are dimensionally compatible (conversion possible)
        'incompatible' - units have different dimensions
        'unspecified' - one or both units are missing/unknown
    '''
    ua = parse_unit(output_unit_str)
    ub = parse_unit(input_unit_str)
    if ua is None or ub is None:
        return 'unspecified'
    if ua == ub:
        return 'exact_match'
    if ua.is_compatible_with(ub):
        return 'compatible'
    return 'incompatible'
