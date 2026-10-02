'''
Version stamps and variable names in openwater HDF5 files.

Model files, and the results files ow-sim writes from them, lay their data out
by position: the second output of a model is whatever that model's second
output was in the openwater-core build that wrote the file. Model descriptions
change between builds, so two things are recorded to keep files readable:

* a **version stamp** at the root of the file, naming the openwater-core build
  the layout belongs to (``openwater_version``, ``signature_hash``) and who
  wrote the file and when (``created_by``, ``created_timestamp``). Results
  files also carry the stamp of the model file they were produced from
  (``model_openwater_version``, ``model_signature_hash``).
* the **variable names** along each dataset's variable axis, as a ``VARIABLES``
  attribute on a model's ``inputs``, ``outputs``, ``states`` and ``parameters``
  datasets. With these a file can be read by name, without the build that
  wrote it.

openwater-core writes the same attributes from Go (``io/hdf5_attrs.go``), in
the same HDF5 layout, so a file reads the same whichever side wrote it.
'''
import logging
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

VERSION_ATTR = 'openwater_version'
SIGNATURE_ATTR = 'signature_hash'
CREATED_BY_ATTR = 'created_by'
CREATED_TIMESTAMP_ATTR = 'created_timestamp'
MODEL_VERSION_ATTR = 'model_openwater_version'
MODEL_SIGNATURE_ATTR = 'model_signature_hash'

STAMP_ATTRS = (VERSION_ATTR, SIGNATURE_ATTR, CREATED_BY_ATTR, CREATED_TIMESTAMP_ATTR)
ALL_STAMP_ATTRS = STAMP_ATTRS + (MODEL_VERSION_ATTR, MODEL_SIGNATURE_ATTR)

VARIABLES_ATTR = 'VARIABLES'

# Datasets under /MODELS/<model> that have a variable axis, and which axis it is.
VARIABLE_AXIS = {
  'inputs': 1,      # (cell, input, timestep)
  'outputs': 1,     # (cell, output, timestep)
  'states': 1,      # (cell, state)
  'parameters': 0,  # (parameter, cell)
}


def _decode(value):
  if isinstance(value, bytes):
    return value.decode('utf-8')
  return value


def _stamp_value(value):
  '''A stamp attribute as a str, or None if it isn't a single string.

  Anything else -- a number, an empty attribute, a list -- is not a stamp, and
  is ignored rather than stringified into something that looks like one.
  openwater-core reads stamps by the same rule.
  '''
  if isinstance(value, np.ndarray):
    if value.size != 1:
      return None
    value = value.reshape(-1)[0]
  value = _decode(value)
  if not isinstance(value, str):
    return None
  return value.rstrip(' \x00')


def current_core():
  '''(version, signature hash) of the openwater-core build in use.

  This is the build whose model descriptions are loaded, and so the one a
  file's layout has to be compared with. Either value is 'unknown' when it
  can't be established.
  '''
  from . import discovery, lib
  version, signature = discovery.core_version()
  if signature == 'unknown':
    # No usable ow-inspect; the library, if there is one, is the next best.
    version, signature = lib.get_core_version(), lib.get_core_signature_hash()
  return version, signature


def read_stamp(h5f):
  '''The version stamp at the root of an open HDF5 file, as a dict of str.

  Attributes the file doesn't carry are left out, so an unstamped file gives
  an empty dict.
  '''
  result = {}
  for name in ALL_STAMP_ATTRS:
    if name not in h5f.attrs:
      continue
    value = _stamp_value(h5f.attrs[name])
    if value is not None:
      result[name] = value
  return result


def write_stamp(h5f, created_by):
  '''Stamp an open HDF5 file as written against the loaded openwater-core.

  The stamp says which build's model descriptions the file is laid out for.
  Failing to stamp is logged and otherwise ignored: the build may not be
  identifiable where the file is being written, and that shouldn't stop the
  write.
  '''
  try:
    version, signature = current_core()
    h5f.attrs[VERSION_ATTR] = version
    h5f.attrs[SIGNATURE_ATTR] = signature
    h5f.attrs[CREATED_TIMESTAMP_ATTR] = datetime.utcnow().isoformat()
    h5f.attrs[CREATED_BY_ATTR] = created_by
  except Exception as e:
    logger.warning(f"Could not write version metadata: {e}")


def copy_stamp(src, dest):
  '''Copy the version stamp from one open HDF5 file to another.

  For files derived from an existing one (clipped, split, substituted): the
  data keeps the layout of the file it came from, so it keeps that file's
  stamp rather than taking one from whichever library happens to be loaded.
  '''
  for name in ALL_STAMP_ATTRS:
    if name in src.attrs:
      dest.attrs[name] = src.attrs[name]


def set_variable_names(dataset, names, kind=None):
  '''Record the names along a dataset's variable axis.

  ``kind`` is one of 'inputs', 'outputs', 'states' or 'parameters' and
  defaults to the dataset's own name. Nothing is written when the number of
  names doesn't match the size of the variable axis: there would be no telling
  which name belonged to which position, and a wrong name is worse than none.

  Returns True if the names were written.
  '''
  if kind is None:
    kind = dataset.name.rsplit('/', 1)[-1]
  axis = VARIABLE_AXIS[kind]
  names = list(names)
  if dataset.shape[axis] != len(names):
    logger.debug('Not naming variables on %s: %d names for %d positions',
                 dataset.name, len(names), dataset.shape[axis])
    return False

  # Fixed-length byte strings, as for the DIMS attribute on a model's map.
  width = max([len(n.encode('utf-8')) for n in names] + [1])
  dataset.attrs[VARIABLES_ATTR] = np.array([n.encode('utf-8') for n in names], dtype='S%d'%width)
  return True


def variable_names(dataset, kind=None):
  '''The recorded names along a dataset's variable axis, or None.

  None means the file doesn't say -- it predates the attribute -- and the
  caller has to fall back on the loaded model description. Names that don't
  match the size of the variable axis are treated the same way.
  '''
  if VARIABLES_ATTR not in dataset.attrs:
    return None
  names = [_decode(n) for n in np.atleast_1d(dataset.attrs[VARIABLES_ATTR])]

  if kind is None:
    kind = dataset.name.rsplit('/', 1)[-1]
  axis = VARIABLE_AXIS.get(kind)
  if axis is not None and dataset.shape[axis] != len(names):
    logger.warning('%s names %d variables but has %d; ignoring the names',
                   dataset.name, len(names), dataset.shape[axis])
    return None
  return names


def parameter_row_names(desc, parameters):
  '''Names for each row of a model's parameters dataset.

  Most parameters occupy one row. A table parameter (one with dimensions)
  occupies a row per element, named ``name[0]``, ``name[1]``, ...

  Returns None when the rows can't be accounted for.
  '''
  from .array_params import get_parameter_locations
  try:
    locations = get_parameter_locations(desc, np.asarray(parameters))
  except Exception as e:
    logger.debug('Cannot locate parameters: %s', e)
    return None

  names = []
  for param, (start, end) in zip(desc['Parameters'], locations):
    if param.get('Dimensions'):
      names += ['%s[%d]'%(param['Name'], i) for i in range(end - start)]
    else:
      names.append(param['Name'])
  return names


def copy_model_metadata(src, dest):
  '''Carry the stamp and variable names from a model file to one derived from it.

  Copies the root version stamp, and the ``VARIABLES`` attribute of every
  dataset under ``/MODELS`` that exists in both files with the same number of
  variables. Datasets that came across with ``Group.copy`` already have their
  attributes; this covers the ones that were rebuilt from sliced data.
  '''
  copy_stamp(src, dest)

  if 'MODELS' not in src or 'MODELS' not in dest:
    return

  for model, dest_grp in dest['MODELS'].items():
    if model not in src['MODELS']:
      continue
    src_grp = src['MODELS'][model]
    for kind, axis in VARIABLE_AXIS.items():
      if kind not in dest_grp or kind not in src_grp:
        continue
      if VARIABLES_ATTR in dest_grp[kind].attrs:
        continue
      names = variable_names(src_grp[kind], kind)
      if names is not None:
        set_variable_names(dest_grp[kind], names, kind)
