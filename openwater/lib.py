import sys
import ctypes
import numpy as np
_the_library = None

def get_core_version():
    """Get the version string from the loaded openwater-core library."""
    global _the_library
    if not _the_library:
        _ensure_library_loaded()
    if _the_library:
        try:
            _the_library.ow_version.restype = ctypes.c_char_p
            return _the_library.ow_version().decode('utf-8')
        except:
            return "unknown"
    return "unknown"

def get_core_signature_hash():
    """Get the model signature hash from the loaded openwater-core library."""
    global _the_library
    if not _the_library:
        _ensure_library_loaded()
    if _the_library:
        try:
            _the_library.ow_signature_hash.restype = ctypes.c_char_p
            return _the_library.ow_signature_hash().decode('utf-8')
        except:
            return "unknown"
    return "unknown"

def extract_signature_hash(version_string):
    """Extract signature hash from a version string.

    Version format: X.Y.Z+BUILD[-BRANCH].SIGHASH
    """
    if not version_string or version_string == "unknown":
        return "unknown"
    parts = version_string.split('.')
    if len(parts) > 0:
        return parts[-1]
    return "unknown"

def is_compatible(file_signature_hash):
    """Check if a model file signature hash is compatible with loaded core."""
    current_hash = get_core_signature_hash()
    return current_hash != "unknown" and file_signature_hash == current_hash

def _ensure_library_loaded():
    """Load the library if not already loaded."""
    from .discovery import _lib_path
    global _the_library
    if not _the_library:
        try:
            _the_library = ctypes.CDLL(_lib_path())
        except:
            pass

def _conv(arr):
    return [arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),*[ctypes.c_int(i) for i in arr.shape]]

def _create_model_func(model_name,description):
  from .discovery import _lib_path, _collect_arguments, _make_model_doc

  global _the_library
  if not _the_library:
    _ensure_library_loaded()

  thismodule = sys.modules[__name__]

  def model_func(*args,**kwargs):
    cpu_profile = kwargs.get('cpu_profile', '')
    kwargs.pop('cpu_profile', None)
    inputs, params, states, _ = _collect_arguments(description,args,kwargs)
    if len(inputs):
      len_first_provided = [len(i) for i in inputs if len(i)][0]
      inputs = [i.reshape(1,len(i)) if len(i)==len_first_provided else np.zeros((1,len_first_provided),'d') for i in inputs]
    inputs = np.stack(inputs,axis=1)

    params = np.array(params)
    if len(params.shape)==1:
      params = params.reshape(params.shape[0],1)

    outputs = np.zeros((inputs.shape[0],
                        len(description['Outputs']),
                        inputs.shape[2]),
                       dtype='d')
    states = np.array(states).transpose()

    init_states = False
    if states.shape[0] != inputs.shape[0]:
      init_states = True
      states = np.zeros((inputs.shape[0],
                        len(description['States'])))

    call = [ctypes.create_string_buffer(bytes(model_name,'ascii')),
            *_conv(inputs),
            *_conv(params),
            *_conv(states),
            *_conv(outputs),
            ctypes.c_bool(init_states),
            ctypes.create_string_buffer(bytes(cpu_profile, 'ascii'))]

    _the_library.RunSingleModel(*call)
    return [outputs[:,i,:] for i in range(len(description['Outputs']))] + \
           [states[:,i] for i in range(len(description['States']))]

  model_func.__name__ = model_name
  model_func.__description__ = description
  def __filter_arguments__(**kwargs):
    pnames = [p['Name'] for p in description['Parameters']]
    return {k:v for k,v in kwargs.items() if (k in description['Inputs']) or (k in pnames)}

  def with_states(*args,**kwargs):
    cpu_profile = kwargs.get('cpu_profile', '')
    kwargs.pop('cpu_profile', None)
    inputs, params, initial_states, _ = _collect_arguments(description,args,kwargs)
    if len(inputs) and len(inputs[0].shape)==1:
      inputs = [i.reshape(1,len(i)) for i in inputs]
    inputs = np.stack(inputs,axis=1)

    params = np.array(params)
    if len(params.shape)==1:
      params = params.reshape(params.shape[0],1)

    n_cells = inputs.shape[0]
    ts_len = inputs.shape[2]

    outputs = np.zeros((n_cells,
                        len(description['Outputs']),
                        ts_len),
                       dtype='d')
    initial_states = np.array(initial_states).transpose()

    init_states = False
    if initial_states.shape[0] != n_cells:
      init_states = True
      initial_states = np.zeros((n_cells,
                        len(description['States'])))

    dest_states = np.zeros((n_cells,
                            len(description['States']),
                            ts_len),
                           dtype='d')

    tmp_outputs = np.zeros((n_cells,len(description['Outputs']),1),dtype='d')
    mod_name = ctypes.create_string_buffer(bytes(model_name,'ascii'))
    c_params = _conv(params)
    for i in range(ts_len):
      call = [mod_name,
              *_conv(inputs[:,:,i].reshape(n_cells,inputs.shape[1],1)),
              *c_params,
              *_conv(initial_states),
              *_conv(tmp_outputs),
              ctypes.c_bool(init_states),
              ctypes.create_string_buffer(bytes(cpu_profile, 'ascii'))]
      _the_library.RunSingleModel(*call)
      dest_states[:,:,i] = initial_states
      outputs[:,:,i] = tmp_outputs[:,:,0] #[:,:,i].reshape(n_cells,outputs.shape[1],1)
      init_states = False

    return [outputs[:,i,:] for i in range(len(description['Outputs']))] + \
           [dest_states[:,i,:] for i in range(len(description['States']))]

  model_func.__filter_arguments__ = __filter_arguments__
  model_func.with_states = with_states
  _make_model_doc(model_func,description,return_states='Final state')
  _make_model_doc(model_func.with_states,description,return_states='State timeseries')
  setattr(thismodule,model_name,model_func)
