import sys
import ctypes
import numpy as np
import logging
logger = logging.getLogger(__name__)
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

def _as_double_array(data, ndim=None):
    """Coerce *data* to a C-contiguous float64 numpy array.

    Accepts lists, scalars, and arrays of any numeric dtype.
    If *ndim* is given the result is reshaped to have exactly that many
    dimensions (trailing size-1 dimensions are appended as needed).
    """
    arr = np.ascontiguousarray(data, dtype=np.float64)
    if ndim is not None:
        while arr.ndim < ndim:
            arr = arr.reshape(arr.shape + (1,))
    return arr

def _conv(arr):
    """Convert a numpy array to ctypes arguments: (pointer, *shape_ints).

    Raises TypeError if the array is not float64 or not C-contiguous so
    that callers get a clear Python exception instead of a segfault.
    """
    if arr.dtype != np.float64:
        raise TypeError(
            f"Expected float64 array, got {arr.dtype}. "
            f"Use _as_double_array() to coerce inputs before calling _conv()."
        )
    if not arr.flags['C_CONTIGUOUS']:
        raise TypeError(
            "Array is not C-contiguous. "
            "Use np.ascontiguousarray() before calling _conv()."
        )
    return [arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),*[ctypes.c_int(i) for i in arr.shape]]

def _get_n_states(model_name, params, n_named_states):
    """Query the shared library for the actual state array width.

    For most models this equals *n_named_states*. For models with
    parameter-dependent state sizes (e.g. GR4J, Lag) the true width
    may be larger.  Falls back to *n_named_states* if the library
    does not expose ``GetStatesSize``.

    *params* can be a list, 1-D array, or 2-D array — it will be
    coerced to a 2-D float64 array before calling the C library.
    """
    global _the_library
    if _the_library is None:
        return n_named_states
    try:
        p = _as_double_array(params, ndim=2)
        n = _the_library.GetStatesSize(
            ctypes.create_string_buffer(bytes(model_name, 'ascii')),
            *_conv(p))
        return n if n > 0 else n_named_states
    except AttributeError:
        logger.warning("Library does not expose GetStatesSize. Assuming %d states for model %s.", n_named_states, model_name, exc_info=True)
        # Library built before GetStatesSize was added
        return n_named_states

class StateDict(dict):
    """A dict of named states with ``.raw`` and ``.names`` attributes.

    Attributes
    ----------
    raw : numpy.ndarray
        The full ``(n_cells, n_states)`` state array.
    names : list[str]
        Ordered state names from the model description.
    """
    raw = None
    names = None

def _get_state_sizes(model_name, params):
    """Query the shared library for the per-state column sizes.

    Returns a list of ints, one per named state, giving the number of
    columns each state occupies in the packed state vector.  For most
    models every entry is 1.  For models with parameter-dependent
    state sizes (e.g. GR4J) some entries may be larger.

    Falls back to all-ones if the library does not expose
    ``GetStateSizes``.
    """
    global _the_library
    if _the_library is None:
        return None
    try:
        p = _as_double_array(params, ndim=2)
        # First call with a zero-length buffer to learn how many
        # states the model has.
        n = _the_library.GetStateSizes(
            ctypes.create_string_buffer(bytes(model_name, 'ascii')),
            *_conv(p),
            ctypes.POINTER(ctypes.c_int)(), ctypes.c_int(0))
        if n <= 0:
            return None
        buf = (ctypes.c_int * n)()
        _the_library.GetStateSizes(
            ctypes.create_string_buffer(bytes(model_name, 'ascii')),
            *_conv(p),
            buf, ctypes.c_int(n))
        return list(buf)
    except AttributeError:
        return None

def _rebuild_raw_states(description, state_kwargs):
    """Reconstruct a raw ``(1, n_states)`` array from named state values.

    Handles both scalar and array-valued states (e.g. from
    ``init_states``).  Scalar states are placed at their column index;
    array states are concatenated after all named scalar/dimension
    columns.
    """
    state_names = list(description['States'])
    # Separate scalars and arrays in declaration order.
    scalars = []
    arrays = []
    for name in state_names:
        val = state_kwargs.get(name, 0.0)
        if hasattr(val, '__len__') and not isinstance(val, str):
            arrays.append(np.asarray(val, dtype=np.float64).ravel())
        else:
            scalars.append(float(val))

    # Layout: [scalar0, ..., scalarN, array0..., array1..., ...]
    parts = scalars.copy()
    for a in arrays:
        parts.extend(a.tolist())

    return np.array([parts], dtype=np.float64)

def _initialise_states(model_name, params, n_cells=1):
    """Ask the shared library to initialise states for a model.

    Parameters
    ----------
    model_name : str
        Registered model name (e.g. ``'GR4J'``).
    params : array-like
        Parameter values — a list, 1-D, or 2-D array.  Coerced to a
        ``(nParameters, nParameterSets)`` float64 array automatically.
    n_cells : int
        Number of cells (rows) to initialise.

    Returns
    -------
    numpy.ndarray
        A ``(n_cells, n_states)`` float64 array containing the
        library-initialised state values.  For models with
        parameter-dependent state sizes (e.g. GR4J) the width reflects
        the actual packed state vector.
    """
    global _the_library
    if _the_library is None:
        raise RuntimeError("Openwater core library not loaded.")

    p = _as_double_array(params, ndim=2)

    # Query width first so we can allocate the right buffer.
    n_states = _the_library.GetStatesSize(
        ctypes.create_string_buffer(bytes(model_name, 'ascii')),
        *_conv(p))
    if n_states <= 0:
        raise RuntimeError(
            f"GetStatesSize returned {n_states} for model {model_name}")

    states = np.zeros((n_cells, n_states), dtype=np.float64)

    _the_library.InitialiseModelStates(
        ctypes.create_string_buffer(bytes(model_name, 'ascii')),
        *_conv(p),
        *_conv(states))

    return states

def _create_model_func(model_name,description):
  from .discovery import _lib_path, _collect_arguments, _make_model_doc

  global _the_library
  if not _the_library:
    _ensure_library_loaded()

  thismodule = sys.modules[__name__]
  n_named_states = len(description['States'])

  def model_func(*args,**kwargs):
    cpu_profile = kwargs.get('cpu_profile', '')
    kwargs.pop('cpu_profile', None)
    raw_states = kwargs.pop('_states', None)
    if isinstance(raw_states, StateDict):
      raw_states = raw_states.raw
    # Check if any state kwargs are arrays (e.g. from init_states dict).
    # _collect_arguments can't handle array-valued states, so we build
    # the raw state vector here and strip them from kwargs.
    if raw_states is None:
      state_kwargs = {}
      for sn in description['States']:
        if sn in kwargs:
          state_kwargs[sn] = kwargs.pop(sn)
      if state_kwargs and any(hasattr(v, '__len__') and not isinstance(v, str)
                              for v in state_kwargs.values()):
        raw_states = _rebuild_raw_states(description, state_kwargs)
    elif raw_states is not None:
      for sn in description['States']:
        kwargs.pop(sn, None)
    inputs, params, states, _ = _collect_arguments(description,args,kwargs)
    if len(inputs):
      len_first_provided = [len(i) for i in inputs if len(i)][0]
      inputs = [_as_double_array(i).reshape(1,len(i)) if len(i)==len_first_provided else np.zeros((1,len_first_provided),'d') for i in inputs]
    inputs = _as_double_array(np.stack(inputs,axis=1), ndim=3)

    params = _as_double_array(params, ndim=2)

    n_states = _get_n_states(model_name, params, n_named_states)

    outputs = np.zeros((inputs.shape[0],
                        len(description['Outputs']),
                        inputs.shape[2]),
                       dtype='d')

    if raw_states is not None:
      states = _as_double_array(raw_states, ndim=2)
      init_states = False
      if states.shape[1] < n_states:
        init_states = True
        states = np.zeros((inputs.shape[0], n_states))
      else:
        states = np.ascontiguousarray(states)
    else:
      states = _as_double_array(states).transpose()
      init_states = False
      if states.shape[0] != inputs.shape[0]:
        init_states = True
        states = np.zeros((inputs.shape[0], n_states))
      elif states.ndim < 2 or states.shape[1] < n_states:
        init_states = True
        states = np.zeros((inputs.shape[0], n_states))
      else:
        states = np.ascontiguousarray(states)

    call = [ctypes.create_string_buffer(bytes(model_name,'ascii')),
            *_conv(inputs),
            *_conv(params),
            *_conv(states),
            *_conv(outputs),
            ctypes.c_bool(init_states),
            ctypes.create_string_buffer(bytes(cpu_profile, 'ascii'))]

    if _the_library is None:
      raise RuntimeError("Openwater core library not loaded. Cannot run model.")

    _the_library.RunSingleModel(*call)
    return [outputs[:,i,:] for i in range(len(description['Outputs']))] + \
           [states[:,i] for i in range(min(n_named_states, states.shape[1]))]

  model_func.__name__ = model_name
  model_func.__description__ = description
  def __filter_arguments__(**kwargs):
    pnames = [p['Name'] for p in description['Parameters']]
    return {k:v for k,v in kwargs.items() if (k in description['Inputs']) or (k in pnames)}

  def with_states(*args,**kwargs):
    cpu_profile = kwargs.get('cpu_profile', '')
    kwargs.pop('cpu_profile', None)
    raw_states = kwargs.pop('_states', None)
    if isinstance(raw_states, StateDict):
      raw_states = raw_states.raw
    if raw_states is None:
      state_kwargs = {}
      for sn in description['States']:
        if sn in kwargs:
          state_kwargs[sn] = kwargs.pop(sn)
      if state_kwargs and any(hasattr(v, '__len__') and not isinstance(v, str)
                              for v in state_kwargs.values()):
        raw_states = _rebuild_raw_states(description, state_kwargs)
    elif raw_states is not None:
      for sn in description['States']:
        kwargs.pop(sn, None)
    inputs, params, initial_states, _ = _collect_arguments(description,args,kwargs)
    if len(inputs) and len(inputs[0].shape)==1:
      inputs = [_as_double_array(i).reshape(1,len(i)) for i in inputs]
    inputs = _as_double_array(np.stack(inputs,axis=1), ndim=3)

    params = _as_double_array(params, ndim=2)

    n_cells = inputs.shape[0]
    ts_len = inputs.shape[2]
    n_states = _get_n_states(model_name, params, n_named_states)

    outputs = np.zeros((n_cells,
                        len(description['Outputs']),
                        ts_len),
                       dtype='d')

    if raw_states is not None:
      initial_states = _as_double_array(raw_states, ndim=2)
      init_states = False
      if initial_states.shape[1] < n_states:
        init_states = True
        initial_states = np.zeros((n_cells, n_states))
      else:
        initial_states = np.ascontiguousarray(initial_states)
    else:
      initial_states = _as_double_array(initial_states).transpose()
      init_states = False
      if initial_states.shape[0] != n_cells:
        init_states = True
        initial_states = np.zeros((n_cells, n_states))
      elif initial_states.ndim < 2 or initial_states.shape[1] < n_states:
        init_states = True
        initial_states = np.zeros((n_cells, n_states))
      else:
        initial_states = np.ascontiguousarray(initial_states)

    # dest_states records the named state columns at each timestep.
    dest_states = np.zeros((n_cells, n_named_states, ts_len), dtype='d')

    tmp_outputs = np.zeros((n_cells,len(description['Outputs']),1),dtype='d')
    mod_name = ctypes.create_string_buffer(bytes(model_name,'ascii'))
    c_params = _conv(params)
    for i in range(ts_len):
      timestep_input = np.ascontiguousarray(
          inputs[:,:,i].reshape(n_cells,inputs.shape[1],1))
      call = [mod_name,
              *_conv(timestep_input),
              *c_params,
              *_conv(initial_states),
              *_conv(tmp_outputs),
              ctypes.c_bool(init_states),
              ctypes.create_string_buffer(bytes(cpu_profile, 'ascii'))]
      if _the_library is None:
        raise RuntimeError("Openwater core library not loaded. Cannot run model.")
      _the_library.RunSingleModel(*call)
      dest_states[:,:,i] = initial_states[:,:n_named_states]
      outputs[:,:,i] = tmp_outputs[:,:,0]
      init_states = False

    return [outputs[:,i,:] for i in range(len(description['Outputs']))] + \
           [dest_states[:,i,:] for i in range(n_named_states)]

  def init_states(*args, n_cells=1, **kwargs):
    """Return initialised states for the given parameters.

    Accepts parameters as positional or keyword arguments (same as the
    model function itself — only parameter names are used, inputs and
    states are ignored).

    Returns a `StateDict` (a dict subclass) mapping each named state
    to a scalar or array extracted from the initialised state vector.
    The dict can be passed directly as ``**kwargs`` to the model
    function or ``with_states``.

    For models whose state vector is wider than the number of named
    states (e.g. GR4J with unit-hydrograph queues), the array-valued
    states are correctly sliced from the raw vector.  The full raw
    array and name list are available as attributes::

        st = GR4J.init_states(X1=350, X2=-1, X3=50, X4=2.0)
        st['q1']            # array([0., 0.])
        st['q9']            # array([0.])
        st.raw              # full (n_cells, n_states) array
        st.names            # ['s', 'r', 'n1', 'n2', 'q1', 'q9']
    """
    _, params_vals, _, _ = _collect_arguments(description, args, kwargs)
    p = _as_double_array(params_vals, ndim=2)
    raw = _initialise_states(model_name, p, n_cells=n_cells)

    state_names = list(description['States'])

    result = StateDict()
    result.raw = raw
    result.names = state_names

    # Get per-state sizes from the library (e.g. [1,1,1,1,2,1] for GR4J).
    sizes = _get_state_sizes(model_name, p)
    if sizes is None:
      sizes = [1] * len(state_names)

    col = 0
    for name, size in zip(state_names, sizes):
      if size == 1:
        result[name] = raw[:, col] if n_cells > 1 else raw[0, col]
      else:
        result[name] = raw[:, col:col + size] if n_cells > 1 else raw[0, col:col + size]
      col += size

    return result

  model_func.__filter_arguments__ = __filter_arguments__
  model_func.with_states = with_states
  model_func.init_states = init_states
  _make_model_doc(model_func,description,return_states='Final state')
  _make_model_doc(model_func.with_states,description,return_states='State timeseries')
  setattr(thismodule,model_name,model_func)
