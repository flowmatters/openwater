import sys
import ctypes
import numpy as np
_the_library = None

def _conv(arr):
    return [arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),*[ctypes.c_int(i) for i in arr.shape]]

def _create_model_func(model_name,description):
  from .discovery import _lib_path, _collect_arguments, _make_model_doc

  global _the_library
  if not _the_library:
    _the_library = ctypes.CDLL(_lib_path())

  thismodule = sys.modules[__name__]

  def model_func(*args,**kwargs):
    inputs, params, _, _ = _collect_arguments(description,args,kwargs)
    if len(inputs) and len(inputs[0].shape)==1:
      inputs = [i.reshape(1,len(i)) for i in inputs]
    inputs = np.stack(inputs,axis=1)

    params = np.array(params)
    if len(params.shape)==1:
      params = params.reshape(params.shape[0],1)

    outputs = np.zeros((inputs.shape[0],
                        len(description['Outputs']),
                        inputs.shape[2]),
                       dtype='d')
    states = np.zeros((inputs.shape[0],
                        len(description['States'])))

    call = [ctypes.create_string_buffer(bytes(model_name,'ascii')),
            *_conv(inputs),
            *_conv(params),
            *_conv(states),
            *_conv(outputs),
            ctypes.c_bool(False)]

    _the_library.RunSingleModel(*call)
    return [outputs[:,i,:] for i in range(len(description['Outputs']))]
  
  model_func.__name__ = model_name

  _make_model_doc(model_func,description)

  setattr(thismodule,model_name,model_func)
