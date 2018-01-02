import sys
import ctypes
_the_library = None

def _create_model_func(model_name,description):
  from .discovery import _lib_path, _collect_arguments, _make_model_doc

  global _the_library
  if not _the_library:
    _the_library = ctypes.CDLL(_lib_path())

  thismodule = sys.modules[__name__]

  def model_func(*args,**kwargs):
    call = [ctypes.create_string_buffer(bytes(model_name,'ascii'))]

    #_the_library.
  
  model_func.__name__ = model_name

  _make_model_doc(model_func,description)

  setattr(thismodule,model_name,model_func)
