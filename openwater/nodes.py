

def _create_model_type(name,description):
  import sys
  thismodule = sys.modules[__name__]

  setattr(thismodule,name,ModelDescription(name,description))

class ModelDescription(object):
  def __init__(self,name,description):
    self.name = name
    self.description = description
