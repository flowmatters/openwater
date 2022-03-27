import h5py
from .template import ModelFile
from openwater.results import OpenwaterSplitResults
from typing import Sequence, Tuple

def create_or_reuse_model_group(fn,existing):
  if fn is None:
    return existing

  print(f'Creating {fn}')  
  h5_f = h5py.File(fn,'w')
  grp = h5_f.create_group('MODELS')
  for model in existing.keys():
    grp.create_group(model)

  return grp

def split_time_series(models_group: h5py.Group,splits:int, windows:Sequence[int]) -> Sequence[Tuple[int,int]]:
  ts_length = -1
  for model_name in models_group.keys():
    model_grp = models_group[model_name]
    if 'inputs' not in model_grp:
      continue

    ts_length = model_grp['inputs'].shape[2]
    break

  if ts_length < 0:
    return None

  if windows is None:
    if splits <= 1:
      return [
        (0,ts_length)
      ]

    split_size = ts_length // splits
    input_windows = [(i*split_size,(i+1)*split_size) for i in range(splits)]
    input_windows[-1] = (input_windows[-1][0],ts_length)
    return input_windows

  if len(windows) and (windows[0]==0):
    windows = windows[1:]

  if len(windows) and (windows[-1]==ts_length):
    windows = windows[:-1]

  result = []
  start_idx = 0
  for end_idx in windows:
    result.append((start_idx,end_idx))
    start_idx = end_idx

  if start_idx < ts_length:
    result.append((start_idx,ts_length))

  return result

def split_model(orig_model: str,
                structure: str,
                parameters: str=None, 
                init_states: str=None,
                inputs: str=None,
                split_ts: int=1,
                input_windows: Sequence[int]=None):

  input_f = structure_f = params_models = init_states_models = None
  inputs_models = []

  try:
    input_f = h5py.File(orig_model,'r')
    structure_f = h5py.File(structure,'w')
    input_f.copy('DIMENSIONS',structure_f)
    input_f.copy('LINKS',structure_f)
    input_f.copy('META',structure_f)

    structure_models = structure_f.create_group('MODELS')
    for model in input_f['MODELS'].keys():
      dest_grp = structure_models.create_group(model)

    params_models = create_or_reuse_model_group(parameters,structure_models)
    init_states_models = create_or_reuse_model_group(init_states,structure_models)

    input_windows = split_time_series(input_f['MODELS'],split_ts,input_windows)

    if len(input_windows)>1:
      inputs_models = [create_or_reuse_model_group(inputs.replace('.h5',f'-{ix}.h5'),structure_models) for ix,_ in enumerate(input_windows)]
    else:
      inputs_models = [create_or_reuse_model_group(inputs,structure_models)]

    for model, grp in input_f['MODELS'].items():
      grp.copy('batches',structure_models[model])
      grp.copy('map',structure_models[model])

      if 'parameters' in grp:
        grp.copy('parameters',params_models[model])

      if 'states' in grp:
        grp.copy('states',init_states_models[model])

      if 'inputs' not in grp:
        print(f'No inputs recorded for {model}. Skipping')
        continue

      print(f'Copying inputs for {model}')
      for ix, ((start_idx,end_idx),ts_dest_grp) in enumerate(zip(input_windows,inputs_models)):
        print(f'Input window {ix}: [{start_idx}:{end_idx}]')
        ts_dest_grp[model]['inputs'] = grp['inputs'][:,:,start_idx:end_idx]
  finally:
    for fp in [input_f,structure_f,params_models,init_states_models]+inputs_models:
      if fp is None:
        continue

      if hasattr(fp,'close'):
        print(f'Closing {fp.filename}')
        fp.close()
      else:
        print(f'Closing {fp.file.filename}')
        fp.file.close()

def run_split_model(structure,params=None,init_states=None,inputs=None,dests=None,final_states=None,**kwargs):
  params = params or structure
  init_states = init_states or structure
  inputs = inputs or [structure]

  all_results = []
  for ix,(input_f, dest_f, states_f) in enumerate(zip(inputs,dests,final_states)):
    print(f'Iteration {ix}: {input_f}/{init_states} => {dest_f}/{states_f}')
    run_results = _run(None,
                        structure,
                        dest_f,
                        initial_states=init_states,
                        final_states=states_f,
                        input_timeseries=input_f,
                        parameters=params,
                        **kwargs)
    init_states = states_f
    all_results.append(run_results)
  return OpenwaterSplitResults(all_results,time_period=TIME_PERIOD)
