import h5py
from .template import ModelFile

def create_or_reuse_model_group(fn,existing):
  if fn is None:
    return existing

  print(f'Creating {fn}')  
  h5_f = h5py.File(fn,'w')
  grp = h5_f.create_group('MODELS')
  for model in existing.keys():
    grp.create_group(model)

  return grp

def split_model(orig_model: str,
                structure: str,
                parameters: str=None, 
                init_states: str=None,
                inputs: str=None,
                split_ts: int=1):
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

  if split_ts and split_ts > 1:
    inputs_models = [create_or_reuse_model_group(inputs.replace('.h5',f'-{ix}.h5'),structure_models) for ix in range(split_ts)]
  else:
    inputs_models = [create_or_reuse_model_group(inputs,structure_models)]

  for model, grp in input_f['MODELS'].items():
    grp.copy('batches',structure_models[model])
    grp.copy('map',structure_models[model])
    grp.copy('parameters',params_models[model])
    grp.copy('states',init_states_models[model])

    if 'inputs' not in grp:
      print(f'No inputs recorded for {model}. Skipping')
      continue

    print(f'Copying inputs for {model}')

    n_splits = len(inputs_models)
    split_size = grp['inputs'].shape[2] // n_splits
    starting_ix = 0
    for ix, ts_grp in enumerate(inputs_models):
      ending_ix = starting_ix + split_size
      if ix == n_splits-1:
        ending_ix = grp['inputs'].shape[2]
      ts_grp[model]['inputs'] = grp['inputs'][:,:,starting_ix:ending_ix]

      starting_ix = ending_ix



