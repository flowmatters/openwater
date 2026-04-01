import h5py
from .template import ModelFile, _run
from openwater.results import OpenwaterSplitResults
from typing import Sequence, Tuple
import logging
logger = logging.getLogger(__name__)


def create_or_reuse_model_group(fn,existing):
  '''
  Create a new HDF5 file with a MODELS group mirroring the structure of `existing`,
  or return `existing` if `fn` is None.

  Parameters
  ----------
  fn : str or None
      Path for the new HDF5 file. If None, returns `existing` directly.
  existing : h5py.Group
      An existing MODELS group whose model subgroups will be replicated (empty) in the new file.

  Returns
  -------
  h5py.Group
      The MODELS group (either newly created or the existing one).
  '''
  if fn is None:
    return existing

  logger.info(f'Creating {fn}')  
  h5_f = h5py.File(fn,'w')
  grp = h5_f.create_group('MODELS')
  for model in existing.keys():
    grp.create_group(model)

  return grp

def split_time_series(models_group: h5py.Group,splits:int, windows:Sequence[int]) -> Sequence[Tuple[int,int]]:
  '''
  Calculate time window boundaries for splitting a model's input time series.

  If `windows` is provided, uses the given breakpoints to define windows.
  Otherwise, divides the time series into `splits` equal-sized windows.

  Parameters
  ----------
  models_group : h5py.Group or dict-like
      The MODELS group from an HDF5 file. Used to determine the total time series length
      from the first model with inputs.
  splits : int
      Number of equal-sized windows to create when `windows` is None.
  windows : sequence of int or None
      Explicit breakpoints (timestep indices) for splitting. Leading 0 and trailing
      ts_length values are handled automatically.

  Returns
  -------
  list of (int, int)
      List of (start_index, end_index) tuples defining each time window,
      or None if no inputs are found.
  '''
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
  '''
  Split a model HDF5 file into separate files for structure, parameters,
  initial states, and time-windowed inputs.

  This enables temporal decomposition: each input window can be simulated
  independently with state chaining (final states from window N become
  initial states for window N+1). See `run_split_model` for execution.

  Parameters
  ----------
  orig_model : str
      Path to the original model HDF5 file.
  structure : str
      Path for the output structure file (receives DIMENSIONS, LINKS, META,
      and model batches/maps).
  parameters : str, optional
      Path for a separate parameters file. If None, parameters are written
      to the structure file.
  init_states : str, optional
      Path for a separate initial states file. If None, states are written
      to the structure file.
  inputs : str, optional
      Path template for input files. For multiple windows, '-N' is inserted
      before '.h5' (e.g. 'inputs.h5' becomes 'inputs-0.h5', 'inputs-1.h5').
      If None, inputs are written to the structure file.
  split_ts : int
      Number of equal-sized time windows (used when `input_windows` is None).
  input_windows : sequence of int, optional
      Explicit breakpoints for time windows. See `split_time_series`.
  '''
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
        logger.info(f'No inputs recorded for {model}. Skipping')
        continue

      logger.info(f'Copying inputs for {model}')
      for ix, ((start_idx,end_idx),ts_dest_grp) in enumerate(zip(input_windows,inputs_models)):
        logger.debug(f'Input window {ix}: [{start_idx}:{end_idx}]')
        ts_dest_grp[model]['inputs'] = grp['inputs'][:,:,start_idx:end_idx]
  finally:
    closed = set()
    for fp in [input_f,structure_f,params_models,init_states_models]+inputs_models:
      if fp is None:
        continue

      # fp may be an h5py.File or an h5py.Group. For groups, accessing .file
      # can raise if the parent file is already closed. Use hasattr('filename')
      # to distinguish Files from Groups.
      if hasattr(fp, 'filename'):
        h5file = fp
      else:
        try:
          h5file = fp.file
        except (ValueError, RuntimeError):
          continue

      if not h5file.id.valid:
        continue
      fn = h5file.filename
      if fn in closed:
        continue
      logger.info(f'Closing {fn}')
      h5file.close()
      closed.add(fn)

def run_split_model(structure,params=None,init_states=None,inputs=None,dests=None,final_states=None,**kwargs):
  '''
  Run a temporally split model, executing each time window sequentially
  and chaining final states from one window as initial states for the next.

  Parameters
  ----------
  structure : str
      Path to the structure file (from `split_model`).
  params : str, optional
      Path to the parameters file. Defaults to the structure file.
  init_states : str, optional
      Path to the initial states file for the first window. Defaults to
      the structure file.
  inputs : list of str, optional
      Paths to input files, one per time window. Defaults to [structure].
  dests : list of str
      Paths for output result files, one per time window.
  final_states : list of str
      Paths for final state files, one per time window. The final states
      from window N become the initial states for window N+1.
  **kwargs
      Additional arguments passed to the simulation runner (e.g. overwrite, verbose).

  Returns
  -------
  OpenwaterSplitResults
      Combined results across all time windows.
  '''
  params = params or structure
  init_states = init_states or structure
  inputs = inputs or [structure]

  all_results = []
  for ix,(input_f, dest_f, states_f) in enumerate(zip(inputs,dests,final_states)):
    logger.info(f'Iteration {ix}: {input_f}/{init_states} => {dest_f}/{states_f}')
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
  return OpenwaterSplitResults(all_results)
