'''
Substitute operation for OpenWater models.

Creates a reduced model by removing nodes and replacing their outputs with
data from a prior simulation. Two main use cases:

1. Remove all nodes of specified model types (e.g. remove water quantity
   models when focusing on water quality parameterisation).
2. Remove everything upstream of specified nodes (complement of clip),
   keeping only a downstream portion of the model for faster repeated runs.

In both cases, links that cross the boundary between removed and kept nodes
are identified. The outputs of removed source nodes are read from prior
simulation results and injected as fixed input time series into the kept
destination nodes.
'''
import logging
import numpy as np
import pandas as pd
import h5py as h5

from openwater import nodes as node_types
from .clip import (
    identify_models_to_keep,
    renumber_links,
    copy_parameters,
    string_data_set,
    check_model_table_consistency,
    resolve_end_nodes,
    read_node_slice,
    _copy_node_slice,
)

logger = logging.getLogger(__name__)


def _all_nodes(model_file):
    '''Return {model_type: sorted list of all run_indices} for every model in the file.'''
    result = {}
    for m in model_file._h5f['MODELS']:
        nodes = model_file.nodes_matching(m)
        result[m] = sorted(int(i) for i in nodes._run_idx)
    return result


def identify_nodes_to_remove(model_file, model_types=None, above_nodes=None):
    '''
    Identify nodes to remove from a model.

    Supports two modes (exactly one must be specified):

    - **model_types**: Remove ALL nodes of the specified model types.
    - **above_nodes**: Remove all nodes that are strictly upstream of the
      specified nodes (i.e. nodes required to compute ``above_nodes``,
      but not ``above_nodes`` themselves). This is the complement of
      ``clip``: clip keeps the upstream subgraph, substitute removes it.

    Parameters
    ----------
    model_file : ModelFile
        The source model file.
    model_types : list of str, optional
        Model type names to remove entirely.
    above_nodes : list of (str, int), optional
        (model_type, run_index) tuples. Everything upstream of these nodes
        is marked for removal. The nodes themselves are kept.

    Returns
    -------
    dict
        {model_type: sorted list of run_indices} for nodes to remove.
    '''
    if model_types is not None and above_nodes is not None:
        raise ValueError('Specify model_types or above_nodes, not both')
    if model_types is None and above_nodes is None:
        raise ValueError('Specify model_types or above_nodes')

    all_nodes = _all_nodes(model_file)

    if model_types is not None:
        return {mt: all_nodes[mt] for mt in model_types if mt in all_nodes}

    # above_nodes mode: find everything upstream, then exclude the above_nodes themselves
    nodes_to_remove, _ = identify_models_to_keep(model_file, above_nodes)
    # nodes_to_remove includes the above_nodes themselves — remove them
    for mt, idx in above_nodes:
        if mt in nodes_to_remove and idx in nodes_to_remove[mt]:
            nodes_to_remove[mt] = [n for n in nodes_to_remove[mt] if n != idx]
            if not nodes_to_remove[mt]:
                del nodes_to_remove[mt]
    return nodes_to_remove


def identify_boundary_links(model_file, nodes_to_remove):
    '''
    Find links that cross the boundary from removed to kept nodes.

    A boundary link has its source in the removal set and its destination
    outside it. These links represent the data flow that must be replaced
    by prior simulation outputs.

    Parameters
    ----------
    model_file : ModelFile
        The source model file.
    nodes_to_remove : dict
        {model_type: list of run_indices} as returned by ``identify_nodes_to_remove``.

    Returns
    -------
    DataFrame
        Subset of the link table where src is removed and dest is kept.
        Columns match the standard link table format.
    '''
    links = model_file.link_table()
    removal_sets = {m: set(ns) for m, ns in nodes_to_remove.items()}

    def is_removed(model, node):
        return model in removal_sets and node in removal_sets[model]

    src_removed = links.apply(lambda r: is_removed(r.src_model, r.src_node), axis=1)
    dest_removed = links.apply(lambda r: is_removed(r.dest_model, r.dest_node), axis=1)

    return links[src_removed & ~dest_removed].copy()


def identify_kept_links(model_file, nodes_to_remove):
    '''
    Find links where both endpoints are kept (not in the removal set).

    Parameters
    ----------
    model_file : ModelFile
        The source model file.
    nodes_to_remove : dict
        {model_type: list of run_indices} as returned by ``identify_nodes_to_remove``.

    Returns
    -------
    DataFrame
        Subset of the link table where neither src nor dest is removed.
    '''
    links = model_file.link_table()
    removal_sets = {m: set(ns) for m, ns in nodes_to_remove.items()}

    def is_removed(model, node):
        return model in removal_sets and node in removal_sets[model]

    src_removed = links.apply(lambda r: is_removed(r.src_model, r.src_node), axis=1)
    dest_removed = links.apply(lambda r: is_removed(r.dest_model, r.dest_node), axis=1)

    return links[~src_removed & ~dest_removed].copy()


def extract_substitute_inputs(results_h5, boundary_links, model_file):
    '''
    Extract output time series from prior results for each boundary link.

    When multiple removed nodes feed the same (dest_model, dest_node, dest_var),
    their contributions are summed (appropriate for additive flow/load variables).

    Parameters
    ----------
    results_h5 : h5py.File or str
        Prior simulation results HDF5 file (or path to it).
    boundary_links : DataFrame
        Boundary links as returned by ``identify_boundary_links``.
    model_file : ModelFile
        The source model file (used for model descriptions).

    Returns
    -------
    dict
        Mapping of (dest_model, dest_node, dest_var_name) -> numpy array
        of shape (n_timesteps,).
    '''
    opened_here = isinstance(results_h5, str)
    if opened_here:
        results_h5 = h5.File(results_h5, 'r')

    try:
        result = {}
        for _, link in boundary_links.iterrows():
            src_model = link.src_model
            src_node = int(link.src_node)
            src_var = link.src_var

            dest_model = link.dest_model
            dest_node = int(link.dest_node)
            dest_var = link.dest_var

            # Read the output from prior results
            desc = getattr(node_types, src_model).description
            var_idx = desc['Outputs'].index(src_var)
            outputs = results_h5['MODELS'][src_model]['outputs']
            ts_data = outputs[src_node, var_idx, :]

            key = (dest_model, dest_node, dest_var)
            if key in result:
                # Fan-in: sum contributions from multiple removed sources
                result[key] = result[key] + ts_data
            else:
                result[key] = ts_data.copy()

        return result
    finally:
        if opened_here:
            results_h5.close()


def substitute(model_file, prior_results, dest_fn,
               model_types_to_remove=None, above_nodes=None):
    '''
    Create a new model file with specified nodes removed and their outputs
    replaced by data from prior simulation results.

    Parameters
    ----------
    model_file : ModelFile
        The original full model.
    prior_results : str or h5py.File
        Path to (or open handle for) the prior simulation results HDF5 file.
    dest_fn : str
        Path for the output substituted HDF5 model file.
    model_types_to_remove : list of str, optional
        Model types to remove entirely.
    above_nodes : list of (str, int), optional
        Remove all nodes upstream of these. Can also pass tag-based queries
        through ``resolve_end_nodes`` first.

    Returns
    -------
    str
        Path to the created model file (same as ``dest_fn``).
    '''
    nodes_to_remove = identify_nodes_to_remove(
        model_file, model_types=model_types_to_remove, above_nodes=above_nodes
    )
    if not nodes_to_remove:
        raise ValueError('No nodes matched for removal')

    all_nodes = _all_nodes(model_file)
    nodes_to_keep = {}
    for mt, indices in all_nodes.items():
        removed = set(nodes_to_remove.get(mt, []))
        kept = [i for i in indices if i not in removed]
        if kept:
            nodes_to_keep[mt] = kept

    if not nodes_to_keep:
        raise ValueError('All nodes would be removed — nothing left to keep')

    boundary_links = identify_boundary_links(model_file, nodes_to_remove)
    kept_links = identify_kept_links(model_file, nodes_to_remove)

    logger.info(f'Removing {sum(len(v) for v in nodes_to_remove.values())} nodes, '
                f'keeping {sum(len(v) for v in nodes_to_keep.values())} nodes, '
                f'{len(boundary_links)} boundary links, {len(kept_links)} kept links')

    # Extract prior outputs for boundary links
    substitute_inputs = extract_substitute_inputs(prior_results, boundary_links, model_file)

    # Renumber kept links
    if len(kept_links):
        new_links = renumber_links(kept_links, nodes_to_keep)
    else:
        new_links = kept_links

    fp = model_file._h5f

    # Determine model types and dimensions
    model_types = set(nodes_to_keep.keys())
    new_model_maps = {}
    for m in model_types:
        nodes = model_file.nodes_matching(m)
        nodes = nodes[nodes._run_idx.isin(nodes_to_keep[m])]
        new_model_maps[m] = nodes

    dimensions_to_match = set()
    for m in model_types:
        dimensions_to_match.update(model_file.dims_for_model(m))

    new_dims = {}
    for dim in dimensions_to_match:
        values = set()
        for m in model_types:
            nodes = new_model_maps[m]
            if dim in nodes.columns:
                values.update(nodes[dim])
        new_dims[dim] = list(values)

    # Write the new model file
    with h5.File(dest_fn, 'w') as new_mod:
        # META
        meta = new_mod.create_group('META')
        string_data_set(meta, 'models', sorted(nodes_to_keep.keys()))
        for key in fp['META'].keys():
            if key != 'models' and key not in meta:
                fp['META'].copy(key, meta)

        # DIMENSIONS
        dim_grp = new_mod.create_group('DIMENSIONS')
        for dim, vals in new_dims.items():
            if not vals:
                continue
            if isinstance(vals[0], str):
                string_data_set(dim_grp, dim, vals)
            else:
                dim_grp.create_dataset(dim, data=vals)

        # MODELS
        model_grp = new_mod.create_group('MODELS')

        # Compute batches from kept links
        if len(new_links):
            link_table_groups = ['src', 'dest']
            link_table_keys = ['generation', 'model', 'node', 'gen_node', 'var']
            tmp = pd.DataFrame()
            for grp in link_table_groups:
                tmp = pd.concat([tmp, new_links[[f'{grp}_{c}' for c in link_table_keys]].rename(
                    columns=lambda c: c.replace(f'{grp}_', '')
                )])
            tmp = tmp.drop_duplicates()
            tmp = tmp.drop_duplicates(subset=['model', 'node'])
            check_model_table_consistency(tmp)
            num_generations = max(new_links.dest_generation) + 1
        else:
            tmp = pd.DataFrame(columns=['model', 'node', 'generation', 'gen_node', 'var'])
            num_generations = 1

        for mod, nodes in nodes_to_keep.items():
            grp = model_grp.create_group(mod)
            batch_sizes = [len(tmp[(tmp.model == mod) & (tmp.generation == g)]) for g in range(num_generations)]
            if not any(batch_sizes):
                batch_sizes = [len(nodes)]
            batches = np.cumsum(batch_sizes)
            grp.create_dataset('batches', dtype=np.uint32, data=batches)

        # Copy parameters, states, and build inputs (with substitutions)
        for mod, nodes in nodes_to_keep.items():
            grp = model_grp[mod]
            src_grp = fp['MODELS'][mod]

            if 'parameters' in src_grp:
                copy_parameters(mod, src_grp, grp, nodes)
            if 'states' in src_grp:
                _copy_node_slice(src_grp['states'], grp, 'states', nodes)

            # Build inputs: start from original, then overwrite boundary link slots
            if 'inputs' in src_grp:
                # read_node_slice returns a numpy array (already a copy).
                inputs_data = np.asarray(read_node_slice(src_grp['inputs'], nodes))
            else:
                # No original inputs — create zeros if we have substitutions for this model
                desc = getattr(node_types, mod).description
                n_inputs = len(desc['Inputs'])
                ts_length = _get_ts_length(fp)
                if n_inputs > 0 and ts_length > 0:
                    inputs_data = np.zeros((len(nodes), n_inputs, ts_length))
                else:
                    inputs_data = None

            if inputs_data is not None:
                desc = getattr(node_types, mod).description
                for (dm, dn, dv), ts_data in substitute_inputs.items():
                    if dm != mod:
                        continue
                    if dn not in nodes:
                        continue
                    new_node_idx = nodes.index(dn)
                    var_idx = desc['Inputs'].index(dv)
                    inputs_data[new_node_idx, var_idx, :] = ts_data

                grp.create_dataset('inputs', data=inputs_data)

        # Model map tables
        for mod, model_map_with_dims in new_model_maps.items():
            grp = model_grp[mod]
            src_grp = fp['MODELS'][mod]
            src_map_dims = src_grp['map'].attrs['DIMS']

            dest_map_dims = [d for d in src_map_dims if len(new_dims.get(d.decode(), []))]
            dest_map = -1 * np.ones(shape=[len(new_dims[d.decode()]) for d in dest_map_dims], dtype=np.int64)
            model_map_with_dims = model_map_with_dims.copy().sort_values('_run_idx')

            for d in dest_map_dims:
                d_str = d.decode()
                model_map_with_dims[d_str] = model_map_with_dims[d_str].apply(
                    lambda dim_val: new_dims[d_str].index(dim_val)
                )

            for _, row in model_map_with_dims.iterrows():
                coords = [row[d.decode()] for d in dest_map_dims]
                dest_map[tuple(coords)] = nodes_to_keep[mod].index(row._run_idx)

            map_var = grp.create_dataset('map', data=dest_map)
            map_var.attrs['DIMS'] = dest_map_dims

        # LINKS (only kept links, boundary links are absorbed as inputs)
        if len(new_links):
            mod_order = [m.decode() for m in new_mod['META']['models'][...]]
            descriptions = {mod: getattr(node_types, mod).description for mod in mod_order}

            new_link_vals = new_links.sort_values('src_generation').copy()
            for grp in ['src', 'dest']:
                mod_col = f'{grp}_model'
                var_col = f'{grp}_var'
                flux_type = 'Inputs' if grp == 'dest' else 'Outputs'
                new_link_vals[var_col] = new_link_vals.apply(
                    lambda row: descriptions[row[mod_col]][flux_type].index(row[var_col]), axis=1
                )
                new_link_vals[mod_col] = new_link_vals[mod_col].apply(lambda m: mod_order.index(m))

            new_mod.create_dataset('LINKS', data=np.array(new_link_vals, dtype=np.uint32))
        else:
            new_mod.create_dataset('LINKS', data=np.zeros((0, 10), dtype=np.uint32))

    return dest_fn


def _get_ts_length(h5f):
    '''Get the time series length from the first model with inputs.'''
    for model_name in h5f['MODELS']:
        grp = h5f['MODELS'][model_name]
        if 'inputs' in grp:
            return grp['inputs'].shape[2]
    return 0
