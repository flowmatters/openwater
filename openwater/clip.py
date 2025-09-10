import logging
from openwater import discovery, template as owt
from openwater.results import OpenwaterResults
from openwater import nodes as node_types
import pandas as pd
import h5py as h5
import numpy as np
from .array_params import get_parameter_locations, param_starts
logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)
logger.propagate = True

def diff_keys_and_types(grp1,grp2,prefix=''):
    for attr in grp1.attrs.keys():
        if attr not in grp2.attrs.keys():
            logger.warning(f'Missing attribute {attr} at {prefix}')
    if not hasattr(grp1,'keys'):
        return

    for key in grp1.keys():
        new_prefix = f'{prefix}/{key}'
        if key not in grp2.keys():
            logger.warning(f'Missing {new_prefix}')
            continue
        if not hasattr(grp1[key],'keys'):
            ds1 = grp1[key]
            ds2 = grp2[key]
            shp1 = ds1.shape
            shp2 = ds2.shape
            if len(shp1) != len(shp2):
                logger.warning(f'Shape mismatch: {shp1} vs {shp2} at {new_prefix}')

            dt1 = ds1.dtype
            dt2 = ds2.dtype
            dts_equal = dt1==dt2
            dts_equal = dts_equal or (str(dt1).startswith('S|') and str(dt2).startswith('S|'))
            if not dts_equal:
                logger.warning(f'Dtype mismatch {ds1.dtype} vs {ds2.dtype} at {new_prefix}')

        diff_keys_and_types(grp1[key],grp2[key],new_prefix)


def identify_models_to_keep(ow_mod, starting):
    links = ow_mod.link_table()
    model_nodes_to_keep = {}
    links_to_keep = pd.DataFrame()

    def visited(mod,idx):
        if not mod in model_nodes_to_keep:
            return False
        return idx in model_nodes_to_keep[mod]

    def keep_node(mod,idx):
        if not mod in model_nodes_to_keep:
            model_nodes_to_keep[mod] = set()

        if idx in model_nodes_to_keep[mod]:
            return

        model_nodes_to_keep[mod] = model_nodes_to_keep[mod].union({idx})
    links['src'] = links['src_model'] + '/' + links['src_node'].astype('str')
    links['dest'] = links['dest_model'] + '/' + links['dest_node'].astype('str')
    to_visit = [f'{model}/{node}' for model,node in starting]
    while len(to_visit):
        new_nodes = []
        for node_path in to_visit:
          mod,idx = node_path.split('/')
          idx = int(idx)
          if visited(mod,idx):
              continue

          keep_node(mod,idx)
          new_nodes.append(node_path)

        links_to_mod = links[links.dest.isin(new_nodes)]
        links_to_keep = pd.concat([links_to_keep,links_to_mod])
        to_visit = list(links_to_mod.src)

    model_nodes_to_keep = {mt:sorted(nodes) for mt,nodes in model_nodes_to_keep.items()}
    links_to_keep = links_to_keep.drop(columns=['src','dest'])
    return model_nodes_to_keep,links_to_keep

def renumber_links(links,nodes):
    links = links.copy()
    min_gen = min(links.src_generation)
    def renumber_node_set(mt):
        return lambda row: nodes[row[f'{mt}_model']].index(row[f'{mt}_node'])    

    def renumber_gen_node(mt):
        def doer(row):
            gen_col = f'{mt}_generation'
            mod_col = f'{mt}_model'
            node_col = f'{mt}_node'
            gen = row[gen_col]
            mod = row[mod_col]

            subset = links[(links[gen_col]==gen)&(links[mod_col]==mod)]
            min_node = min(subset[node_col])
            return row[node_col]-min_node
        return doer

    links['src_node'] = links.apply(renumber_node_set('src'),axis=1)
    links['dest_node'] = links.apply(renumber_node_set('dest'),axis=1)
    
    links['src_generation'] -= min_gen
    links['dest_generation'] -= min_gen

    links['src_gen_node'] = links.apply(renumber_gen_node('src'),axis=1)
    links['dest_gen_node'] = links.apply(renumber_gen_node('dest'),axis=1)
    return links

def string_data_set(grp,name,values):
    max_len = max([len(s) for s in values])
    str_type = f'|S{max_len}'
    return grp.create_dataset(name,dtype=str_type,data=values)

def check_model_table_consistency(df):
    for mt in set(df.model):
        ss = df[df.model==mt]
        for n in set(ss.node):
            sss = ss[ss.node==n]
            assert len(set(sss.generation))==1
            assert len(set(sss.gen_node))==1

def copy_parameters(mod,src_grp,dest_grp,nodes):
  desc = getattr(node_types,mod).description
  subset = src_grp['parameters'][:,nodes]
  dimensions = desc.get('Dimensions',[])
  if not len(dimensions):
    dest_grp.create_dataset('parameters',data=subset)
    return

  new_param_locs = get_parameter_locations(desc,subset)
  len_params = new_param_locs[-1][1]
  existing_param_locs = get_parameter_locations(desc,src_grp['parameters'])
  new_sizes = [window[1] - window[0] for window in new_param_locs]
  source_windows = [[old[0],old[0]+sz] for old,sz in zip(existing_param_locs,new_sizes)]

  dest_grp.create_dataset('parameters',shape=[len_params,len(nodes)],dtype=subset.dtype)
  for src,dest in zip(source_windows,new_param_locs):
    dest_grp['parameters'][dest[0]:dest[1],:] = subset[src[0]:src[1],:]


def clip(model,dest_fn,end_nodes):
  nodes_to_keep, links_to_keep = identify_models_to_keep(model,end_nodes)
  node_count = sum([len(v) for v in nodes_to_keep.values()])
  logger.info(f'Keeping {node_count} nodes and {len(links_to_keep)} links')
  new_links = renumber_links(links_to_keep,nodes_to_keep)

  model_types = set.union(set(new_links.src_model),set(new_links.dest_model))
  dimensions_to_match = set.union(*[set(model.dims_for_model(m)) for m in model_types])

  fp = model._h5f
  new_model_maps = {}
  for m in model_types:
      nodes = model.nodes_matching(m)
      len_before = len(nodes)
      nodes = nodes[nodes._run_idx.isin(nodes_to_keep[m])]
      len_after = len(nodes)
      logger.info(f'Using {len_after} of {len_before} {m} nodes')
      new_model_maps[m] = nodes

  new_dims = {} 
  for dim in dimensions_to_match:
      orig_values = fp['DIMENSIONS'][dim][...]
      values = set()
      for m in model_types:
          nodes = new_model_maps[m]
          if dim not in nodes.columns:
              continue

          values = values.union(set(nodes[dim]))
      new_dims[dim] = list(values)

  new_mod = h5.File(dest_fn,'w')
  meta = new_mod.create_group('META')
  meta_models = string_data_set(meta,'models',sorted(nodes_to_keep.keys()))
  model_grp = new_mod.create_group('MODELS')
  dim_grp = new_mod.create_group('DIMENSIONS')

  logger.info(f'Creating {len(new_dims)} dimensions')
  for dim,vals in new_dims.items():
      logger.debug(f'{dim}, {vals}')
      if isinstance(vals[0],str):
          string_data_set(dim_grp,dim,vals)
      else:
          dim_grp.create_dataset(dim,data=vals)

  link_table_groups = ['src','dest']
  link_table_keys =['generation','model','node','gen_node','var']
  tmp = pd.DataFrame()
  for grp in link_table_groups:
      tmp = pd.concat([tmp,new_links[[f'{grp}_{c}' for c in link_table_keys]].rename(columns=lambda c:c.replace(f'{grp}_',''))])
  tmp = tmp.drop_duplicates()
  tmp = tmp.drop_duplicates(subset=['model','node'])
  check_model_table_consistency(tmp)

  num_generations = max(new_links.dest_generation)+1

  logger.info('Creating model batches')
  for mod,nodes in nodes_to_keep.items():
      grp = model_grp.create_group(mod)
      batch_sizes = [len(tmp[(tmp.model==mod)&(tmp.generation==g)]) for g in range(num_generations)]
      batches = np.cumsum(batch_sizes)
      grp.create_dataset('batches',dtype=np.uint32,data=batches)

  logger.info('Copying parameters, inputs and states')
  for mod,nodes in nodes_to_keep.items():
      grp = model_grp[mod]
      src_grp = fp['MODELS'][mod]
      logger.debug(f'{mod}, {list(src_grp.keys())}')
      if 'inputs' in src_grp:
          grp.create_dataset('inputs',data=src_grp['inputs'][nodes,:,:])
      if 'parameters' in src_grp:
          copy_parameters(mod,src_grp,grp,nodes)
      if 'states' in src_grp:
          grp.create_dataset('states',data=src_grp['states'][nodes,:])

  logger.info('Creating model map tables')
  for mod,model_map_with_dims in new_model_maps.items():
      grp = model_grp[mod]
      src_grp = fp['MODELS'][mod]
      src_map = src_grp['map']
      src_map_dims = src_map.attrs['DIMS']
      src_map_proc = src_map.attrs['PROCESSES']

      dest_map_dims = [d for d in src_map_dims if len(new_dims[d.decode()])]
      dest_map = -1 * np.ones(shape=[len(new_dims[d.decode()]) for d in dest_map_dims],dtype=np.int64)
      model_map_with_dims = model_map_with_dims.copy().sort_values('_run_idx')

      for d in dest_map_dims:
          d = d.decode()
          model_map_with_dims[d] = model_map_with_dims[d].apply(lambda dim_val:new_dims[d].index(dim_val))

      for _,row in model_map_with_dims.iterrows():
          coords = [row[d.decode()] for d in dest_map_dims]
          dest_map[*coords] = nodes_to_keep[mod].index(row._run_idx)
          assert dest_map[*coords] == nodes_to_keep[mod].index(row._run_idx)

      map_var = grp.create_dataset('map',data=dest_map)
      map_var.attrs['DIMS']=dest_map_dims

  mod_order =[m.decode() for m in new_mod['META']['models'][...]]
  descriptions = {mod:getattr(node_types,mod).description for mod in mod_order}

  logger.info('Creating LINKS table')
  new_link_vals = new_links.sort_values('src_generation').copy()
  for grp in ['src','dest']:
      mod_col = f'{grp}_model'
      var_col = f'{grp}_var'
      flux_type = 'Inputs' if grp=='dest' else 'Outputs'
      new_link_vals[var_col] = new_link_vals.apply(lambda row: descriptions[row[mod_col]][flux_type].index(row[var_col]),axis=1)
      new_link_vals[mod_col] = new_link_vals[mod_col].apply(lambda m: mod_order.index(m))

  new_mod.create_dataset('LINKS',data=np.array(new_link_vals,dtype=np.uint32))

  new_mod.close()
