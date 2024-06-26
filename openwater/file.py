import pandas as pd
from . import nodes as node_types

def _tabulate_model_scalars_from_file(file,model,model_map,scalar_type,**tags):
    vals = file['MODELS'][model][scalar_type][...]
    desc = getattr(node_types,model)
    if scalar_type=='parameters':
        names = [p['Name'] for p in desc.description['Parameters']]
    else:
        names = desc.description['States']

    result = pd.DataFrame(model_map)

    order = list(result['_run_idx'])
    for ix,name in enumerate(names):
        if scalar_type=='parameters':
            sl1 = ix
            sl2 = slice(None)
        else:
            sl1 = slice(None)
            sl2 = ix
        result[name] = vals[sl1,sl2][order]

    # result = pd.DataFrame(vals.transpose(),columns=names)
    # for k,vals in dims.items():
    #     result = result[:len(vals)]
    #     result[k] = vals

    for k,v in tags.items():
        result = result[result[k]==v]

    return result
