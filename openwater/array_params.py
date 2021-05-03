
import numpy as np

def get_parameter_locations(desc,current):
    result = []

    dim_sizes = {}
    current_parameter_idx = 0

    for p in desc['Parameters']:
        p_start = current_parameter_idx
        size = 1
        if ('Dimensions' in p) and len(p['Dimensions']):
            for d in p['Dimensions']:
                size *= dim_sizes[d]
        size = int(size)

        if p['Name'] in desc['Dimensions']:
            dim_size = max(current[current_parameter_idx,:])
            dim_sizes[p['Name']] = dim_size

        current_parameter_idx += size
        p_end = current_parameter_idx
        result.append((p_start,p_end))
    return result

def param_starts(desc,current):
    param_locs = get_parameter_locations(desc,np.array(current))
    param_starts = {}
    current_idx=0
    for ix,param in enumerate(desc['Parameters']):
        p_start = param_locs[ix][0]
        param_starts[param['Name']]=p_start
    return param_starts

        