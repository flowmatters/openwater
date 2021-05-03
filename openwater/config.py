from string import Template
import numpy as np
import pandas as pd
import string
from .array_params import get_parameter_locations

def _models_match(configured,trial):
    if configured is None:
        return True

    if configured == trial:
        return True

    return configured == trial.name

def _locate_parameter_in_description(model_desc,parameter):
    matching_params = [i for i,p in enumerate(model_desc.description['Parameters']) if p['Name']==parameter]
    if len(matching_params):
        return 'parameters',matching_params[0],slice(None)

    if parameter in model_desc.description['States']:
        state_index = model_desc.description['States'].index(parameter)
        #[i for i,s in enumerate(model_desc.description['States']) if s['Name']==parameter]
        return 'states',slice(None),state_index

    raise Exception('Unknown parameter or state: %s'%parameter)

def _matches_constraints(constraint_tags,present_tags):
    for k,v in (constraint_tags or {}).items():
        if not k in present_tags:
            return False
        if kwargs[k] != v:
            return False
    return True

def initialise_model_inputs(model_grp,n_cells,n_inputs,n_timesteps):
    if not 'inputs' in model_grp:
        model_grp.create_dataset('inputs',shape=(n_cells,n_inputs,n_timesteps),dtype=np.float64,fillvalue=0)

class Parameteriser(object):
    def __init__(self):
        self._parameterisers = []

    def append(self,parameteriser):
        self._parameterisers.append(parameteriser)

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df):
        for p in self._parameterisers:
            p.parameterise(model_desc,grp,instances,dims,nodes,nodes_df)


class DataframeInput(object):
    def __init__(self,dataframe,column_format,model,constraint_tags):
        self.df = dataframe
        self.model = model
        self.constraint_tags = constraint_tags

        if isinstance(column_format,str):
            column_format = Template(column_format)
        self.column_format = column_format

    def applies(self,model):
        return _models_match(self.model,model)

    def get_series(self,**kwargs):
        # TODO Check against constraint tags
        if not _matches_constraints(self.constraint_tags,kwargs):
            return None

        col_name = self.column_format.substitute(**kwargs)
        if col_name in self.df.columns:
            return np.array(self.df[col_name])
        return None
    
class DataframeInputs(object):
    def __init__(self):
        self._inputs = {}
    
    def inputter(self,df,input_name,col_format,model=None,**kwargs):
        if not input_name in self._inputs:
            self._inputs[input_name] = []
        self._inputs[input_name].append(DataframeInput(df,col_format,model,kwargs))

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df):
        description = model_desc.description
        inputs = description['Inputs']
        if not len(set(inputs).intersection(set(self._inputs.keys()))):
            return

        print('==== Called for %s ===='%model_desc.name)
        print(inputs)
        print(grp)
        print(dims)
        print(list(nodes.items())[0])
        print(instances.shape)
        print()
        i = 0
        applied = 0
        for node_name,node in nodes.items():
            run_idx = node['_run_idx']
            for input_num,input_name in enumerate(inputs):
                if not input_name in self._inputs:
                    continue

                inputters = self._inputs[input_name]
                for inputter in inputters:
                    if not inputter.applies(model_desc):
                        continue
                    initialise_model_inputs(grp,len(nodes_df),len(inputs),len(self._inputs[input_name][0].df))

                    data = inputter.get_series(**node)
                    if data is None:
                        continue
                    applied += 1
                    grp['inputs'][run_idx,input_num,:] = data

            if i%100 == 0:
                print('Processing %s. Applied %d inputs'%(node_name,applied))
            i += 1

class SingleTimeseriesInput(object):
    def __init__(self,series,the_input,model=None,**tags):
        self.series = series
        self.model = model
        self.the_input = the_input
        self.tags = tags

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df):
        if not _models_match(self.model,model_desc):
            return

        description = model_desc.description
        inputs = description['Inputs']
        matching_inputs = [i for i,nm in enumerate(inputs) if nm==self.the_input]
        if len(matching_inputs)==0:
            return

        print('==== SingleTimeseriesInput(%s) called for %s ===='%(self.the_input,model_desc.name))
        input_num = matching_inputs[0]
        data = np.array(self.series)

        i = 0
        for node_name,node in nodes.items():
            applies_to_node = True
            for tag_name,tag_value in self.tags.items():
                if node[tag_name] != tag_value:
                    applies_to_node = False
                    break

            if not applies_to_node:
                continue

            run_idx = node['_run_idx']
            grp['inputs'][run_idx,input_num,:] = data

            if i%100 == 0:
                print('Processing %s'%node_name)
            i += 1

class ParameterTableAssignment(object):
    '''
    Parameterise OpenWater models from a DataFrame.


    '''
    def __init__(self,df,model,parameter=None,column_dim=None,row_dim=None,dim_columns=None,complete=True,skip_na=False):
        self.df = df
        self.column_dim = column_dim
        self.row_dim = row_dim
        self.model = model
        self.parameter = parameter
        self.dim_columns = dim_columns
        self.complete = complete
        self.skip_na = skip_na

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df):
        if not _models_match(self.model,model_desc):
            return

        print('Applying parameter table to %s'%model_desc.name)

        if None in [self.column_dim,self.row_dim,self.parameter is None]:
           self._parameterise_nd(model_desc,grp,instances,dims,nodes,nodes_df)
        else:
            self._parameterise_2d(model_desc,grp,instances,dims,nodes)

    def _parameterise_nd(self,model_desc,grp,instances,dims,nodes,nodes_df):
        names = [p['Name'] for p in model_desc.description['Parameters']] + model_desc.description['States']
        current_data = {}
        for p in names:
            if not p in self.df.columns:
                continue
            dest_grp,dest_idx0,dest_idx1 = self.locate(model_desc,p)
            existing = grp[dest_grp][dest_idx0,dest_idx1]
            try:
                assert len(existing.shape)==1
                assert existing.shape[0]>=len(nodes)
            except:
                print(dest_grp,dest_idx0,dest_idx1,p,existing.shape,len(nodes))
                raise
            current_data[p] = existing

        # param_data = {p:np.zeros(instances.size,dtype='float64') for i,p in enumerate(param_names) if p in self.df.columns}
        ignored = []
        # print('model',model_desc)
        # print('nodes_df',len(nodes_df),nodes_df.columns)
        # print('df',len(self.df),self.df.columns)
        join_keys = set(dims.keys()).intersection(set(self.df.columns))
        # print('join_keys',join_keys)
        joined = pd.merge(nodes_df,self.df,how='inner',on=list(join_keys))
        # print('joined',len(joined),joined.columns)

        indices = np.array(joined._run_idx)
        for p,arr in current_data.items():
            srs = joined[p]
            if self.skip_na:
                srs.fillna(0.0,inplace=True)
            arr[indices] = np.array(srs)

        # if True:
        #     raise Exception('BOO')

        # for _,node in nodes.items():
        #     # TODO: very slow. could we have a dataframe of all the nodes with tags and join them?
        #     subset = self.df
        #     for dim in dims.keys():
        #         if not dim in subset.columns:
        #             if not dim in ignored:
        #                 print('%s not specified in table, ignoring'%dim)
        #             ignored.append(dim)
        #             continue
        #         subset = subset[subset[dim]==node[dim]]

        #     if len(subset)==0 and not self.complete:
        #         continue

        #     if len(subset)>1:
        #         for dc in self.dim_columns:
        #             if not dc in subset.columns:
        #                 continue
        #             subset = subset[subset[dc]==node[dc]]

        #     if not len(subset)==1:
        #         print('=== Model: %s ==='%model_desc.name)
        #         print('=== Dims ===')
        #         print(list(dims.keys()))
        #         print('=== Node ===')
        #         print(node)
        #         print('=== Subset ===')
        #         print(subset)
        #         # print('=== Original ===')
        #         # print(self.df)
        #         assert len(subset)==1

        #     run_idx = node['_run_idx']
        #     for p,arr in current_data.items():
        #         val = subset[p]
        #         if self.skip_na:
        #             val.fillna(0.0,inplace=True)
        #         arr[run_idx] = val

        for p,vals in current_data.items():
            # if not p in param_data:
            #     print('--> No parameters for %s'%p)
            #     continue
            dest_grp, dest_idx0, dest_idx1 = self.locate(model_desc,p)
            print('Applying %s for %s'%(dest_grp,p))
            grp[dest_grp][dest_idx0,dest_idx1]=vals

    def _parameterise_2d(self,model_desc,grp,instances,dims,nodes):
        dest_grp, dest_idx0, dest_idx1 = self.locate(model_desc,self.parameter)
        print(dest_grp,dest_idx0,dest_idx1)

        param_data = np.zeros(len(nodes),dtype='float64')
        for _,node in nodes.items():
            run_idx = node['_run_idx']
            col = node[self.column_dim]
            row = node[self.row_dim]

            if col not in self.df.columns and self.skip_na:
                continue

            series = self.df[col]

            if row not in series.index and self.skip_na:
                continue

            param = series[row]
            if np.isnan(param) and self.skip_na:
                continue

            param_data[run_idx] = param

        print(param_data)
        grp[dest_grp][dest_idx0,dest_idx1]=param_data

    def locate(self,model_desc,parameter):
        return _locate_parameter_in_description(model_desc,parameter)

class DefaultParameteriser(object):
    def __init__(self,model_name=None,**kwargs):
        self._model = model_name
        self._params = kwargs

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df):
        if not _models_match(self._model,model_desc):
            return

        print('Applying default parameters: %s'%model_desc.name)
        for param_num, param in enumerate(model_desc.description['Parameters']):
            pname = param['Name']
            pdefault = param['Default']
            grp['parameters'][param_num,:] = self._params.get(pname,pdefault)

class UniformParameteriser(object):
    def __init__(self,model_name=None,**kwargs):
        self._model = model_name
        self._params = kwargs

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df):
        if not _models_match(self._model,model_desc):
            return

        print('Applying uniform parameters: %s'%model_desc.name)
        for param_num, param in enumerate(model_desc.description['Parameters']):
            pname = param['Name']
            if not pname in self._params:
                continue
            grp['parameters'][param_num,:] = self._params[pname]

class UniformInput(object):
    def __init__(self,input_name,val,length):
        self.input_name = input_name
        self.value = val
        self._length = length

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df):
      inputs = model_desc.description['Inputs']
      for input_num,input_name in enumerate(inputs):
          if input_name!=self.input_name:
            continue
          initialise_model_inputs(grp,len(nodes_df),len(inputs),self._length)
          print('Uniform %s = %f'%(self.input_name,self.value))
          for cell in range(len(nodes_df)):
            if hasattr(self.value,'__call__'):
              grp['inputs'][cell,input_num,:] = self.value(cell)
            else:
              grp['inputs'][cell,input_num,:] = self.value

class DictParameteriser(object):
    def __init__(self,parameter,key_format,model=None,parameters={},constraints={},**kwargs):
        self.parameter = parameter
        self.key_format = string.Template(key_format)
        self.model = model
        self.constraints = constraints
        self.parameters = parameters
        self.parameters.update(**kwargs)

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df):
        if not _models_match(self.model,model_desc):
            return

        dest_grp,dest_idx0,dest_idx1 = _locate_parameter_in_description(model_desc,self.parameter)
        for ix, row in nodes_df.iterrows():
            if not _matches_constraints(self.constraints,row):
                continue

            if dest_grp=='parameters':
                dest_idx1 = row._run_idx
            else:
                dest_idx0 = row._run_idx
            grp[dest_grp][dest_idx0,dest_idx1] = self.parameters[self.key_format.substitute(row)]

class DimensionParameterSizer(object):
    def __init__(self):
        pass

    def applies(self,desc):
        if not 'Dimensions' in desc:
            return False

        if not len(desc['Dimensions']):
            return False

        return True

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df):
        desc = model_desc.description
        if not self.applies(desc):
            return

        new_param_locs = get_parameter_locations(desc,grp['parameters'][...])
        len_params = new_param_locs[-1][1]

        grp.create_dataset('new_params',shape=(len_params,len(nodes_df)),dtype=np.float64,fillvalue=0)#,compression='gzip')
        grp['new_params'][0:grp['parameters'].shape[0],:] = grp['parameters'][:,:]
        grp.pop('parameters')
        grp.move('new_params','parameters')

class NestedParameteriser(object):
    def __init__(self,nested=[]):
        self.nested = nested[:]

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df):
        for np in self.nested:
            np.parameterise(model_desc,grp,instances,dims,nodes,nodes_df)

class CustomParameteriser(object):
    def __init__(self,fn,model=None,filter=None):
        self.model = model
        self.fn = fn
        self.filter = None

    def parameterise(self,model_desc,grp,instances,dims,nodes,nodes_df):
        if not _models_match(self.model,model_desc):
            return

        self.fn(model_desc,grp,instances,dims,nodes,nodes_df)
