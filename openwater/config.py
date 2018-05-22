from string import Template
import numpy as np

class Parameteriser(object):
    def __init__(self):
        self._parameterisers = []

    def parameterise(self,model_desc,grp,instances,dims,nodes):
        for p in self._parameterisers:
            p.parameterise(model_desc,grp,instances,dims,nodes)

class DataframeInput(object):
    def __init__(self,dataframe,column_format):
        self.df = dataframe
        if isinstance(column_format,str):
            column_format = Template(column_format)
        self.column_format = column_format

    def get_series(self,**kwargs):
        col_name = self.column_format.substitute(**kwargs)
        return np.array(self.df[col_name])
    
class DataframeInputs(object):
    def __init__(self):
        self._inputs = {}
    
    def inputter(self,df,input_name,col_format):
        self._inputs[input_name] = DataframeInput(df,col_format)

    def parameterise(self,model_desc,grp,instances,dims,nodes):
        print('==== Called for %s ===='%model_desc.name)
        description = model_desc.description
        inputs = description['Inputs']
        if not len(set(inputs).intersection(set(self._inputs.keys()))):
            print('No inputs configured')
            return

        print(inputs)
        print(grp)
        print(dims)
        print(list(nodes.items())[0])
        print(instances.shape)
        print()
        i = 0
        for node_name,node in nodes.items():
            run_idx = node['_run_idx']
            for input_num,input_name in enumerate(inputs):
                if not input_name in self._inputs:
                    continue
                data = self._inputs[input_name].get_series(**node)
                grp['inputs'][run_idx,input_num,:] = data

            if i%100 == 0:
                print('Processing %s'%node_name)
            i += 1

class ParameterTableAssignment(object):
    def __init__(self,df,column_dim,row_dim,model='DepthToRate',parameter='area'):
        self.df = df
        self.column_dim = column_dim
        self.row_dim = row_dim
        self.model = model
        self.parameter = parameter

    def parameterise(self,model_desc,grp,instances,dims,nodes):
        if model_desc.name != self.model:
            return
        param_idx = [i for i,p in enumerate(model_desc.description['Parameters']) if p['Name']==self.parameter][0]
        print(param_idx)
        print(model_desc.description)
        param_data = np.zeros(instances.size,dtype='float64')
        for _,node in nodes.items():
            run_idx = node['_run_idx']
            col = node[self.column_dim]
            row = node[self.row_dim]

            param = self.df[col][row]
            param_data[run_idx] = param

        print(param_data)
        grp['parameters'][param_idx,:]=param_data

class DefaultParameteriser(object):
    def __init__(self,model_name,**kwargs):
        self._model = model_name
        self._params = kwargs
    
    def parameterise(self,model_desc,grp,instances,dims,nodes):
        if model_desc.name != self._model:
            return
        
        print('Applying default parameters: %s'%self._model)
        print(model_desc.description['Parameters'])
        for param_num, param in enumerate(model_desc.description['Parameters']):
            pname = param['Name']
            pdefault = param['Default']
            grp['parameters'][param_num,:] = self._params.get(pname,pdefault)

class  UniformInput(object):
    def __init__(self,input_name,val):
        self.input_name = input_name
        self.value = val

    def parameterise(self,model_desc,grp,instances,dims,nodes):
      inputs = model_desc.description['Inputs']
      for input_num,input_name in enumerate(inputs):
          if input_name!=self.input_name:
            continue
          print('Uniform %s = %f'%(self.input_name,self.value))
          for cell in range(instances.size):
            if hasattr(self.value,'__call__'):
              grp['inputs'][cell,input_num,:] = self.value(cell)
            else:
              grp['inputs'][cell,input_num,:] = self.value