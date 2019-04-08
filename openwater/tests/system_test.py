import unittest
import json
import os
import io
from openwater import OWTemplate, OWLink, debugging, compute_simulation_order
import openwater.template as templating
import openwater.nodes as node_types
import openwater.config as config
import openwater.discovery as discovery
from openwater.config import *
from openwater.results import OpenwaterResults
from datetime import datetime

from numpy.testing import assert_array_equal
import math
import re
import numpy as np
import netCDF4 as nc
import h5py as h5
import pandas as pd

N_LANDUSES=3
N_LEVELS=8
N_TIMESTEPS=500
LU_SCALE = 10**math.ceil(math.log10(N_TIMESTEPS)) * 10
CATCHMENT_SCALE = 10**math.ceil(math.log10(N_LANDUSES)) * LU_SCALE * 10
N_CATCHMENTS = 2**N_LEVELS - 1
MODEL_FN='test_model.h5'
RESULTS_FN='test_model_outputs.h5'
discovery.set_exe_path(os.environ.get('OW_BIN',os.getcwd()))

def clean(fn):
    if os.path.exists(fn):
        os.remove(fn)
    assert not os.path.exists(fn)

def create_catchment_tree(n_levels,template):
    n_nodes = 2**n_levels - 1
    ids = list(range(1,n_nodes+1))
    labels = ['{0:b}'.format(i) for i in ids]
    nodes_to_create = list(zip(ids,labels))
    units = [(i,l,template) for i,l in nodes_to_create]
    return units

def build_catchment_graph(catchments):
    g = None
    for i,_,tpl in catchments:
        g = templating.template_to_graph(g,tpl,catchment=i)

    for i,label,_ in catchments:
        dest_label = label[:-1]
        if not len(dest_label):
            continue
        dest_catchment = int(dest_label,2)

        
        src_node =   [n for n in g.nodes if n.startswith('%d-transport'%i)][0]
        dest_node =  [n for n in g.nodes if n.startswith('%d-transport'%dest_catchment)][0]
        g.add_edge(src_node,dest_node,src=['out'],dest=['i2'])
    return g

class ScaledTimeSeriesParameteriser(object):
    def __init__(self,ts,the_input,model=None,**kwargs):
        self.ts = ts
        self.the_input = the_input
        self.model = model
        self.scales = kwargs
    
    def parameterise(self,model_desc,grp,instances,dims,nodes):
        if not config._models_match(self.model,model_desc):
            return

        description = model_desc.description
        inputs = description['Inputs']
        matching_inputs = [i for i,nm in enumerate(inputs) if nm==self.the_input]
        if len(matching_inputs)==0:
            return

        print('==== SingleTimeseriesInput(%s) called for %s ===='%(self.the_input,model_desc.name))
        input_num = matching_inputs[0]
        data = np.array(self.ts)

        i = 0
        for node_name,node in nodes.items():
            offset = 0
            for tag_name,scale_value in self.scales.items():
                offset += node[tag_name] * scale_value

            run_idx = node['_run_idx']
            grp['inputs'][run_idx,input_num,:] = data + offset

            if i%100 == 0:
                print('Processing %s'%node_name)
            i += 1

class TestOWSim(unittest.TestCase):
    def test_split(self):
        clean(MODEL_FN)
        clean(RESULTS_FN)

        discovery.discover()
        template = OWTemplate()
        link = template.add_node(node_types.Sum,process='transport')

        for lu in range(N_LANDUSES):
            n = template.add_node(node_types.RunoffCoefficient,process='runoff',lu=lu)
            template.add_link(OWLink(n,'runoff',link,'i1'))
        catchments = create_catchment_tree(N_LEVELS,template)
        g = build_catchment_graph(catchments)
        model = templating.ModelGraph(g)
        params = Parameteriser()
        coeffs = DefaultParameteriser(node_types.RunoffCoefficient,coeff=1.0)
        params.append(coeffs)
        base_rain = np.arange(N_TIMESTEPS,dtype='f')
        rainfall_input = ScaledTimeSeriesParameteriser(base_rain,'rainfall',node_types.RunoffCoefficient,
                                                    catchment=CATCHMENT_SCALE,lu=LU_SCALE)
        params.append(rainfall_input)
        model._parameteriser = params

        model.write_model(MODEL_FN,N_TIMESTEPS)
        os.system('%s -overwrite %s %s'%(os.path.join(discovery._exe_path('sim')),MODEL_FN,RESULTS_FN))
        res = OpenwaterResults(MODEL_FN,RESULTS_FN)
        models = res.models()
        self.assertIn('RunoffCoefficient', models)
        self.assertIn('Sum', models)
        runoff = res.time_series('RunoffCoefficient','runoff','catchment',lu=1)
        streamflow = res.time_series('Sum','out','catchment')
        def check_runoff(c,l=1):
            c_factor = c * CATCHMENT_SCALE
            l_factor = l * LU_SCALE
            values = np.array(runoff[c])
            orig_values = values - c_factor - l_factor
            assert_array_equal(orig_values,base_rain,'Runoff in catchment %d'%c)

        def expected_streamflow(c):
            factor = 0
            for l_scale in range(N_LANDUSES):
                factor += c * CATCHMENT_SCALE
                factor += l_scale * LU_SCALE
            result = N_LANDUSES * base_rain + factor
            
            N_CATCHMENTS = 2**N_LEVELS - 1
            inflows = [c*2,c*2+1]
            if inflows[0] <= N_CATCHMENTS:
                result += expected_streamflow(inflows[0]) + expected_streamflow(inflows[1])

            return result

        def check_streamflow(c):
            expected = expected_streamflow(c)
            assert_array_equal(streamflow[c],expected,'Streamflow in catchment %d'%c)

        for catchment in runoff.columns:
            check_runoff(catchment)

        for i in streamflow.columns:
            check_streamflow(i)

if __name__ == '__main__':
    unittest.main()
