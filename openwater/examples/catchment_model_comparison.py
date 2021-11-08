import pandas as pd
import numpy as np
from scipy import stats
import os
from .const import *

def sum_squares(l1,l2):
    return sum((np.array(l1)-np.array(l2))**2)

class SourceImplementation(object):
    def __init__(self, directory):
        self.directory = directory

    def _load_csv(self, f):
        return pd.read_csv(os.path.join(self.directory, f.replace(' ', '')+'.csv'), index_col=0, parse_dates=True)


class ImplementationComparison(object):
    def __init__(self, expected, actual):
        self.expected = expected
        self.actual = actual
        self.results = {}
        self.comparison = SourceOWComparison(actual.meta,
                                             actual,
                                             actual.results,
                                             expected.directory,
                                             actual.catchments)

def run(self):
        pass

class SourceOWComparison(object):
    def __init__(self,meta,ow_process_map,ow_results,source_results_path,catchments):
        self.meta = meta
        # self.ow_model_fn = ow_model_fn
        # self.ow_results_fn = ow_results_fn
        self.source_results_path = source_results_path
        self.time_period = pd.date_range(self.meta['start'],self.meta['end'])
        self.catchments = catchments
        self.process_map = ow_process_map
        self.results = ow_results

        self.comparison_flows = None
        self.link_outflow = None
        self.source_timeseries_cache = {}
        self.ow_timeseries_cache = {}

    def _load_csv(self,f):
        # fn = os.path.join(self.source_results_path,f.replace(' ','')+'.csv')
        fn = os.path.join(self.source_results_path,f+'.csv')
        if not os.path.exists(fn):
            fn = fn + '.gz'
        return pd.read_csv(fn,index_col=0,parse_dates=True)

    def _load_flows(self):
        if self.comparison_flows is not None:
            return

        routing = 'StorageRouting'
        self.link_outflow = self.results.time_series(routing,'outflow','catchment')
        self.comparison_flows = self.get_source_timeseries('downstream_flow_volume')
        self.comparison_flows = self.comparison_flows.rename(columns={c:c.replace('link for catchment ','') for c in self.comparison_flows.columns})
        self.comparison_flows = self.comparison_flows * PER_DAY_TO_PER_SECOND

    def plot_flows(self,sc,time_period=None):
        import matplotlib.pyplot as plt
        orig, ow = self.comparable_flows(sc,time_period)
        plt.figure()
        orig.plot(label='orig')
        ow.plot(label='ow')
        plt.legend()

        plt.figure()
        plt.scatter(orig,ow)

    def get_source_timeseries(self,fn):
        if not fn in self.source_timeseries_cache:
            self.source_timeseries_cache[fn] = self._load_csv('Results/%s'%fn).reindex(self.time_period)
        return self.source_timeseries_cache[fn]

    def get_generation(self,constituent,catchment,fu):
        from_source = self.get_source_timeseries('%sgeneration'%constituent)

        model,output = self.process_map.generation_model(constituent,fu)
        # print('OW results in %s.%s'%(model,output))
        from_ow = self.get_ow_timeseries(model,output,'catchment',cgu=fu,constituent=constituent)

        comparison_column = '%s: %s'%(fu,catchment)
        return from_source[comparison_column], from_ow[catchment]

    def compare_fu_level_results(self,elements,s_pattern,ow_fn,tag,progress=print):
        errors = []
        for e in elements:
            progress(e)
            comparison = self.get_source_timeseries(s_pattern%e)

            for fu in self.meta['fus']:
                # model,output = ow_fn(e,fu)
                # ts_tags = {
                #     'cgu':fu
                # }
                # ts_tags[tag]=e
                # ow = self.results.time_series(model,output,'catchment',**ts_tags)
                ow = ow_fn(e,fu)
                ow_columns = [c for c in ow.columns if not c=='dummy-catchment']
                comparison_columns = ['%s: %s'%(fu,catchment) for catchment in ow_columns ]
                try:
                    fu_comparison = comparison[comparison_columns]
                except:
                    print(comparison.columns)
                    print(comparison.index)
                    raise
                if ow.sum().sum()==0 and fu_comparison.sum().sum()==0:
                    for sc in ow_columns:
                        errors.append({'catchment':sc,'cgu':fu,tag:e,'ssquares':0,
                                       'sum-ow':0,'sum-orig':0,'r-squared':1,
                                       'delta':0})
                else:
                    for sc in ow_columns:
                        res = {'catchment':sc,'cgu':fu,tag:e,'ssquares':0,'r-squared':1,'sum-ow':0,'sum-orig':0,'delta':0}
                        ow_sc = ow[sc]
                        orig_sc = fu_comparison['%s: %s'%(fu,sc)]
                        if ow_sc.sum()>0 or orig_sc.sum()>0:
                            orig_scaled = (orig_sc*PER_SECOND_TO_PER_DAY)
                            ow_scaled = (ow_sc*PER_SECOND_TO_PER_DAY)
                            res['ssquares'] = sum_squares(orig_scaled,ow_scaled)
                            _,_,r_value,_,_ = stats.linregress(orig_scaled,ow_scaled)
                            res['r-squared'] = r_value**2
                            res['sum-ow'] = ow_scaled.sum()
                            res['sum-orig'] = orig_scaled.sum()
                            res['delta'] = res['sum-orig'] - res['sum-ow']
                        errors.append(res)
        return pd.DataFrame(errors)

    def get_paired_generation(self,catchment,fu,constituent):
        source = self.get_source_timeseries('%sgeneration'%constituent)['%s: %s'%(fu,catchment)]
        ow = self.get_ow_gen(constituent,fu)[catchment]
        return source, ow

    def get_ow_gen(self,c,fu):
        mod,flux = self.process_map.generation_model(c,fu)
        return self.results.time_series(mod,flux,'catchment',cgu=fu,constituent=c)

    def compare_constituent_generation(self,constituents=None,progress=print):
        if constituents is None:
            constituents = self.meta['constituents']

        def get_gen(c,fu):
            return self.get_ow_gen(c,fu)

        return self.compare_fu_level_results(constituents,'%sgeneration',get_gen,'constituent',progress)

    def get_ow_runoff(self,c,fu):
        if c=='Slow_Flow':
            c = 'Baseflow'
        elif c=='Quick_Flow':
            c = 'Quickflow'
        else:
            c = 'Runoff'
        return self.get_ow_timeseries('DepthToRate','outflow','catchment',cgu=fu,component=c)

    def get_paired_runoff(self,catchment,fu,component):
        source = self.get_source_timeseries(component)['%s: %s'%(fu,catchment)]
        ow = self.get_ow_runoff(component,fu)[catchment]
        return source, ow

    def compare_runoff(self,progress=print):
        def get_runoff(c,fu):
            return self.get_ow_runoff(c,fu)

        return self.compare_fu_level_results(['Slow_Flow','Quick_Flow','Total_Flow'],
                                             '%s',
                                             get_runoff,
                                             'component',
                                             progress)

    def comparable_flows(self,sc,time_period=None):
        self._load_flows()

        if not sc in self.comparison_flows.columns or not sc in self.link_outflow.columns:
            return None,None

        orig = self.comparison_flows[sc]
        ow = self.link_outflow[sc]
        common = orig.index.intersection(ow.index)
        orig = orig[common]
        ow = ow[common]
        if time_period is not None:
            return orig[time_period],ow[time_period]
        return orig,ow

    def compare_flow(self,sc):
        orig, ow = self.comparable_flows(sc)
        if orig is None or ow is None:
            print('Problem in %s'%sc)
            return {}#np.nan

        _,_,r_value,_,_ = stats.linregress(orig,ow)
        res = {
            'r-squared':r_value**2,
            'ssquares': sum_squares(orig,ow),
            'sum-ow': ow.sum(),
            'sum-orig': orig.sum()
        }

        res['delta'] = res['sum-orig'] - res['sum-ow']
        return res

    def compare_flows(self):
        self._load_flows()
        columns = self.link_outflow.columns
        return pd.DataFrame([self.compare_flow(c) for c in columns],index=columns)

    def get_ow_timeseries(self,*args,**kwargs):
        cache_key = tuple([args] + list(kwargs.items()))
        if cache_key not in self.ow_timeseries_cache:
            self.ow_timeseries_cache[cache_key] = self.results.time_series(*args,**kwargs)
        return self.ow_timeseries_cache[cache_key]

    def get_routed_constituent(self,constituent,catchment):
        from_source = self.get_source_timeseries('%snetwork'%constituent)

        model,output = self.process_map.transport_model(constituent)
        # print('OW results in %s.%s'%(model,output))
        from_ow = self.get_ow_timeseries(model,output,'catchment',constituent=constituent)

        comparison_column = 'link for catchment %s'%catchment
        print(from_source.columns)
        return (from_source[comparison_column]*PER_DAY_TO_PER_SECOND), from_ow[catchment]

    def compare_constituent_transport(self,constituents=None,progress=print):
        if constituents is None:
            constituents = self.meta['constituents']

        SOURCE_COL_PREFIX='link for catchment '
        errors = []
        for c in constituents:
            progress(c)
            comparison = self.get_source_timeseries('%snetwork'%c)
            comparison = comparison[[catchment for catchment in comparison.columns if catchment.startswith(SOURCE_COL_PREFIX)]]
            comparison = comparison.rename(columns={catchment:catchment.replace(SOURCE_COL_PREFIX,'') for catchment in comparison.columns})
            comparison = comparison * PER_DAY_TO_PER_SECOND
            model,output = self.process_map.transport_model(c)
            if 'constituent' in self.results.dims_for_model(model):
                ow = self.get_ow_timeseries(model,output,'catchment',constituent=c)
            else:
                ow = self.get_ow_timeseries(model,output,'catchment')
            # progress(comparison.columns)
            # progress(ow.columns)
            for sc in ow.columns:
                if not sc in comparison:
                    continue

                res = {'catchment':sc,'constituent':c,'r-squared':1,'ssquares':0,'sum-ow':0,'sum-orig':0,'delta':0}
                ow_sc = ow[sc]
                orig_sc = comparison[sc]
                if ow_sc.sum()>0 or orig_sc.sum()>0:
                    orig_scaled = (orig_sc*PER_SECOND_TO_PER_DAY)
                    ow_scaled = (ow_sc*PER_SECOND_TO_PER_DAY)
                    _,_,r_value,_,_ = stats.linregress(orig_scaled,ow_scaled)
                    res['r-squared'] = r_value**2
                    res['ssquares'] = sum_squares(orig_scaled,ow_scaled)
                    res['sum-ow'] = ow_scaled.sum()
                    res['sum-orig'] = orig_scaled.sum()
                    res['delta'] = res['sum-orig'] - res['sum-ow']
                errors.append(res)
        #    break
        return pd.DataFrame(errors)
