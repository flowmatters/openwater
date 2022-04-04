import pandas as pd
import numpy as np
import json
import os
from scipy import stats
from glob import glob
from itertools import product
from multiprocessing import Pool
from .const import *
from datetime import datetime
from openwater.discovery import set_exe_path, discover
from openwater.timing import init_timer, report_time, close_timer
from openwater.template import ModelFile
from .catchment_model_results import OpenwaterCatchmentModelResults
import veneer
import logging

COMPONENTS=[
  'Generation','Routing','Transport','Quick_Flow','Slow_Flow','Total_Flow'
]
FACTORS = ['model', 'component', 'catchment', 'cgu', 'constituent','regulated']
R_SQD_THRESHOLD=0.99
SSQ_THRESHOLD=1e10 # 1e2
PC_ERR_THRESHOLD=1e-1 # 0.1%
SUM_THRESHOLD=2.5e-2*28*365

def sum_squares(l1,l2):
    return sum((np.array(l1)-np.array(l2))**2)

class SourceImplementation(object):
    def __init__(self, directory):
      self.path = directory
      self.source_timeseries_cache = {}
      self.link_outflow = None
      self.comparison_flows = None
      self.time_period = None

    def _load_csv(self,f):
        fn = os.path.join(self.path,f+'.csv')
        if not os.path.exists(fn):
            fn = fn + '.gz'
        return pd.read_csv(fn,index_col=0,parse_dates=True)

    def get_source_timeseries(self,fn):
        if not fn in self.source_timeseries_cache:
            self.source_timeseries_cache[fn] = self._load_csv('Results/%s'%fn).reindex(self.time_period)
        return self.source_timeseries_cache[fn]

    def _load_flows(self):
        if self.comparison_flows is not None:
            return

        self.comparison_flows = self.get_source_timeseries('downstream_flow_volume')
        self.comparison_flows = self.comparison_flows.rename(columns={c:c.replace('link for catchment ','') for c in self.comparison_flows.columns})
        self.comparison_flows = self.comparison_flows * PER_DAY_TO_PER_SECOND

    def flows(self,sc):
      self._load_flows()
      if not sc in self.comparison_flows:
        logging.warn('No comparison flows for %s. Link may not conform to naming convention'%sc)
        return None

      return self.comparison_flows[sc]

class SourceVeneerImplementation(object):
  def __init__(self,url):
    self.path = url
    from veneer import from_url
    self.veneer = from_url(url)
    self.run_data = veneer.expand_run_results_metadata(
      self.veneer.retrieve_run(),self.veneer.network())
    self.dataframe_cache = {}

  def get_source_timeseries(self,fn):
    if not fn in self.dataframe_cache:
      self.dataframe_cache[fn] = self._load(fn)
    return self.dataframe_cache[fn]

  def _load(self,fn):
    full_fn = os.path.join('tmp-results',f'{fn}.csv')
    if not os.path.exists(full_fn):
      full_fn += '.gz'
    if os.path.exists(full_fn):
      print(f'Loading pre-formatted data {full_fn}')
      return pd.read_csv(full_fn,index_col=0,parse_dates=True)

    print(f'Retrieving data from source results for {fn}')
    constituent_markers = ['generation','network']
    recording_variables = {
      'generation':'Constituents@%s@Total Flow Mass',
      'network':'Constituents@%s@Downstream Flow Mass'
    }
    name_functions = {
      'generation':veneer.name_for_fu_and_sc,
      'network':veneer.name_for_location
    }

    constituent = None
    for marker in constituent_markers:
      if fn.endswith(marker):
        constituent = fn.replace(marker,'')
        break

    if constituent is not None:
      criteria = {
        'RecordingVariable':recording_variables[marker]%constituent,
        'RecordingElement':'Constituents'
      }
      name_fn = name_functions[marker]
    elif fn.endswith('_Flow'):
      flow_component = fn.replace('_',' ')
      criteria = {
        'RecordingVariable':flow_component,
        'RecordingElement':flow_component
      }
      name_fn = veneer.name_for_fu_and_sc
    else:
      raise Exception('not implemented')

    data_frame = self.veneer.retrieve_multiple_time_series(run_data=self.run_data,
                                                           criteria=criteria,
                                                           name_fn=name_fn)
    if not os.path.exists('tmp-results'): os.makedirs('tmp-results')
    data_frame.to_csv(full_fn)
    return data_frame

  def _load_flows(self):
    fn = 'link_flow'
    if fn in self.dataframe_cache:
      return self.dataframe_cache[fn]

    full_fn = os.path.join('tmp-results',f'{fn}.csv')
    if not os.path.exists(full_fn):
      full_fn += '.gz'
    if os.path.exists(full_fn):
      print(f'Loading pre-formatted data {full_fn}')
      df = pd.read_csv(full_fn,index_col=0,parse_dates=True)
    else:
      criteria = {
        'RecordingElement':'Downstream Flow Volume',
        'RecordingVariable':'Downstream Flow Volume',
        'feature_type':'link'
      }
      df = self.veneer.retrieve_multiple_time_series(
        run_data=self.run_data,
        criteria=criteria,
        name_fn=veneer.name_for_location)
      df = df.rename(columns=lambda c:c.replace('Link for catchment ',''))
      timestep = (df.index[1]-df.index[0]).total_seconds()
      df /= timestep
      df.to_csv(full_fn)
    self.dataframe_cache[fn] = df
    return df

  def flows(self,sc):
    flows = self._load_flows()
    if sc in flows.columns:
      return flows[sc]
    return None

class ImplementationComparison(object):
    def __init__(self, expected, actual):
        self.expected = expected
        self.actual = actual
        self.results = {}
        self.comparison = SourceOWComparison(actual.meta,
                                             actual,
                                             actual.results,
                                             expected,
                                             actual.catchments)

def run(self):
        pass

class SourceOWComparison(object):
    def __init__(self,meta,ow_process_map,ow_results,source_results,catchments):
        self.meta = meta
        self.source_results = source_results
        self.time_period = pd.date_range(self.meta['start'],self.meta['end'],freq=self.meta.get('timestep',None))
        self.catchments = catchments
        self.process_map = ow_process_map
        self.results = ow_results

        self.ow_timeseries_cache = {}
        self.link_outflow = None
    def plot_flows(self,sc,time_period=None):
        import matplotlib.pyplot as plt
        orig, ow = self.comparable_flows(sc,time_period)
        plt.figure()
        orig.plot(label='orig')
        ow.plot(label='ow')
        plt.legend()

        plt.figure()
        plt.scatter(orig,ow)

    def get_generation(self,constituent,catchment,fu):
        from_source = self.source_results.get_source_timeseries('%sgeneration'%constituent)
        comparison_column = '%s: %s'%(fu,catchment)

        model,output = self.process_map.generation_model(constituent,fu)
        if model is None:
          from_ow = from_source[comparison_column]*0.0
        else:
          from_ow = self.get_ow_timeseries(model,output,'catchment',cgu=fu,constituent=constituent)
          from_ow = from_ow[catchment]

        return from_source[comparison_column], from_ow

    def compare_fu_level_results(self,elements,s_pattern,ow_fn,tag,progress=print):
        errors = []
        for e in elements:
            progress(e)
            comparison = self.source_results.get_source_timeseries(s_pattern%e)

            for fu in self.meta['fus']:
                progress(f'- {fu}')
                ow = ow_fn(e,fu)
                if ow is None: # No results from openwater - fill as 0
                  ow_columns = list(set([c.split(': ')[-1] for c in comparison.columns]))
                  comparison_columns = [f'{fu}: {c}' for c in ow_columns]
                  ow = comparison[comparison_columns] * 0.0
                  ow = ow.rename(columns=lambda c:c.split(': ')[-1])
                else:
                  ow_columns = [c for c in ow.columns if not c=='dummy-catchment']
                  comparison_columns = ['%s: %s'%(fu,catchment) for catchment in ow_columns ]

                try:
                    fu_comparison = comparison[comparison_columns]
                except:
                    print('Looking for',comparison_columns)
                    print('Have',comparison.columns)
                    print('Index',comparison.index)
                    raise
                if ((ow is None) or (ow.sum().sum()==0)) and fu_comparison.sum().sum()==0:
                    for sc in ow_columns:
                        errors.append({'catchment':sc,'cgu':fu,tag:e,'ssquares':0,
                                       'sum-ow':0,'sum-orig':0,'r-squared':1,
                                       'delta':0})
                else:
                    for sc in ow_columns:
                        res = {'catchment':sc,'cgu':fu,tag:e,'ssquares':0,'r-squared':1,'sum-ow':0,'sum-orig':0,'delta':0}
                        ow_sc = ow[sc]
                        orig_sc = fu_comparison['%s: %s'%(fu,sc)]
                        ow_sc, orig_sc = common_period(ow_sc,orig_sc)
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
        source = self.source_results.get_source_timeseries('%sgeneration'%constituent)['%s: %s'%(fu,catchment)]
        ow = self.get_ow_gen(constituent,fu)[catchment]
        return source, ow

    def get_ow_gen(self,c,fu):
        mod,flux = self.process_map.generation_model(c,fu)
        if mod is None:
          return None
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
        source = self.source_results.get_source_timeseries(component)['%s: %s'%(fu,catchment)]
        ow = self.get_ow_runoff(component,fu)[catchment]
        return source, ow

    def compare_runoff(self,progress=print):
        def get_runoff(c,fu):
            return self.get_ow_runoff(c,fu)

        return self.compare_fu_level_results(['Total_Flow','Slow_Flow','Quick_Flow'],
                                             '%s',
                                             get_runoff,
                                             'component',
                                             progress)

    def comparable_flows(self,sc,time_period=None):
        self._load_flows()
        orig = self.source_results.flows(sc)
        if orig is None or (sc not in self.link_outflow.columns):
            return None,None

        ow = self.link_outflow[sc]
        orig, ow = common_period(orig,ow)
        if time_period is not None:
            return orig[time_period],ow[time_period]
        return orig,ow

    def compare_flow(self,sc):
        orig, ow = self.comparable_flows(sc)
        error = False
        if orig is None:
            print('No flows for %s in source implementation'%sc)
            error = True
        if ow is None:
            print('No flows for %s in openwater implementation'%sc)
            error = True
        if error:
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

    def _load_flows(self):
      self.source_results._load_flows()
      routing = 'StorageRouting'
      if self.link_outflow is None:
        self.link_outflow = self.results.time_series(routing,'outflow','catchment')

    def compare_flows(self):
        self._load_flows()
        columns = [c for c in self.link_outflow.columns if c != 'dummy-catchment']
        return pd.DataFrame([self.compare_flow(c) for c in columns],index=columns)

    def get_ow_timeseries(self,*args,**kwargs):
        cache_key = tuple([args] + list(kwargs.items()))
        if cache_key not in self.ow_timeseries_cache:
            self.ow_timeseries_cache[cache_key] = self.results.time_series(*args,**kwargs)
        return self.ow_timeseries_cache[cache_key]

    def get_routed_constituent(self,constituent,catchment):
        from_source = self.source_results.get_source_timeseries('%snetwork'%constituent)

        model,output = self.process_map.transport_model(constituent)
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
            comparison = self.source_results.get_source_timeseries('%snetwork'%c)
            comparison = comparison[[catchment for catchment in comparison.columns if catchment.startswith(SOURCE_COL_PREFIX)]]
            comparison = comparison.rename(columns={catchment:catchment.replace(SOURCE_COL_PREFIX,'') for catchment in comparison.columns})
            comparison = comparison * PER_DAY_TO_PER_SECOND
            model,output = self.process_map.transport_model(c)
            if 'constituent' in self.results.dims_for_model(model):
                ow = self.get_ow_timeseries(model,output,'catchment',constituent=c)
            else:
                ow = self.get_ow_timeseries(model,output,'catchment')
            for sc in ow.columns:
                if not sc in comparison:
                    continue

                res = {'catchment':sc,'constituent':c,'r-squared':1,'ssquares':0,'sum-ow':0,'sum-orig':0,'delta':0}
                ow_sc = ow[sc]
                orig_sc = comparison[sc]
                ow_sc, orig_sc = common_period(ow_sc,orig_sc)
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
        return pd.DataFrame(errors)

def constituents_for_model(m):
  fn = f'{m}.meta.json'
  if os.path.exists(fn):
    with open(fn) as f:
      meta = json.load(f)
      return meta['constituents']

  fn = f'{m}.h5'
  with ModelFile(fn) as mf:
    return mf._dimensions['constituent']

def default_model_comparison(m,source_files,ow_dir,component=None,con=None):
  return model_comparison(OpenwaterCatchmentModelResults,m,source_files,ow_dir,component,con)

def check_factors(df,factors=FACTORS[1:]):
    for f in factors:
        if np.nan in set(df[f]):
            print('!!! NaN in set of %s values'%f)

def model_comparison(ow_results_class,m,source_files,ow_dir,component=None,con=None):
  ow_fn     = os.path.join(ow_dir,m)
  print(f'===== Comparing results for {m}/{component}/{con} from {source_files} and {ow_fn} =====')

  ow     = ow_results_class(ow_fn+'.h5')

  if '://' in source_files:
    source = SourceVeneerImplementation(source_files)
  else:
    source_fn = os.path.join(source_files,m)
    source = SourceImplementation(source_fn)
    source.time_period = ow.time_period

  test   = ImplementationComparison(source,ow)

  regulated_links = ow.regulated_links()
  data_frames = []
  if (component is None) or component == 'Runoff':
    print('- Runoff')
    runoff_comparison = test.comparison.compare_runoff().reset_index()
    runoff_comparison['constituent'] = '-'
    runoff_comparison['regulated'] = False
    check_factors(runoff_comparison)
    data_frames.append(runoff_comparison)

  if (component is None) or component == 'Routing':
    print('- Streamflow')
    flow_comparison = test.comparison.compare_flows()
    flow_comparison['component'] = 'Routing'
    flow_comparison['catchment'] = flow_comparison.index
    flow_comparison['constituent'] = '-'
    flow_comparison['cgu'] = '-'
    flow_comparison['regulated'] = flow_comparison['catchment'].isin(regulated_links)
    flow_comparison = flow_comparison.reset_index()
    check_factors(flow_comparison)
    data_frames.append(flow_comparison)

  if con is not None:
    cons = [con]
  else:
    cons = None

  if (component is None) or component == 'Generation':
    print('- Constituent generation')
    generation_comparison = test.comparison.compare_constituent_generation(cons).reset_index()
    generation_comparison['component'] = 'Generation'
    generation_comparison['regulated'] = False
    check_factors(generation_comparison)
    data_frames.append(generation_comparison)

  if (component is None) or component == 'Transport':
    print('- Constituent transport')
    transport_comparison = test.comparison.compare_constituent_transport(cons).reset_index()
    transport_comparison['component'] = 'Transport'
    transport_comparison['cgu'] = '-'
    transport_comparison['regulated'] = transport_comparison['catchment'].isin(regulated_links)
    check_factors(transport_comparison)
    data_frames.append(transport_comparison)

  print('- Combining results')
  res = pd.concat(data_frames)
  res['model'] = m
  res = res.reset_index()
  check_factors(res,FACTORS)
  res['regulated'] = np.where(res['regulated'],'Regulated','Not Regulated')

  res['pc-err'] = 100.0*np.where(res['sum-orig']==0.0,np.minimum(res['sum-ow'],1.0),np.abs(res['delta']/res['sum-orig']))
  print(f'===== Comparison {m}/{component}/{con} complete =====')
  return res

def compare_all(comparison_fn,models,source_files,ow_dir,processes=1,component=None):
    model_constituents = [constituents_for_model(os.path.join(ow_dir,m)) for m in models]
    model_constituent_combos = sum([list(product([mod],cons)) for (mod,cons) in zip(models,model_constituents)],[])
    water_quality_combos = [(mod,source_files,ow_dir,comp,con) for (mod,con),comp in product(model_constituent_combos,['Generation','Transport'])]

    water_quantity_combos = [(mod,source_files,ow_dir,comp,None) for (mod,comp) in product(models,['Runoff','Routing'])]
    combos = water_quality_combos + water_quantity_combos

    if component is not None:
      print(f'Only comparing component={component}')
      print(f'Before filter = {len(combos)}')
      combos = [c for c in combos if c[3]==component]
      print(f'After filter = {len(combos)}')

    if processes>1:
        with Pool(processes=processes) as pool:
            model_data = pool.starmap(comparison_fn,combos)
    else :
        model_data = []
        for combo in combos:
            model_data.append(comparison_fn(*combo))

    return pd.concat(model_data)

def common_period(*dataframes):
  common = dataframes[0].index
  for df in dataframes[1:]:
    common = common.intersection(df.index)
  return [df.loc[common] for df in dataframes]

def all_results_files(pattern='all-models'):
  return sorted(list(glob('model-comparison-summary-%s-2*.csv'%pattern)))

def read_results(fn):
  dtypes={
    'level_0':'i',
    'index':'str',
    'catchment':'str',
    'cgu':'string',
    'constituent':'string',
    'ssquares':'f',
    'sum-ow':'f',
    'sum-orig':'f',
    'r-squared':'f',
    'delta':'f',
    'component':'string',
    'regulated':'string',
    'model':'string',
    'pc-err':'f'
  }
  return pd.read_csv(fn,index_col=0,dtype=dtypes)

def latest_results(pattern='all-models'):
  latest = all_results_files(pattern)[-1]
  print('Reading from %s'%latest)
  return read_results(latest)

def get_problematic_rows(df,pc_err_threshold=PC_ERR_THRESHOLD,ssq_threshold=SSQ_THRESHOLD,r_sqd_threshold=0.99):
    df = df[(df['sum-orig']>=SUM_THRESHOLD)]
    return df[(df['r-squared']<r_sqd_threshold)|np.isnan(df['r-squared']) | (df['pc-err']>pc_err_threshold)|(df['ssquares']>ssq_threshold)]

def get_good_rows(df,pc_err_threshold=PC_ERR_THRESHOLD,ssq_threshold=SSQ_THRESHOLD,r_sqd_threshold=0.99):
    return df[(df['pc-err']<=pc_err_threshold)&(df['r-squared']>=r_sqd_threshold)|np.isnan(df['r-squared'])&(df['ssquares']<=ssq_threshold)]

def subset_data(df,**kwargs):
    for k,v in kwargs.items():
        df = df[df[k]==v]
    return df

def proportion_bad(df,factors=['model'],pc_err_threshold=PC_ERR_THRESHOLD,ssq_threshold=SSQ_THRESHOLD,r_sqd_threshold=0.99,**kwargs):
    df = subset_data(df,**kwargs)
    column_list = factors + ['index']
    bad_rows = get_problematic_rows(df,pc_err_threshold,ssq_threshold,r_sqd_threshold)[column_list]
    grouped = df[column_list].groupby(factors).count()
    bad_grouped = bad_rows.groupby(factors).count()
    result = grouped
    result['total'] = result['index']
    result['bad'] = bad_grouped['index']
    result = result.fillna(0)
    result['good'] = result['total'] - result['bad']
    result['pcGood'] = 100.0 * result['good'] / result['total']
    result['pcBad'] = 100.0 * result['bad'] / result['total']
    return result

def get_relevant_parameters(component_model,df,ow_models):
    model_files = list(set(df.model))
    all_joined = []
    for mf in model_files:
        rows = df[df.model==mf]
        model = ow_models[mf]
        dims = model.dims_for_model(component_model)
        params = model.parameters(component_model)
        if 'hru' in params:
            rows['hru'] = rows['cgu']
        joined = rows.merge(params,how='inner',on=dims)
        all_joined.append(joined)
    return pd.concat(all_joined)

def _arg_parser():
  from veneer import extract_config as ec
  parser = ec._base_arg_parser(model=False)
  parser.add_argument('-o','--openwater',help='Path to Openwater binaries',default=None)
  parser.add_argument('-r','--replay', help='Replay the results of hydrology from the Source implementation', action='store_true')
  parser.add_argument('--processes',type=int,help='Number of parallel processes',default=1)
  parser.add_argument('--veneer',default=None,help='Path to a Veneer instance (live or saved) for results')
  parser.add_argument('--converted',default='.',help='Path to converted (openwater) model files')
  parser.add_argument('--component',default=None,help=f'Only compare results for defined component ({",".join(COMPONENTS)})')
  parser.add_argument('models',nargs='+',help='Model names to compare')
  return parser

def compare_main(comparison_fn,models,**kwargs):
  now = datetime.now()

  set_exe_path(kwargs.get('openwater',os.getcwd()))
  discover()

  if not len(models):
    raise Exception('Need at least one model')
  else:
    init_timer('Model comparison for %s'%(','.join(models)))
    lbl = '-'.join(models)

  process_count = kwargs.get('processes',1)
  source_files = kwargs.get('veneer',None)
  if source_files is None:
    source_files = kwargs.get('extractedfiles','.')

  all_results = compare_all(comparison_fn=comparison_fn,
                            models=models,
                            processes=process_count,
                            source_files=source_files,
                            ow_dir=kwargs.get('converted','.'),
                            component=kwargs.get('component',None))
  all_results.to_csv('model-comparison-summary-%s-%s.csv'%(lbl,now.strftime('%Y%m%d-%H%M')))

if __name__=='__main__':
  import veneer.extract_config as ec
  args = ec._parsed_args(_arg_parser())
  compare_main(default_model_comparison,**args)
