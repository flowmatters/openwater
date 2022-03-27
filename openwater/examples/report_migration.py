import sys
from .catchment_model_comparison import all_results_files, latest_results, proportion_bad
from openwater.discovery import discover, set_exe_path
import pandas as pd

OW_BIN='/home/joelrahman/src/projects/openwater/bin'
DEFAULT_ELEMENTS=['component']
NEXT_ELEMENT={
  'Generation':['cgu','constituent'],
  'Total_Flow':['cgu'],
  'Routing':['regulated'],
  'Transport':['constituent','regulated']
}
COMPONENT_PRIORITY_ORDER=['Total_Flow','Generation','Routing','Transport']

def report_df(df,label):
  nz_bad = df[df.bad!=0]
  if not len(nz_bad):return
  print(f'\n\n================== {label} ==================')
  print(nz_bad)

if __name__ == '__main__':
  pd.options.display.max_rows = None
  pd.options.display.max_columns = None
  pd.options.display.width = None

  set_exe_path(OW_BIN)
  _ = discover()

  MODEL = sys.argv[1]
  print(MODEL)

  all_results = all_results_files(MODEL)
  REFERENCE_RESULTS_FN=all_results[-2]
  LATEST_RESULTS_FN=all_results[-1]
  print(f'Comparing {LATEST_RESULTS_FN} to {REFERENCE_RESULTS_FN}')

  ref_results = pd.read_csv(REFERENCE_RESULTS_FN,index_col=0)
  all_results = latest_results(MODEL)

  def pb_delta(*args,**kwargs):
    from_all = proportion_bad(all_results,*args,**kwargs)
    from_ref = proportion_bad(ref_results,*args,**kwargs)
    return from_all - from_ref

  def make_label(*args,**kwargs):
    lbl = ''
    if len(args):
      lbl = 'by ' + ','.join(args[0])
    if len(kwargs):
      lbl += ' where ' + ' and '.join(f'{k}={v}' for k,v in kwargs.items())
    return lbl

  def report_delta(*args,**kwargs):
    report_df(pb_delta(*args,**kwargs),'Change ' + make_label(*args,**kwargs))

  def report(*args,**kwargs):
    report_df(proportion_bad(all_results,*args,**kwargs),'Proportion bad ' + make_label(*args,**kwargs))


  report(DEFAULT_ELEMENTS)
  report_delta(DEFAULT_ELEMENTS)

  bad_results = proportion_bad(all_results,DEFAULT_ELEMENTS)
  bad_components = list(bad_results[bad_results.bad>0].index)
  for c in COMPONENT_PRIORITY_ORDER[::-1]:
    if not c in bad_components:
      continue
    constraint = {'component':c}
    report(NEXT_ELEMENT[c],component=c)
    report_delta(NEXT_ELEMENT[c],component=c)


