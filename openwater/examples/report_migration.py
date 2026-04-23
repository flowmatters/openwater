import sys
from .catchment_model_comparison import all_results_files, latest_results, proportion_bad,read_results
from openwater.discovery import discover, set_exe_path
import pandas as pd
import logging
logger = logging.getLogger(__name__)


OW_BIN='/home/joelrahman/src/projects/openwater/bin'
DEFAULT_ELEMENTS=['component']
NEXT_ELEMENT={
  'Generation':['cgu','constituent'],
  'Total_Flow':['cgu'],
  'Routing':['regulated'],
  'Transport':['constituent','regulated']
}
COMPONENT_PRIORITY_ORDER=['Total_Flow','Generation','Routing','Transport']

def report_df(df,label,always=False):
  if not always:
    df = df[df.bad!=0]
  if not len(df) and not always:return
  print(f'\n\n================== {label} ==================')
  print(df)


def make_label(*args,**kwargs):
  lbl = ''
  if len(args):
    lbl = 'by ' + ','.join(args[0])
  if len(kwargs):
    lbl += ' where ' + ' and '.join(f'{k}={v}' for k,v in kwargs.items())
  return lbl


def main():
  pd.options.display.max_rows = None
  pd.options.display.max_columns = None
  pd.options.display.width = None

  set_exe_path(OW_BIN)
  _ = discover()

  if len(sys.argv) < 2:
    print('Usage: report_migration MODEL [ELEMENT ...] [--KEY=VALUE ...]')
    sys.exit(1)

  MODEL = sys.argv[1]
  user_elements = []
  user_filters = {}
  for arg in sys.argv[2:]:
    if arg.startswith('--'):
      if '=' not in arg:
        print(f'Invalid filter {arg!r}; expected --KEY=VALUE')
        sys.exit(1)
      k, v = arg[2:].split('=', 1)
      user_filters[k] = v
    else:
      user_elements.append(arg)
  logger.info(MODEL)

  all_results_fns = all_results_files(MODEL)
  LATEST_RESULTS_FN = all_results_fns[-1]
  have_reference = len(all_results_fns) >= 2
  if have_reference:
    REFERENCE_RESULTS_FN = all_results_fns[-2]
    logger.info(f'Comparing {LATEST_RESULTS_FN} to {REFERENCE_RESULTS_FN}')
    ref_results = read_results(REFERENCE_RESULTS_FN)
  else:
    logger.info(f'Only one results file available ({LATEST_RESULTS_FN}); reporting current state only.')
    ref_results = None

  all_results = latest_results(MODEL)

  def pb_delta(*args,**kwargs):
    from_all = proportion_bad(all_results,*args,**kwargs)
    from_ref = proportion_bad(ref_results,*args,**kwargs)
    return from_all - from_ref

  def report_delta(*args,**kwargs):
    if not have_reference:
      return
    report_df(pb_delta(*args,**kwargs),'Change ' + make_label(*args,**kwargs))

  def report(always=False,*args,**kwargs):
    report_df(proportion_bad(all_results,*args,**kwargs),'Proportion bad ' + make_label(*args,**kwargs),always=always)

  if user_elements or user_filters:
    elements = user_elements or DEFAULT_ELEMENTS
    report(True, elements, **user_filters)
    report_delta(elements, **user_filters)
    return

  if MODEL=='all-models':
    report(True,['model'])
    report_delta(['model'])

  report(True,DEFAULT_ELEMENTS)
  report_delta(DEFAULT_ELEMENTS)

  if MODEL=='all-models':
    report(True,['model']+DEFAULT_ELEMENTS)
    report_delta(['model']+DEFAULT_ELEMENTS)

  bad_results = proportion_bad(all_results,DEFAULT_ELEMENTS)
  bad_components = list(bad_results[bad_results.bad>0].index)
  for c in COMPONENT_PRIORITY_ORDER[::-1]:
    if not c in bad_components:
      continue
    report(False,['model']+NEXT_ELEMENT[c],component=c)
    report_delta(['model']+NEXT_ELEMENT[c],component=c)

    report(False,NEXT_ELEMENT[c],component=c)
    report_delta(NEXT_ELEMENT[c],component=c)


if __name__ == '__main__':
  main()
