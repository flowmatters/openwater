
INDENT=2
INDENT_CHAR=' '
INDENT_SUFFIX=' * '

from datetime import datetime
class Timer(object):
    def __init__(self,label,parent=None):
        self.label = label
        self.parent = parent
        self.subtimers = []
        self.start = datetime.now()
        self.end = None
        self.elapsed = None
    
    def close(self):
        self.end = datetime.now()
        self.elapsed = (self.end - self.start).total_seconds()
        self.report()
    
    def report(self,indent=INDENT):
        indent_txt = INDENT_CHAR*indent + INDENT_SUFFIX
        if self.end is None:
            msg = ' not closed'
        else:
            msg =' %f seconds'%self.elapsed

        txt = indent_txt+self.label + msg
        print(indent_txt+self.label + msg)
        messages = []
        for st in self.subtimers:
            m = st.report(indent+INDENT)
            messages.append(m)
        return '\n'.join([txt] + messages)

root_timer = Timer('root timer')
current_timer = root_timer

def init_timer(label):
    global current_timer
    new_timer = Timer(label,parent=current_timer)
    current_timer.subtimers.append(new_timer)
    current_timer = new_timer

def report_time(msg):
    close_timer()
    init_timer(msg)
    # global t_last, timings
    # now = datetime.now()
    # d = now - t_last
    # sec = d.total_seconds()
    # txt = '##### %s: %s seconds'%(msg,sec)
    # print(txt)
    # timings.append(txt)
    # t_last = now

def close_timer():
    global current_timer
    current_timer.close()
    current_timer = current_timer.parent
