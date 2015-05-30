import sys
import datetime

firststamp = datetime.datetime.now()
laststamp = firststamp - firststamp

def log(msg, p=True):
    if not p:
        return
    global laststamp
    dn = datetime.datetime.now() - firststamp
    ts = datetime.datetime.isoformat(datetime.datetime.now())
    frame = sys._getframe().f_back
    while frame.f_code.co_name == '<lambda>':
        frame = frame.f_back
    caller = frame.f_code.co_name
    print ts, "[%s]" % (dn), "%s:" % caller, msg, "(+%s)" % (dn-laststamp)
    laststamp = dn

