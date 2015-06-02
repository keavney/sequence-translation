import sys
import datetime
import logging

glob = logging.getLogger('')
glob.setLevel(logging.DEBUG)

stat = logging.getLogger('stat')
stat.setLevel(logging.DEBUG)

info = logging.getLogger('info')
info.setLevel(logging.DEBUG)
ih = logging.StreamHandler(stream=sys.stdout)
ih.setLevel(logging.DEBUG)
info.addHandler(ih)

def log_to_file(logname, filename):
    try:
        h = logging.FileHandler(filename)
        h.setLevel(logging.DEBUG)
        l = globals()[logname]
        l.addHandler(h)
        info.info("logging {0} to {1}".format(logname, filename))
    except Exception:
        info.info("failed to log {0} to {1}".format(logname, filename))

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
    #print ts, "[%s]" % (dn), "%s:" % caller, msg, "(+%s)" % (dn-laststamp)
    s = "{0} [{1}] {2}: {3} (+{4})".format(ts, dn, caller, msg, (dn-laststamp))
    info.info(s)
    laststamp = dn

