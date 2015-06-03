# reads in from a new-style training log

import sys
from matplotlib.pyplot import *
from itertools import cycle

if len(sys.argv) < 4:
    print "usage: {0} metric(loss, etc) [epoch|time] logs...".format(sys.argv[0])
    exit()

colors = cycle(['r', 'b', 'g', 'k', 'c', 'm', 'y'])
sets = [('train', '-'), ('validate', '--')]
maxval = 1e10
m = 10000
convert = lambda x: min(float(x*m), maxval)
showmax = False
showmin = False

plotsets = []
metric = sys.argv[1]
xtype = sys.argv[2]

for arg in sys.argv[3:]:
    print 'reading', arg
    
    with open(arg) as f:
        text = f.read().replace('\n', '').replace('\r', '').strip()

    if text == '':
        print 'empty'
        continue

    dicts = eval(text)

    color = colors.next()
    for setname, line in sets:
        name = "{0}_{1}".format(arg, setname)
        if metric == 'loss':
            X = [d[xtype] for d in dicts]
            Y = [convert(d['sets'][setname]['loss']) for d in dicts]
        else:
            tag = 'avg_correct_pct'
            clause = lambda d: 'summary' in d['sets'][setname] and metric in d['sets'][setname]['summary'] and tag in d['sets'][setname]['summary'][metric]
            X = [d[xtype] for d in dicts if clause(d)]
            Y = [convert(d['sets'][setname]['summary'][metric][tag]) for d in dicts if clause(d)]
        linetype = color + line
        maxval = sorted(zip(X, Y), key=lambda (x, y): y, reverse=True)[0]
        minval = sorted(zip(X, Y), key=lambda (x, y): y, reverse=True)[-1]
        plotsets.append((name, X, Y, linetype, maxval))
        print "{0:15} {1:10} {2:4} ({3} points) max: {4} min: {5}".format(arg+':', setname, linetype, len(X), maxval, minval)

for name, X, Y, linetype, maxval in plotsets:
    plot(X, Y, linetype)
    if showmax:
        annotate("{0}: {1}".format(name, maxval), xy=maxval)

xlabel(xtype)
ylabel(metric)

show()