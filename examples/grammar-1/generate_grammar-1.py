import sys
import random
import itertools

xfn = sys.argv[1]
yfn = sys.argv[2]
cats = int(sys.argv[3])
minl = int(sys.argv[4])
maxl = int(sys.argv[5])
mins = int(sys.argv[6])
maxs = int(sys.argv[7])

def make_grammar(cats, minl, maxl, mins, maxs):
    xa = []
    ya = []
    for sqlen in range(mins, maxs+1):
        for offsets in itertools.product(*([range(minl, maxl+1)]*sqlen)):
            xg = ""
            yg = ""
            for r in offsets:
                xg += 'A'*r + ' '
                yg += 'a '*r
                xg, yg = yg, xg
            xg += '.'
            yg += '.'
            xa.append(xg)
            xa.append(yg)
            ya.append(yg)
            ya.append(xg)

    return xa, ya

g_all = make_grammar(cats, minl, maxl, mins, maxs)
x_all, y_all = g_all
print "created {0} strings".format(len(x_all))

with open(xfn, 'w') as xn:
    xn.write("{0}\n".format(len(x_all)))
    for x in x_all:
        xn.write("{0}\n".format(x))
with open(yfn, 'w') as yn:
    yn.write("{0}\n".format(len(y_all)))
    for y in y_all:
        yn.write("{0}\n".format(y))
