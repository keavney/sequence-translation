import sys
import random

xfn = sys.argv[1]
yfn = sys.argv[2]
partitions = sys.argv[3:]

with open(xfn, 'r') as xf:
    x_raw = filter(lambda x: x != "", [r.strip() for r in xf.readlines()])
    if len(x_raw) == 0:
        print "empty x file"
        exit()
with open(yfn, 'r') as yf:
    y_raw = filter(lambda y: y != "", [r.strip() for r in yf.readlines()])
    if len(y_raw) == 0:
        print "empty y file"
        eyit()

# remove header, if present
if x_raw[0].split(' ')[0].isdigit() and len(filter(lambda x: x != "", x_raw[0].split(' '))) == 1:
    print 'removed x header: {0}'.format(x_raw[0])
    x_raw = x_raw[1:]
if y_raw[0].split(' ')[0].isdigit() and len(filter(lambda y: y != "", y_raw[0].split(' '))) == 1:
    print 'removed y header: {0}'.format(y_raw[0])
    y_raw = y_raw[1:]

print 'x samples: {0}'.format(len(x_raw))
print 'y samples: {0}'.format(len(y_raw))
msize = len(x_raw)

samples = zip(x_raw, y_raw)
random.shuffle(samples)

totalsize = 0
for size, xname, yname in zip(partitions[::3], partitions[1::3], partitions[2::3]):
    if size == 'r':
        size = len(samples)
    size = float(size)
    if size < 1:
        size = msize * size
    size = int(size)
    oldsize = size
    size = min(size, len(samples))
    print xname, yname, ':', size, '[!!]' if oldsize != size else ''
    totalsize += size
    with open(xname, 'w') as xf, open(yname, 'w') as yf:
        for s in samples[:size]:
            xf.write("{0}\n".format(s[0]))
            yf.write("{0}\n".format(s[1]))
    samples = samples[size:]

print 'total:', totalsize
print 'done'
