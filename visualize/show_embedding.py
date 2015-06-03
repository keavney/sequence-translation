import sys
from sklearn.manifold import *
from matplotlib.pyplot import *

fname = sys.argv[1]
algo = 'mds'
if len(sys.argv) > 2:
    algo = sys.argv[2].lower()

with open(fname) as f:
    a = [x.strip() for x in f.readlines()]
b = [x.split(' ') for x in a]
c = b[1:]

labels = [x[0] for x in c]
data = [map(float, x[1:]) for x in c]

proj = None
if algo == 'mds':
    m = MDS(); proj = m.fit_transform(data)
if algo == 'lle':
    proj, err = locally_linear_embedding(data, 5, 2)

def vis(D):
    X = [x[0] for x in D]
    Y = [y[1] for y in D]
    scatter(X, Y)
    p = zip(labels, X, Y)
    for l,x,y in p:
        print l, x, y
        annotate(l,xy=(x,y))
    show()
 
if proj is not None:
    vis(proj)
else:
    print 'invalid algorithm specified'

