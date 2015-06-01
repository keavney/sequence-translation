import numpy

epsilon = 1.0e-15

def numpy_softmax(x): 
    e = numpy.exp(x) # prevent overflow
    return e / numpy.sum(e, axis=0) 

def numpy_norm(x):
    v = numpy.linalg.norm(x)
    if abs(v) < epsilon:
        return v
    return x / v

def mean(x):
    return sum(x) / float(len(x))

def LN(X, Y, n=2):
    return numpy.linalg.norm(numpy.array(X)-numpy.array(Y), ord=n)

# like .index() but with a predicate
def pred_index(e, l):
    for i, li in enumerate(l):
        if e(li):
            return i
    return len(l)

# like .split() but with a predicate
def pred_split(e, l):
    a = []
    tail = 0
    for i, li in enumerate(l):
        if e(li):
            a.append(l[tail:i])
            tail = i+1
    a.append(l[tail:])
    return a

