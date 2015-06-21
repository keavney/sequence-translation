import numpy

ftype = numpy.float32
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

def one_hot(size, index):
    a = numpy.zeros(size, dtype=ftype)
    a[index] = 1
    return a

def to_one_hotN(z, nb_class):
    b = numpy.zeros(list(z.shape) + [nb_class])
    b.flat[numpy.arange(0, b.size, nb_class) + z.flat] = 1
    return b

class LazyArray(object):
    '''
        Allows evaluation of massive strings containing arrays (hundreds of MB
        into the low GB range) with significantly reduced memory usage.

    '''

    def __init__(self, s, limit=None, verbose=False):
        '''
            limit: recursive limit of array index (default: infinite)

        '''
        self.array_str = s 
        self.index = self.build_index(limit=limit, verbose=verbose)

    def build_index(self, limit=None, verbose=False):
        '''
            Constructs an index from the stored string.
            Index nodes have the following form:

            (starting character, child ... , ending character)

        '''
        c = 0 
        stack = [[]]
        for i, s in enumerate(self.array_str):
            if s == '[':
                if c < limit or limit is None:
                    if verbose:
                        print ' '*c + '[ ' + str(i)
                    active = [i] 
                    stack[-1].append(active)
                    stack.append(active)
                c += 1
            if s == ']':
                c -= 1
                if c < limit or limit is None:
                    if verbose:
                        print ' '*c + '] ' + str(i)
                    stack[-1].append(i)
                    stack = stack[:-1]
        return stack[0][0]

    def get_indices(self, indices):
        '''
            Given a list of indices, returns the start and end (inclusive)
            character index of the stored string.

        '''
        res = self.index
        for i in indices:
            res = res[1:-1][i]
        return [res[0], res[-1]]

    def get(self, indices):
        '''
            Given a list of indices, eval()s the corresponding piece of
            the underlying string.

            The "shape" of the array expressed in the string is computed
            in build_index.
        
        '''
        start, end = self.get_indices(indices)
        return eval(self.array_str[start:end+1])

    def materialize(self):
        '''
            Evaluates the entire stored string by constructing it from the
            index in a DFS traversal.

        '''
        def _materialize(self, ar):
            if len(ar) > 2:
                return [_materialize(self, a) for a in ar[1:-1]]
            else:
                return eval(self.array_str[ar[0]: ar[-1]+1])
        return _materialize(self, self.index)


