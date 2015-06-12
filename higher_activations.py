import theano
import theano.tensor as T

def softmaxN(x, axis=-1):
    '''
        Applies softmax over the specified axis to a higher-dimensional tensor.
    '''
    e = T.exp(x)
    return e / T.shape_padright(T.sum(e, axis=axis))

