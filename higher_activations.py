import theano
import theano.tensor as T

def softmaxN(x, axis=-1):
    '''
        Applies softmax over the specified axis to a higher-dimensional tensor.
    '''
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out

