import theano
import theano.tensor as T

import keras.optimizers as optimizers
import keras.objectives as objectives
from keras.utils.generic_utils import Progbar
from keras.models import *
from keras import activations, initializations
from keras.utils.theano_utils import shared_zeros, alloc_zeros_matrix, sharedX
from keras.layers.core import Layer

from itertools import izip, chain

import numpy

def create_masked_loss(objective):
    def masked_loss(y_true, y_pred, mask):
        y_diff = objective(y_true, y_pred)
        y_masked = y_diff * mask
        return y_masked.mean()
    return masked_loss

# Sequence with an additional dimension
class SequentialSequence(object):
    '''
        Sequential Sequence: Keras' Sequence model type isn't
        flexible with its input and output dimensions. This version
        adds one dimension to X (taking the entire sequence instead of
        a single item) and two dimensions to Y (taking sequences in higher-
        dimensional space instead of a scalar label)

    '''
    def __init__(self):
        self.layers = []
        self.params = []

    def get_weights(self):
        return [l.get_weights() for l in self.layers]

    def set_weights(self, weights):
        assert len(self.layers) == len(weights)
        for l, w in izip(self.layers, weights):
            l.set_weights(w)

    def add(self, layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])
        self.params += [p for p in layer.params]

    def compile(self, optimizer, loss, log_fcn=lambda x, y: (x, y),
            joint_model=False, skiplist = []):
        log = lambda x: log_fcn(x, True)

        log("Entering compile...")

        log("Compiling functions...")

        self.CH_layers = filter(lambda x: hasattr(x, 'C1') and hasattr(x, 'H1'), self.layers)

        self.optimizer = optimizers.get(optimizer)
        self.old_lr = self.optimizer.lr if 'lr' in dir(self.optimizer) else 0
        self.lr = T.scalar()
        self.optimizer.lr = self.lr
        #self.loss = objectives.get(loss)

        objective = objectives.get(loss)
        self.loss = create_masked_loss(objective)

        self.X = self.layers[0].input # input of model 
        self.Y = T.tensor3() # vector word labels
        self.M = T.tensor3() # mask

        self.y_train = self.layers[-1].output(train=True)[0]
        self.y_test = self.layers[-1].output(train=False)[0]

        print 'y_train', self.y_train

        self.train_loss = self.loss(self.Y, self.y_train, self.M)
        self.test_score = self.loss(self.Y, self.y_test, self.M)
        log("Getting updates...")
        self.updates = self.optimizer.get_updates(self.params, self.train_loss)

        self.train_result = self.y_train
        self.predict_result = self.y_test

        #if 'train' not in skiplist:
        #    log("Creating train function...")
        #    self._train = theano.function([self.X, self.Y, self.lr], self.train_loss, 
        #        updates=self.updates, allow_input_downcast=True)
        #if 'predict' not in skiplist:
        #    log("Creating predict function...")
        #    self._predict = theano.function([self.X], self.y_test, 
        #        allow_input_downcast=True)
        #if 'test' not in skiplist:
        #    log("Creating test function...")
        #    self._test = theano.function([self.X, self.Y], self.test_score, 
        #        allow_input_downcast=True)

        log("Done compiling functions")

    def train(self, X, y, lr):
        if lr is None:
            lr = self.old_lr
        y = standardize_y(y)
        loss = self._train(X, y, lr)
        return loss

    def test(self, X, y):
        y = standardize_y(y)
        score = self._test(X, y)
        return score

    def fit(self, X, y, batch_size=128, nb_epoch=100, verbose=1,
            validation_split=0., lr=None, shuffle=True):
        # If a validation split size is given (e.g. validation_split=0.2)
        # then split X into smaller X and X_val,
        # and split y into smaller y and y_val.
        y = standardize_y(y)

        do_validation = False
        if validation_split > 0 and validation_split < 1:
            do_validation = True
            split_at = int(len(X) * (1 - validation_split))
            (X, X_val) = (X[0:split_at], X[split_at:])
            (y, y_val) = (y[0:split_at], y[split_at:])
            if verbose:
                print "Train on %d samples, validate on %d samples" % (len(y), len(y_val))
        
        index_array = numpy.arange(len(X))
        for epoch in range(nb_epoch):
            if verbose:
                print 'Epoch', epoch
            if shuffle:
                numpy.random.shuffle(index_array)

            nb_batch = int(numpy.ceil(len(X)/float(batch_size)))
            progbar = Progbar(target=len(X))
            for batch_index in range(0, nb_batch):
                batch_start = batch_index*batch_size
                batch_end = min(len(X), (batch_index+1)*batch_size)
                batch_ids = index_array[batch_start:batch_end]

                X_batch = X[batch_ids]
                y_batch = y[batch_ids]
                loss = self._train(X_batch, y_batch, lr)
                
                if verbose:
                    is_last_batch = (batch_index == nb_batch - 1)
                    if not is_last_batch or not do_validation:
                        progbar.update(batch_end, [('loss', loss)])
                    else:
                        progbar.update(batch_end, [('loss', loss), ('val. loss', self.test(X_val, y_val))])
            
    def predict_batch(self, X, batch_size=128):
        preds = None
        for batch_index in range(0, len(X)/batch_size+1):
            batch = range(batch_index*batch_size, min(len(X), (batch_index+1)*batch_size))
            if not batch:
                break
            #batch_preds = self._predict(X[batch])
            batch_preds = self._predict(X[batch[0]:batch[-1]+1])

            if batch_index == 0:
                shape = (len(X),) + batch_preds.shape[1:]
                preds = numpy.zeros(shape)
            preds[batch] = batch_preds
        return preds



class MemLSTM(Layer):
    '''
        This version of Keras' LSTM Layer returns memories as
        well as output. This is required for a dependent model
        to read the LSTM's state.

        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    '''
    def __init__(self, input_dim, output_dim=128, 
        init='uniform', inner_init='orthogonal', 
        activation='tanh', inner_activation='hard_sigmoid',
        weights=None, truncate_gradient=-1, return_sequences=False,
        return_memories=False):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.return_memories = return_memories

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_i = self.init((self.input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((self.input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = shared_zeros((self.output_dim))

        self.W_c = self.init((self.input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((self.input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]

        if weights is not None:
            self.set_weights(weights)

        self.C1 = None
        self.H1 = None

    def _step(self, 
        xi_t, xf_t, xo_t, xc_t, 
        h_tm1, c_tm1, 
        u_i, u_f, u_o, u_c): 
        i_t = self.inner_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def get_input(self, train):
        if hasattr(self, 'previous_layer'):
            return self.previous_layer.output(train=train)
        else:
            # Need to include empty sequences for memories passing
            return self.input, []

    def output(self, train):
        print 'mem'
        print 'C1', self.C1
        print 'H1', self.H1

        X, m = self.get_input(train) 
        X = X.dimshuffle((1,0,2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o
        
        [outputs, memories], updates = theano.scan(
            self._step, 
            sequences=[xi, xf, xo, xc],
            outputs_info=[
                self.C1 if self.C1 else alloc_zeros_matrix(X.shape[1], self.output_dim), 
                self.H1 if self.H1 else alloc_zeros_matrix(X.shape[1], self.output_dim)
            ], 
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c], 
            truncate_gradient=self.truncate_gradient 
        )

        return outputs.dimshuffle((1,0,2)), m + [memories[-1]]



class MemLSTM_NoInput(Layer):
    '''
        This version of Keras' LSTM Layer returns memories as
        well as output. This is required for a dependent model
        to read the LSTM's state.

        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    '''
    def __init__(self, input_dim, output_dim=128, 
        init='uniform', inner_init='orthogonal', 
        activation='tanh', inner_activation='hard_sigmoid',
        weights=None, truncate_gradient=-1, return_sequences=False,
        return_memories=False):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.return_memories = return_memories

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        #self.input = T.tensor3()
        self.input = None

        self.W_i = self.init((self.input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((self.input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = shared_zeros((self.output_dim))

        self.W_c = self.init((self.input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((self.input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]

        self.params = [
            self.U_i,
            self.U_c,
            self.U_f,
            self.U_o,
        ]


        if weights is not None:
            self.set_weights(weights)

        self.C1 = T.matrix()
        self.H1 = T.matrix()
        self.n = None

    def _step(self, 
        h_tm1, c_tm1,
        u_i, u_f, u_o, u_c): 
        i_t = self.inner_activation(T.dot(h_tm1, u_i))
        f_t = self.inner_activation(T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(T.dot(h_tm1, u_c))
        o_t = self.inner_activation(T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def get_input(self, train):
        return None
        #if hasattr(self, 'previous_layer'):
        #    return self.previous_layer.output(train=train)
        #else:
        #    # Need to include empty sequences for memories passing
        #    return [], []

    def output(self, train):
        print 'mem NI'
        print 'C1', self.C1
        print 'H1', self.H1
        print 'n', self.n
        [outputs, memories], updates = theano.scan(
            self._step, 
            outputs_info=[self.H1, self.C1], 
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c], 
            n_steps=self.n,
            truncate_gradient=self.truncate_gradient 
        )

        return outputs.dimshuffle((1,0,2)), [memories[-1]]
        


from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams()

class MemDropout(Layer):
    '''
        Dropout layer adapted for use with other Mem layers (incomplete)

        Hinton's dropout. 
    '''
    def __init__(self, p, l):
        self.p = p
        self.output_dim = l
        self.params = []

        # temporary
        self.C1 = T.matrix()
        self.H1 = T.matrix()

    def output(self, train):
        X, m = self.get_input(train)
        if self.p > 0.:
            retain_prob = 1. - self.p
            if train:
                X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            else:
                X *= retain_prob
        return X, m + [m[0]]



class MemDense(Layer):
    '''
        Dense layer adapted for use with other Mem layers (incomplete)

        Just your regular fully connected NN layer.
    '''
    def __init__(self, input_dim, output_dim, init='uniform', activation='linear', weights=None):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.tensor3()
        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.params = [self.W, self.b]

        if weights is not None:
            self.set_weights(weights)

    def get_input(self, train):
        if hasattr(self, 'previous_layer'):
            return self.previous_layer.output(train=train)
        else:
            # Need to include empty sequences for memories passing
            return self.input, []

    def output(self, train):
        X, m = self.get_input(train)
        output = self.activation(T.dot(X, self.W) + self.b)
        return output, m + [None]

