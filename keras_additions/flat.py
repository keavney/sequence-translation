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


def create_masked_loss_old(objective):
    def masked_loss(y_true, y_pred, mask):
        y_diff = objective(y_true, y_pred)
        y_masked = y_diff * mask
        return y_masked.mean()
    return masked_loss

#def create_masked_loss(objective, nb_class):
#    # assumes y_true is of shape (n_sentences, n_words, 1)
#    # assumes mask if of shape (n_sentences, 1)
#    def masked_loss(y_true, y_pred, mask):
#        y_1h = to_one_hotN(y_true, nb_class, dtype=theano.config.floatX)
#        y_diff = objective(y_1h, y_pred)
#        y_masked = y_diff * mask
#        return y_masked.mean()
#    return masked_loss

class RecurrentSequence(object):
    '''
        Recurrent sequence: instead of specifying an entire sequence to be
        passed into the network, specify a starting word and the number of 
        outputs to be generated. This is accomplished by moving the LSTM
        loop out of the layer level and into the network's main predict
        and test functions.
    
        The model receives a starting value of X1. At each step, it generates
        an output and uses it as the next time step's input.
    
        This model is incompatable with Keras' normal layers, as 
        recurrent layers pass cell memories as well as outputs through
        each iteration.
    
    '''

    def __init__(self, n_steps=5):
        self.layers = []
        self.params = []
        self.steps = n_steps

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

    def prep_embedding(self, embedding):
        self.embed_matrix = sharedX(embedding.embed_matrix)

    def compile(self, optimizer, loss, log_fcn=lambda x, y: (x, y),
            joint_model=False, skiplist = []):
        log = lambda x: log_fcn(x, True)

        log("Entering compile...")

        self.optimizer = optimizers.get(optimizer)
        self.old_lr = self.optimizer.lr if 'lr' in dir(self.optimizer) else 0
        self.lr = T.scalar()
        self.optimizer.lr = self.lr

        objective = objectives.get(loss)
        self.loss = create_masked_loss(objective)

        v = theano.shared(numpy.array([1]))

        # output of model
        self.Y  = T.tensor3() # vector word labels
        self.M  = T.tensor3() # mask
        self.X1 = T.tensor3() # first sequence

        log("Compiling functions...")

        self.CH_layers = filter(lambda x: hasattr(x, 'C1') and hasattr(x, 'H1'), self.layers)

        # Loop inner function
        def make_step(train):
            # create closure around train
            def _step(last_X, *last_S):
                # set top layer's input = last output
                self.layers[0].input = last_X

                # C and H have to be manually passed into FlatLSTM
                # layers for each iteration of the loop.
                # last_S is variadic, as inputs to _step need to be
                # tensors.
                last_C = last_S[:len(self.CH_layers)]
                last_H = last_S[len(self.CH_layers):]
                for i, layer in enumerate(self.CH_layers):
                    layer.c_tm1 = last_C[i]
                    layer.h_tm1 = last_H[i]

                # Get the following:
                # - final layer's output
                # - each layer's C (cell memory)
                # - each layer's H (layer's last output)
                out, C, H = self.layers[-1].output(train=train)

                return [out] + C + H

            return _step

        # Create train, predict functions
        train_step = make_step(True)
        predict_step = make_step(False)

        # Train and predict result: loop over step function n_steps times.
        # Initial values are set by the calling function: the first sequence
        # token, and an initial C and H for each layer.
        # (this produces a sequence of length n_steps)

        # Train result can take an extremely long time to compile.

        if 'train' not in skiplist:
            log("Creating train result (n_steps={0})...".format(self.steps))
            self._train_result_scan, _ = theano.scan(fn=train_step,
                    outputs_info = [dict(initial=self.X1, taps=[-1])] + 
                                   [dict(initial=layer.C1, taps=[-1]) for layer in self.CH_layers] +
                                   [dict(initial=layer.H1, taps=[-1]) for layer in self.CH_layers],
                    n_steps=self.steps)

        if 'predict' not in skiplist or 'test' not in skiplist:
            log("Creating predict result (n_steps={0})...".format(self.steps))
            self._predict_result_scan, _ = theano.scan(fn=predict_step,
                    outputs_info = [dict(initial=self.X1, taps=[-1])] + 
                                   [dict(initial=layer.C1, taps=[-1]) for layer in self.CH_layers] +
                                   [dict(initial=layer.H1, taps=[-1]) for layer in self.CH_layers],
                    n_steps=self.steps)

        # Fixes dimensions from result function to produce the
        # correct ordering of (sequence, token, vector)
        # (dimension #2 is an artefact of porting the functions to a loop)
        if not 'train' in skiplist:
            self._train_result = self._train_result_scan[0]
            self._train_result = self._train_result.dimshuffle(1, 0, 3, 2)
            self._train_result = self._train_result.flatten(ndim=3)

        if not ('predict' in skiplist and 'test' in skiplist):
            self._predict_result = self._predict_result_scan[0]
            self._predict_result = self._predict_result.dimshuffle(1, 0, 3, 2)
            self._predict_result = self._predict_result.flatten(ndim=3)

        # Create train, predict, testing functions
        if not 'train' in skiplist:
            log("Setting train loss and updates...")
            self.train_loss = self.loss(self.Y, self._train_result, self.M)
            self.updates = self.optimizer.get_updates(self.params, self.train_loss)
            if not joint_model:
                log("Creating train function...")
                self.__train = theano.function([self.X1, self.Y, self.M] +
                    [self.lr] + 
                    [layer.C1 for layer in self.CH_layers] +
                    [layer.H1 for layer in self.CH_layers],
                    self.train_loss, 
                    updates=self.updates, allow_input_downcast=True)

        if not 'predict' in skiplist:
            self.predict_result = self._predict_result
            if not joint_model:
                log("Creating predict function...")
                self.__predict = theano.function([self.X1] +
                    [layer.C1 for layer in self.CH_layers] +
                    [layer.H1 for layer in self.CH_layers],
                    self.predict_result,
                    allow_input_downcast=True)

        if not 'test' in skiplist:
            self.test_score = self.loss(self.Y, self._predict_result, self.M)
            if not joint_model:
                log("Creating test function...")
                self.__test = theano.function([self.X1, self.Y, self.M] +
                    [layer.C1 for layer in self.CH_layers] +
                    [layer.H1 for layer in self.CH_layers],
                    self.test_score, 
                    allow_input_downcast=True)

        log("Done compiling functions")

    def makeCH(self, X):
        '''
            Create C, H from X. Need a matrix of (batch size * layer's output size)

        '''
        C = [numpy.zeros((len(X), layer.output_dim), dtype=numpy.float32) for layer in self.layers]
        H = [numpy.zeros((len(X), layer.output_dim), dtype=numpy.float32) for layer in self.layers]
        return C, H

    def _train(self, X, y, M, lr):
        '''
            Wrapper to compiled test function, called by other member functions.

        '''
        if lr is None:
            lr = self.old_lr
        C, H = self.makeCH(X)
        return self.__train(*([X, y, M] + [lr] + C + H))

    def _predict(self, X):
        '''
            Wrapper to compiled test function, called by other member functions.

        '''
        C, H = self.makeCH(X)
        return self.__predict(*([X] + C + H))

    def _test(self, X, y, M):
        '''
            Wrapper to compiled test function, called by other member functions.

        '''
        C, H = self.makeCH(X)
        return self.__test(*([X, y, M] + C + H))

    def train(self, X, y, M, lr):
        y = standardize_y(y)
        loss = self._train(X, y, M, lr)
        return loss

    def test(self, X, y, M):
        y = standardize_y(y)
        score = self._test(X, y, M)
        return score
   
    def fit(self, X, y, M, batch_size=128, nb_epoch=100, verbose=1,
            validation_split=0., lr=None, shuffle=True):
        y = standardize_y(y)

        # If a validation split size is given (e.g. validation_split=0.2)
        # then split X into smaller X and X_val,
        # and split y into smaller y and y_val.
        do_validation = False
        if validation_split > 0 and validation_split < 1:
            do_validation = True
            split_at = int(len(X) * (1 - validation_split))
            (X, X_val) = (X[0:split_at], X[split_at:])
            (y, y_val) = (y[0:split_at], y[split_at:])
            (M, M_val) = (M[0:split_at], M[split_at:])
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
                M_batch = M[batch_ids]
                loss = self._train(X_batch, y_batch, M_batch, lr)
                
                if verbose:
                    is_last_batch = (batch_index == nb_batch - 1)
                    if not is_last_batch or not do_validation:
                        progbar.update(batch_end, [('loss', loss)])
                    else:
                        progbar.update(batch_end, [('loss', loss), ('val. loss', self.test(X_val, y_val, M_val))])
            
    def predict_batch(self, X, batch_size=128):
        preds = None
        for batch_index in range(0, len(X)/batch_size+1):
            batch = range(batch_index*batch_size, min(len(X), (batch_index+1)*batch_size))
            if not batch:
                break
            batch_preds = self._predict(X[batch[0]:batch[-1]+1])

            if batch_index == 0:
                shape = (len(X),) + batch_preds.shape[1:]
                preds = numpy.zeros(shape)
            preds[batch] = batch_preds
        return preds



class FlatLSTM(Layer):
    '''
        Copy of LSTM layer that works with RecurrentSequence.

        The main difference is that this layer processes tokens one
        token at a time, instead of receiving an entire sequence.
        The scan() call is moved from inside of an individual layer
        to the surrounding model itself. 

        This is required in order to support models where the input
        at t_n depends on the output at t_(n-1).

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
        weights=None, truncate_gradient=-1, return_sequences=False):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

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

        # C1, H1: starting C, H values
        self.C1 = T.matrix()
        self.H1 = T.matrix()

        if weights is not None:
            self.set_weights(weights)

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
            # Need to include empty sequences for C, H
            return self.input, [], []

    def output(self, train):
        Xo, parent_c, parent_h = self.get_input(train)
        X = Xo.dimshuffle((1,0,2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o
        
        # Call to theano.scan has been moved to the model instead of
        # the layer: only calculate the result from this token
        outputs, memories = self._step(xi, xf, xc, xo,
                self.h_tm1, self.c_tm1,
                self.U_i, self.U_f, self.U_o, self.U_c)

        # Add a first dimension to produce expected output.
        # This is a holdover from when this function used theano.scan
        # and should probably be refactored
        outputs.reshape(*[[1] + outputs.shape], ndim=3)

        # Store the output from this layer
        layer_output = X[0]

        return outputs.dimshuffle((1,0,2)), parent_c + [memories[0]], parent_h + [layer_output]
        


class FlatDropout(Layer):
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
        X, c, h = self.get_input(train)
        # TODO: fix dropout by adding a way to pass dropout vector between steps
        #if self.p > 0.:
        #    retain_prob = 1. - self.p
        #    if train:
        #        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        #    else:
        #        X *= retain_prob
        return X, c + [c[0]], h + [h[0]]



class FlatDense(Layer):
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

        # temporary
        #self.C1 = T.matrix()
        #self.H1 = T.matrix()

    def get_input(self, train):
        if hasattr(self, 'previous_layer'):
            return self.previous_layer.output(train=train)
        else:
            # Need to include empty sequences for C, H
            return self.input, [], []

    def output(self, train):
        Xo, parent_c, parent_h = self.get_input(train)
        output = self.activation(T.dot(Xo, self.W) + self.b)
        return output, parent_c, parent_h

