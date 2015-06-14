from log import log

import sys
import numpy
import random
import atexit
import traceback
import math

from itertools import izip, chain
from cPickle import load, dump
from collections import Counter

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.datasets import imdb

from keras_additions.joint import *
from keras_additions.mem import *
from keras_additions.flat import *

from kvembed import KVEmbed
from utils import *
from output_dumps import *

from higher_activations import softmaxN

ftype = numpy.float32


def import_model(filename):
    model = None
    try:
        with open(filename, 'rb') as f:
            model = load(f)
    except Exception, e:
        print "Error in import_model: could not import model"
        print traceback.format_exc()
    return model

def export_model(model, filename):
    try:
        sys.setrecursionlimit(50000)
        with open(filename, 'wb') as f:
            # clear functions from model (they aren't needed and can't be pickled)
            model.model_A.loss = None
            model.model_A.optimizer = None
            model.model_B.loss = None
            model.model_B.optimizer = None
            dump(model, f)
    except Exception, e:
        print "Error in export_model: could not export model"
        print traceback.format_exc()

def nparrays_to_lists(w):
    try:
        for i in range(len(w)):
            if w[i].__class__ == numpy.array([]).__class__:
                w[i] = list(w[i])
        for ww in w:
            nparrays_to_lists(ww)
    except TypeError:
        pass

def import_weights(filename):
    weights = None
    try:
        with open(filename, 'r') as f:
            weights = eval(f.read().strip())
    except Exception, e:
        print "Error in import_weights: could not import weights"
        print traceback.format_exc()
    return weights

def export_weights(model, filename):
    try:
        with open(filename, 'w') as f:
            weights = model.get_weights()
            nparrays_to_lists(weights)
            f.write(str(weights))
    except Exception, e:
        print "Error in export_weights: could not export weights"
        print traceback.format_exc()

def create_embed(input_file, min_count=1):
    log("creating kv embed")
    embed = KVEmbed(input_file, min_count=min_count)
    log("done")
    return embed


def build_model(layer_size, layer_count, wc_src, wc_dst, maxlen, start_token, loss, optimizer, compile_train):
    log('Building model...')
    
    model_A = SequentialSequence()
    model_A.add(MemDense(wc_src, layer_size))
    for i in range(layer_count):
        model_A.add(MemLSTM(layer_size, layer_size, return_memories=True))
    
    model_B = RecurrentSequence(n_steps=maxlen)
    model_B.add(FlatDense(wc_dst, layer_size))
    for i in range(layer_count):
        model_B.add(FlatLSTM(layer_size, layer_size, return_sequences=True))
    model_B.add(FlatDense(layer_size, wc_dst, activation=softmaxN))
    
    model = JointModel(model_A, model_B)
    
    log("Compiling model...")
    model.compile(loss=loss, optimizer=optimizer, log_fcn=log, compile_train=compile_train)
    log("Done compiling model")

    return model


def _load_dataset(embed, infile, maxlen, reverse=False):
    with open(infile) as f:
        raw_lines = f.readlines()

    # skip header, if applicable
    h = raw_lines[0].strip().split(' ')
    if len(h) == 1 and h[0].isdigit():
        raw_lines = raw_lines[1:]
        
    variable_lines = map(lambda x: x.split(' '), \
            filter(lambda x: x, [x.strip() for x in raw_lines]))
    count = len(variable_lines)
    token_lines = [x[:min(len(x), maxlen)] for x in variable_lines]

    vectors = numpy.zeros((len(token_lines), maxlen), dtype=int)
    for i, line in enumerate(token_lines):
        vectors[i] = numpy.array(embed.sentence_to_ids(line, reverse=reverse, pad_length=maxlen))

    return vectors


def _load_dataset_tokens(embed, infile, maxlen, reverse=False):
    with open(infile) as f:
        raw_lines = f.readlines()

    # skip header, if applicable
    h = raw_lines[0].strip().split(' ')
    if len(h) == 1 and h[0].isdigit():
        raw_lines = raw_lines[1:]
        
    variable_lines = map(lambda x: x.split(' '), \
            filter(lambda x: x, [x.strip() for x in raw_lines]))
    token_lines = [x[:min(len(x), maxlen)] for x in variable_lines]

    return token_lines


def load_dataset(embed_src, embed_dst, infile_src, infile_dst, maxlen):
    raise NotImplementedError("load_dataset: deprecated")


def load_dataset_test(embed_src, embed_dst, infile_src, infile_dst, maxlen):
    raise NotImplementedError("load_dataset_test: deprecated")

def padright(a, n=1):
    return a.reshape(list(a.shape)+[1]*n)

def load_datasets(req, embed_src, embed_dst, infile_src, infile_dst, maxlen):
    res = {}
    if 'X_emb' in req:
        X_vectors = _load_dataset(embed_src, infile_src, maxlen, reverse=True)
        print 'loaded X ({0} bytes)'.format(X_vectors.nbytes)
        res['X_emb'] = X_vectors
    
    if 'X_tokens' in req:
        X_tokens = _load_dataset_tokens(embed_src, infile_src, maxlen, reverse=True)
        res['X_tokens'] = X_tokens

    if 'Y_emb' in req:
        Y_vectors = _load_dataset(embed_dst, infile_dst, maxlen, reverse=False)

        if 'M' in req:
            # create mask from Y vectors
            s = embed_dst.token_count
            e = embed_dst.end
            mask = [[1-int(v == e) for v in y] for y in Y_vectors]
            M = numpy.array(mask, dtype=int)
            M = padright(M)
            print 'loaded M ({0} bytes)'.format(M.nbytes)
            res['M'] = M

        print 'loaded Y ({0} bytes)'.format(Y_vectors.nbytes)
        res['Y_emb'] = Y_vectors

    if 'Y_tokens' in req:
        Y_tokens = _load_dataset_tokens(embed_dst, infile_dst, maxlen, reverse=False)
        res['Y_tokens'] = Y_tokens

    if 'maxlen' in req:
        res['maxlen'] = maxlen

    def sl(v):
        try:
            return str(len(v))
        except TypeError:
            return 'n/a'

    print 'loaded sets {0} with lengths {1}'.format(', '.join(res.keys()), ', '.join(map(sl, res.itervalues())))

    return res


def generate_D_L1_Usage(embed_src, embed_dst, model, X, Y):
    # get regular translations
    R = model.predict_batch(X, batch_size=len(X))
    R_translations = [embed_dst.match_sentence(r) for r in R]

    # get word counts
    count = Counter(chain(*R_translations))
    s = sum(count.itervalues())

    # create dictionary from softmax
    soft = numpy_softmax(list({k: (v / float(s)) for k, v in count.iteritems()}.itervalues()))
    D = {k: 1. / -math.log(min(1.-1e-10, v)) for k, v in zip(count.iterkeys(), soft)}

    return D

