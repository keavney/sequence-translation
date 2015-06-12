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

def create_embed(input_file):
    log("creating kv embed")
    #eol = [('.', True), ('!', True), ('?', True)]
    eol = []
    embed = KVEmbed(input_file, eol_tokens=eol)
    log("done")
    return embed


def build_model(layer_size, layer_count, wc_src, wc_dst, maxlen, start_token, loss, optimizer, compile_train):
    log('Building model...')

    #primer_v = numpy.zeros(wc_dst, dtype=ftype)
    #primer_v[0] = 1.
    
    model_A = SequentialSequence()
    model_A.add(MemDense(wc_src, layer_size))
    for i in range(layer_count):
        model_A.add(MemLSTM(layer_size, layer_size, return_memories=True))
    
    model_B = RecurrentSequence(n_steps=maxlen)
    model_B.add(FlatDense(wc_dst, layer_size))
    for i in range(layer_count):
        model_B.add(FlatLSTM(layer_size, layer_size, return_sequences=True))
    model_B.add(FlatDense(layer_size, wc_dst, activation='softmax'))
    
    model = JointModel(model_A, model_B)
    #model.X1 = [[primer_v]]
    
    log("Compiling model...")
    model.compile(loss=loss, optimizer=optimizer, log_fcn=log, compile_train=compile_train)
    log("Done compiling model")

    return model


def _load_dataset(embed, infile, maxlen, reverse=False, convert=True):
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

    if convert:
        vectors = [embed.convert_sentence(line, reverse=reverse) for line in token_lines]
    else:
        vectors = token_lines
    eol_token = embed.eol()

    return vectors, eol_token


def load_dataset(embed_src, embed_dst, infile_src, infile_dst, maxlen):
    # load X, Y
    vectors_X, eol_token_X = _load_dataset(embed_src, infile_src, maxlen, reverse=True)
    vectors_Y, eol_token_Y = _load_dataset(embed_dst, infile_dst, maxlen, reverse=False)

    # create mask from Y vectors
    s = embed_dst.embedding_size
    mask = [[[1.]*s]*len(y) + [[0.]*s]*(maxlen-len(y)) for y in vectors_Y]

    # pad X and Y vectors with EOL tokens (AFTER the mask has been created)
    vectors_X = [[eol_token_X]*(maxlen-len(x)) + x for x in vectors_X]
    vectors_Y = [x + [eol_token_Y]*(maxlen-len(x)) for x in vectors_Y]

    # convert to array
    X = numpy.array(vectors_X, dtype=ftype)
    print 'loaded X'
    print X.nbytes
    Y = numpy.array(vectors_Y, dtype=ftype)
    print 'loaded Y'
    print Y.nbytes
    M = numpy.array(mask, dtype=ftype)
    print 'loaded M'
    print M.nbytes

    print 'loaded sets X, Y, M with lengths: {0}, {1}, {2}'.format(len(X), len(Y), len(M))

    return X, Y, M, maxlen


def load_dataset_test(embed_src, embed_dst, infile_src, infile_dst, maxlen):
    X_vectors, X_eol_token = _load_dataset(embed_src, infile_src, maxlen, reverse=True)
    X_words, _ = _load_dataset(embed_src, infile_src, maxlen, reverse=True, convert=False)
    Y_words, _ = _load_dataset(embed_dst, infile_dst, maxlen, reverse=False, convert=False)

    # pad X with EOL tokens
    X_vectors = [[X_eol_token]*(maxlen-len(x)) + x for x in X_vectors]

    # convert X to array (Y is only used here as a word label)
    X = numpy.array(X_vectors, dtype=ftype)
    print 'loaded X'
    print X.nbytes

    print 'loaded sets X, X_words, Y_words with lengths: {0}, {1}, {2}'.format(len(X), len(X_words), len(Y_words))

    return X, X_words, Y_words, maxlen


def ds_request(req, embed_src, embed_dst, infile_src, infile_dst, maxlen):
    res = {}
    if 'X_emb' in req:
        X_vectors, X_eol_token = _load_dataset(embed_src, infile_src, maxlen, reverse=True)
        X_padded_vectors = [[X_eol_token]*(maxlen-len(x)) + x for x in X_vectors]
        X = numpy.array(X_padded_vectors, dtype=ftype)
        print 'loaded X'
        print X.nbytes
        res['X_emb'] = X
    
    if 'X_tokens' in req:
        X_tokens, _ = _load_dataset(embed_src, infile_src, maxlen, reverse=True, convert=False)
        res['X_tokens'] = X_tokens

    if 'Y_emb' in req:
        Y_vectors, Y_eol_token = _load_dataset(embed_dst, infile_dst, maxlen, reverse=False)

        if 'M' in req:
            # create mask from Y vectors
            s = embed_dst.embedding_size
            mask = [[[1.]*s]*len(y) + [[0.]*s]*(maxlen-len(y)) for y in Y_vectors]
            M = numpy.array(mask, dtype=ftype)
            print 'loaded M'
            print M.nbytes
            res['M'] = M

        Y_padded_vectors = [y + [Y_eol_token]*(maxlen-len(y)) for y in Y_vectors]
        Y = numpy.array(Y_padded_vectors, dtype=ftype)
        print 'loaded Y'
        print Y.nbytes
        res['Y_emb'] = Y

    if 'Y_tokens' in req:
        Y_tokens, _ = _load_dataset(embed_dst, infile_dst, maxlen, reverse=False, convert=False)
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

