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
    eol = [('.', True), ('!', True), ('?', True)]
    embed = KVEmbed(input_file, eol_tokens=eol)
    log("done")
    return embed


def build_model(layer_size, layer_count, maxlen, start_token, loss, optimizer, compile_train):
    log('Building model...')

    primer_v = [0.]*layer_size
    
    model_A = SequentialSequence()
    for i in range(layer_count):
        model_A.add(MemLSTM(layer_size, layer_size, return_memories=True))
    
    model_B = RecurrentSequence(n_steps=maxlen)
    for i in range(layer_count):
        model_B.add(FlatLSTM(layer_size, layer_size, return_sequences=True))
    
    model = JointModel(model_A, model_B)
    model.X1 = [[primer_v]]
    
    log("Compiling model...")
    model.compile(loss=loss, optimizer=optimizer, log_fcn=log, compile_train=compile_train)
    log("Done compiling model")

    return model


def load_dataset(embed_src, embed_dst, infile_src, infile_dst, maxlen):
    def _load_dataset(embed, infile, reverse):
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

        vectors = [embed.convert_sentence(line, reverse=reverse) for line in token_lines]
        eol_token = embed.eol()

        return vectors, eol_token

    # load X, Y
    vectors_X, eol_token_X = _load_dataset(embed_src, infile_src, True)
    vectors_Y, eol_token_Y = _load_dataset(embed_dst, infile_dst, False)

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

