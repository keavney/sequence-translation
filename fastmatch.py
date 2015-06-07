import numpy
import theano
import theano.tensor as T
from utils import *
from itertools import izip

ftype = numpy.float32

class FastMatch(object):
    '''
        Wrapper around KVEmbed with a fast matching function (useful for 
        large batches with a large embedding dictionary)
    
    '''

    def __init__(self, kv):
        self.kv = kv
        self.kvd = theano.shared(self.kv.embed_matrix, borrow=True)
        self.f_match = None

    def compile(self, batch_size=None, sentence_length=None, kshape=None, mshape=None, metric='l2'):
        if kshape is None:
            word_count = self.kv.vocab_size
            embedding_size = self.kv.embedding_size
            kshape = [word_count, embedding_size]
        if mshape is None:
            embedding_size = self.kv.embedding_size
            mshape = [batch_size, sentence_length, embedding_size]

        self.batch_size = mshape[0]

        #kvd = T.matrix()
        kvd = self.kvd
        mat = T.tensor3()
        self.f_match = None

        xkvd = T.tile(kvd.reshape([1] + kshape), list(mshape[:-1])+[1])
        xmat = T.repeat(mat, kshape[0], axis=mat.ndim-2)
        diffs = (xkvd - xmat).reshape(list(mshape[:-1]) + list(kshape))
        distances = T.sum(diffs ** 2, axis=mat.ndim)
        matches = T.argmin(distances, axis=mat.ndim-1)

        self.f_match = theano.function([mat], matches, allow_input_downcast=True)
        #self.f_match = theano.function([kvd, mat], matches, allow_input_downcast=True)

        print 'done'


    def match(self, mat, log=lambda x: None):
        '''
            Accepts input of shape [batch size, sentence length, embedding size]

        '''
        
        if self.f_match is None:
            raise Exception("Error: called match without first compiling function")

        if len(mat) != self.batch_size:
            raise Exception("Error: received matrix of size {0} (expected {1})".format(len(mat), self.batch_size))

        log('start')
        matches = self.f_match(mat)
        #matches = self.f_match(self.kv.get_matrix(), mat)
        log('done matches')
        words = [[self.kv.get_index(word) for word in match] for match in matches]
        log('done translation')
        return words

