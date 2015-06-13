import numpy
from utils import *
from itertools import izip, chain
from collections import Counter

ftype = numpy.float32

class KVEmbed(object):
    '''
        Object with a dictionary of {token: embedding value} pairs.

    '''

    invalid_token = "<<INVALID>>"
    start_token   = "<<START>>"
    end_token     = "<<END>>"

    def __init__(self, infile, eol_tokens=None, min_count=1):
        '''
            Create KV embed from a tokenized (space-separated) file of sentences.

        '''
        with open(infile) as f:
            c = Counter(chain(*(line.strip().split(' ') for line in f)))

        blacklist = ['\n', '', ' ']
        words = [word for word, count in c.iteritems() if count >= min_count and word not in blacklist]

        alltokens = [self.invalid_token, self.start_token, self.end_token] + words

        self.word_count = len(words)
        self.special_token_count = len(alltokens) - len(words)
        self.token_count = len(alltokens)

        self.word_to_int = {word: i for i, word in enumerate(alltokens)}
        self.int_to_word = {i: word for i, word in enumerate(alltokens)}

        self.eol_tokens = [(self.end_token, False)]
        if eol_tokens is not None:
            self.eol_tokens += eol_tokens

        self.start = self.word_to_int[self.start_token]
        self.end = self.word_to_int[self.end_token]
        self.start_1h = one_hot(self.token_count, self.start)
        self.end_1h = one_hot(self.token_count, self.start)
    
    def get(self, token):
        '''
            token -> index

        '''
        i = self.word_to_int.get(token, self.word_to_int[self.invalid_token])
        return i

    def get_1h(self, token):
        '''
            token -> one-hot vector

        '''
        i = self.word_to_int.get(token, self.word_to_int[self.invalid_token])
        return one_hot(self.token_count, i)

    def get_index(self, i):
        '''
            matrix index -> token

        '''
        return self.int_to_word[i]

    def clip(self, sentence):
        '''
            Clips sentence at first occurrence of an eol token.
            eol_tokens contains (token, inclusive=bool) pairs.

        '''
        for i, token in enumerate(sentence):
            for eol, inclusive in self.eol_tokens:
                if token == eol:
                    return sentence[:i + int(inclusive)]

        return sentence

    # used to be convert_sentence
    def sentence_to_1h(self, tokens, reverse=False, pad_length=None):
        vectors = [self.get_1h(x) for x in tokens]
        if pad_length:
            vectors += [self.end_1h]*(pad_length-len(vectors))
        if reverse:
            vectors = list(reversed(vectors))
        return vectors

    def sentence_to_ids(self, tokens, reverse=False, pad_length=None):
        vectors = [self.get(x) for x in tokens]
        if pad_length:
            vectors += [self.end]*(pad_length-len(vectors))
        if reverse:
            vectors = list(reversed(vectors))
        return vectors

    # used to be match_sentence
    def vectors_to_sentence(self, vectors, clip=True):
        tokens = [self.int_to_word[numpy.argmax(v)] for v in vectors]
        if clip:
            tokens = self.clip(tokens)
        return tokens

    def matchN(self, vector, n):
        '''
            Takes a probability vector of length token_count and returns the top n tokens.

        '''
        if len(vec) != self.token_count:
            raise Exception("matchN: invalid input: expected vector of length {0} (received length {1})".format(self.token_count, len(vec)))

        matches = [(self.int_to_word[i], i, prob) for i, prob in sorted(enumerate(vector), key=lambda (i, prob): prob, reverse=True)[:n]]
        return tokens

    def get_matrix(self, *args, **kwargs):
        raise NotImplementedError("get_matrix: Not implemented")

    def convert_sentence(self, *args, **kwargs):
        raise NotImplementedError("convert_sentence: not implemented")

    def match_sentence(self, *args, **kwargs):
        raise NotImplementedError("match_sentence: not implemented")

    def topN(self, *args, **kwargs):
        raise NotImplementedError("topN: not implemented")

    def match1(self, *args, **kwargs):
        raise NotImplementedError("match1: not implemented")

    def eol(self, *args, **kwargs):
        raise NotImplementedError("eol: not implemented")

