import numpy
from utils import *
from itertools import izip

class KVEmbed(object):
    '''
        Object with a dictionary of {token: embedding value} pairs.

    '''

    default_rep = lambda x: numpy.array(x, dtype=numpy.float32)

    invalid_token = "<INVALID>"

    def __init__(self, infile, representation=default_rep, eol_tokens=None):
        '''
            Create KV embed from a file.
            The keyword arg "representation" is used to cast input arguments
            (by default, as numpy arrays)

        '''
        with open(infile) as f:
            header = f.readline().strip()
            self.vocab_size, self.embedding_size = map(int, header.split(" "))

            # token: embedding dictionary
            # (token -> embedding)
            self.embedding_dict = {}
            for tokens in (line.strip().split(" ") for line in f):
                key = tokens[0]
                value = representation(map(float, tokens[1:]))
                self.embedding_dict[key] = value

        self.unknown_token = representation([0.]*self.embedding_size)

        # embeddings arranged in a matrix form, for calculation efficiency
        # (index -> embedding)
        self.embed_matrix = representation(
                [x for x in self.embedding_dict.itervalues()])

        # mapping of embedding matrix row indices to tokens
        # (index -> token)
        self.embed_matrix_lookup = {k: v for k, v in 
                [x for x in enumerate(self.embedding_dict.iterkeys())]}
        
        self.eol_tokens = eol_tokens

        self.default_metric = lambda vec, x: -LN(vec, x, n=2)
    
    def get(self, token):
        '''
            token -> embedding

        '''
        return self.embedding_dict.get(token, self.unknown_token)

    def get_index(self, i):
        '''
            matrix index -> token

        '''
        return self.embed_matrix_lookup[i]

    def eol(self):
        return [0.]*self.embedding_size

    def clip(self, sentence):
        '''
            Clips sentence at first occurrence of an eol token.
            eol_tokens contains (token, inclusive=bool) pairs.

        '''
        for i, token in enumerate(sentence):
            for eol, inclusive in self.eol_tokens:
                if token == eol:
                    return sentence[:i + (1 if inclusive else 0)]

        return sentence

    def convert_sentence(self, s, reverse=False):
        if reverse:
            s = reversed(s) 
        return [self.embedding_dict.get(x, self.unknown_token) for x in s]


    def topN(self, vec, n):
        '''
            Given an input vector of probabilities or distances with length
            self.vocab_size, returns the top N matches.

        '''
        if len(vec) != self.vocab_size:
            raise Exception("topN: invalid input: expected vector of length {0} (received length {1})".format(self.vocab_size, len(vec)))

        sorted_pairs = sorted(enumerate(vec), key=lambda (i, m): m, reverse=True)[:n]
        lookups = [(self.embed_matrix_lookup[i], i, m) for i, m in sorted_pairs]
        return lookups


    def matchN(self, vec, n, w=None, default=0, metric=None):
        '''
            Given an input vector of length self.embedding size, finds the top N closest
            words. Word distance is calculated using metric (default: -L2 distance)
            and weighted by w[word]. Softmax is calculated over all words and the
            top N are returned.
            
            Outputs are of the form [(word, word id, 0 <= probability <= 1), ...]

            w should be either None or a dictionary of the form {word: weight}

        '''

        if len(vec) != self.embedding_size:
            raise Exception("matchN: invalid input: expected vector of length {0} (received length {1})".format(self.embedding_size, len(vec)))

        if metric is None:
            metric = self.default_metric

        match_raw = (metric(vec, x) for x in self.embed_matrix)
        pairs = [(self.embed_matrix_lookup[i], i, m) for i, m in enumerate(match_raw)]

        if w is not None:
            pairs = [(word, i, m * (w[word] if word in w else default)) for word, i, m in pairs]

        pairs_softmax = [(word, i, sm) for (word, i), sm in izip(
                ((word, i) for word, i, m in pairs),
                numpy_softmax(m for word, i, m in pairs)
                )]

        sorted_pairs = sorted(pairs_softmax, key=lambda (word, i, m): m, reverse=True)[:n]
        return sorted_pairs

    def match1(self, vec, w=None, default=0, metric=None):
        return self.matchN(vec, 1, w=w, default=default, metric=metric)[0]

    def match_sentence(self, sentence, w=None, default=0, metric=None):
        return self.clip([self.matchN(word, 1, default=default, metric=metric)[0][0] \
                for word in sentence])
        

