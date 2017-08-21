import os
import sys 
import time
from math import sqrt
import ctypes
from ctypes import cdll, c_void_p

import numpy as np
import pandas as pd
from numba import guvectorize

cblas = cdll.LoadLibrary("/usr/lib/libopenblas.so")

CblasRowMajor = 101
CblasNoTrans = 111
CblasTrans = 112
incX = incY = 1

class WordVecSpaceException(Exception):
    pass

class UnknownWord(WordVecSpaceException):
    def __init__(self, word):
        self.word = word

    def __str__(self):
        return '"%s"' % self.word

class UnknownIndex(WordVecSpaceException):
    def __init__(self, index):
        self.index = index

    def __str__(self):
        return '"%s"' % self.index

@guvectorize(['void(float32[:], float32[:])'], '(n) -> ()', nopython=True, target='parallel')
def normalize_vectors(vec, m):
    '''
    Converts each vector in `vectors` into a
    unit vector and stores the magnitude in
    `magnitudes` vector in the corresponding
    location
    '''

    _m = 0.0
    for i in xrange(len(vec)):
	_m += vec[i]**2

    _m = np.sqrt(_m)

    for i in xrange(len(vec)):
	vec[i] /= _m

    m[0] = _m


class WordVecSpace(object):
    VECTOR_FNAME = 'vectors.npy'
    VOCAB_FNAME = 'vocab.txt'

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # Two dimensional array holding the vector data
        # for the entire vector space
        self.vectors = None

        #List holding all the words in the word vector space
        self.words = []

        #Dictionary holds word as key and it's index from word list as value
        self.word_indices = {}

        #One dimensional array holding occurrences of words.
        self.word_occurrences = None

        #One dimensional array holding magnitudes of vectors.
        self.magnitudes = None

    @property
    def num_vectors(self):
        rows, _ = self.vectors.shape
        return rows

    @property
    def num_dimensions(self):
        _, cols = self.vectors.shape
        return cols

    def load(self):
        vec_fpath = os.path.join(self.data_dir, self.VECTOR_FNAME)
	self.vectors = np.load(vec_fpath)

        vocab_fpath = os.path.join(self.data_dir, self.VOCAB_FNAME)
        self.words, self.word_indices, self.word_occurrences = \
            self._load_vocab(vocab_fpath, self.num_vectors)

        self.magnitudes = np.ndarray(self.num_vectors, dtype=np.float32)
        normalize_vectors(self.vectors, self.magnitudes)

    def _load_vocab(self, fpath, nvecs):
        '''
        Load information in the vocab file into memory.

        # TODO: explain file format with example
        '''
        word_indices = {}
        words = []
        word_occurrences = np.ndarray(nvecs, dtype=np.uint64)

        for index, line in enumerate(open(fpath)):
            word, n = line[:-1].split(' ', 1)
            word_occurrences[index] = int(n)
            word_indices[word] = index
            words.append(word)

        return words, word_indices, word_occurrences

    def does_word_exist(self, word):
        '''
        >>> from wordvecspace import WordVecSpace, UnknownWord, UnknownIndex
        >>> data_dir = "/home/deepcompute/alpha/data/w2v_new_sharded/shard_0"
        >>> wv = WordVecSpace(data_dir)
        >>> wv.load()
        >>> print wv.does_word_exist("imatljfsnib")
        False
        >>> print wv.does_word_exist("imatinib")
        True

        '''
        return word in self.word_indices

    def get_word_index(self, word, raise_exc=True):
        # if `word` is an integer already
        # and it is a valid index (i.e. in range)
        # then return that
        '''
        >>> from wordvecspace import WordVecSpace, UnknownWord, UnknownIndex
        >>> data_dir = "/home/deepcompute/alpha/data/w2v_new_sharded/shard_0"
        >>> wv = WordVecSpace(data_dir)
        >>> wv.load()
        >>> try:
        ...     print wv.get_word_index("iskjf4s")
        ... except UnknownWord, e:
        ...     print "Word %s was not found" % e.word
        ...
        Word iskjf4s was not found

        >>> try:
        ...     print wv.get_word_index("for")
        ... except UnknownWord, e:
        ...     print "Word %s was not found" % e.word
        ...
        2
        '''

        if isinstance(word, int):
            if word < self.num_vectors:
                return word
            else:
                raise UnknownIndex(word)

        try:
            return self.word_indices[word]

        except KeyError:
            raise UnknownWord(word)

    def get_word_at_index(self, index):

        '''
        >>> from wordvecspace import WordVecSpace, UnknownWord, UnknownIndex
        >>> data_dir = "/home/deepcompute/alpha/data/w2v_new_sharded/shard_0"
        >>> wv = WordVecSpace(data_dir)
        >>> wv.load()
        >>> try:
        ...     print wv.get_word_at_index(10)
        ... except UnknownIndex, e:
        ...     print "Index %d was not in the range" % e.index
        ...
        pubmed

        '''

	try:
	    return self.words[index]

        except IndexError:
            raise UnknownIndex(index)

    def get_word_vector(self, word_or_index, normalized=False):

        '''
        >>> from wordvecspace import WordVecSpace, UnknownWord, UnknownIndex
        >>> data_dir = "/home/deepcompute/alpha/data/w2v_new_sharded/shard_0"
        >>> wv = WordVecSpace(data_dir)
        >>> wv.load()
        >>> try:
        ...     print wv.get_word_vector(10, normalized=False)
        ... except UnknownIndex, e:
        ...     print "Index %d was not found" % e.index
        ...
	[0.08531315624713898 0.014989836141467094 0.2868293821811676 ... -0.05077797546982765]



        >>> try:
        ...     print wv.get_word_vector(10, normalized=True)
        ... except UnknownIndex, e:
        ...     print "Index %d was not found" % e.index
        ...
	[ 0.01522985  0.00267594 ... -0.00906473]



        '''

        if (normalized == True):
            index = self.get_word_index(word_or_index)
            return self.vectors[index]

        else:
            index = self.get_word_index(word_or_index)
            norm_vec = pd.Series(self.vectors[index])
            return np.array((norm_vec * self.magnitudes[index]), dtype=pd.Series)


    def get_vector_magnitudes(self, words_or_indices):
        '''
        >>> from wordvecspace import WordVecSpace, UnknownWord, UnknownIndex
        >>> data_dir = "/home/deepcompute/alpha/data/w2v_new_sharded/shard_0"
        >>> wv = WordVecSpace(data_dir)
        >>> wv.load()

        >>> print wv.get_vector_magnitudes(["hi", 500])
        [5.1710873, 3.8780088]


        '''

        if not isinstance(words_or_indices, (tuple, list)):
            words_or_indices = [words_or_indices]

        mag = np.ndarray(len(words_or_indices), dtype=np.float32)

        for i, w in enumerate(words_or_indices):
            try:
                windex = self.get_word_index(w)
                mag[i] = self.magnitudes[windex]

            except (UnknownIndex, UnknownWord):
                mag[i] = 0.0

        return mag

    def get_word_occurrences(self, word_or_index):

        '''
        >>> from wordvecspace import WordVecSpace, UnknownWord, UnknownIndex
        >>> data_dir = "/home/deepcompute/alpha/data/w2v_new_sharded/shard_0"
        >>> wv = WordVecSpace(data_dir)
        >>> wv.load()
        >>> print wv.get_word_occurrences(5327)
        664333

        >>> try:
        ...     print wv.get_word_occurrences("to")
        ... except UnknownWord, e:
        ...     print "Word %s was not found" % e.word
        616385965
        '''

        index = self.get_word_index(word_or_index)
        return self.word_occurrences[index]

    def get_word_vectors(self, words_or_indices):
        '''
        >>> from wordvecspace import WordVecSpace, UnknownWord, UnknownIndex
        >>> data_dir = "/home/deepcompute/alpha/data/w2v_new_sharded/shard_0"
        >>> wv = WordVecSpace(data_dir)
        >>> wv.load()
        >>> print wv.get_word_vectors(["hi", "imatinib"])
	[[  4.58009765e-02   2.27097664e-02 ...  -4.50771116e-02]
	 [  2.15231422e-02   7.32142106e-02 ...  -7.41100591e-03]]


        '''

        n = len(words_or_indices)
        wmat = np.ndarray(dtype=np.float32, shape=(n, self.num_dimensions))

        for i, w in enumerate(words_or_indices):
            try:
                windex = self.get_word_index(w)
                np.copyto(wmat[i], self.vectors[windex])
            except (UnknownWord, UnknownIndex):
                wmat[i].fill(0.0)

        return wmat

    def get_distance(self, word1, word2):
        '''
        Get cosine distance between two words
        >>> from wordvecspace import WordVecSpace, UnknownWord, UnknownIndex
        >>> data_dir = "/home/deepcompute/alpha/data/w2v_new_sharded/shard_0"
        >>> wv = WordVecSpace(data_dir)
        >>> wv.load()
        >>> print wv.get_distance("250", "500")
        0.817561

        >>> print wv.get_distance("250", "imatinib")
        0.13943

        '''
        return np.dot(self.get_word_vector(word1, normalized=True),
                self.get_word_vector(word2, normalized=True).T)

    def get_distances(self, row_words, col_words=None):
        '''
        get_distances(word)
        get_distances(words)
        get_distances(word, words)
        get_distances(words_x, words_y)
        >>> from wordvecspace import WordVecSpace, UnknownWord, UnknownIndex
        >>> data_dir = "/home/deepcompute/alpha/data/w2v_new_sharded/shard_0"
        >>> wv = WordVecSpace(data_dir)
        >>> wv.load()
        >>> print wv.get_distances("for", ["to", "for", "imatinib"])
        [[ 0.80703819]
         [ 0.99999988]
         [ 0.27108291]]

        >>> print wv.get_distances(["nilotinib", "for"], ["to", "for", "imatinib"])
        [[ 0.23537777  0.20905481  0.88973904]
         [ 0.80703843  0.99999982  0.27108291]]

        >>> print wv.get_distances(["nilotinib", "hi"])
        [[ 0.0601333   0.23537777  0.20905481 ...,  0.22716512  0.2496517
           0.25396603]
         [ 0.05879326  0.36978272  0.35755485 ...,  0.21065465  0.21103515  0.19593   ]]

        >>> print wv.get_distances(["imatinib"])
        [[ 0.03310118  0.27105609  0.27108291 ...,  0.25952423  0.22930798
           0.22244862]]


        '''

        only_single_row_word = False

        if not isinstance(row_words, (tuple, list)):
            row_words = [row_words]
            only_single_row_word = True

        row_vectors = self.get_word_vectors(row_words)

        col_vectors = self.vectors
        if col_words is not None:
            col_vectors = self.get_word_vectors(col_words)

        if only_single_row_word:
            mat_a = col_vectors
            v = row_vectors
	    mat_c = np.ndarray((len(mat_a), len(v)), dtype=np.float32)

            nvecs, dim = mat_a.shape

	    cblas.cblas_sgemv(CblasRowMajor,
                             CblasNoTrans,
                             nvecs,
                             dim,
                             ctypes.c_float(1.0),
                             mat_a.ctypes.data_as(c_void_p),
                             dim,
                             v.ctypes.data_as(c_void_p),
                             incX,
                             ctypes.c_float(0.0),
                             mat_c.ctypes.data_as(c_void_p),
                             incY)

        else:
            mat_a = row_vectors
            mat_b = col_vectors

            mat_c = np.ndarray((len(mat_a), len(mat_b)), dtype=np.float32)

	    cblas.cblas_sgemm(CblasRowMajor,
                             CblasNoTrans,
			     CblasTrans,
			     len(mat_a),
			     len(mat_b),
			     self.num_dimensions,
			     ctypes.c_float(1.0),
			     mat_a.ctypes.data_as(c_void_p),
			     self.num_dimensions,
			     mat_b.ctypes.data_as(c_void_p),
			     self.num_dimensions,
			     ctypes.c_float(0.0),
			     mat_c.ctypes.data_as(c_void_p),
			     len(mat_b))

        return mat_c

    DEFAULT_K = 512

    # TODO : Return ndarrays insted of list
    def get_nearest_neighbors(self, word, k=DEFAULT_K):
        '''
        >>> from wordvecspace import WordVecSpace, UnknownWord, UnknownIndex
        >>> data_dir = "/home/deepcompute/alpha/data/w2v_new_sharded/shard_0"
        >>> wv = WordVecSpace(data_dir)
        >>> wv.load()
        >>> print wv.get_nearest_neighbors(374, 20)
        Int64Index([    374,     284,     684,   14522,     226,     170,    1887,
                    1208052,     236,    7144,     484,   33908,    5514,    1475,
                    3617979, 1533950, 3490404,    1411, 1341958, 1400502],
                   dtype='int64')


        '''

        distances = self.get_distances(word)
        distances = distances.reshape((len(distances),))

        distances = pd.Series(distances)
        distances = distances.nlargest(k)

        return distances.keys()
