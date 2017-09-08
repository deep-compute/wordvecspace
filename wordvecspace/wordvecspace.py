import os
import sys
import time
from math import sqrt
import ctypes
from ctypes import cdll, c_void_p

import numpy as np
import pandas as pd
from numba import guvectorize

np.set_printoptions(precision=4)
check_equal = np.testing.assert_array_almost_equal

# Add data dir path to environment variables

# $export WORDVECSPACE_DATADIR=/path/to/data/
DATADIR_ENV_VAR = os.environ.get('WORDVECSPACE_DATADIR', ' ')

# $export WORDVECSPACE_BLAS_FPATH=/usr/lib/libopenblas.so.0
BLAS_LIBRARY_FPATH = os.environ.get('WORDVECSPACE_BLAS_FPATH',
        '/usr/lib/libopenblas.so.0')
cblas = cdll.LoadLibrary(BLAS_LIBRARY_FPATH)

# Some OpenBlas constants
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
    Compute magnitude and store in `m` and then
    modify the vector `vec` into a unit vector.

    We are using `guvectorize` from `numba` to make this
    computation almost C-fast and to parallelize it
    across all available CPU cores.

    To understand more about `numba` and `guvectorize`,
    read this - http://numba.pydata.org/numba-doc/0.17.0/reference/compilation.html
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

        # List holding all the words in the word vector space
        self.words = []

        # Dictionary holds word as key and its index from word list as value
        self.word_indices = {}

        # One dimensional array holding occurrences of words
        self.word_occurrences = None

        # One dimensional array holding magnitudes of vectors
        self.magnitudes = None


    def _make_array(self, shape, dtype):
        return np.ndarray(shape, dtype)

    def _perform_dot(self, v1, v2):
        return np.dot(v1, v2)

    def _perform_sgemv(self, mat, v, vec_out, nvecs, dim):
	'''
        cblas_sgemv is used to multiply vector and matrix.

        CblasRowMajor                   -> Multiiply in a row major order
        CblasNoTrans                    -> Whether to transpose matix or not
        nvecs, dim                      -> Rows, columns of Matrix
        ctypes.c_float(1.0)             -> Scaling factor for the product of matrix and vector
        mat.ctypes.data_as(c_void_p)    -> matrix
        dim                             -> Columns of matirx
        v.ctypes.data_as(c_void_p)      -> vector
        incX                            -> Stride within X. For example, if incX is 7, every 7th element is used.
        ctypes.c_float(0.0)             -> Scaling factor for vector.
        vec_out.ctypes.data_as(c_void_p)-> result vector
        incY                            -> Stride within Y. For example, if incY is 7, every 7th element is used

        Read more                       -> https://developer.apple.com/documentation/accelerate/1513065-cblas_sgemv?language=objc
        '''
        cblas.cblas_sgemv(CblasRowMajor,
                         CblasNoTrans,
                         nvecs,
                         dim,
                         ctypes.c_float(1.0),
                         mat.ctypes.data_as(c_void_p),
                         dim,
                         v.ctypes.data_as(c_void_p),
                         incX,
                         ctypes.c_float(0.0),
                         vec_out.ctypes.data_as(c_void_p),
                         incY)
        return vec_out

    def _perform_sgemm(self, mat_a, mat_b, mat_out):
	'''
        cblas_sgemm is for multiplying matrix and matrix

        CblasRowMajor                   -> Specifies row-major (C)
        CblasNoTrans                    -> Specifies whether to transpose matrix mat_a
        CblasTrans                      -> Specifies whether to transpose matrix mat_b
        len(mat_a)                      -> Rows of result(Rows of mat_out)
        len(mat_b)                      -> Columns of result(Columns of mat_out)
        self.num_dimensions             -> Common dimension in mat_a and mat_b
        ctypes.c_float(1.0)             -> Scaling factor for the product of matrices mat_a and mat_b
        mat_a.ctypes.data_as(c_void_p)  -> matrix mat_a
        self.num_dimensions             -> Columns of mat_a
        mat_b.ctypes.data_as(c_void_p)  -> matrix mat_b
        self.num_dimensions             -> Columns of mat_b
        ctypes.c_float(0.0)             -> Scaling factor for matrix mat_out
        mat_out.ctypes.data_as(c_void_p)-> matirx mat_out
        len(mat_b)                      -> Columns of mat_out

        Read more                       -> https://developer.apple.com/documentation/accelerate/1513264-cblas_sgemm?language=objc
        '''
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
                         mat_out.ctypes.data_as(c_void_p),
                         len(mat_b))
        return mat_out


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
        vocab_file = open(vocab_fpath)
        self.words, self.word_indices, self.word_occurrences = \
            self._load_vocab(vocab_file, self.num_vectors)

        # The vectors present in the data file are not unit vectors
        # thus have magnitudes. For distance computation in this
        # library we use "cosine distance" which can be efficiently
        # computed by using Linear Algebra based BLAS operations such
        # as `dot`, `sgemv` etc. These operations for the purpose of
        # computing cosine distance need the input vector data to be
        # normalized to unit vectors. We ensure that we store the
        # magnitude information in a separate array.
        self.magnitudes = np.ndarray(self.num_vectors, dtype=np.float32)
        normalize_vectors(self.vectors, self.magnitudes)

    def _load_vocab(self, vocab_file, nvecs):
        '''
        Load information in the vocab file into memory.
        >>> import StringIO
        >>> wv = WordVecSpace(DATADIR_ENV_VAR)
        >>> s = StringIO.StringIO('the 10\\nand 50\\n')
        >>> wv._load_vocab(s, 2)
        (['the', 'and'], {'and': 1, 'the': 0}, array([10, 50], dtype=uint64))
        '''
        word_indices = {}
        words = []
        word_occurrences = np.ndarray(nvecs, dtype=np.uint64)

        for index, line in enumerate(vocab_file):
            word, n = line[:-1].split(' ', 1)
            word_occurrences[index] = int(n)
            word_indices[word] = index
            words.append(word)

        return words, word_indices, word_occurrences

    def does_word_exist(self, word):
        '''
        >>> wv = WordVecSpace(DATADIR_ENV_VAR)
        >>> wv.load()
        >>> print wv.does_word_exist("india")
        True
        >>> print wv.does_word_exist("inidia")
        False
        '''
        return word in self.word_indices

    def get_word_index(self, word, raise_exc=False):
        '''
        if `word` is an integer already
        and it is a valid index (i.e. in range)
        then return that

        >>> wv = WordVecSpace(DATADIR_ENV_VAR)
	>>> wv.load()
	>>> print wv.get_word_index("india")
        509
        >>> print wv.get_word_index("inidia")
        None
        >>> print wv.get_word_index("inidia", raise_exc=True) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        wordvecspace.UnknownWord: "inidia"
        '''

        if isinstance(word, int):
            if word < self.num_vectors:
                return word

            raise UnknownIndex(word)

        try:
            return self.word_indices[word]

        except KeyError:
            if raise_exc == True:
                raise UnknownWord(word)

    def get_word_at_index(self, index, raise_exc=False):
        '''
        >>> wv = WordVecSpace(DATADIR_ENV_VAR)
        >>> wv.load()
	>>> print wv.get_word_at_index(509)
	india
        '''

	try:
	    return self.words[index]

        except IndexError:
            if raise_exc == True:
                raise UnknownIndex(index)

    def get_word_vector(self, word_or_index, normalized=False, raise_exc=False):
        '''
        >>> wv = WordVecSpace(DATADIR_ENV_VAR)
        >>> wv.load()
	>>> print wv.get_word_vector('india')
        [-6.4482 -2.1636  5.7277 -3.7746  3.583 ]
        >>> print wv.get_word_vector(509, normalized=True)
        [-0.6259 -0.21    0.5559 -0.3664  0.3478]
        >>> print wv.get_word_vector('inidia', normalized=True)
        [ 0.  0.  0.  0.  0.]
        '''

        if (normalized == True):
            index = self.get_word_index(word_or_index, raise_exc)
            word_vec = self._make_array(shape=self.num_dimensions, dtype=np.float32)
            if index:
                word_vec = self.vectors[index]
            else:
                word_vec.fill(0.0)

            return word_vec

        else:
            index = self.get_word_index(word_or_index, raise_exc)
            norm_vec = self._make_array(shape=self.num_dimensions, dtype=np.float32)

            if index:
                norm_vec = self.vectors[index] * self.magnitudes[index]
            else:
                norm_vec.fill(0.0)

            return norm_vec

    def get_vector_magnitudes(self, words_or_indices, raise_exc=False):
        '''
        >>> wv = WordVecSpace(DATADIR_ENV_VAR)
        >>> wv.load()
        >>> print wv.get_vector_magnitudes(["hi", "india"])
        [  8.7948  10.303 ]
        >>> print wv.get_vector_magnitudes(["inidia", "india"])
        [  0.     10.303]
        >>> print wv.get_vector_magnitudes(["inidia", "india"], raise_exc=True) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        wordvecspace.UnknownWord: "inidia"
        '''

        if not isinstance(words_or_indices, (tuple, list)):
            words_or_indices = [words_or_indices]

        mag = np.ndarray(len(words_or_indices), dtype=np.float32)

        for i, w in enumerate(words_or_indices):
            windex = self.get_word_index(w, raise_exc)
            mag[i] = self.magnitudes[windex] if windex is not None else 0.0

        return mag

    def get_word_occurrences(self, word_or_index, raise_exc=False):
        '''
        >>> wv = WordVecSpace(DATADIR_ENV_VAR)
        >>> wv.load()
        >>> print wv.get_word_occurrences(5327)
        297
        >>> print wv.get_word_occurrences("india")
        3242
        >>> print wv.get_word_occurrences("inidia")
        None
        '''

        index = self.get_word_index(word_or_index, raise_exc)
        return self.word_occurrences[index] if index is not None else None

    def get_word_vectors(self, words_or_indices, raise_exc=False):
        '''
        >>> wv = WordVecSpace(DATADIR_ENV_VAR)
        >>> wv.load()
	>>> print wv.get_word_vectors(["hi", "india"])
        [[ 0.2473  0.2535 -0.3206  0.8058  0.3501]
         [-0.6259 -0.21    0.5559 -0.3664  0.3478]]
        >>> print wv.get_word_vectors(["hi", "inidia"])
        [[ 0.2473  0.2535 -0.3206  0.8058  0.3501]
         [ 0.      0.      0.      0.      0.    ]]
        >>> print wv.get_word_vectors(["hi", "inidia"], raise_exc=True) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        wordvecspace.UnknownWord: "inidia"
        '''

        n = len(words_or_indices)
        wmat = self._make_array(dtype=np.float32, shape=(n, self.num_dimensions))

        for i, w in enumerate(words_or_indices):
            windex = self.get_word_index(w, raise_exc)
            if windex:
                wmat[i] = self.vectors[windex]
            else:
                wmat[i].fill(0.0)


        return wmat

    def get_distance(self, word1, word2, raise_exc=False):
        '''
        Get cosine distance between two words
        >>> wv = WordVecSpace(DATADIR_ENV_VAR)
        >>> wv.load()
	>>> print wv.get_distance(250, "india")
	1.16397565603
        '''
        return 1 - self._perform_dot(\
            self.get_word_vector(word1, normalized=True, raise_exc=raise_exc),\
            self.get_word_vector(word2, normalized=True, raise_exc=raise_exc).T)


    def get_distances(self, row_words, col_words=None, raise_exc=False):
        '''
        get_distances(word)
        get_distances(words)
        get_distances(word, words)
        get_distances(words_x, words_y)
        >>> wv = WordVecSpace(DATADIR_ENV_VAR)
        >>> wv.load()
	>>> res = wv.get_distances("for", ["to", "for", "india"])
        >>> check_equal(res, np.array([[  1.4990e-01], [ -1.1921e-07], [  1.3855e+00]], dtype=np.float32), decimal=4)
        >>> res = wv.get_distances("for", ["to", "for", "inidia"])
        >>> check_equal(res, np.array([[  1.4990e-01], [ -1.1921e-07], [  1.0000e+00]], dtype=np.float32), decimal=4)
	>>> print wv.get_distances(["india", "for"], ["to", "for", "usa"])
        [[  1.1830e+00   1.3855e+00   4.8380e-01]
         [  1.4990e-01  -1.1921e-07   1.4975e+00]]
	>>> print wv.get_distances(["india", "usa"])
        [[ 1.4903  0.4202  0.269  ...,  1.2041  1.3539  0.6154]
         [ 1.8084  0.9541  1.1678 ...,  0.5963  1.0458  1.1608]]
	>>> print wv.get_distances(["andhra"])
        [[ 1.3432  0.5781  0.2306 ...,  1.0937  1.1369  0.4284]]
        '''

        only_single_row_word = False

        if not isinstance(row_words, (tuple, list)):
            row_words = [row_words]
            only_single_row_word = True

        row_vectors = self.get_word_vectors(row_words, raise_exc)

        col_vectors = self.vectors
        if col_words is not None:
            col_vectors = self.get_word_vectors(col_words, raise_exc)

        if only_single_row_word:
            mat = col_vectors
            v = row_vectors
	    vec_out = self._make_array((len(mat), len(v)), dtype=np.float32)

            nvecs, dim = mat.shape

            res = self._perform_sgemv(mat, v, vec_out, nvecs, dim)

        else:
            mat_a = row_vectors
            mat_b = col_vectors

            mat_out = self._make_array((len(mat_a), len(mat_b)), dtype=np.float32)

            res = self._perform_sgemm(mat_a, mat_b, mat_out)

        return 1 - res

    DEFAULT_K = 512

    def get_nearest_neighbors(self, word, k=DEFAULT_K):
        '''
        >>> wv = WordVecSpace(DATADIR_ENV_VAR)
        >>> wv.load()
        >>> print wv.get_nearest_neighbors("india", 20)
        ...
        Int64Index([  509,   486, 14208, 20639,  8573,  3389,  5226, 20919, 10172,
                     6866,  9772, 24149, 13942,  1980, 20932, 28413, 17910,  2196,
                    28738, 20855],
                   dtype='int64')
        '''

        distances = self.get_distances(word)
        distances = distances.reshape((len(distances),))

        distances = pd.Series(distances)
        distances = distances.nsmallest(k)

        return distances.keys()
