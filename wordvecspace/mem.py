import os
from math import sqrt
from ctypes import cdll, c_void_p, c_float

import numpy as np
import pandas as pd
from numba import guvectorize

from .fileformat import WordVecSpaceFile
from .base import WordVecSpace

np.set_printoptions(precision=4)
check_equal = np.testing.assert_array_almost_equal

# $export WORDVECSPACEMEM_DATADIR=/path/to/data/
DATADIR_ENV_VAR = os.environ.get('WORDVECSPACE_DATADIR', ' ')
DIMENSION = 5

# export blas path if your system has different path for blas
# $export WORDVECSPACE_BLAS_FPATH=/path/to/blas
# ex: $export WORDVECSPACE_BLAS_FPATH='/usr/lib/x86_64-linux-gnu/libopenblas.so.0'
BLAS_LIBRARY_FPATH = os.environ.get('WORDVECSPACE_BLAS_FPATH',\
        '/usr/lib/libopenblas.so.0')
cblas = cdll.LoadLibrary(BLAS_LIBRARY_FPATH)

# Some OpenBlas constants
CblasRowMajor = 101
CblasNoTrans = 111
CblasTrans = 112
incX = incY = 1

class WordVecSpaceMemException(Exception):
    pass

class UnknownWord(WordVecSpaceMemException):
    def __init__(self, word):
        self.word = word

    def __str__(self):
        return '"%s"' % self.word

class UnknownIndex(WordVecSpaceMemException):
    def __init__(self, index):
        self.index = index

    def __int__(self):
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
    for i in range(len(vec)):
        _m += vec[i]**2

    _m = np.sqrt(_m)

    for i in range(len(vec)):
        vec[i] /= _m

    m[0] = _m

class WordVecSpaceMem(WordVecSpace):

    def __init__(self, input_file, dim):
        self._f = WordVecSpaceFile(input_file, dim, 'r')
        self.nvecs = len(self._f)
        self.dim = dim

        self.vectors = self._f.getmany(0, self.nvecs)
        self.words = self._f.getmany(0, self.nvecs, self._f.WORD)
        self.word_occurrences = self._f.getmany(0, self.nvecs, self._f.OCCURRENCE)

        self.word_indices = {}
        for index, word in enumerate(self.words):
            self.word_indices[word] = int(index)

        self.magnitudes = np.ndarray(self.nvecs, dtype=np.float32)
        normalize_vectors(self.vectors, self.magnitudes)

    def _make_array(self, shape, dtype):
        return np.ndarray(shape, dtype)

    def _perform_sgemv(self, mat, v, vec_out, nvecs, dim):
        '''
        cblas_sgemv is used to multiply vector and matrix.

        CblasRowMajor                   -> Multiiply in a row major order
        CblasNoTrans                    -> Whether to transpose matix or not
        nvecs, dim                      -> Rows, columns of Matrix
        c_float(1.0)                    -> Scaling factor for the product of matrix and vector
        mat.ctypes.data_as(c_void_p)    -> matrix
        dim                             -> Columns of matirx
        v.ctypes.data_as(c_void_p)      -> vector
        incX                            -> Stride within X. For example, if incX is 7, every 7th element is used.
        c_float(0.0)                    -> Scaling factor for vector.
        vec_out.ctypes.data_as(c_void_p)-> result vector
        incY                            -> Stride within Y. For example, if incY is 7, every 7th element is used

        Read more                       -> https://developer.apple.com/documentation/accelerate/1513065-cblas_sgemv?language=objc
        '''
        cblas.cblas_sgemv(CblasRowMajor,
                         CblasNoTrans,
                         nvecs,
                         dim,
                         c_float(1.0),
                         mat.ctypes.data_as(c_void_p),
                         dim,
                         v.ctypes.data_as(c_void_p),
                         incX,
                         c_float(0.0),
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
        self.dim                        -> Common dimension in mat_a and mat_b
        c_float(1.0)                    -> Scaling factor for the product of matrices mat_a and mat_b
        mat_a.ctypes.data_as(c_void_p)  -> matrix mat_a
        self.dim                        -> Columns of mat_a
        mat_b.ctypes.data_as(c_void_p)  -> matrix mat_b
        self.dim                        -> Columns of mat_b
        c_float(0.0)                    -> Scaling factor for matrix mat_out
        mat_out.ctypes.data_as(c_void_p)-> matirx mat_out
        len(mat_b)                      -> Columns of mat_out

        Read more                       -> https://developer.apple.com/documentation/accelerate/1513264-cblas_sgemm?language=objc
        '''
        cblas.cblas_sgemm(CblasRowMajor,
                         CblasNoTrans,
                         CblasTrans,
                         len(mat_a),
                         len(mat_b),
                         self.dim,
                         c_float(1.0),
                         mat_a.ctypes.data_as(c_void_p),
                         self.dim,
                         mat_b.ctypes.data_as(c_void_p),
                         self.dim,
                         c_float(0.0),
                         mat_out.ctypes.data_as(c_void_p),
                         len(mat_b))
        return mat_out

    def does_word_exist(self, word):
        '''
        >>> ws = WordVecSpaceMem(DATADIR_ENV_VAR, DIMENSION)
        >>> print(ws.does_word_exist("india"))
        True
        >>> print(ws.does_word_exist("inidia"))
        False
        '''

        return word in self.word_indices

    def get_word_index(self, word, raise_exc=False):
        '''
        >>> ws = WordVecSpaceMem(DATADIR_ENV_VAR, DIMENSION)
        >>> print(ws.get_word_index("india"))
        509
        >>> print(ws.get_word_index("inidia"))
        None
        >>> print(ws.get_word_index("inidia", raise_exc=True)) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        wordvecspace.wordvecspace_mem.UnknownWord: "inidia"
        '''

        if isinstance(word, int):
            if word < self.nvecs:
                return word

        try:
            return self.word_indices[word]

        except KeyError:
            if raise_exc == True:
                raise UnknownWord(word)

    def get_word_indices(self, words, raise_exc=False):
        '''
        >>> ws = WordVecSpaceMem(DATADIR_ENV_VAR, DIMENSION)
        >>> print(ws.get_word_indices(['the', 'deepcompute', 'india']))
        [1, None, 509]
        '''

        indices = []

        if isinstance(words, (list, tuple)):
            for word in words:
                index = self.get_word_index(word, raise_exc=raise_exc)
                indices.append(index)

        return indices

    def get_word_at_index(self, index, raise_exc=False):
        '''
        >>> ws = WordVecSpaceMem(DATADIR_ENV_VAR, DIMENSION)
        >>> print(ws.get_word_at_index(509))
        india
        >>> print(ws.get_word_at_index(72000))
        None
        >>> print(ws.get_word_at_index(72000, raise_exc=True)) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        wordvecspace_mem.UnknownIndex: "72000"
        '''

        try:
            return self.words[index]

        except IndexError:
            if raise_exc == True:
                raise UnknownIndex(index)

    def get_word_at_indices(self, indices, raise_exc=False):
        '''
        >>> ws = WordVecSpaceMem(DATADIR_ENV_VAR, DIMENSION)
        >>> print(ws.get_word_at_indices([1,509,71190,72000]))
        ['the', 'india', 'reka', None]
        '''

        words = []

        if isinstance(indices, (list, tuple)):
            for index in indices:
                word = self.get_word_at_index(index, raise_exc=raise_exc)
                words.append(word)

        return words

    def get_vector_magnitude(self, word_or_index, raise_exc=False):
        '''
        >>> ws = WordVecSpaceMem(DATADIR_ENV_VAR, DIMENSION)
        >>> print(ws.get_vector_magnitude("india"))
        10.1417
        '''

        index = self.get_word_index(word_or_index, raise_exc)

        return self.magnitudes[index] if index is not None else 0.0

    def get_vector_magnitudes(self, words_or_indices, raise_exc=False):
        '''
       >>> wv = WordVecSpaceMem(DATADIR_ENV_VAR, DIMENSION)
       >>> print(wv.get_vector_magnitudes(["hi", "india"]))
       [9.36555, 10.141716]
       >>> print(wv.get_vector_magnitudes(["inidia", "india"]))
       [0.0, 10.141716]
        '''

        mag = []
        if isinstance(words_or_indices, (tuple, list)):
            for w in words_or_indices:
                mag.append(self.get_vector_magnitude(w, raise_exc=raise_exc))

        return mag

    def get_word_vector(self, word_or_index, normalized=False, raise_exc=False):
        '''
        >>> ws = WordVecSpaceMem(DATADIR_ENV_VAR, DIMENSION)
        >>> print(ws.get_word_vector('india'))
        [-8.4037  4.2569  2.7932  0.6523 -2.4258]
        >>> print(ws.get_word_vector('inidia', normalized=True))
        [ 0.  0.  0.  0.  0.]
        >>> print(ws.get_word_vector('india', normalized=True))
        [-0.8286  0.4197  0.2754  0.0643 -0.2392]
        '''

        index = self.get_word_index(word_or_index, raise_exc)
        word_vec = self._make_array(shape=self.dim, dtype=np.float32)

        if normalized and index:
            return self.vectors[index]

        if index:
            return self.vectors[index] * self.magnitudes[index]

        word_vec.fill(0.0)

        return word_vec

    def get_word_vectors(self, words_or_indices, normalized=False, raise_exc=False):
        '''
        >>> ws = WordVecSpaceMem(DATADIR_ENV_VAR, DIMENSION)
        >>> print(ws.get_word_vectors(["hi", "india"]))
        [[ 2.94   -3.3523 -6.4059 -2.1225 -4.7214]
         [-8.4037  4.2569  2.7932  0.6523 -2.4258]]
        >>> print(ws.get_word_vectors(["hi", "inidia"]))
        [[ 2.94   -3.3523 -6.4059 -2.1225 -4.7214]
         [ 0.      0.      0.      0.      0.    ]]
        '''

        wmat = []
        if isinstance(words_or_indices, (list, tuple)):
            n = len(words_or_indices)
            wmat = self._make_array(dtype=np.float32, shape=(n, self.dim))

            for i, w in enumerate(words_or_indices):
                wmat[i] = self.get_word_vector(w, normalized=normalized, raise_exc=raise_exc)

        return wmat

    def get_word_occurrence(self, word_or_index, raise_exc=False):
        '''
        >>> ws = WordVecSpaceMem(DATADIR_ENV_VAR, DIMENSION)
        >>> print(ws.get_word_occurrence("india"))
        3242
        >>> print(ws.get_word_occurrence("inidia"))
        None
        '''

        index = self.get_word_index(word_or_index, raise_exc)
        return self.word_occurrences[index] if index is not None else None

    def get_word_occurrences(self, words_or_indices, raise_exc=False):
        '''
        >>> ws = WordVecSpaceMem(DATADIR_ENV_VAR, DIMENSION)
        >>> print(ws.get_word_occurrences(['the', 'india', 'Deepcompute']))
        [1061396, 3242, None]
        >>> print(ws.get_word_occurrences(['the', 'india', 'pakistan' ]))
        [1061396, 3242, 819]
        '''

        words_occur = []

        if isinstance(words_or_indices, (list, tuple)):
            for word in words_or_indices:
                occur = self.get_word_occurrence(word, raise_exc=raise_exc)
                words_occur.append(occur)

        return words_occur

    def get_distance(self, word_or_index1, word_or_index2, raise_exc=False):
        '''
        >>> ws = WordVecSpaceMem(DATADIR_ENV_VAR, DIMENSION)
        >>> print(ws.get_distance("india", "usa"))
        0.325127840042
        '''

        return 1 - np.dot(\
            self.get_word_vector(word_or_index1, normalized=True, raise_exc=raise_exc),\
            self.get_word_vector(word_or_index2, normalized=True, raise_exc=raise_exc).T)

    def get_distances(self, row_words_or_indices, col_words_or_indices=None, raise_exc=False):
        '''
        >>> ws = WordVecSpaceMem(DATADIR_ENV_VAR, DIMENSION)
        >>> res = ws.get_distances("for", ["to", "for", "india"])
        >>> check_equal(res, np.array([[ 0.381], [ 0.    ], [ 0.9561]], dtype=np.float32), decimal=4)
        >>> res = ws.get_distances("for", ["to", "for", "inidia"])
        >>> check_equal(res, np.array([[ 0.381], [ 0.    ], [ 1.]], dtype=np.float32), decimal=4)
        >>> print(ws.get_distances(["india", "for"], ["to", "for", "usa"]))
        [[ 1.0685  0.9561  0.3251]
         [ 0.381   0.      1.4781]]
        >>> print(ws.get_distances(["india", "usa"]))
        [[ 1.3853  0.4129  0.3149 ...,  1.1231  1.4595  0.7912]
         [ 1.3742  0.9549  1.0354 ...,  0.5556  1.0847  1.0832]]

        >>> print(ws.get_distances(["andhra"]))
        [[ 1.2817  0.6138  0.2995 ...,  0.9945  1.224   0.6137]]
        '''

        only_single_row_word = False

        if not isinstance(row_words_or_indices, (tuple, list)):
            row_words_or_indices = [row_words_or_indices]
            only_single_row_word = True

        row_vectors = self.get_word_vectors(row_words_or_indices, normalized=True, raise_exc=raise_exc)
        col_vectors = self.vectors

        if col_words_or_indices is not None:
            col_vectors = self.get_word_vectors(col_words_or_indices, normalized=True, raise_exc=raise_exc)

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
    def get_nearest(self, words_or_indices, k=DEFAULT_K, combination=False, raise_exc=False):
        '''
        >>> ws = WordVecSpaceMem(DATADIR_ENV_VAR, DIMENSION)
        >>> print(ws.get_nearest("india", 20))
        [[509, 486, 523, 4343, 14208, 13942, 42424, 25578, 6212, 2475, 3560, 13508, 20919, 3389, 4484, 19995, 8776, 7012, 12191, 16619]]
        '''
        distances = self.get_distances(words_or_indices, raise_exc=raise_exc)

        if isinstance(words_or_indices, (tuple, list)):
            n = len(words_or_indices)

            l = []
            for i in range(n):
                dist = pd.Series(distances[i].reshape((len(distances[i],))))
                dist = dist.nsmallest(k).keys()
                l.append(list(dist))

            if combination:
                return list(set(l[0]).intersection(*l))

            else:
                return l

        distances = pd.Series(distances.reshape((len(distances),)))
        distances = distances.nsmallest(k)

        return [list(distances.keys())]
