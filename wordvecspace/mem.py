import os
from math import sqrt

from scipy.spatial import distance
import numpy as np
import pandas as pd
from numba import guvectorize

from .fileformat import WordVecSpaceFile
from .base import WordVecSpace
from .exception import UnknownIndex, UnknownWord

np.set_printoptions(precision=4)
check_equal = np.testing.assert_array_almost_equal

# export data directory path for test cases
# $export WORDVECSPACE_DATADIR=/path/to/data/
DATAFILE_ENV_VAR = os.environ.get('WORDVECSPACE_DATADIR', '')

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
    METRIC = 'angular'

    def __init__(self, input_dir, metric=METRIC):
        '''
        >>> _f = WordVecSpaceFile(DATAFILE_ENV_VAR, mode='r')
        >>> nvecs = len(_f)
        >>> print(nvecs)
        71291
        >>> dim = int(_f.dim)
        >>> print(dim)
        5
        >>> vectors = _f.getmany(0, nvecs)
        >>> word_occurrences = _f.getmany(0, nvecs, _f.OCCURRENCE)
        >>> print(word_occurrences)
        [      0 1061396  593677 ...,       5       5       5]
        >>> magnitudes = np.ndarray(nvecs, dtype=np.float32)
        >>> normalize_vectors(vectors, magnitudes)
        array([ 1.,  1.,  1., ...,  1.,  1.,  1.], dtype=float32)
        '''

        self.metric = metric

        self._f = WordVecSpaceFile(input_dir, mode='r')
        self.nvecs = len(self._f)
        self.dim = int(self._f.dim)

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

    def does_word_exist(self, word):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.does_word_exist("india"))
        True
        >>> print(wv.does_word_exist("inidia"))
        False
        '''

        return word in self.word_indices

    def get_word_index(self, word, raise_exc=False):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_index("india"))
        509
        >>> print(wv.get_word_index("inidia"))
        None
        >>> print(wv.get_word_index("inidia", raise_exc=True)) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        wordvecspace.exception.UnknownWord: "inidia"
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
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_indices(['the', 'deepcompute', 'india']))
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
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_at_index(509))
        india
        >>> print(wv.get_word_at_index(72000))
        None
        >>> print(wv.get_word_at_index(72000, raise_exc=True)) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        wordvecspace.exception.UnknownIndex: "72000"
        '''

        try:
            return self.words[index]

        except IndexError:
            if raise_exc == True:
                raise UnknownIndex(index)

    def get_word_at_indices(self, indices, raise_exc=False):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_at_indices([1,509,71190,72000]))
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
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_vector_magnitude("india"))
        1.0
        '''

        index = self.get_word_index(word_or_index, raise_exc)

        return self.magnitudes[index] if index is not None else 0.0

    def get_vector_magnitudes(self, words_or_indices, raise_exc=False):
        '''
       >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
       >>> print(wv.get_vector_magnitudes(["hi", "india"]))
       [1.0, 1.0]
       >>> print(wv.get_vector_magnitudes(["inidia", "india"]))
       [0.0, 1.0]
        '''

        mag = []
        if isinstance(words_or_indices, (tuple, list)):
            for w in words_or_indices:
                mag.append(self.get_vector_magnitude(w, raise_exc=raise_exc))

        return mag

    def get_word_vector(self, word_or_index, normalized=False, raise_exc=False):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_vector('india'))
        [-0.7871 -0.2993  0.3233 -0.2864  0.323 ]
        >>> print(wv.get_word_vector('inidia', normalized=True))
        [ 0.  0.  0.  0.  0.]
        >>> print(wv.get_word_vector('india', normalized=True))
        [-0.7871 -0.2993  0.3233 -0.2864  0.323 ]
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
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_vectors(["hi", "india"]))
        [[ 0.6342  0.2268 -0.3904  0.0368  0.6266]
         [-0.7871 -0.2993  0.3233 -0.2864  0.323 ]]
        >>> print(wv.get_word_vectors(["hi", "inidia"]))
        [[ 0.6342  0.2268 -0.3904  0.0368  0.6266]
         [ 0.      0.      0.      0.      0.    ]]
        '''

        wmat = []
        if isinstance(words_or_indices, (list, tuple)):
            n = len(words_or_indices)
            wmat = self._make_array(shape=(n, self.dim), dtype=np.float32)

            for i, w in enumerate(words_or_indices):
                wmat[i] = self.get_word_vector(w, normalized=normalized, raise_exc=raise_exc)

        return wmat

    def get_word_occurrence(self, word_or_index, raise_exc=False):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_occurrence("india"))
        3242
        >>> print(wv.get_word_occurrence("inidia"))
        None
        '''
        index = self.get_word_index(word_or_index, raise_exc)
        return self.word_occurrences[index] if index is not None else None

    def get_word_occurrences(self, words_or_indices, raise_exc=False):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_occurrences(['the', 'india', 'Deepcompute']))
        [1061396, 3242, None]
        >>> print(wv.get_word_occurrences(['the', 'india', 'pakistan' ]))
        [1061396, 3242, 819]
        '''

        words_occur = []

        if isinstance(words_or_indices, (list, tuple)):
            for word in words_or_indices:
                occur = self.get_word_occurrence(word, raise_exc=raise_exc)
                words_occur.append(occur)

        return words_occur

    def get_distance(self, word_or_index1, word_or_index2, metric=None, raise_exc=False):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_distance("india", "usa"))
        0.37698328495
        >>> print(wv.get_distance("india", "usa", metric='euclidean'))
        0.8683125376701355
        '''

        if not metric:
            metric = self.metric

        if metric == 'angular':
            vec1 = self.get_word_vector(word_or_index1, normalized=True, raise_exc=raise_exc)
            vec2 = self.get_word_vector(word_or_index2, normalized=True, raise_exc=raise_exc)

            return 1 - np.dot(vec1, vec2.T)

        elif metric == 'euclidean':
            vec1 = self.get_word_vector(word_or_index1, raise_exc=raise_exc)
            vec2 = self.get_word_vector(word_or_index2, raise_exc=raise_exc)

            return distance.euclidean(vec1, vec2)

    def get_distances(self, row_words_or_indices, col_words_or_indices=None, metric=None, raise_exc=False):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> res = wv.get_distances("for", ["to", "for", "india"])
        >>> check_equal(res, np.array([[  2.7428e-01,   5.9605e-08,   1.1567e+00]], dtype=np.float32), decimal=4)
        >>> res = wv.get_distances("for", ["to", "for", "inidia"])
        >>> check_equal(res, np.array([[  2.7428e-01,   5.9605e-08,   1.0000e+00]], dtype=np.float32), decimal=4)
        >>> print(wv.get_distances(["india", "for"], ["to", "for", "usa"]))
        [[  1.1445e+00   1.1567e+00   3.7698e-01]
         [  2.7428e-01   5.9605e-08   1.6128e+00]]
        >>> print(wv.get_distances(["india", "usa"]))
        [[ 1.5464  0.4876  0.3017 ...,  1.2492  1.2451  0.8925]
         [ 1.0436  0.9995  1.0913 ...,  0.6996  0.8014  1.1608]]
        >>> print(wv.get_distances(["andhra"]))
        [[ 1.5418  0.7153  0.277  ...,  1.1657  1.0774  0.7036]]
        >>> print(wv.get_distances(["andhra"], metric='euclidean'))
        [[ 1.756   1.1961  0.7443 ...,  1.5269  1.4679  1.1862]]
        '''

        r = row_words_or_indices
        c = col_words_or_indices

        if not metric:
            metric = self.metric

        if not isinstance(r, (tuple, list)):
            r = [r]

        if c:
            if not isinstance(c, (tuple, list)):
                c = [c]

        if metric == 'angular':
            row_vectors = self.get_word_vectors(r, normalized=True, raise_exc=raise_exc)

            col_vectors = self.vectors
            if c:
                col_vectors = self.get_word_vectors(c, normalized=True, raise_exc=raise_exc)

            if len(r) == 1:
                nvecs, dim = col_vectors.shape

                vec_out = self._make_array((len(col_vectors), len(row_vectors)), dtype=np.float32)
                res = self._perform_sgemv(col_vectors, row_vectors, vec_out, nvecs, dim).T

            else:
                mat_out = self._make_array((len(row_vectors), len(col_vectors)), dtype=np.float32)
                res = self._perform_sgemm(row_vectors, col_vectors, mat_out)

            return 1 - res

        elif metric == 'euclidean':
            row_vectors = self.get_word_vectors(r, raise_exc=raise_exc)

            if c:
                col_vectors = self.get_word_vectors(c, raise_exc=raise_exc)
            else:
                col_vectors = self._f.getmany(0, self.nvecs)

            return distance.cdist(row_vectors, col_vectors, 'euclidean')

    DEFAULT_K = 512

    def get_nearest(self, words_or_indices, k=DEFAULT_K, combination=False, metric=None, raise_exc=False):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_nearest("india", 20))
        [509, 3389, 486, 523, 7125, 16619, 4491, 12191, 6866, 8776, 15232, 14208, 5998, 21916, 5226, 6322, 4343, 6212, 10172, 6186]
        >>> print(wv.get_nearest("india", 20, metric='euclidean'))
        [509, 3389, 486, 523, 7125, 16619, 4491, 12191, 6866, 8776, 15232, 14208, 5998, 21916, 5226, 6322, 4343, 6212, 10172, 6186]
        '''

        if not metric:
            metric = self.metric

        distances = self.get_distances(words_or_indices, metric=metric, raise_exc=raise_exc)

        ner = []
        for dist in distances:
            dist = pd.Series(dist.reshape((len(dist,))))
            dist = dist.nsmallest(k).keys()
            ner.append(list(dist))

        if combination:
            return list(set(ner[0]).intersection(*ner))

        return ner if isinstance(words_or_indices, (list, tuple)) else ner[0]
