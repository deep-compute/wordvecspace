import os
from math import sqrt

import numpy as np
import pandas as pd
from scipy.spatial import distance

from .fileformat import WordVecSpaceFile
from .base import WordVecSpace

np.set_printoptions(precision=4)
check_equal = np.testing.assert_array_almost_equal

# export data directory path for test cases
# export WORDVECSPACE_DATADIR=/path/to/data
DATAFILE_ENV_VAR = os.environ.get('WORDVECSPACE_DATADIR', '')

class WordVecSpaceDisk(WordVecSpace):
    METRIC = 'angular'

    def __init__(self, input_dir, metric=METRIC):
        self.metric = metric

        self._f = WordVecSpaceFile(input_dir, mode='r')

        self.dim = int(self._f.dim)
        self.nvecs = len(self._f)

    def _make_array(self, shape, dtype):
        return np.ndarray(shape, dtype)

    def does_word_exist(self, word):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.does_word_exist("india"))
        True
        >>> print(wv.does_word_exist("inidia"))
        False
        '''

        if isinstance(word, int) or not self._f.get_word_index(word):
            return False
        else:
            return True

    def get_word_index(self, word, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_index("india"))
        509
        >>> print(wv.get_word_index("inidia"))
        None
        '''

        return self._f.get_word_index(word, raise_exc=raise_exc)

    def get_word_indices(self, words, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_indices(['the', 'deepcompute', 'india']))
        [1, None, 509]
        '''

        indices = []

        if isinstance(words, (list, tuple)):
            for word in words:
                index = self._f.get_word_index(word, raise_exc=raise_exc)
                indices.append(index)

        return indices

    def get_word_at_index(self, index, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_at_index(509))
        india
        >>> print(wv.get_word_at_index(72000))
        None
        '''

        return self._f.get_word(index, raise_exc=raise_exc)

    def get_word_at_indices(self, indices, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> wv.get_word_at_indices([1,509,71190])
        ['the', 'india', 'reka']
        >>> wv.get_word_at_indices([1,509,71190,72000])
        ['the', 'india', 'reka', None]
        '''

        words = []

        if isinstance(indices, (list, tuple)):
            for i in indices:
                word = self._f.get_word(i, raise_exc=raise_exc)
                words.append(word)

        return words

    def get_vector_magnitude(self, word_or_index, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_vector_magnitude("hi"))
        1.0000000188010743
        '''

        vec = self.get_word_vector(word_or_index, raise_exc=raise_exc)

        res = 0
        for val in vec:
            res += (val * val)

        return sqrt(res)

    def get_vector_magnitudes(self, words_or_indices, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_vector_magnitudes(["hi", "india"]))
        [1.0000000188010743, 1.000000011175871]
        '''

        mag = []
        if isinstance(words_or_indices, (list, tuple)):
            for w in words_or_indices:
                mag.append(self.get_vector_magnitude(w, raise_exc=raise_exc))

        return mag

    def get_word_vector(self, word_or_index, normalized=False, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_vector('india'))
        [-0.7871 -0.2993  0.3233 -0.2864  0.323 ]
        >>> wv.get_word_vector('inidia')
        array([ 0.,  0.,  0.,  0.,  0.], dtype=float32)
        '''
        if normalized:
            return self._f.get_word_vector(word_or_index, raise_exc=raise_exc) \
                                / self.get_vector_magnitude(word_or_index)

        return self._f.get_word_vector(word_or_index, raise_exc=raise_exc)

    def get_word_vectors(self, words_or_indices, normalized=False, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
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
            wmat = self._make_array(dtype=np.float32, shape=(n, self.dim))

            for i, w in enumerate(words_or_indices):
                wmat[i] = self.get_word_vector(w, normalized=normalized, raise_exc=raise_exc)

        return wmat

    def get_word_occurrence(self, word_or_index, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_occurrence('india'))
        3242
        >>> print(wv.get_word_occurrence('inidia'))
        None
        '''

        return self._f.get_word_occurrence(word_or_index, raise_exc=raise_exc)

    def get_word_occurrences(self, words_or_indices, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_occurrences(['hi', 'india', 'bye']))
        [210, 3242, 82]
        >>> print(wv.get_word_occurrences(('hi', 'Deepcompute')))
        [210, None]
        '''

        word_occur = []

        if isinstance(words_or_indices, (list, tuple)):
            for word in words_or_indices:
                occur = self._f.get_word_occurrence(word, raise_exc=raise_exc)
                word_occur.append(occur)

        return word_occur

    def get_distance(self, word_or_index1, word_or_index2, metric=None, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
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
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> res = wv.get_distances("for", ["to", "for", "india"])
        >>> check_equal(res, np.array([[  2.7428e-01,   5.9605e-08,   1.1567e+00]], dtype=np.float32), decimal=4)
        >>> res = wv.get_distances("for", ["to", "for"])
        >>> check_equal(res, np.array([[  2.7428e-01,   5.9605e-08]], dtype=np.float32), decimal=4)
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

            col_vectors = self._f.getmany(0, self.nvecs)
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
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
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
