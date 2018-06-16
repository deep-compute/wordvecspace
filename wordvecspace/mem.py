import os
import json

from scipy.spatial import distance
import numpy as np
import bottleneck

from .fileformat import WordVecSpaceFile
from .base import WordVecSpace

np.set_printoptions(precision=4)
check_equal = np.testing.assert_array_almost_equal

# export data directory path for test cases
# $export WORDVECSPACE_DATADIR=/path/to/data/
DATAFILE_ENV_VAR = os.environ.get('WORDVECSPACE_DATADIR', '')

class WordVecSpaceMem(WordVecSpace):
    METRIC = 'cosine'

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
        '''

        self.input_dir = input_dir
        self.metric = metric

        self._f = WordVecSpaceFile(input_dir, mode='r')
        self.nvecs = len(self._f)
        self.dim = int(self._f.dim)

        self.vectors = self._f._vectors['vector']
        #self.words = self._f.getmany(0, self.nvecs, self._f.WORD)

        self.word_occurrences = self._f._occurrences['occurrence']
        self.magnitudes = self._f._magnitudes['magnitude']

        #self.word_indices = {}
        #for index, w in enumerate(self.words):
        #    self.word_indices[w] = int(index)

    def _make_array(self, shape, dtype):
        return np.ndarray(shape, dtype)

    def _check_index_or_word(self, item):
        if isinstance(item, str):
            return self.get_index(item)
        return item

    def _check_indices_or_words(self, items):
        w = items

        if len(w) == 0:
            return []

        if isinstance(w, np.ndarray):
            assert(w.dtype == np.uint32 and len(w.shape) == 1)

        if isinstance(w, (list, tuple)):
            if isinstance(w[0], str):
                return self.get_indices(w)
        return w

    def get_manifest(self):
        manifest_info = open(os.path.join(self.input_dir, 'manifest.json'), 'r')
        manifest_info = json.loads(manifest_info.read())

        return manifest_info

    def does_word_exist(self, word):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.does_word_exist("india"))
        True
        >>> print(wv.does_word_exist("inidia"))
        False
        '''

        return word in self.word_indices

    def get_index(self, word):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_index("india"))
        509
        >>> print(wv.get_word_index("inidia"))
        None
        >>> print(wv.get_word_index("inidia")) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        wordvecspace.exception.UnknownWord: "inidia"
        '''
        assert(isinstance(word, str))

        return self.word_indices[word]

    def get_indices(self, words):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_indices(['the', 'deepcompute', 'india']))
        [1, None, 509]
        '''

        assert(isinstance(words, (tuple, list)) and len(words) != 0)

        indices = [self.word_indices[w] for w in words]

    def get_word(self, index):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_at_index(509))
        india
        >>> print(wv.get_word_at_index(72000))
        None
        >>> print(wv.get_word_at_index(72000)) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        wordvecspace.exception.UnknownIndex: "72000"
        '''
        return self.words[index]

    def get_words(self, indices):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_at_indices([1,509,71190,72000]))
        ['the', 'india', 'reka', None]
        '''
        return [self.words[i] for i in indices)

    def get_magnitude(self, word_or_index):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_vector_magnitude("india"))
        1.0
        '''

        index = self._check_index_or_word(word_or_index)

        return self.magnitudes[index]

    def get_magnitudes(self, words_or_indices):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_vector_magnitudes(["hi", "india"]))
        [1.0, 1.0]
        >>> print(wv.get_vector_magnitudes(["inidia", "india"]))
        [0.0, 1.0]
        '''

        w = self._check_indices_or_words(words_or_indices)

        return self.magnitudes.take(w)

    def get_occurrence(self, word_or_index):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_occurrence("india"))
        3242
        >>> print(wv.get_word_occurrence("inidia"))
        None
        '''
        index = self._check_index_or_word(word_or_index)

        return self.word_occurrences[index]

    def get_occurrences(self, words_or_indices):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_occurrences(['the', 'india', 'Deepcompute']))
        [1061396, 3242, None]
        >>> print(wv.get_word_occurrences(['the', 'india', 'pakistan' ]))
        [1061396, 3242, 819]
        '''

        w = self._check_indices_or_words(words_or_indices)

        return self.word_occurrences.take(w)

    def get_vector(self, word_or_index, normalized=False):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_vector('india'))
        [-0.7871 -0.2993  0.3233 -0.2864  0.323 ]
        >>> print(wv.get_word_vector('inidia', normalized=True))
        [ 0.  0.  0.  0.  0.]
        >>> print(wv.get_word_vector('india', normalized=True))
        [-0.7871 -0.2993  0.3233 -0.2864  0.323 ]
        '''

        index = self._check_index_or_word(word_or_index)

        if normalized:
            return self.vectors[index]

        return self.vectors[index] * self.magnitudes[index]

    def get_vectors(self, words_or_indices, normalized=False):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_vectors(["hi", "india"]))
        [[ 0.6342  0.2268 -0.3904  0.0368  0.6266]
         [-0.7871 -0.2993  0.3233 -0.2864  0.323 ]]
        >>> print(wv.get_word_vectors(["hi", "inidia"]))
        [[ 0.6342  0.2268 -0.3904  0.0368  0.6266]
         [ 0.      0.      0.      0.      0.    ]]
        '''

        w = self._check_indices_or_words(words_or_indices)

        if normalized:
            return self.vectors.take(w, axis=0)

        vecs = self.vectors.take(w, axis=0)
        mags = self.magnitudes.take(w)

        return np.multiply(vecs.T, mags).T

    def get_distance(self, word_or_index1, word_or_index2, metric=None):
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
            vec1 = self.get_vector(word_or_index1, normalized=True)
            vec2 = self.get_vector(word_or_index2, normalized=True)

            return 1 - np.dot(vec1, vec2.T)

        elif metric == 'euclidean':
            vec1 = self.get_vector(word_or_index1)
            vec2 = self.get_vector(word_or_index2)

            return distance.euclidean(vec1, vec2)

    def get_distances(self, row_words_or_indices, col_words_or_indices=None, metric=None):
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

        if not isinstance(r, (tuple, list, np.ndarray)):
            r = [r]

        if c is not None and len(c):
            if not isinstance(c, (tuple, list, np.ndarray)):
                c = [c]

        if metric == 'cosine':
            if isinstance(r, np.ndarray) and len(r.shape) == 2 and r.dtype==np.float32:
                row_vectors = r
            else:
                row_vectors = self.get_vectors(r, normalized=True)

            col_vectors = self.vectors
            if c is not None and len(c):
                if isinstance(c, np.ndarray) and len(c.shape) == 2 and c.dtype==np.float32:
                    col_vectors = c
                else:
                    col_vectors = self.get_vectors(c, normalized=True)

            if len(r) == 1:
                nvecs, dim = col_vectors.shape

                vec_out = self._make_array((len(col_vectors), len(row_vectors)), dtype=np.float32)

                res = self._perform_sgemv(row_vectors, col_vectors, vec_out, nvecs, dim)

            else:
                mat_out = self._make_array((len(row_vectors), len(col_vectors)), dtype=np.float32)
                res = self._perform_sgemm(row_vectors, col_vectors, mat_out)

            return 1 - res

        elif metric == 'euclidean':
            row_vectors = self.get_vectors(r)

            if c:
                col_vectors = self.get_vectors(c)
            else:
                col_vectors = self.vectors

            return distance.cdist(row_vectors, col_vectors, 'euclidean')

    DEFAULT_K = 512

    def get_nearest(self, v_w_i, k=DEFAULT_K, distances=False, combination=False, metric=None):
        '''
        >>> wv = WordVecSpaceMem(DATAFILE_ENV_VAR)
        >>> print(wv.get_nearest("india", 20))
        [509, 3389, 486, 523, 7125, 16619, 4491, 12191, 6866, 8776, 15232, 14208, 5998, 21916, 5226, 6322, 4343, 6212, 10172, 6186]
        >>> print(wv.get_nearest("india", 20, metric='euclidean'))
        [509, 3389, 486, 523, 7125, 16619, 4491, 12191, 6866, 8776, 15232, 14208, 5998, 21916, 5226, 6322, 4343, 6212, 10172, 6186]
        '''

        d = self.get_distances(v_w_i, metric=metric)

        ner = self._make_array(shape=(len(d), k), dtype=np.uint32)
        dist = self._make_array(shape=(len(d), k), dtype=np.float32)

        for index, p in enumerate(d):
            b_sort = bottleneck.argpartition(p, k)[:k]
            _sort = np.take(d, b_sort)
            _sorted = np.argsort(_sort)
            ner[index] = _sorted
            dist[index] = np.take(p, _sorted)

        if combination:
            ner = set(ner[0]).intersection(*ner)
            return (ner, dist) if distances else ner

        if isinstance(v_w_i, (list, tuple)):
            return (ner, dist) if distances else ner
        else:
            return (ner[0], dist[0]) if distances else ner[0]
