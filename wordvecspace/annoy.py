import os

import numpy as np
from annoy import AnnoyIndex

from .disk import WordVecSpaceDisk
from .fileformat import WordVecSpaceFile

# export data directory path for test cases
# export WORDVECSPACE_DATADIR=/path/to/data
DATAFILE_ENV_VAR = os.environ.get('WORDVECSPACE_DATADIR', '')

check_equal = np.testing.assert_array_almost_equal

class WordVecSpaceAnnoy(WordVecSpaceDisk):

    N_TREES = 1
    METRIC = 'angular'
    ANN_FILE = 'vectors.ann'

    def __init__(self, input_dir, n_trees=N_TREES, metric=METRIC, index_fpath=None):
        super(WordVecSpaceAnnoy, self).__init__(input_dir)
        self.ann = AnnoyIndex(self.dim, metric=metric)

        self.ann_file = os.path.join(input_dir, self.ANN_FILE)
        if index_fpath:
            self.ann_file = os.path.join(index_fpath, self.ANN_FILE)

        self._create_annoy_file(n_trees)

        self.ann.load(self.ann_file)

    def _create_annoy_file(self, n_trees):
        for i in range(self.nvecs):
            v = self._f.get(i, self._f.VECTOR).reshape(self.dim, )
            self.ann.add_item(i, v)

        self.ann.build(n_trees)
        self.ann.save(self.ann_file)

    def get_distance(self, word_or_index1, word_or_index2, raise_exc=False):
        '''
        >>> wv = WordVecSpaceAnnoy(DATAFILE_ENV_VAR)
        >>> print(wv.get_distance(250, 'india'))
        1.5112241506576538
        '''

        v1 = self.get_word_index(word_or_index1, raise_exc=raise_exc)
        v2 = self.get_word_index(word_or_index2, raise_exc=raise_exc)

        return self.ann.get_distance(v1, v2)

    def get_distances(self, row_words_or_indices, col_words_or_indices=None, raise_exc=False):
        '''
        get_distances(word)
        get_distances(words)
        get_distances(word, words)
        get_distances(words_x, words_y)

        >>> wv = WordVecSpaceAnnoy(DATAFILE_ENV_VAR)
        >>> res = wv.get_distances("for", ["to", "for", "india"])
        >>> check_equal(res, np.array([[ 0.7407,  0.    ,  1.521 ]], dtype=np.float32), decimal=4)
        '''

        r = row_words_or_indices
        c = col_words_or_indices

        if not isinstance(r, (list, tuple)):
            r = [r]

        if c:
            if not isinstance(c, (list, tuple)):
                c = [c]

            mat = self._make_array(shape=((len(r)), len(c)), dtype=np.float32)

            for i, row_word in enumerate(r):
                dist = []
                for col_word in c:
                    dist.append(self.get_distance(row_word, col_word))

                mat[i] = np.asarray(dist, dtype=np.float32)

        else:
            mat = self._make_array(shape=((len(r)), self.nvecs), dtype=np.float32)
            dist = {}

            for i, row_word in enumerate(r):
                index = self.get_word_index(row_word, raise_exc=raise_exc)
                key, val = self.ann.get_nns_by_item(index, self.nvecs, include_distances=True)

                for k, v in zip(key, val):
                    dist[k] = v

                mat[i] = np.asarray([dist[key] for key in sorted(dist.keys(), reverse=False)], dtype=np.float32)

        return mat

    DEFAULT_K = 512
    def get_nearest(self, words_or_indices, k=DEFAULT_K, combination=False, raise_exc=False):
        '''
        >>> wv = WordVecSpaceAnnoy(DATAFILE_ENV_VAR)
        >>> print(wv.get_nearest(509, 10))
        [509, 16619, 4491, 6866, 8776, 14208, 5998, 2325, 4622, 3546]
        '''

        if isinstance(words_or_indices, (tuple, list)):
            res = []
            for word in words_or_indices:
                index = self.get_word_index(word, raise_exc=raise_exc)

                if index:
                    res.append(self.ann.get_nns_by_item(index, k)) # will find the k nearest neighbors

            # will find common nearest neighbors among given words
            if combination and len(words_or_indices) > 1:
                return list(set(res[0]).intersection(*res))

            return res

        index = self.get_word_index(words_or_indices, raise_exc=raise_exc)

        return self.ann.get_nns_by_item(index, k)
