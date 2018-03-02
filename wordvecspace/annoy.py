import os

import numpy as np
from annoy import AnnoyIndex

from .disk import WordVecSpaceDisk
from .fileformat import WordVecSpaceFile

# export WORDVECSPACE_DATADIR=/path/to/data
DATADIR_ENV_VAR = os.environ.get('WORDVECSPACE_DATADIR', ' ')

class WordVecSpaceAnnoy(WordVecSpaceDisk):

    N_TREES = 1
    METRIC = 'angular'
    ANN_FILE = 'vectors.ann'

    def __init__(self, input_file, n_trees=N_TREES, metric=METRIC):
        super(WordVecSpaceAnnoy, self).__init__(input_file)
        self.ann = AnnoyIndex(self.dim, metric=metric)
        self.ann_file = os.path.join(os.path.dirname(input_file), self.ANN_FILE)

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
        >>> wa = WordVecSpaceAnnoy(DATADIR_ENV_VAR)
        >>> print(wa.get_distance(250, 'india'))
        1.5029966831207275
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

        >>> wa = WordVecSpaceAnnoy(DATADIR_ENV_VAR)
        >>> print(wa.get_distances("for", ["to", "for", "india"]))
        [[ 0.8729  0.      1.3828]]
        '''

        if not isinstance(row_words_or_indices, (list, tuple)):
            row_words_or_indices = [row_words_or_indices]

        if col_words_or_indices:
            if not isinstance(col_words_or_indices, (list, tuple)):
                col_words_indices = [col_words_indices]

            mat = self._make_array(shape=((len(row_words_or_indices)), len(col_words_or_indices)), dtype=np.float32)

            for i, row_word in enumerate(row_words_or_indices):
                dist = []
                for col_word in col_words_or_indices:
                    dist.append(self.get_distance(row_word, col_word))

                mat[i] = np.asarray(dist, dtype=np.float32)

        else:
            mat = self._make_array(shape=((len(row_words_or_indices)), self.nvecs), dtype=np.float32)
            dist = {}

            for i, row_word in enumerate(row_words_or_indices):
                index = self.get_word_index(row_word, raise_exc=raise_exc)
                key, val = self.ann.get_nns_by_item(index, self.nvecs, include_distances=True)
                for k, v in zip(key, val):
                    dist[k] = v

                mat[i] = np.asarray([dist[key] for key in sorted(dist.keys(), reverse=False)], dtype=np.float32)

        return mat

    DEFAULT_k = 512
    def get_nearest(self, words_or_indices, k=DEFAULT_k, combination=False, raise_exc=False):
        '''
        >>> wa = WordVecSpaceAnnoy(DATADIR_ENV_VAR)
        >>> print(wa.get_nearest(509, 10))
        [509, 486, 4343, 25578, 6049, 4137, 41883, 18617, 10172, 35704]
        '''

        if isinstance(words_or_indices, (tuple, list)):
            res = []
            for word in words_or_indices:
                index = self.get_word_index(word)

                if index:
                    res.append(self.ann.get_nns_by_item(index, k)) # will find the k nearest neighbors

            if combination and len(words_or_indices) > 1:
                return list(set(res[0]).intersection(*res))

            return res

        index = self.get_word_index(words_or_indices)

        return self.ann.get_nns_by_item(index, k)
