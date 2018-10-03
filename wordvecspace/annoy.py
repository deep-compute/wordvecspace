import os
from typing import Union

import numpy as np
from annoy import AnnoyIndex

from .disk import WordVecSpaceDisk

# export data directory path for test cases
# export WORDVECSPACE_DATADIR=/path/to/data
DATAFILE_ENV_VAR = os.environ.get('WORDVECSPACE_DATADIR', '')

check_equal = np.testing.assert_array_almost_equal

class WordVecSpaceAnnoy(WordVecSpaceDisk):

    N_TREES = 1
    METRIC = 'angular'
    ANN_FILE = 'vectors.ann'

    def __init__(self, input_dir, n_trees=N_TREES, metric=METRIC, index_fpath=None):
        super().__init__(input_dir)
        self.ann = AnnoyIndex(self.dim, metric=metric)

        self.ann_file = self.J(input_dir, self.ANN_FILE)
        if index_fpath:
            self.ann_file = self.J(index_fpath, self.ANN_FILE)

        self._create_annoy_file(n_trees)

        self.ann.load(self.ann_file)

    def J(self, p1, p2):
        return os.path.join(p1, p2)

    def _create_annoy_file(self, n_trees):
        for i in range(self.nvecs):
            v = self.vecs[i]
            self.ann.add_item(i, v)

        self.ann.build(n_trees)
        self.ann.save(self.ann_file)

    def get_distance(self, word_or_index1: Union[int, str, np.ndarray],\
                    word_or_index2: Union[int, str, np.ndarray]):
        v1 = self._check_index_or_word(word_or_index1)
        v2 = self._check_index_or_word(word_or_index2)

        return self.ann.get_distance(v1, v2)

    def get_distances(self, row_words_or_indices: Union[int, str, np.ndarray],\
                    col_words_or_indices: Union[int, str, np.ndarray, None]=None):

        r = row_words_or_indices
        c = col_words_or_indices

        if not isinstance(r, (list, tuple, np.ndarray)):
            r = [r]

        if c:
            if not isinstance(c, (list, tuple, np.ndarray)):
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
                index = self._check_index_or_word(row_word)
                key, val = self.ann.get_nns_by_item(index, self.nvecs, include_distances=True)

                for k, v in zip(key, val):
                    dist[k] = v

                mat[i] = np.asarray([dist[key] for key in sorted(dist.keys(), reverse=False)], dtype=np.float32)

        return mat

    DEFAULT_K = 512
    def get_nearest(self, v_w_i: Union[int, str, np.ndarray], k: int=DEFAULT_K,\
                    combination: bool=False):
        if isinstance(v_w_i, (tuple, list)):
            res = []
            for word in v_w_i:
                index = self._check_index_or_word(word)

                if index:
                    res.append(self.ann.get_nns_by_item(index, k)) # will find the k nearest neighbors

            # will find common nearest neighbors among given words
            if combination and len(v_w_i) > 1:
                return list(set(res[0]).intersection(*res))

            return res

        index = self._check_index_or_word(v_w_i)

        return self.ann.get_nns_by_item(index, k)
