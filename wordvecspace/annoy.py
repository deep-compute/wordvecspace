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
    DEFAULT_K = 512

    def __init__(self, input_dir, n_trees=N_TREES, metric=METRIC, index_fpath=None):
        super().__init__(input_dir)
        self.ann = AnnoyIndex(self.dim, metric=metric)

        self.ann_file = self.J(input_dir, self.ANN_FILE)

        if index_fpath:
            self.ann_file = self.J(index_fpath, self.ANN_FILE)

        if not os.path.exists(self.ann_file):
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

    def get_nearest(self, v_w_i: Union[int, str, list], k: int=DEFAULT_K,
                    metric: str=METRIC, include_distances: bool=False, combination: bool=False,
                    combination_method: str='vec_intersect', weights: list=None) -> np.ndarray:

        if not combination or combination_method == 'vec_intersect':
            return self._get_brute(v_w_i, k, combination, include_distances)

        if combination and combination_method == 'vector':
            return self._get_resultant_vector_nearest(v_w_i, k, weights, metric, include_distances)

    def _get_resultant_vector_nearest(self, v_w_i, k, weights, metric, include_distances):
        v = self._check_vec(v_w_i)

        res = None
        if not weights:
            weights = np.ones(len(v_w_i))
        else:
            weights = np.array(weights)

        if isinstance(v, (list, np.ndarray)):
            resultant_vec = (v * weights[:, None]).sum(axis=0)
            res = self.ann.get_nns_by_vector(resultant_vec, k, include_distances=include_distances)

        return res

    # FIXME: add include_distances for list of vectors
    def _get_brute(self, v_w_i, k, combination, include_distances):

        if not isinstance(v_w_i, (tuple, list, np.ndarray)):
            index = self._check_index_or_word(v_w_i)

            return self.ann.get_nns_by_item(index, k, include_distances=include_distances)
        else:
            res = list()
            for item in v_w_i:
                if isinstance(item, (int, str)):
                    index = self._check_index_or_word(item)
                    res.append(self.ann.get_nns_by_item(index, k))
                else:
                    res.append(self.ann.get_nns_by_vector(item, k))

            if combination:
                return list(set(res[0]).intersection(*res))
            else:
                return res
