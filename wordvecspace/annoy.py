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
                    examine_k: int=None, combination: bool=False,
                    combination_method: str='vec_intersect',weights: list=None,
                    metric: str=METRIC, include_distances: bool=False) -> np.ndarray:

        """Returns nearest indices from the vector space.

        Arguments:
        v_w_i -- vector(s), word(s), index(es)
        k -- number of nearest items to be retrieved
        examine_k -- number of items to be examined only in set_intersect and set_union
        combination -- combination to be done or not
        combination_method -- set_intersect/set_union/distance/vector
        weights -- importance factor
        metric -- angular/cosine/euclidean
        include_distances -- know the distance value or not

        Tests:
        >>> wv = WordVecSpaceAnnoy(DATAFILE_ENV_VAR)
        >>> wv.get_nearest('india', k=5)
        array([  509,   486,  6186, 22151,  1980], dtype=uint32)
        >>> wv.get_nearest(709, k=5)
        array([  709, 15824, 26752,  7597,   242], dtype=uint32)
        >>> vecs = wv.get_vectors(['india', 'pakistan'])
        >>> vecs
        array([[-2.4544, -0.1309,  8.9653, -3.1779,  3.2016],
               [-3.3736,  0.4845,  9.7016, -3.5337,  1.0142]], dtype=float32)
        >>> wv.get_nearest(vecs, k=5)
        array([[  509,   486,  6186, 22151,  1980],
               [ 2224,  5281, 11087, 16342, 38760]], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, examine_k=500,
        ... combination=True, combination_method='set_intersect')
        array([[ 2224,   523,  5969, 24149,  9772]])
        >>> wv.get_nearest(vecs, k=5, examine_k=500,
        ... combination=True, combination_method='set_intersect', weights=[0.1, 0.9])
        array([[ 2224,  5281,  3560,  4622, 11886]])
        >>> wv.get_nearest(vecs, k=5, examine_k=500, combination=True,
        ... combination_method='set_intersect', weights=[0.1, 0.9], include_distances=True)[1]
        array([[0.024 , 0.0844, 0.1039, 0.1054, 0.1067]], dtype=float32)
        >>> wv.get_nearest(vecs, k=5, examine_k=500,
        ... combination=True, combination_method='set_union')
        array([[  509,   486,  6186, 22151,  1980]], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, examine_k=500,
        ... combination=True, combination_method='set_union', weights=[0.9, -0.9])
        array([[64001, 12629, 65521, 56911,  9621]], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, examine_k=500, combination=True,
        ... combination_method='set_union', weights=[0.1, 0.9], include_distances=True)[1]
        array([[0.    , 0.0058, 0.0077, 0.01  , 0.011 ]], dtype=float32)
        >>> wv.get_nearest(vecs, k=10, combination=True, combination_method='vector')
        array([  523,  5969,  4622, 18072,  2224, 10018,  5226,  8012,  5281,
               17910], dtype=uint32)
        >>> wv.get_nearest(vecs, k=10, combination=True, combination_method='vector', include_distances=True)[1]
        array([0.0066, 0.0658, 0.1132, 0.1134, 0.1154, 0.1315, 0.1343, 0.1544,
               0.1579, 0.169 ], dtype=float32)
        """
        if not combination:
            return self._get_brute(v_w_i, k, include_distances)

        if combination and combination_method == 'set_intersect':
            return self._get_set_intersect(v_w_i, k, examine_k, metric, weights, include_distances)

        if combination and combination_method == 'set_union':
            return self._get_set_union(v_w_i, k, examine_k, metric, weights, include_distances)

        if combination and combination_method == 'vector':
            return self._get_resultant_vector_nearest(v_w_i, k, weights, metric, include_distances)

    def _get_resultant_vector_nearest(self, v_w_i, k, weights, metric, include_distances):
        """Retrieves nearest indices based on the resultant vector.

        Algorithm:
        i) Compute the resultant vector by summing per dimension across vectors
        ii) Commpute distances based on this resultant vector
        iii) Return nearest indices after sorting
        """

        v = self._check_vec(v_w_i)

        res = None
        if not weights:
            weights = np.ones(len(v_w_i))
        else:
            weights = np.array(weights)

        if isinstance(v, (list, np.ndarray)):
            resultant_vec = (v * weights[:, None]).sum(axis=0, dtype=np.float32)
            res, distances= self.ann.get_nns_by_vector(resultant_vec, k, include_distances=True)

        return (np.array(res, dtype=np.uint32), np.array(distances, dtype=np.float32)) if include_distances else np.array(res, dtype=np.uint32)

    # FIXME: add include_distances for list of vectors
    def _get_brute(self, v_w_i, k, include_distances):
        """Retrives nearest indices when combination=False"""

        if not isinstance(v_w_i, (tuple, list, np.ndarray)):
            index = self._check_index_or_word(v_w_i)

            nearest_indices, distances =  self.ann.get_nns_by_item(index, k, include_distances=True)
            return ((np.array(nearest_indices, dtype=np.uint32), np.array(distances, dtype=np.float32)) if include_distances else np.array(nearest_indices, dtype=np.uint32))
        else:
            res = list()
            for item in v_w_i:
                if isinstance(item, (int, str)):
                    index = self._check_index_or_word(item)
                    res.append(self.ann.get_nns_by_item(index, k))
                else:
                    res.append(self.ann.get_nns_by_vector(item, k))

            return np.asarray(res, dtype=np.uint32)

    def _get_set_intersect(self, v_w_i, k, examine_k, metric, weights, include_distances):
        """Performs set_intersect combination method.

        Algorithm:
        i) retrive nearest indices for each of the item
        ii) do a set intersection operation
        iii) find the corresponding distance of each item in interection set
        iv) retrive nearest indices from intersection set
        v) return
        """

        v = self._check_vec(v_w_i)
        indices = list()
        distances = list()
        if not weights:
            weights = np.ones(len(v))
        else:
            weights = np.array(weights)
        for item in v:
            idx, dist = self.ann.get_nns_by_vector(item, examine_k, include_distances=True)
            indices.append(idx)
            distances.append(dist)

        intersect = np.array(list(set(indices[0]).intersection(*indices)))
        corresponding_dists = list()

        indices = np.array(indices, dtype=np.uint32)
        distances = np.array(distances, dtype=np.float32)

        for index in intersect:
            corresponding_dists.append(self._get_corresponding_dist(v, index, indices, distances, weights))

        corresponding_dists = np.array(corresponding_dists, np.float32)
        intersect_indices, distances = self._nearest_sorting(corresponding_dists.reshape(1, len(corresponding_dists)), k)
        nearest_indices = intersect[intersect_indices]

        return (nearest_indices, distances) if include_distances else nearest_indices

    def _get_set_union(self, v_w_i, k, examine_k, metric, weights, include_distances):
        """Performs set_union combination method.

        Algorithm:
        i) retrive nearest indices for each item
        ii) do a set union operation
        iii) find the corresponding distance of each of the in union set
        iv) retrieve nearest indices from union set
        v) return
        """

        v = self._check_vec(v_w_i)
        indices = list()
        distances = list()
        if not weights:
            weights = np.ones(len(v))
        else: weights = np.array(weights)

        for item in v:
            idx, dist = self.ann.get_nns_by_vector(item, examine_k, include_distances=True)
            indices.append(idx)
            distances.append(dist)

        union = np.array(list(set().union(*indices)), dtype=np.uint32)
        corresponding_dists = list()

        indices = np.array(indices, dtype=np.uint32)
        distances = np.array(distances, dtype=np.float32)

        for index in union:
            corresponding_dists.append(self._get_corresponding_dist(v, index, indices, distances, weights))

        corresponding_dists = np.array(corresponding_dists, np.float32)
        union_indices, distances = self._nearest_sorting(corresponding_dists.reshape(1, len(corresponding_dists)), k)
        nearest_indices = union[union_indices]

        return (nearest_indices, distances) if include_distances else nearest_indices
