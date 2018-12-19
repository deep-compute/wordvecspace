import os
from typing import Union

import numpy as np
import pandas as pd
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
        """Joins two paths.

        Arguments:
        p1 -- first path
        p2 -- second path

        Tests:
        >>> wv = WordVecSpaceAnnoy(DATAFILE_ENV_VAR)
        >>> wv.J('path', 'sub_path')
        'path/sub_path'
        """

        return os.path.join(p1, p2)

    def _create_annoy_file(self, n_trees):
        for i in range(self.nvecs):
            v = self.vecs[i]
            self.ann.add_item(i, v)

        self.ann.build(n_trees)
        self.ann.save(self.ann_file)

    def get_distance(self, word_or_index1: Union[int, str, np.ndarray],
                     word_or_index2: Union[int, str, np.ndarray]):
        """Get distance between two vectors given corresponding word/index.

        Arguments:
        word_or_index1 -- first word/index
        word_or_index2 -- second word/index

        Tests:
        >>> wv = WordVecSpaceAnnoy(DATAFILE_ENV_VAR)
        >>> wv.get_distance('timon', 'king')
        1.0100256204605103
        """
        v1 = self._check_index_or_word(word_or_index1)
        v2 = self._check_index_or_word(word_or_index2)

        return self.ann.get_distance(v1, v2)

    def get_distances(self, row_words_or_indices: Union[int, str, np.ndarray],
                      col_words_or_indices: Union[int, str, np.ndarray, None]=None):
        """Get distance between multiple indices/words and all the vectors in the vector space.

        Arguments:
        row_words_or_indices -- row words/indices
        col_words_or_indices -- columns words/indices

        Tests:
        >>> wv = WordVecSpaceAnnoy(DATAFILE_ENV_VAR)
        >>> wv.get_distances('timon')
        array([[1.3154, 1.6462, 1.5289, ..., 0.5316, 0.2373, 1.0765]],
              dtype=float32)
        >>> wv.get_distances(['timon', 'lion'])
        array([[1.3154, 1.6462, 1.5289, ..., 0.5316, 0.2373, 1.0765],
               [1.5923, 1.6938, 1.7747, ..., 0.3426, 0.8104, 1.5716]],
              dtype=float32)
        """

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
                    combination_method: str='vec_intersect', weights: list=None,
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
        >>> wv.get_nearest('laptop', k=5)
        array([15890, 32156, 31197, 32342, 20573], dtype=uint32)
        >>> wv.get_nearest('phone', k=5)
        array([ 3826, 10458,  2686, 20515, 39195], dtype=uint32)
        >>> vecs = wv.get_vectors(['laptop', 'phone'])
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='vector')
        array([10101, 27639, 25905, 28303,  8971], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='vector',
        ... include_distances=True)
        (array([10101, 27639, 25905, 28303,  8971], dtype=uint32), array([0.0529, 0.0622, 0.1041, 0.1386, 0.1539], dtype=float32))
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='vector',
        ... include_distances=True)
        (array([10101, 27639, 25905, 28303,  8971], dtype=uint32), array([0.0529, 0.0622, 0.1041, 0.1386, 0.1539], dtype=float32))
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='vector',
        ... weights=[0.2, 0.5], include_distances=True)
        (array([ 8350, 29213, 24296, 25509,  8618], dtype=uint32), array([0.0429, 0.0499, 0.0811, 0.0986, 0.1031], dtype=float32))
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_intersect',
        ... weights=[0.2, 0.5], examine_k=500)
        array([15445, 14669, 12198,  3801, 22196], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_intersect',
        ... examine_k=500)
        array([11044, 10665, 41600, 17728, 15445], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_intersect',
        ... examine_k=500, weights=[0.2, 0.8])
        array([ 5177, 14669, 15189, 15445,  3801], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_intersect',
        ... examine_k=500, weights=[0.2, 0.8], include_distances=True)
        (array([ 5177, 14669, 15189, 15445,  3801], dtype=uint32), array([0.2725, 0.2727, 0.2743, 0.275 , 0.2766], dtype=float32))
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_union',
        ... examine_k=500)
        array([11044, 10665, 41600, 17728, 15445], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_union',
        ... examine_k=500, weights=[0.2, 0.5])
        array([15445, 14669, 12198,  3801, 22196], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_union',
        ... examine_k=500, weights=[0.2, 0.5])
        array([15445, 14669, 12198,  3801, 22196], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_union',
        ... examine_k=500, weights=[0.2, 0.5], include_distances=True)
        (array([15445, 14669, 12198,  3801, 22196], dtype=uint32), array([0.4889, 0.4891, 0.4924, 0.4938, 0.4944], dtype=float32))
        >>> u = wv.get_nearest(vecs, k=1000, combination=True, combination_method='set_union',
        ... examine_k=1500, weights=[0.2, 0.5])
        >>> i = wv.get_nearest(vecs, k=1000, combination=True, combination_method='set_intersect',
        ... examine_k=1500, weights=[0.2, 0.5])
        >>> u == i
        False
        """

        if not combination:
            return self._get_nearest(v_w_i, k, include_distances)
        elif combination_method == 'set_intersect':
            return self._get_set_intersect(v_w_i, k, examine_k, metric, weights, include_distances)
        elif combination_method == 'set_union':
            return self._get_set_union(v_w_i, k, examine_k, metric, weights, include_distances)
        elif combination_method == 'vector':
            return self._get_resultant_vector_nearest(v_w_i, k, weights, metric, include_distances)
        else:
            raise InvalidCombination()

    def _get_resultant_vector_nearest(self, v_w_i, k, weights, metric, include_distances):
        """Retrieves nearest indices based on the resultant vector.

        Algorithm:
        i) Compute the resultant vector by summing per dimension across vectors
        ii) Commpute distances based on this resultant vector
        iii) Return nearest indices after sorting

        Tests:
        >>> wv = WordVecSpaceAnnoy(DATAFILE_ENV_VAR)
        >>> vecs = wv.get_vectors(['laptop', 'phone'])
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='vector',
        ... include_distances=True)
        (array([10101, 27639, 25905, 28303,  8971], dtype=uint32), array([0.0529, 0.0622, 0.1041, 0.1386, 0.1539], dtype=float32))
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='vector',
        ... include_distances=True)
        (array([10101, 27639, 25905, 28303,  8971], dtype=uint32), array([0.0529, 0.0622, 0.1041, 0.1386, 0.1539], dtype=float32))
        """

        v = self._check_vec(v_w_i)

        res = None
        if not weights:
            weights = np.ones(len(v_w_i))
        else:
            weights = np.array(weights)

        if isinstance(v, (list, np.ndarray)):
            resultant_vec = (v * weights[:, None]).sum(axis=0, dtype=np.float32)
            res, distances = self.ann.get_nns_by_vector(resultant_vec, k, include_distances=True)

        if include_distances:
            return (np.array(res, dtype=np.uint32), np.array(distances, dtype=np.float32))
        else:
            return np.array(res, dtype=np.uint32)

    def _get_nearest(self, v_w_i, k, include_distances):
        """Retrieves nearest indices when combination=False.

        Tests:
        >>> wv = WordVecSpaceAnnoy(DATAFILE_ENV_VAR)
        >>> vecs = wv.get_vectors(['laptop', 'phone'])
        >>> wv.get_nearest(vecs, k=5)
        array([[15890, 32156, 31197, 32342, 20573],
               [ 3826, 10458,  2686, 20515, 39195]], dtype=uint32)
        """

        if not isinstance(v_w_i, (tuple, list, np.ndarray)):
            index = self._check_index_or_word(v_w_i)

            nearest_indices, distances = self.ann.get_nns_by_item(index, k, include_distances=True)
            if include_distances:
                return (np.array(nearest_indices, dtype=np.uint32), np.array(distances, dtype=np.float32))
            else:
                return np.array(nearest_indices, dtype=np.uint32)
        else:
            res = list()
            dist = list()
            for item in v_w_i:
                if isinstance(item, (int, str)):
                    index = self._check_index_or_word(item)
                    indices, distances = self.ann.get_nns_by_item(index, k, include_distances=True)
                    res.append(indices)
                    dist.append(distances)
                else:
                    indices, distances = self.ann.get_nns_by_vector(item, k, include_distances=True)
                    res.append(indices)
                    dist.append(distances)

            if include_distances:
                return (np.asarray(res, dtype=np.uint32), np.asarray(dist, dtype=np.float32))
            else:
                return np.asarray(res, dtype=np.uint32)

    def _get_set_intersect(self, v_w_i, k, examine_k, metric, weights, include_distances):
        """Performs set_intersect combination method.

        Algorithm:
        i) retrive nearest indices for each of the item
        ii) do a set intersection operation
        iii) find the corresponding distance of each item in interection set
        iv) retrive nearest indices from intersection set
        v) return

        Tests:
        >>> wv = WordVecSpaceAnnoy(DATAFILE_ENV_VAR)
        >>> vecs = wv.get_vectors(['laptop', 'phone'])
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_intersect',
        ... examine_k=500, weights=[0.2, 0.8])
        array([ 5177, 14669, 15189, 15445,  3801], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_intersect',
        ... examine_k=500, weights=[0.2, 0.8], include_distances=True)
        (array([ 5177, 14669, 15189, 15445,  3801], dtype=uint32), array([0.2725, 0.2727, 0.2743, 0.275 , 0.2766], dtype=float32))
        """

        v = self._check_vec(v_w_i)
        indices = list()
        similarities = list()
        if not weights: weights = np.ones(len(v))
        else: weights = np.array(weights)
        for item in v:
            idx, dist = self.ann.get_nns_by_vector(item, examine_k, include_distances=True)
            indices.append(idx)
            dist = [1 - x for x in dist]
            similarities.append(dist)

        indices = np.array(indices, dtype=np.uint32)
        similarities = np.array(similarities, dtype=np.float32)

        idxes, scores = self._get_index_scores(indices, similarities, examine_k, weights, intersect=True)
        nearest_indices, nearest_distances = self._f_sorting(idxes, (1-scores), k)

        return (nearest_indices, nearest_distances) if include_distances else nearest_indices

    def _get_set_union(self, v_w_i, k, examine_k, metric, weights, include_distances):
        """Performs set_union combination method.

        Algorithm:
        i) retrive nearest indices for each item
        ii) do a set union operation
        iii) find the corresponding distance of each of the in union set
        iv) retrieve nearest indices from union set
        v) return

        Tests:
        >>> wv = WordVecSpaceAnnoy(DATAFILE_ENV_VAR)
        >>> vecs = wv.get_vectors(['laptop', 'phone'])
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_union',
        ... examine_k=500, weights=[0.2, 0.5])
        array([15445, 14669, 12198,  3801, 22196], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_union',
        ... examine_k=500, weights=[0.2, 0.5], include_distances=True)
        (array([15445, 14669, 12198,  3801, 22196], dtype=uint32), array([0.4889, 0.4891, 0.4924, 0.4938, 0.4944], dtype=float32))
        """

        v = self._check_vec(v_w_i)
        indices = list()
        similarities = list()
        if not weights: weights = np.ones(len(v))
        else: weights = np.array(weights)

        for item in v:
            idx, dist = self.ann.get_nns_by_vector(item, examine_k, include_distances=True)
            indices.append(idx)
            dist = [1 - x for x in dist]
            similarities.append(dist)

        indices = np.array(indices, dtype=np.uint32)
        similarities = np.array(similarities, dtype=np.float32)

        idxes, scores = self._get_index_scores(indices, similarities, examine_k, weights)
        nearest_indices, nearest_distances = self._f_sorting(idxes, (1-scores), k)

        return (nearest_indices, nearest_distances) if include_distances else nearest_indices
