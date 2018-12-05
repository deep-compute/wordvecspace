import os
import json
from typing import Union

from scipy.spatial import distance
import numpy as np
import bottleneck

from .fileformat import WordVecSpaceFile
from .base import WordVecSpaceBase

np.set_printoptions(precision=4)
check_equal = np.testing.assert_array_almost_equal

# export data directory path for test cases
# $export WORDVECSPACE_DATADIR=/path/to/data/
DATAFILE_ENV_VAR = os.environ.get('WORDVECSPACE_DATADIR', '')


class WordVecSpace(WordVecSpaceBase):
    METRIC = 'cosine'
    DEFAULT_K = 512

    def __init__(self, input_dir: str, metric: str=METRIC) -> None:
        self._f = WordVecSpaceFile(input_dir, mode='r')

        self.input_dir = input_dir
        self.metric = metric
        self.nvecs = len(self._f)
        self.dim = int(self._f.dim)

        self.vecs = self._f.vecs
        self.wtoi = self._f.wtoi
        self.itow = self._f.itow

        self.occurs = self._f.occurs
        self.mags = self._f.mags

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

    def _check_vec(self, v, normalised=False):
        if isinstance(v, np.ndarray) and len(v.shape) == 2 and v.dtype == np.float32:
            if normalised:
                m = np.linalg.norm(v)
                return v / m

            return v

        else:
            if isinstance(v, (list, tuple)):
                return self.get_vectors(v, normalized=normalised)

            return self.get_vector(v, normalized=normalised)

    def get_manifest(self) -> dict:
        manifest_info = open(os.path.join(self.input_dir, 'manifest.json'), 'r')
        manifest_info = json.loads(manifest_info.read())

        return manifest_info

    def does_word_exist(self, word: str) -> bool:
        return word in self.wtoi

    def get_index(self, word: str) -> int:
        assert(isinstance(word, str))

        return self.wtoi[word]

    def get_indices(self, words: list) -> list:
        assert(isinstance(words, (tuple, list)) and len(words) != 0)

        indices = [self.wtoi[w] for w in words]
        return indices

    def get_word(self, index: int) -> str:
        return self.itow[index]

    def get_words(self, indices: list) -> list:
        return [self.itow[i] for i in indices]

    def get_magnitude(self, word_or_index: Union[int, str]) -> np.float32:
        index = self._check_index_or_word(word_or_index)

        return self.mags[index]

    def get_magnitudes(self, words_or_indices: list) -> np.ndarray:
        w = self._check_indices_or_words(words_or_indices)

        return self.mags.take(w)

    def get_occurrence(self, word_or_index: Union[int, str]) -> int:
        index = self._check_index_or_word(word_or_index)

        return self.occurs[index]

    def get_occurrences(self, words_or_indices: list) -> list:
        w = self._check_indices_or_words(words_or_indices)

        return self.occurs.take(w)

    def get_vector(self, word_or_index: Union[int, str], normalized: bool=False) -> np.ndarray:
        index = self._check_index_or_word(word_or_index)

        if normalized:
            return self.vecs[index]

        return self.vecs[index] * self.mags[index]

    def get_vectors(self, words_or_indices: list, normalized: bool=False) -> np.ndarray:
        w = self._check_indices_or_words(words_or_indices)

        if normalized:
            return self.vecs.take(w, axis=0)

        vecs = self.vecs.take(w, axis=0)
        mags = self.mags.take(w)

        return np.multiply(vecs.T, mags).T

    def get_distance(self, word_or_index1: Union[int, str],
                     word_or_index2: Union[int, str], metric: str='cosine') -> float:

        w1 = word_or_index1
        w2 = word_or_index2

        if not metric:
            metric = self.metric

        if metric == 'cosine' or 'angular':
            vec1 = self._check_vec(w1, True)
            vec2 = self._check_vec(w2, True)

            return 1 - np.dot(vec1, vec2.T)

        elif metric == 'euclidean':
            vec1 = self._check_vec(w1)
            vec2 = self._check_vec(w2)

            return distance.euclidean(vec1, vec2)

    def _check_r_and_c(self, r, c, m):
        if not m:
            m = self.metric

        if not isinstance(r, (tuple, list, np.ndarray)):
            r = [r]

        if c is not None and len(c):
            if not isinstance(c, (tuple, list, np.ndarray)):
                c = [c]

        return m, r, c

    def get_distances(self, row_words_or_indices: Union[list, np.ndarray],
                      col_words_or_indices: Union[list, None, np.ndarray]=None,
                      metric=None) -> np.ndarray:

        r = row_words_or_indices
        c = col_words_or_indices
        metric, r, c = self._check_r_and_c(r, c, metric)

        if metric == 'cosine' or 'angular':
            row_vectors = self._check_vec(r, True)

            col_vectors = self.vecs
            if c is not None and len(c):
                col_vectors = self._check_vec(c, True)

            if len(r) == 1:
                nvecs, dim = col_vectors.shape

                vec_out = self._make_array((len(col_vectors), len(row_vectors)), dtype=np.float32)
                res = self._perform_sgemv(row_vectors, col_vectors, vec_out, nvecs, dim)

            else:
                mat_out = self._make_array((len(row_vectors), len(col_vectors)), dtype=np.float32)
                res = self._perform_sgemm(row_vectors, col_vectors, mat_out)

            return 1 - res

        elif metric == 'euclidean':
            row_vectors = self._check_vec(r)

            if c:
                col_vectors = self._check_vec(c)
            else:
                col_vectors = self.vecs

            return distance.cdist(row_vectors, col_vectors, 'euclidean')

    def _nearest_sorting(self, d, k):
        ner = self._make_array(shape=(len(d), k), dtype=np.uint32)
        dist = self._make_array(shape=(len(d), k), dtype=np.float32)

        for index, p in enumerate(d):
            # FIXME: better variable name for b_sort
            b_sort = bottleneck.argpartition(p, k)[:k]
            pr_dist = np.take(p, b_sort)

            # FIXME: better variable name for a_sorted
            a_sorted = np.argsort(pr_dist)
            indices = np.take(b_sort, a_sorted)

            ner[index] = indices
            dist[index] = np.take(p, indices)

        return ner, dist

    def get_nearest(self, v_w_i: list, k: int=DEFAULT_K, examine_k: int=None,
                    combination: bool=False, combination_method: str='distance',
                    weights: list=None, metric: str='cosine',
                    include_distances: bool=False) -> np.ndarray:

        """Retrives most similar indices from the vector space.

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
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv.get_nearest('india', k=5)
        array([  509,  3389,   486,  6186, 20932], dtype=uint32)
        >>> wv.get_nearest(709, k=5)
        array([  709,  1398,  8758,  7892, 15824], dtype=uint32)
        >>> vecs = wv.get_vectors(['india', 'pakistan'])
        >>> vecs
        array([[-2.4544, -0.1309,  8.9653, -3.1779,  3.2016],
               [-3.3736,  0.4845,  9.7016, -3.5337,  1.0142]], dtype=float32)
        >>> wv.get_nearest(vecs, k=5)
        array([[  509,  3389,   486,  6186, 20932],
               [ 2224,  5281,  3560,  4622, 11886]], dtype=uint32)
	>>> wv.get_nearest(vecs, k=5, examine_k=500,
	... combination=True, combination_method='set_intersect')
	array([[  523,  5969, 24149,  9772,   486]], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, examine_k=500,
        ... combination=True, combination_method='set_intersect', weights=[0.1, 0.9])
        array([[ 2224,  5281,  3560,  4622, 11886]], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, examine_k=500, combination=True,
        ... combination_method='set_intersect', weights=[0.1, 0.9], include_distances=True)[1]
        array([[0.2795, 0.2814, 0.2819, 0.282 , 0.2821]], dtype=float32)
	>>> wv.get_nearest(vecs, k=5, examine_k=500,
	... combination=True, combination_method='set_union')
	array([[11087, 11304, 14483,  4538, 10737]], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, examine_k=500,
        ... combination=True, combination_method='set_union', weights=[0.1, 0.9])
        array([[13942,   503, 28413, 27280, 26180]], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, examine_k=500,
        ... combination=True, combination_method='set_union', weights=[0.9, -0.9])
        array([[ 2822,  2765, 12515, 66318,  8591]], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, examine_k=500, combination=True,
        ... combination_method='set_union', weights=[0.1, 0.9], include_distances=True)[1]
        array([[0.0321, 0.0321, 0.0322, 0.0322, 0.0322]], dtype=float32)
	>>> wv.get_nearest(vecs, k=5, combination=True, combination_method='distance')
	array([[  523,  5969, 24149,  9772,   486]], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True,
        ... combination_method='distance', weights=[0.5, 0.3])
        array([[  523,  5969,   486, 24149,  9772]], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True,
        ... combination_method='distance', weights=[0.5, 0.3], include_distances=True)[1]
        array([[0.2425, 0.2435, 0.2435, 0.2439, 0.2441]], dtype=float32)
	>>> wv.get_nearest(vecs, k=5, combination=True, combination_method='vector')
	array([[  523,  5969, 24149,  9772,   486]], dtype=uint32)
	>>> wv.get_nearest(vecs, k=5, combination=True,
	... combination_method='vector', weights=[0.8, 0.2])
	array([[  486,   509,   523, 24149,  3389]], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True,
        ... combination_method='vector', weights=[0.5, 0.3])
        array([[  523,  5969,   486, 24149,  9772]], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True,
        ... combination_method='vector', weights=[0.5, 0.3], include_distances=True)[1]
        array([[0.0006, 0.0023, 0.0023, 0.003 , 0.0034]], dtype=float32)
        """

        if not combination:
            return self._get_brute(v_w_i, k, metric,
                                   combination, include_distances)

        if combination and combination_method == 'set_intersect':
            return self._get_set_intersect(v_w_i, k, examine_k, metric,
                                           weights, include_distances)

        if combination and combination_method == 'set_union':
            return self._get_set_union(v_w_i, k, examine_k, metric,
                                       weights, include_distances)

        if combination and combination_method == 'distance':
            return self._get_weighted_brute(v_w_i, k, metric,
                                            weights, include_distances)

        if combination and combination_method == 'vector':
            return self._get_resultant_vector_nearest(v_w_i, k, metric,
                                                      weights, include_distances)

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

        if not weights:
            weights = np.ones(len(v))
        else:
            weights = np.array(weights)

        d = self.get_distances(v, metric=metric)
        indices, distances = self._nearest_sorting(d, examine_k)

        intersect = np.array(list(set(indices[0]).intersection(*indices)))
        corresponding_dists = list()
        for index in intersect:
            corresponding_dists.append(self._get_corresponding_dist(v, index, indices, distances, weights))

        corresponding_dists = np.array(corresponding_dists, dtype=np.float32)
        intersect_indices, distances = self._nearest_sorting(corresponding_dists.reshape(1, len(corresponding_dists)), k)
        nearest_indices = intersect[intersect_indices]

        # FIXME: get_distances twice is computationally expensive
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

        if not weights:
            weights = np.ones(len(v))
        else:
            weights = np.array(weights)

        d = self.get_distances(v, metric=metric)
        indices, distances = self._nearest_sorting(d, examine_k)

        union = np.array(list(set().union(*indices)), dtype=np.uint32)

        corresponding_dists = list()
        for index in union:
            corresponding_dists.append(self._get_corresponding_dist(v, index, indices, distances, weights))

        corresponding_dists = np.array(corresponding_dists, dtype=np.float32)
        union_indices, distances = self._nearest_sorting(corresponding_dists.reshape(1, len(corresponding_dists)), k)
        nearest_indices = union[union_indices]

        return (nearest_indices, distances) if include_distances else nearest_indices

    def _get_corresponding_dist(self, v, index, indices, distances, weights):
        """Return corresponding distance.

        While returning the corresponding distance we need make sure that,
        importance factor, i.e weights must applied to that distance.
        The following numpy operations helps understand will accomplish this task.

        Tests:
        >>> index = 3
        >>> indices = np.array([[3,6,9], [4, 3, 8]])
        >>> distances = np.array([[0.1, 0.2, 0.3], [0.2, 0.5, 0.8]])
        >>> weights = np.array([0.6, 0.4])
        >>> weight_array = np.zeros(len(indices))
        >>> dist_array = np.zeros(len(indices))
        >>> loc = np.where(indices==index)
        >>> loc
        (array([0, 1]), array([0, 1]))
        >>> # First item in tuple is row and second is column
        >>> dist = distances[loc]
        >>> dist
        array([0.1, 0.5])
        >>> np.put(dist_array, loc[0], dist)
        >>> dist_array
        array([0.1, 0.5])
        >>> np.put(weight_array, loc[0], weights[loc[0]])
        >>> weight_array
        array([0.6, 0.4])
        >>> index = 6
        >>> weight_array = np.zeros(len(indices))
        >>> dist_array = np.zeros(len(indices))
        >>> loc = np.where(indices==index)
        >>> loc
        (array([0]), array([1]))
        >>> dist = distances[loc]
        >>> dist
        array([0.2])
        >>> np.put(dist_array, loc[0], dist)
        >>> dist_array
        array([0.2, 0. ])
        >>> np.put(weight_array, loc[0], weights[loc[0]])
        >>> weight_array
        array([0.6, 0. ])
        >>> index = 8
        >>> weight_array = np.zeros(len(indices))
        >>> dist_array = np.zeros(len(indices))
        >>> loc = np.where(indices==index)
        >>> loc
        (array([1]), array([2]))
        >>> dist = distances[loc]
        >>> dist
        array([0.8])
        >>> np.put(dist_array, loc[0], dist)
        >>> dist_array
        array([0. , 0.8])
        >>> np.put(weight_array, loc[0], weights[loc[0]])
        >>> weight_array
        array([0. , 0.4])
        >>> indices = np.array([[3,6,9], [4, 3, 8], [7, 3, 3]])
        >>> distances = np.array([[0.1, 0.2, 0.3], [0.2, 0.5, 0.8], [0.3, 0.4, 0.5]])
        >>> weights = np.array([0.6, 0.4, 0.9])
        >>> weight_array = np.zeros(len(indices))
        >>> dist_array = np.zeros(len(indices))
        >>> loc = np.where(indices==7)
        >>> loc
        (array([2]), array([0]))
        >>> dist = distances[loc]
        >>> dist
        array([0.3])
        >>> np.put(dist_array, loc[0], dist)
        >>> dist_array
        array([0. , 0. , 0.3])
        >>> np.put(weight_array, loc[0], weights[loc[0]])
        >>> weight_array
        array([0. , 0. , 0.9])
        """

        w_a = np.zeros(len(v))
        d_a = np.zeros(len(v))
        loc = np.where(indices == index)
        dist = distances[loc]
        np.put(d_a, loc[0], dist)
        np.put(w_a, loc[0], weights[loc[0]])
        s = np.dot(w_a, d_a)

        return s

    def _get_resultant_vector_nearest(self, v_w_i, k, metric, weights, include_distances):
        """Retrieves nearest indices based on the resultant vector.

        Algorithm:
        i) Compute the resultant vector by summing per dimension across vectors
        ii) Commpute distances based on this resultant vector
        iii) Return nearest indices after sorting
        """

        v = self._check_vec(v_w_i)

        if not weights:
            weights = np.ones(len(v_w_i))
        else:
            weights = np.array(weights)

        resultant_vec = (v * weights[:, None]).sum(axis=0, dtype=np.float32).reshape(1, self.dim)

        d = self.get_distances(resultant_vec, metric=metric)
        nearest_indices, distances = self._nearest_sorting(d, k)

        return (nearest_indices, distances) if include_distances else nearest_indices

    def _get_weighted_brute(self, v_w_i, k, metric, weights, include_distances):
        """Retrieves nearest indices based on weights applied to per vector's distance.

        Algorithm:
        i) Compute the distance matrix .
        ii) Perform dot operation with this distance matrix and weights
        iii) Retrive nearest indices on the weighted distance matrix
        iv) Return
        """

        v = self._check_vec(v_w_i)
        d = self.get_distances(v, metric=metric)
        if not weights:
            weights = np.ones(len(v))
        else:
            weights = np.array(weights)

        w_d = np.dot(weights, d)
        nearest_indices, distances = self._nearest_sorting(w_d.reshape(1, len(w_d)), k)

        return (nearest_indices, distances) if include_distances else nearest_indices

    def _get_brute(self, v_w_i, k, metric, combination, include_distances):
        """Retrives nearest indices when combination=False"""

        d = self.get_distances(v_w_i, metric=metric)
        nearest_indices, distances = self._nearest_sorting(d, k)

        if isinstance(v_w_i, (list, tuple)) or isinstance(v_w_i, np.ndarray) and len(v_w_i) > 1:
            return (nearest_indices, distances) if include_distances else nearest_indices
        else:
            return (nearest_indices[0], distances[0]) if include_distances else nearest_indices[0]
