import os
import json
from typing import Union

from scipy.spatial import distance
import numpy as np
import pandas as pd
import bottleneck

from .fileformat import WordVecSpaceFile
from .base import WordVecSpaceBase
from .exception import InvalidCombination, InvalidMetric

np.set_printoptions(precision=4)
check_equal = np.testing.assert_array_almost_equal

# export data directory path for test cases
# $export WORDVECSPACE_DATADIR=/path/to/data/
DATAFILE_ENV_VAR = os.environ.get('WORDVECSPACE_DATADIR', '')
SEED_ENV_VAR = os.environ.get('NUMPY_RANDOM_SEED', 1)
np.random.seed(SEED_ENV_VAR)


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
        """Makes an ndarray of given shape and dtype.

        Arguments:
        shape -- shape of the ndarray
        dtype -- datatype of the ndarray

        Returns:
        numpy ndarray of given shape and datatype

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> t = wv._make_array((2, 3), dtype=np.uint32)
        >>> t.dtype
        dtype('uint32')
        >>> t = wv._make_array((3, 1), dtype=np.float32)
        >>> t.shape
        (3, 1)
        """

        return np.ndarray(shape, dtype)

    def _check_index_or_word(self, item):
        """Checks if item is index or word

        Arguments:
        item -- str/int

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv._check_index_or_word('india')
        509
        >>> wv._check_index_or_word(509)
        509
        >>> wv._check_index_or_word('pumba')
        """

        if isinstance(item, str):
            return self.get_index(item)

        return item

    def _check_indices_or_words(self, items):
        """Returns indices of the given items.

        Arguments:
        items -- list/array words or indices

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv._check_indices_or_words(['apple', 'banana', 'fruits'])
        [1221, 10968, 5441]
        """

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
        """Returns vector/normalised vector.

        Arguments:
        v -- vector/word/index
        normalised -- whether to return normalised vector

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> v = wv.get_vector('india')
        >>> wv._check_vec(v)
        array([-2.4544, -0.1309,  8.9653, -3.1779,  3.2016], dtype=float32)
        >>> wv._check_vec(509)
        array([-2.4544, -0.1309,  8.9653, -3.1779,  3.2016], dtype=float32)
        >>> wv._check_vec('india')
        array([-2.4544, -0.1309,  8.9653, -3.1779,  3.2016], dtype=float32)
        """

        if isinstance(v, np.ndarray) and len(v.shape) == 2 and v.dtype == np.float32:
            if normalised:
                m = np.linalg.norm(v)
                return v / m

            return v
        else:
            if isinstance(v, (list, tuple)):
                return self.get_vectors(v, normalized=normalised)
            elif isinstance(v, np.ndarray):
                if normalised:
                    m = np.linalg.norm(v)

                    return v / m
                else:
                    return v
            else:
                return self.get_vector(v, normalized=normalised)

    def get_manifest(self) -> dict:
        """Shows the manifest information about the vector space.

        Tests:
        >>> from pprint import pprint
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> pprint(wv.get_manifest())
        {'dimension': 5,
         'dt_creation': '2018-10-10T12:41:00.194158',
         'input_path': '/tmp/Ram/word2vec/word2vec',
         'manifest_info': {},
         'num_shards': 1,
         'num_vecs': 71291,
         'num_vecs_in_shard': 71290,
         'num_vecs_per_shard': 0,
         'num_words': 16718843}
        """

        manifest_info = open(os.path.join(self.input_dir, 'manifest.json'), 'r')
        manifest_info = json.loads(manifest_info.read())

        return manifest_info

    def does_word_exist(self, word: str) -> bool:
        """Check if the word exists in the vocabulary.

        Argument:
        word -- str

        Returns:
        boolean value, True if word exists else False

        Tests:
        Pending as diskdict does not support in operation on disk.
        """

        return word in self.wtoi

    def get_index(self, word: str) -> int:
        """Get the index for a given word.

        Argument:
        word -- str

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv.get_index('timon')
        64538
        >>> wv.get_index('pumba')
        """

        assert(isinstance(word, str))

        return self.wtoi[word]

    def get_indices(self, words: list) -> list:
        """Get multiple indices at onces.

        Argument:
        words -- list/array of words

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv.get_indices(['timon', 'king', 'jungle'])
        [64538, 187, 7200]
        >>> wv.get_indices(['timon'])
        [64538]
        """

        assert(isinstance(words, (tuple, list)) and len(words) != 0)

        indices = [self.wtoi[w] for w in words]
        return indices

    def get_word(self, index: int) -> str:
        """Get the word corresponding at an index.

        Argument:
        index -- int

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv.get_word(509)
        'india'
        >>> wv.get_word(987)
        'bank'
        """

        return self.itow[index]

    def get_words(self, indices: list) -> list:
        """Get words corresponding to multiple indices.

        Argument:
        indices -- list of int indices

        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv.get_words([509, 987])
        ['india', 'bank']
        """

        return [self.itow[i] for i in indices]

    def get_magnitude(self, word_or_index: Union[int, str]) -> np.float32:
        """Get the magnitude of vector corresponding to an index/word.

        Arguments:
        word_or_index -- int/str

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv.get_magnitude('timon')
        1.5348816
        >>> wv.get_magnitude(64538)
        1.5348816
        """

        index = self._check_index_or_word(word_or_index)

        return self.mags[index]

    def get_magnitudes(self, words_or_indices: list) -> np.ndarray:
        """Get magnitudes of multiple vectors corresponding to index/word.

        Arguments:
        words_or_indices -- list of indices/words

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv.get_magnitudes(['timon', 'lion', 'king', 'jungle'])
        memmap([ 1.5349,  4.141 , 11.3422,  5.9619], dtype=float32)
        >>> wv.get_magnitudes(wv.get_indices(['timon', 'lion', 'king', 'jungle']))
        memmap([ 1.5349,  4.141 , 11.3422,  5.9619], dtype=float32)
        """

        w = self._check_indices_or_words(words_or_indices)

        return self.mags.take(w)

    def get_occurrence(self, word_or_index: Union[int, str]) -> int:
        """Get occurrence of a index/word.

        Arguments:
        word_or_index -- int/str

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv.get_occurrence('lion')
        348
        >>> wv.get_occurrence(wv.get_index('king'))
        7456
        """

        index = self._check_index_or_word(word_or_index)

        return self.occurs[index]

    def get_occurrences(self, words_or_indices: list) -> list:
        """Get occurrences of multiple indices/words

        Arguments:
        words_or_indices -- list/array of indices or words

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv.get_occurrences(['timon', 'lion', 'king', 'jungle'])
        memmap([   5,  348, 7456,  199], dtype=uint64)
        >>> wv.get_occurrences(wv.get_indices(['timon', 'lion', 'king', 'jungle']))
        memmap([   5,  348, 7456,  199], dtype=uint64)
        """

        w = self._check_indices_or_words(words_or_indices)

        return self.occurs.take(w)

    def get_vector(self, word_or_index: Union[int, str], normalized: bool=False) -> np.ndarray:
        """Get the vector corresponding to a word or a index.

        Arguments:
        word_or_index -- int/str
        normalised -- get the vector in normalised form, magnitude = 1

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv.get_vector('timon')
        array([-0.949 ,  0.44  , -0.9174,  0.2542,  0.5963], dtype=float32)
        >>> wv.get_vector('timon', normalized=True)
        memmap([-0.6183,  0.2867, -0.5977,  0.1656,  0.3885], dtype=float32)
        """

        index = self._check_index_or_word(word_or_index)

        if normalized:
            return self.vecs[index]

        return self.vecs[index] * self.mags[index]

    def get_vectors(self, words_or_indices: list, normalized: bool=False) -> np.ndarray:
        """Get vector corresponding to multiple words or indices.

        Arguments:
        word_or_index -- list/array of indices or words
        normalized -- get vectors in normalised form, magnitude = 1

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv.get_vectors(['timon', 'lion', 'king'])
        array([[ -0.949 ,   0.44  ,  -0.9174,   0.2542,   0.5963],
               [ -3.1758,   1.17  ,  -1.4279,   1.6389,  -0.984 ],
               [-10.0505,  -1.8819,  -1.5939,  -4.5718,  -0.8063]], dtype=float32)
        >>> wv.get_vectors(['timon', 'lion', 'king'], normalized=True)
        memmap([[-0.6183,  0.2867, -0.5977,  0.1656,  0.3885],
                [-0.7669,  0.2825, -0.3448,  0.3958, -0.2376],
                [-0.8861, -0.1659, -0.1405, -0.4031, -0.0711]], dtype=float32)
        >>> wv.get_vectors(wv.get_indices(['timon', 'lion', 'king']), normalized=True)
        memmap([[-0.6183,  0.2867, -0.5977,  0.1656,  0.3885],
                [-0.7669,  0.2825, -0.3448,  0.3958, -0.2376],
                [-0.8861, -0.1659, -0.1405, -0.4031, -0.0711]], dtype=float32)
        >>> wv.get_vectors(wv.get_indices(['timon', 'lion', 'king']))
        array([[ -0.949 ,   0.44  ,  -0.9174,   0.2542,   0.5963],
               [ -3.1758,   1.17  ,  -1.4279,   1.6389,  -0.984 ],
               [-10.0505,  -1.8819,  -1.5939,  -4.5718,  -0.8063]], dtype=float32)
        """

        w = self._check_indices_or_words(words_or_indices)

        if normalized:
            return self.vecs.take(w, axis=0)

        vecs = self.vecs.take(w, axis=0)
        mags = self.mags.take(w)

        return np.multiply(vecs.T, mags).T

    def get_distance(self, word_or_index1: Union[int, str],
                     word_or_index2: Union[int, str], metric: str='cosine') -> float:
        """Get distance between two vectors corresponding to either index/word

        Arguments:
        word_or_index1 -- index/word
        word_or_index2 -- index/word
        metric -- distance funtion

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv.get_distance('timon', 'king')
        0.5100758969783783
        >>> wv.get_distance('timon', 'king', metric='euclidean')
        10.674457550048828
        """

        w1 = word_or_index1
        w2 = word_or_index2

        if not metric:
            metric = self.metric

        if metric == 'cosine' or metric == 'angular':
            vec1 = self._check_vec(w1, True)
            vec2 = self._check_vec(w2, True)

            return 1 - np.dot(vec1, vec2.T)

        elif metric == 'euclidean':
            vec1 = self._check_vec(w1)
            vec2 = self._check_vec(w2)

            return distance.euclidean(vec1, vec2)

        else:
            raise InvalidMetric()

    def _check_r_and_c(self, r, c, m):
        """Check row, column and metric.

        Arguments:
        r -- row
        c -- column
        m -- metric

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv._check_r_and_c([1,2,3], [[3,2,1], [9, 8, 7]], 'cosine')
        ('cosine', [1, 2, 3], [[3, 2, 1], [9, 8, 7]])
        """

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
                      metric=None, similarity: bool=False) -> np.ndarray:
        """Get distance between vectors to all the vectors in the vector space.

        Arguments:
        row_words_or_indices -- row words/indices
        col_words_or_indices -- columns words/indices
        metric -- distance function
        similarity -- compute in similarity space or not, only for cosine metric

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
	>>> wv.get_distances('timon')
	array([[0.8651, 1.3549, 1.1688, ..., 0.1413, 0.0282, 0.5794]],
	      dtype=float32)
	>>> wv.get_distances(['timon', 'lion'])
	array([[0.8651, 1.3549, 1.1688, ..., 0.1413, 0.0282, 0.5794],
	       [1.2677, 1.4346, 1.5748, ..., 0.0587, 0.3284, 1.2349]],
	      dtype=float32)
        >>> wv.get_distances(['timon', 'lion'], metric='euclidean')
        array([[1.7151, 2.1084, 1.9682, ..., 0.8485, 0.6103, 1.4369],
               [4.5127, 4.6633, 4.7862, ..., 3.2174, 3.5476, 4.4825]])
	>>> wv.get_distances(['timon', 'lion'], similarity=True)
	memmap([[ 0.1349, -0.3549, -0.1688, ...,  0.8587,  0.9718,  0.4206],
		[-0.2677, -0.4346, -0.5748, ...,  0.9413,  0.6716, -0.2349]],
	       dtype=float32)
        """

        r = row_words_or_indices
        c = col_words_or_indices
        metric, r, c = self._check_r_and_c(r, c, metric)

        if metric == 'cosine' or metric == 'angular':
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

            if similarity:
                return res
            else:
                return 1 - res

        elif metric == 'euclidean':
            row_vectors = self._check_vec(r)

            if c:
                col_vectors = self._check_vec(c)
            else:
                col_vectors = self.vecs

            return distance.cdist(row_vectors, col_vectors, 'euclidean')

        else:
            raise InvalidMetric()

    def _nearest_sorting(self, d, k, similarity=False):
        """Sorts the distance in ascending order if not similarity else in descending,
        and retrieve the corresponding indices.

        Arguments:
        d -- distances
        k -- slice value
        similarity -- perform operation in similarity space.

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> distances = np.array([[0.3, 0.4, 0.5], [0.1, 0.2, 0.3]])
        >>> wv._nearest_sorting(distances, 2)
        (array([[0, 1],
               [0, 1]], dtype=uint32), array([[0.3, 0.4],
               [0.1, 0.2]], dtype=float32))
        >>> wv._nearest_sorting(distances, 2, similarity=True)
        (array([[2, 1],
               [2, 1]], dtype=uint32), array([[0.5, 0.4],
               [0.3, 0.2]], dtype=float32))
        """

        ner = self._make_array(shape=(len(d), k), dtype=np.uint32)
        dist = self._make_array(shape=(len(d), k), dtype=np.float32)

        for index, p in enumerate(d):
            # FIXME: better variable name for b_sort
            # FIXME: bad implementation
            b_sort = bottleneck.argpartition(p, k)[:k] if not similarity else bottleneck.argpartition(-p, k)[:k]
            pr_dist = np.take(p, b_sort)

            # FIXME: better variable name for a_sorted
            a_sorted = np.argsort(pr_dist) if not similarity else np.argsort(-pr_dist)
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
        >>> wv.get_nearest('laptop', k=5)
        array([15890, 18487, 32156,  3917, 31197], dtype=uint32)
        >>> wv.get_nearest('phone', k=5)
        array([ 3826, 10458,  2686, 17794, 20515], dtype=uint32)
        >>> vecs = wv.get_vectors(['laptop', 'phone'])
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='full_matrix')
        array([10101, 27639,  3656, 15445, 11044], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='full_matrix',
        ... include_distances=True)
        (array([10101, 27639,  3656, 15445, 11044], dtype=uint32), array([0.6346, 0.6353, 0.6358, 0.6362, 0.6369], dtype=float32))
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='vector')
        array([10101, 27639,  3656, 15445, 11044], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='vector',
        ... include_distances=True)
        (array([10101, 27639,  3656, 15445, 11044], dtype=uint32), array([0.0014, 0.0019, 0.0023, 0.0026, 0.0031], dtype=float32))
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='vector',
        ... weights=[0.2, 0.5])
        array([ 8350, 29213, 10101, 22307, 25905], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='vector',
        ... weights=[0.2, 0.5], include_distances=True)
        (array([ 8350, 29213, 10101, 22307, 25905], dtype=uint32), array([0.0009, 0.0012, 0.0028, 0.0028, 0.0029], dtype=float32))
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='full_matrix',
        ... weights=[0.2, 0.5])
        array([ 8350, 29213, 10101, 22307, 25905], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_intersect',
        ... weights=[0.2, 0.5], examine_k=500)
        array([10101, 25905, 25509,  8618, 28400], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_intersect',
        ... examine_k=500)
        array([10101, 27639, 15445, 12198, 14669], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_intersect',
        ... examine_k=500, weights=[0.2, 0.8])
        array([25905, 10101, 25509,  8618,  6063], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_intersect',
        ... examine_k=500, weights=[0.2, 0.8], include_distances=True)
        (array([25905, 10101, 25509,  8618,  6063], dtype=uint32), array([0.3446, 0.3456, 0.3463, 0.3467, 0.3482], dtype=float32))
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_union',
        ... examine_k=500)
        array([10101, 27639, 15445, 12198, 14669], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_union',
        ... examine_k=500, weights=[0.2, 0.5])
        array([10101, 25905, 25509,  8618, 28400], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_union',
        ... examine_k=500, weights=[0.2, 0.5], include_distances=True)
        (array([10101, 25905, 25509,  8618, 28400], dtype=uint32), array([0.5362, 0.5363, 0.5372, 0.5374, 0.5384], dtype=float32))
        >>> u = wv.get_nearest(vecs, k=1000, combination=True, combination_method='set_union',
        ... examine_k=1500, weights=[0.2, 0.5])
        >>> i = wv.get_nearest(vecs, k=1000, combination=True, combination_method='set_intersect',
        ... examine_k=1500, weights=[0.2, 0.5])
        >>> u == i
        False
        >>> d = wv.get_nearest(vecs, k=5000, combination=True, combination_method='full_matrix')
        >>> v = wv.get_nearest(vecs, k=5000, combination=True, combination_method='vector')
        >>> v == d
        array([ True,  True,  True, ...,  True,  True,  True])
        >>> (v.tolist() == d.tolist())
        False
        """

        if not combination:
            return self._get_nearest(v_w_i, k, metric, include_distances)
        elif combination_method == 'set_intersect':
            return self._get_set_intersect(v_w_i, k, examine_k, metric,
                                           weights, include_distances)
        elif combination_method == 'set_union':
            return self._get_set_union(v_w_i, k, examine_k, metric,
                                       weights, include_distances)
        elif combination_method == 'full_matrix':
            return self._get_weighted_brute(v_w_i, k, metric,
                                            weights, include_distances)
        elif combination_method == 'vector':
            return self._get_resultant_vector_nearest(v_w_i, k, metric,
                                                      weights, include_distances)
        else:
            raise InvalidCombination()

    def _get_set_intersect(self, v_w_i, k, examine_k, metric, weights, include_distances):
        """Performs set_intersect combination method.

        Algorithm:
        i) retrieve nearest indices for each of the item
        ii) do a set intersection operation
        iii) find the corresponding distance of each item in interection set
        iv) retrieve nearest indices from intersection set
        v) return

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> vecs = wv.get_vectors(['laptop', 'phone'])
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_intersect',
        ... examine_k=500, weights=[0.2, 0.8])
        array([25905, 10101, 25509,  8618,  6063], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_intersect',
        ... examine_k=500, weights=[0.2, 0.8], include_distances=True)
        (array([25905, 10101, 25509,  8618,  6063], dtype=uint32), array([0.3446, 0.3456, 0.3463, 0.3467, 0.3482], dtype=float32))
        """

        v = self._check_vec(v_w_i, normalised=True)

        if not weights: weights = np.ones(len(v))
        else: weights = np.array(weights)

        s = self.get_distances(v, metric=metric, similarity=True)
        indices, similarities = self._nearest_sorting(s, examine_k, similarity=True)

        idxes, scores = self._get_index_scores(indices, similarities, examine_k, weights, intersect=True)
        nearest_indices, nearest_distances = self._f_sorting(idxes, (1-scores), k)
        return (nearest_indices, nearest_distances) if include_distances else nearest_indices

    def _get_set_union(self, v_w_i, k, examine_k, metric, weights, include_distances):
        """Performs set_union combination method.

        Algorithm:
        i) retrieve nearest indices for each item
        ii) do a a type of union operation
        iii) find the corresponding distance of each of the in this set
        iv) retrieve nearest indices from union set
        v) return

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> vecs = wv.get_vectors(['laptop', 'phone'])
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='set_union',
        ... examine_k=500, weights=[0.2, 0.5], include_distances=True)
        (array([10101, 25905, 25509,  8618, 28400], dtype=uint32), array([0.5362, 0.5363, 0.5372, 0.5374, 0.5384], dtype=float32))
        >>> u = wv.get_nearest(vecs, k=1000, combination=True, combination_method='set_union',
        ... examine_k=1500, weights=[0.2, 0.5])
        """

        v = self._check_vec(v_w_i, normalised=True)

        if not weights: weights = np.ones(len(v))
        else: weights = np.array(weights)

        s = self.get_distances(v, metric=metric, similarity=True)
        indices, similarities = self._nearest_sorting(s, examine_k, similarity=True)

        idxes, scores = self._get_index_scores(indices, similarities, examine_k, weights)
        nearest_indices, nearest_distances = self._f_sorting(idxes, (1-scores), k)

        return (nearest_indices, nearest_distances) if include_distances else nearest_indices

    def _f_sorting(self, indices, distances, k):
        """Final sorting on distances to find nearest indices

        Arguments:
        indices -- array(1d)/list of indices
        distances -- array(1d)/list of corresponding distances

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> indices = np.array([2,4,521, 1, 5, 76, 90])
        >>> distances = np.array([0.9, 0.99, 0.1, 0.3, 0.6, 0.2, 0.87])
        >>> wv._f_sorting(indices, distances, 4)
        (array([521,  76,   1,   5], dtype=uint32), array([0.1, 0.2, 0.3, 0.6], dtype=float32))
        """

        idx = pd.Series(distances.reshape(len(distances))).nsmallest(k).keys().tolist()
        dist = pd.Series(distances.reshape(len(distances))).nsmallest(k).values.tolist()

        return (indices[idx].astype(np.uint32), np.array(dist, dtype=np.float32))

    def _get_index_scores(self, indices, similarities, examine_k, weights, intersect=False):
        """Computes scores for given indices in the similarity space

        Arguments:
        indices -- ndarray of integer numbers represented as indices
        similarities -- ndarray of floats represented as corresponding similarity measure to indices
        examine_k -- count of the items for each query word/index/vector
        weights -- importance factor for computing scores

        Algorithm:
        i) Form temporary table
        ii) Fill the table with indices, and distances
        iii) Collapse the table
        iv) Compute the score based on the weights

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
	>>> indices = np.array([[1,2,3], [3, 4, 5]])
	>>> similarities = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
        >>> examine_k = 3
        >>> weights = np.array([0.2, 0.8])
	>>> wv._get_index_scores(indices, similarities, examine_k, weights)
        (array([1., 2., 3., 4., 5.]), array([0.02, 0.04, 0.22, 0.24, 0.32]))
        >>> wv._get_index_scores(indices, similarities, examine_k, weights, intersect=True)
        (array([3.]), array([0.22]))
        """

        temp_table = np.zeros(shape=(len(indices.ravel()), len(indices)+1))
        temp_table[:, 0] = indices.ravel()
        for i in range(1, len(indices)+1):
            z = np.zeros(examine_k*len(indices))
            np.put(z, np.arange((i*examine_k)-examine_k, i*examine_k), similarities[i-1])
            temp_table[:, i] = z

        temp_sorted = temp_table[temp_table[:, 0].argsort(kind='quicksort')]

        # collapsing table
        df = pd.DataFrame(temp_sorted)
        collapsed_df = df.groupby(0).sum() if not intersect else df.groupby(0).sum().loc[~(df.groupby(0).sum() == 0).any(axis=1)]
        collapsed_df_dist = collapsed_df.values
        collapsed_df_idx = collapsed_df.index.values

        # compute scores
        scores = np.dot(collapsed_df_dist, weights)

        return (collapsed_df_idx, scores)

    def _get_resultant_vector_nearest(self, v_w_i, k, metric, weights, include_distances):
        """Retrieves nearest indices based on the resultant vector.

        Algorithm:
        i) Compute the resultant vector by summing per dimension across vectors
        ii) Compute distances based on this resultant vector
        iii) Return nearest indices after sorting

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> vecs = wv.get_vectors(['laptop', 'phone'])
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='vector',
        ... weights=[0.2, 0.5])
        array([ 8350, 29213, 10101, 22307, 25905], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='vector',
        ... weights=[0.2, 0.5], include_distances=True)
        (array([ 8350, 29213, 10101, 22307, 25905], dtype=uint32), array([0.0009, 0.0012, 0.0028, 0.0028, 0.0029], dtype=float32))
        """

        v = self._check_vec(v_w_i)

        if not weights:
            weights = np.ones(len(v_w_i))
        else:
            weights = np.array(weights)

        resultant_vec = (v * weights[:, None]).sum(axis=0, dtype=np.float32).reshape(1, self.dim)

        d = self.get_distances(resultant_vec, metric=metric)
        nearest_indices, distances = self._nearest_sorting(d, k)

        return (nearest_indices[0], distances[0]) if include_distances else nearest_indices[0]

    def _get_weighted_brute(self, v_w_i, k, metric, weights, include_distances):
        """Retrieves nearest indices based on weights applied to per vector's distance.

        Algorithm:
        i) Compute the distance matrix .
        ii) Perform dot operation with this distance matrix and weights
        iii) Retrive nearest indices on the weighted distance matrix
        iv) Return

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> vecs = wv.get_vectors(['laptop', 'phone'])
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='full_matrix')
        array([10101, 27639,  3656, 15445, 11044], dtype=uint32)
        >>> wv.get_nearest(vecs, k=5, combination=True, combination_method='full_matrix',
        ... include_distances=True)
        (array([10101, 27639,  3656, 15445, 11044], dtype=uint32), array([0.6346, 0.6353, 0.6358, 0.6362, 0.6369], dtype=float32))
        """

        v = self._check_vec(v_w_i)
        d = self.get_distances(v, metric=metric)
        if not weights:
            weights = np.ones(len(v))
        else:
            weights = np.array(weights)

        w_d = np.dot(weights, d)
        nearest_indices, nearest_distances = self._nearest_sorting(w_d.reshape(1, len(w_d)), k)

        return (nearest_indices[0], nearest_distances[0]) if include_distances else nearest_indices[0]

    def _get_nearest(self, v_w_i, k, metric, include_distances):
        """Retrives nearest indices when combination=False.

        Tests:
        >>> wv = WordVecSpace(DATAFILE_ENV_VAR)
        >>> wv.get_nearest('laptop', k=5)
        array([15890, 18487, 32156,  3917, 31197], dtype=uint32)
        >>> wv.get_nearest('phone', k=5)
        array([ 3826, 10458,  2686, 17794, 20515], dtype=uint32)
        """

        d = self.get_distances(v_w_i, metric=metric)
        nearest_indices, distances = self._nearest_sorting(d, k)

        if isinstance(v_w_i, (list, tuple)) or isinstance(v_w_i, np.ndarray) and len(v_w_i) > 1:
            return (nearest_indices, distances) if include_distances else nearest_indices
        else:
            return (nearest_indices[0], distances[0]) if include_distances else nearest_indices[0]
