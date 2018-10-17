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

    def get_distances(self,
                    row_words_or_indices: Union[list, np.ndarray],
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

    def get_nearest(self, v_w_i: list,
                    k: int=DEFAULT_K,
                    distances: bool=False,
                    combination: bool=False,
                    weights: list=None,
                    metric: str='cosine') -> np.ndarray:

        d = self.get_distances(v_w_i, metric=metric)

        if not weights:
            weights = np.ones(len(v_w_i))

        if combination and len(weights) == len(v_w_i):
            weights = np.array(weights)
            w_d = np.dot(weights, d)
            nearest_indices, dist = self._nearest_sorting(w_d.reshape(1, len(w_d)), k)

            if distances:
                return nearest_indices, dist

            else:
                return nearest_indices

        nearest_indices, dist = self._nearest_sorting(d, k)

        if isinstance(v_w_i, (list, tuple)) or isinstance(v_w_i, np.ndarray) and len(v_w_i) > 1:
            return (nearest_indices, dist) if distances else nearest_indices

        else:
            return (nearest_indices[0], dist[0]) if distances else nearest_indices[0]
