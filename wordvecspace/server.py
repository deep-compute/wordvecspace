from typing import Union

import tornado.ioloop
import tornado.web
from kwikapi.tornado import RequestHandler
from kwikapi import API
from deeputil import Dummy

from .mem import WordVecSpaceMem
from .annoy import WordVecSpaceAnnoy

DUMMY_LOG = Dummy()

class APIFunctions(object):
    def __init__(self, _type, input_file, n_trees, metric, index_fpath):
        self._type = _type

        if self._type == 'mem':
            self.wv = WordVecSpaceMem(input_file, metric=metric)

        elif self._type == 'annoy':
            self.wv = WordVecSpaceAnnoy(input_file, n_trees=n_trees, metric=metric, index_fpath=index_fpath)

    def does_word_exist(self, word: str) -> bool:
        '''
        Check if a word exists in the vector space

        does_word_exist("india") => True
        does_word_exist("sdaksjl") => False
        '''

        return self.wv.does_word_exist(word)

    def get_word_index(self, word: str, raise_exc: bool=False) -> int:
        '''
        Get the index of a word

        if `word` is an integer already
        and it is a valid index (i.e. in range)
        then it will be returned

        get_word_index("india") => 509
        get_word_index("inidia") => None
        '''

        return self.wv.get_word_index(word, raise_exc)

    def get_word_indices(self, words: list, raise_exc: bool=False) -> list:
        '''
        Get indices for given words

        get_word_indices(['the', 'deepcompute', 'india']) => [1, None, 509]
        '''

        return self.wv.get_word_indices(words, raise_exc)

    def get_word_at_index(self, index: int, raise_exc: bool=False) -> str:
        '''
        Get the word for an index

        get_word_at_index(509) => india
        '''

        return self.wv.get_word_at_index(index, raise_exc)

    def get_word_at_indices(self, indices: list, raise_exc: bool=False) -> list:
        '''
        Get words for given indices

        get_word_at_indices([1,509,71190,72000]) => ['the', 'india', 'reka', None]
        '''

        return self.wv.get_word_at_indices(indices, raise_exc)

    def get_word_vector(self, word_or_index: Union[str, int], normalized: bool=False, raise_exc: bool=False) -> list:
        '''
        Get vector for a given word or index

        get_word_vector('india') => [-6.4482 -2.1636  5.7277 -3.7746  3.583 ]
        get_word_vector(509, normalized=True) => [-0.6259 -0.21    0.5559 -0.3664  0.3478]
        get_word_vector('inidia', normalized=True) => [ 0.  0.  0.  0.  0.]
        '''

        return self.wv.get_word_vector(word_or_index, normalized=normalized, raise_exc=raise_exc).tolist()

    def get_vector_magnitude(self, word_or_index: Union[str, int], raise_exc: bool=False) -> int:
        '''
        Get magnitude for given word

        get_vector_magnitude("hi") => 8.7948
        '''

        return self.wv.get_vector_magnitude(self, word_or_index, raise_exc=raise_exc)

    def get_vector_magnitudes(self, words_or_indices: Union[list, tuple], raise_exc: bool=False) -> list:
        '''
        Get vector magnitudes for given words or indices

        get_vector_magnitudes(["hi", "india"]) => [  8.7948  10.303 ]
        get_vector_magnitudes(["inidia", "india"]) => [  0.     10.303]
        '''

        return self.wv.get_vector_magnitudes(words_or_indices, raise_exc).tolist()

    def get_word_occurrence(self, word_or_index: Union[str, int], raise_exc: bool=False) -> Union[int, None]:
        '''
        Get word occurrence for given word

        get_word_occurrences(5327) => 297
        get_word_occurrences("india") => 3242
        get_word_occurrences("inidia") => None
        '''

        occur = self.wv.get_word_occurrence(word_or_index, raise_exc)

        return int(occur) if occur else None

    def get_word_occurrences(self, words_or_indices: list, raise_exc: bool=False) -> list:
        '''
        Get occurences for a given word or index

        get_word_occurrences(["the", "india", "Deepcompute"]) => [1061396, 3242, None]
        '''

        res = self.wv.get_word_occurrences(words_or_indices, raise_exc)
        for val in res:
            res[val] = int(res[val])

        return res

    def get_word_vectors(self, words_or_indices: Union[list, tuple], normalized: bool=False, raise_exc: bool=False) -> list:
        '''
        Get vectors for given words or indices

        get_word_vectors(["hi", "india"]) => [[ 0.2473  0.2535 -0.3206  0.8058  0.3501], [-0.6259 -0.21    0.5559 -0.3664  0.3478]]
        get_word_vectors(["hi", "inidia"]) => [[ 0.2473  0.2535 -0.3206  0.8058  0.3501], [ 0.      0.      0.      0.      0.    ]]
        '''

        return self.wv.get_word_vectors(words_or_indices, normalized=normalized, raise_exc=raise_exc).tolist()

    def get_distance(self, word_or_index1: Union[str, int], word_or_index2: Union[str, int], metric: Union[str, None]=None, raise_exc: bool=False) -> float:
        '''
        Get cosine distance between two words

        get_distance(250, "india") => 1.16397565603
        get_distance(250, "india", metric='euclidean') => 1.5029966831207275
        '''

        if self._type == 'mem':
            return self.wv.get_distance(word_or_index1, word_or_index2, metric=metric, raise_exc=raise_exc)

        return self.wv.get_distance(word_or_index1, word_or_index2, raise_exc=raise_exc)

    def get_distances(self, row_words_or_indices: Union[str, int, tuple, list], col_words_or_indices: Union[list, None]=None, metric: Union[str, None]=None, raise_exc: bool=False) -> list:
        '''
        Get distances between given words and all words in the vector space

        get_distances(word)
        get_distances(words)
        get_distances(word, words)
        get_distances(words_x, words_y)

        get_distances("for", ["to", "for", "india"] => [[  1.4990e-01], [ -1.1921e-07], [  1.3855e+00]]
        get_distances("for", ["to", "for", "inidia"]) => [[  1.4990e-01], [ -1.1921e-07], [  1.0000e+00]]
        get_distances(["india", "for"], ["to", "for", "usa"]) => [[  1.1830e+00,   1.3855e+00,   4.8380e-01], [  1.4990e-01,  -1.1921e-07,   1.4975e+00]]
        get_distances(["india", "usa"]) => [[ 1.4903,  0.4202,  0.269 , ...,  1.2041,  1.3539,  0.6154], [ 1.8084,  0.9541,  1.1678, ...,  0.5963,  1.0458,  1.1608]]
        get_distances(["andhra"]) => [[ 1.3432,  0.5781,  0.2306, ...,  1.0937,  1.1369,  0.4284]]
        get_distances(["andhra"], metric='euclidean') => [[ 1.601   1.108   0.7739 ...,  1.4103  1.5646  1.1079]]
        '''
        if self._type == 'mem':
            return self.wv.get_distances(row_words_or_indices, col_words_or_indices=col_words_or_indices, metric=metric, raise_exc=raise_exc).tolist()

        return self.wv.get_distances(row_words_or_indices, col_words_or_indices=col_words_or_indices, raise_exc=raise_exc).tolist()

    def get_nearest(self, words_or_indices: Union[str, int, list, tuple], k: int=512, metric: Union[str, None]=None, combination: bool=False, raise_exc: bool=False) -> list:
        '''
        get_nearest_neighbors("india", 20) => [509, 486, 14208, 20639, 8573, 3389, 5226, 20919, 10172, 6866, 9772, 24149, 13942, 1980, 20932, 28413, 17910, 2196, 28738, 20855]
        get_nearest(["ram", "india"], 5, metric='euclidean') => [[3844, 38851, 25381, 10830, 17049], [509, 486, 523, 4343, 14208]]
        get_nearest(['india', 'bosnia'], 10, combination=True) => [14208, 486, 523, 4343, 42424, 509]
        '''

        if self._type == 'mem':
            neg = self.wv.get_nearest(words_or_indices, k, raise_exc=raise_exc, metric=metric)

            if isinstance(words_or_indices, (tuple, list)) and len(words_or_indices) > 1:
                for neg_key, item in enumerate(neg):
                    for item_key, val in enumerate(item):
                        item[item_key] = int(val)
                    neg[neg_key] = item

            else:
                for key, val in enumerate(neg):
                    neg[key] = int(val)
        else:
            neg = self.wv.get_nearest(words_or_indices, k, raise_exc=raise_exc)

        return neg

class WordVecSpaceServer(object):
    N_TREES = 1
    METRIC = 'angular'

    def __init__(self, _type, input_file, port, n_trees=N_TREES, metric=METRIC, index_fpath=None, log=DUMMY_LOG):
        self._type = _type
        self.input_file = input_file
        self.port = port
        self.n_trees = n_trees
        self.metric = metric
        self.index_fpath = index_fpath
        self.log = log

    def start(self):
        self.api = API(log=self.log)
        self.api.register(APIFunctions(self._type,
                                    self.input_file,
                                    self.n_trees,
                                    self.metric,
                                    self.index_fpath), 'v1')

        app = self._make_app()
        app.listen(self.port)
        tornado.ioloop.IOLoop.current().start()

    def _make_app(self):
        return tornado.web.Application([
            (r'^/api/.*', RequestHandler, dict(api=self.api)),
        ])
