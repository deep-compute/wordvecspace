from typing import Union

import tornado.ioloop
import tornado.web
from kwikapi.tornado import RequestHandler
from kwikapi import API
from deeputil import Dummy

from .mem import WordVecSpaceMem
from .disk import WordVecSpaceDisk
from .annoy import WordVecSpaceAnnoy

DUMMY_LOG = Dummy()

class APIFunctions(object):
    def __init__(self, _type, input_dir, n_trees, metric, index_fpath):
        self._type = _type

        if self._type == 'mem':
            self.wv = WordVecSpaceMem(input_dir, metric=metric)

        elif self._type == 'annoy':
            self.wv = WordVecSpaceAnnoy(input_dir, n_trees=n_trees, metric=metric, index_fpath=index_fpath)

        elif self._type == 'disk':
            self.wv = WordVecSpaceDisk(input_dir, metric=metric)

    def does_word_exist(self, word: str) -> bool:
        '''
        Check if a word exists in the vector space

        does_word_exist("india") => True
        does_word_exist("sdaksjl") => False
        '''

        return self.wv.does_word_exist(word)

    def get_index(self, word: str) -> int:
        '''
        Get the index of a word

        if `word` is an integer already
        and it is a valid index (i.e. in range)
        then it will be returned

        get_index("india") => 509
        get_index("inidia") => None
        '''

        return self.wv.get_index(word)

    def get_indices(self, words: list) -> list:
        '''
        Get indices for given words

        get_indices(['the', 'deepcompute', 'india']) => [1, None, 509]
        '''

        return self.wv.get_indices(words)

    def get_word(self, index: int) -> str:
        '''
        Get the word for an index

        get_word(509) => india
        '''

        return self.wv.get_word(index)

    def get_words(self, indices: list) -> list:
        '''
        Get words for given indices

        get_words([1,509,71190,72000]) => ['the', 'india', 'reka', None]
        '''

        return self.wv.get_words(indices)

    def get_magnitude(self, word_or_index: Union[str, int]) -> int:
        '''
        Get magnitude for given word

        get_magnitude("hi") => 1.0
        '''

        return self.wv.get_magnitude(self, word_or_index)

    def get_magnitudes(self, words_or_indices: Union[list, tuple]) -> list:
        '''
        Get vector magnitudes for given words or indices

        get_magnitudes(["hi", "india"]) => [1.0, 1.0]
        get_magnitudes(["inidia", "india"]) => [0.0, 1.0]
        '''

        return self.wv.get_magnitudes(words_or_indices).tolist()

    def get_occurrence(self, word_or_index: Union[str, int]) -> Union[int, None]:
        '''
        Get word occurrence for given word

        get_occurrences(5327) => 297
        get_occurrences("india") => 3242
        get_occurrences("inidia") => None
        '''

        occur = self.wv.get_occurrence(word_or_index)

        return int(occur) if occur else None

    def get_occurrences(self, words_or_indices: list) -> list:
        '''
        Get occurences for a given word or index

        get_occurrences(["the", "india", "Deepcompute"]) => [1061396, 3242, None]
        '''

        res = self.wv.get_occurrences(words_or_indices).tolist()

        return res

    def get_vector(self, word_or_index: Union[str, int], normalized: bool=False) -> list:
        '''
        Get vector for a given word or index

        get_vector('india') => [-0.7871 -0.2993  0.3233 -0.2864  0.323 ]
        get_vector(509, normalized=True) => [-0.7871 -0.2993  0.3233 -0.2864  0.323 ]
        get_vector('inidia', normalized=True) => [ 0.  0.  0.  0.  0.]
        '''

        return self.wv.get_vector(word_or_index, normalized=normalized).tolist()

    def get_vectors(self, words_or_indices: Union[list, tuple], normalized: bool=False) -> list:
        '''
        Get vectors for given words or indices

        get_vectors(["hi", "india"]) => [[ 0.6342  0.2268 -0.3904  0.0368  0.6266], [-0.7871 -0.2993  0.3233 -0.2864  0.323 ]]
        get_vectors(["hi", "inidia"]) => [[[ 0.6342  0.2268 -0.3904  0.0368  0.6266], [ 0.      0.      0.      0.      0.    ]]
        '''

        return self.wv.get_vectors(words_or_indices, normalized=normalized, raise_exc=raise_exc).tolist()

    def get_distance(self, word_or_index1: Union[str, int], word_or_index2: Union[str, int], metric: str='angular') -> float:
        '''
        Get cosine distance between two words

        get_distance(250, "india") => 1.1418992727994919
        get_distance(250, "india", metric='euclidean') => 1.5112241506576538
        '''

        if self._type == 'mem' or 'disk':
            return self.wv.get_distance(word_or_index1, word_or_index2, metric=metric)

        return self.wv.get_distance(word_or_index1, word_or_index2)

    def get_distances(self, row_words_or_indices: Union[str, int, tuple, list], col_words_or_indices: Union[list, None]=None, metric: str='angular') -> list:
        '''
        Get distances between given words and all words in the vector space

        get_distances(word)
        get_distances(words)
        get_distances(word, words)
        get_distances(words_x, words_y)

        get_distances("for", ["to", "for", "india"] => [[  2.7428e-01,   5.9605e-08,   1.1567e+00]]
        get_distances("for", ["to", "for", "inidia"]) => [[  2.7428e-01,   5.9605e-08,   1.0000e+00]]
        get_distances(["india", "for"], ["to", "for", "usa"]) => [[[  1.1445e+00   1.1567e+00   3.7698e-01], [  2.7428e-01   5.9605e-08   1.6128e+00]]
        get_distances(["india", "usa"]) => [[ 1.5464  0.4876  0.3017 ...,  1.2492  1.2451  0.8925], [ 1.0436  0.9995  1.0913 ...,  0.6996  0.8014  1.1608]]
        get_distances(["andhra"]) => [[ 1.5418  0.7153  0.277  ...,  1.1657  1.0774  0.7036]]
        get_distances(["andhra"], metric='euclidean') => [[ 1.756   1.1961  0.7443 ...,  1.5269  1.4679  1.1862]]
        '''
        c = col_words_or_indices
        if self._type == 'mem' or 'disk':
            return self.wv.get_distances(row_words_or_indices, col_words_or_indices=c, metric=metric).tolist()

        return self.wv.get_distances(row_words_or_indices, col_words_or_indices=c).tolist()

    def get_nearest(self, v_w_i: Union[str, int, list, tuple], k: int=512, metric: str='angular', combination: bool=False) -> list:
        '''
        get_nearest("india", 20) => [509, 3389, 486, 523, 7125, 16619, 4491, 12191, 6866, 8776, 15232, 14208, 5998, 21916, 5226, 6322, 4343, 6212, 10172, 6186]
        get_nearest(["ram", "india"], 5, metric='euclidean') => [[3844, 16727, 15811, 42731, 41516], [509, 3389, 486, 523, 7125]]
        get_nearest(['india', 'bosnia'], 10, combination=True) => [523, 509, 486]
        '''
        if self._type == 'mem' or self._type == 'disk':
            neg = self.wv.get_nearest(v_w_i, k, metric=metric, combination=combination)
            neg = neg.tolist()

        else:
            neg = self.wv.get_nearest(v_w_i, k)

        return neg

class WordVecSpaceServer(object):
    N_TREES = 1
    METRIC = 'angular'

    def __init__(self, _type, input_dir, port, n_trees=N_TREES, metric=METRIC, index_fpath=None, log=DUMMY_LOG):
        self._type = _type
        self.input_dir = input_dir
        self.port = port
        self.n_trees = n_trees
        self.metric = metric
        self.index_fpath = index_fpath
        self.log = log

    def start(self):
        self.api = API(log=self.log)
        self.api.register(APIFunctions(self._type,
                                       self.input_dir,
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
