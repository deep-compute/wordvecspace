import os
import numpy as np
import tornado.ioloop
import tornado.web
from typing import Union

# FIXME: use dummy logger using deeputil
from logging import Logger

from kwikapi.tornado import RequestHandler
from kwikapi import API

from .wordvecspace import WordVecSpace

np.set_printoptions(precision=4)
check_equal = np.testing.assert_array_almost_equal

DATADIR_ENV_VAR = os.environ.get('WORDVECSPACE_DATADIR', ' ')

class APIFunctions(object):
    def __init__(self, input_dir):
        self.wv = WordVecSpace(input_dir)
        self.wv.load()

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

    def get_word_at_index(self, index: int, raise_exc: bool=False) -> str:
        '''
        Get the word for an index

        get_word_at_index(509) => india
        '''

        return self.wv.get_word_at_index(index, raise_exc)

    def get_word_vector(self, word_or_index: Union[str, int], raise_exc: bool=False) -> list:
        '''
        Get vector for a given word or index

        get_word_vector('india') => [-6.4482 -2.1636  5.7277 -3.7746  3.583 ]
        get_word_vector(509, normalized=True) => [-0.6259 -0.21    0.5559 -0.3664  0.3478]
        get_word_vector('inidia', normalized=True) => [ 0.  0.  0.  0.  0.]
        '''

        return self.wv.get_word_vector(word_or_index, raise_exc).tolist()

    def get_vector_magnitudes(self, words_or_indices: Union[int, str, list, tuple], raise_exc: bool=False) -> list:
        '''
        Get vector magnitudes for given words or indices

        get_vector_magnitudes(["hi", "india"]) => [  8.7948  10.303 ]
        get_vector_magnitudes(["inidia", "india"]) => [  0.     10.303]
        '''

        return self.wv.get_vector_magnitudes(words_or_indices, raise_exc).tolist()

    def get_word_occurrences(self, word_or_index: Union[str, int], raise_exc: bool=False) -> int:
        '''
        Get occurences for a given word or index

        get_word_occurrences(5327) => 297
        get_word_occurrences("india") => 3242
        get_word_occurrences("inidia") => None
        '''

        return self.wv.get_word_occurrences(word_or_index, raise_exc)

    def get_word_vectors(self, words_or_indices: Union[list, tuple], raise_exc: bool=False) -> list:
        '''
        Get vectors for given words or indices

        get_word_vectors(["hi", "india"]) => [[ 0.2473  0.2535 -0.3206  0.8058  0.3501], [-0.6259 -0.21    0.5559 -0.3664  0.3478]]
        get_word_vectors(["hi", "inidia"]) => [[ 0.2473  0.2535 -0.3206  0.8058  0.3501], [ 0.      0.      0.      0.      0.    ]]
        '''

        return self.wv.get_word_vectors(words_or_indices, raise_exc).tolist()

    def get_distance(self, word1: Union[str, int], word2: Union[str, int], raise_exc: bool=False) -> float:
        '''
        Get cosine distance between two words

        get_distance(250, "india") => 1.16397565603
        '''

        return self.wv.get_distance(word1, word2, raise_exc)

    def get_distances(self, row_words: Union[str, int, tuple, list], raise_exc: bool=False) -> list:
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
        '''

        return self.wv.get_distances(row_words, raise_exc).tolist()

    def get_nearest_neighbors(self, word: Union[str, int], k: int=512, raise_exc: bool=False) -> list:
        '''
        get_nearest_neighbors("india", 20) => [509, 486, 14208, 20639, 8573, 3389, 5226, 20919, 10172, 6866, 9772, 24149, 13942, 1980, 20932, 28413, 17910, 2196, 28738, 20855]
        '''

        neg = self.wv.get_nearest_neighbors(word, k, raise_exc).tolist()
        for key, val in enumerate(neg):
            neg[key] = int(val)

        return neg

class WordVecSpaceServer(object):
    def __init__(self, input_dir, port, log=Logger):
        self.input_dir = input_dir
        self.port = port
        self.log = log

    def start(self):
        self.api = API(log=self.log)
        self.api.register(APIFunctions(self.input_dir), 'v1')

        app = self._make_app()
        app.listen(self.port)
        tornado.ioloop.IOLoop.current().start()

    def _make_app(self):
        return tornado.web.Application([
            (r'^/api/.*', RequestHandler, dict(api=self.api)),
        ])
