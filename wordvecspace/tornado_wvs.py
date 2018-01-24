import numpy as np
import tornado.ioloop
import tornado.web

from kwikapi.tornado import RequestHandler
from kwikapi import API
from logging import Logger

from typing import Union

from wordvecspace import WordVecSpace

class Tornado_wvs(WordVecSpace):

    def does_word_exist(self, word: str) -> bool:
        return super().does_word_exist(word)

    def get_word_index(self, word: str, raise_exc=False) -> int:
       return super().get_word_index(word, raise_exc=False)

    def get_word_at_index(self, index: int, raise_exc=False) -> str:
        return super().get_word_at_index(index, raise_exc=False)

    def get_word_vector(self, word_or_index: Union[str, int], normalized=False, raise_exc=False) -> np.ndarray:
        return super().get_word_vector(word_or_index, normalized=False, raise_exc=False)

    def get_vector_magnitudes(self, words_or_indices: Union[tuple, list], raise_exc=False) -> list:
        return super().get_vector_magnitudes(words_or_indices, raise_exc=False)

    def get_word_occurrences(self, word_or_index: Union[str, int], raise_exc=False) -> int:
        return super().get_word_occurrences(word_or_index, raise_exc=False)

    def get_word_vectors(self, words_or_indices: list, raise_exc=False) -> np.ndarray:
        return super().get_word_vectors(words_or_indices, raise_exc=False)

    def get_distance(self, word1: Union[str, int], word2: Union[str, int], raise_exc=False) -> float:
        return super().get_distance(word1, word2, raise_exc=False)

    def get_distances(self, row_words: Union[tuple, list], col_words=None, raise_exc=False) -> np.ndarray:
        return super().get_distances(row_words, col_words=None, raise_exc=False)

    #DEFAULT_K = 512
    def get_nearest_neighbors(self, word: str, DEFAULT_K: int) -> np.ndarray:
        return super().get_nearest_neighbors(word, k=DEFAULT_K)

wv = Tornado_wvs("/home/ram/Ram/data/shard_0")
wv.load()

api = API(Logger, default_version='v1')
api.register(wv, 'v1', 'dc')

def make_app():
    return tornado.web.Application([
        (r'^/api/.*', RequestHandler, dict(api=api)),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
