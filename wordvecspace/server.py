from typing import Union

import numpy as np
import tornado.ioloop
import tornado.web
from logging import Logger

from kwikapi.tornado import RequestHandler
from kwikapi import API

from .wordvecspace import WordVecSpace

class WVSAPIFunctions(object):
    def __init__(self, input_dir):
        self.wv = WordVecSpace(input_dir)
        self.wv.load()

    def does_word_exist(self, word: str) -> bool:
        return self.wv.does_word_exist(word)

    def get_word_index(self, word: str) -> int:
        return self.wv.get_word_index(word)

    def get_word_at_index(self, index: int) -> str:
        return self.wv.get_word_at_index(index)

    def get_word_vector(self, word_or_index: Union[str, int]) -> list:
        return self.wv.get_word_vector(word_or_index).tolist()

    # int, str, list[int, str], tuple(int, str)
    def get_vector_magnitudes(self, words_or_indices: Union[int, str, list, tuple]) -> list:
        return self.wv.get_vector_magnitudes(words_or_indices).tolist()

    def get_word_occurrences(self, word_or_index: Union[str, int]) -> int:
        return self.wv.get_word_occurrences(word_or_index)

    # list[int, str], tuple(int, str)
    def get_word_vectors(self, words_or_indices: Union[list, tuple], raise_exc=False) -> list:
        return self.wv.get_word_vectors(words_or_indices).tolist()

    def get_distance(self, word1: Union[str, int], word2: Union[str, int]) -> float:
        return self.wv.get_distance(word1, word2)

    # str, int, list[int, str], tuple(int, str)
    def get_distances(self, row_words: Union[str, int, tuple, list]) -> list:
        return self.wv.get_distances(row_words).tolist()

    def get_nearest_neighbors(self, word: Union[str, int], k=512) -> list:
        neg = self.wv.get_nearest_neighbors(word, k).tolist()
        for key, val in enumerate(neg):
            neg[key] = int(val)

        return neg

class WVSServer(object):
    def __init__(self, input_dir, port):
        self.input_dir = input_dir
        self.port = port

    def start(self):
        self.api = API(Logger, default_version='wvs')
        self.api.register(WVSAPIFunctions(self.input_dir), 'wvs')

        app = self._make_app()
        app.listen(self.port)
        tornado.ioloop.IOLoop.current().start()

    def _make_app(self):
        return tornado.web.Application([
            (r'^/api/.*', RequestHandler, dict(api=self.api)),
        ])
