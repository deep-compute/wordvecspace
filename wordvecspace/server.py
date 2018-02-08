from typing import Union

import numpy as np
import tornado.ioloop
import tornado.web
from logging import Logger

from kwikapi.tornado import RequestHandler
from kwikapi import API
from deeputil import Dummy

from .wordvecspace import WordVecSpace

DUMMY_LOG = Dummy()

class APIFunctions(object):
    def __init__(self, input_dir):
        self.wv = WordVecSpace(input_dir)
        self.wv.load()

    def does_word_exist(self, word):
        return self.wv.does_word_exist(word)

    def get_word_index(self, word):
        return self.wv.get_word_index(word)

    def get_word_at_index(self, index):
        return self.wv.get_word_at_index(index)

    def get_word_vector(self, word_or_index):
        return self.wv.get_word_vector(word_or_index).tolist()

    def get_vector_magnitudes(self, words_or_indices):
        return self.wv.get_vector_magnitudes(words_or_indices).tolist()

    def get_word_occurrences(self, word_or_index):
        return self.wv.get_word_occurrences(word_or_index)

    def get_word_vectors(self, words_or_indices, raise_exc=False):
        return self.wv.get_word_vectors(words_or_indices).tolist()

    def get_distance(self, word1, word2):
        return self.wv.get_distance(word1, word2)

    def get_distances(self, row_words):
        return self.wv.get_distances(row_words).tolist()

    def get_nearest_neighbors(self, word, k=512):
        neg = self.wv.get_nearest_neighbors(word, k).tolist()
        for key, val in enumerate(neg):
            neg[key] = int(val)

        return neg

class WordVecSpaceServer(object):
    def __init__(self, input_dir, port, log=DUMMY_LOG):
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
