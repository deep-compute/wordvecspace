from abc import ABCMeta, abstractmethod

import numpy as np

class WordVecSpaceBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def _perform_sgemv(self, mat, v, vec_out, nvecs, dim):
        res = np.dot(mat, v.T)

        return res

    @abstractmethod
    def _perform_sgemm(self, mat_a, mat_b, mat_out):
        res = np.dot(mat_a, mat_b.T)

        return res

    @abstractmethod
    def does_word_exist(self, word):
        pass

    @abstractmethod
    def get_index(self, word):
        pass

    @abstractmethod
    def get_indices(self, words):
        pass

    @abstractmethod
    def get_word(self, index):
        pass

    @abstractmethod
    def get_words(self, indices):
        pass

    @abstractmethod
    def get_magnitude(self, word_or_index):
        pass

    @abstractmethod
    def get_magnitudes(self, words_or_indices):
        pass
    
    @abstractmethod
    def get_occurrence(self, word_or_index):
        pass

    @abstractmethod
    def get_occurrences(self, words_or_indices):
        pass

    @abstractmethod
    def get_vector(self, word_or_index):
        pass

    @abstractmethod
    def get_vectors(self, words_or_indices):
        pass

    @abstractmethod
    def get_distance(self, word_or_index1, word_or_index2):
        pass

    @abstractmethod
    def get_distances(self, row_words_or_indices):
        pass

    @abstractmethod
    def get_nearest(self, v_w_i):
        pass
