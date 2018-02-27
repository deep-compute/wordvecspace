from abc import ABCMeta, abstractmethod

class WordVecSpace(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def does_word_exist(self, word):
        pass

    @abstractmethod
    def get_word_index(self, word):
        pass

    @abstractmethod
    def get_word_indices(self, words):
        pass

    @abstractmethod
    def get_word_at_index(self, index):
        pass

    @abstractmethod
    def get_word_at_indices(self, indices):
        pass

    @abstractmethod
    def get_vector_magnitude(self, word_or_index):
        pass

    @abstractmethod
    def get_vector_magnitudes(self, words_or_indices):
        pass

    @abstractmethod
    def get_word_vector(self, word_or_index):
        pass

    @abstractmethod
    def get_word_vectors(self, words_or_indices):
        pass

    @abstractmethod
    def get_word_occurrence(self, word_or_index):
        pass

    @abstractmethod
    def get_word_occurrences(self, words_or_indices):
        pass

    @abstractmethod
    def get_distance(self, word_or_index1, word_or_index2):
        pass

    @abstractmethod
    def get_distances(self, row_words_or_indices):
        pass

    @abstractmethod
    def get_nearest(self, words_or_indices, k):
        pass
