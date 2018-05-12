import os
from abc import ABCMeta, abstractmethod
from ctypes import cdll, c_void_p, c_float

# export blas path if your system has different path for blas
# $export WORDVECSPACE_BLAS_FPATH=/path/to/blas
# ex: $export WORDVECSPACE_BLAS_FPATH='/usr/lib/x86_64-linux-gnu/libopenblas.so.0'
BLAS_LIBRARY_FPATH = os.environ.get('WORDVECSPACE_BLAS_FPATH',\
        '/usr/lib/libopenblas.so.0')
cblas = cdll.LoadLibrary(BLAS_LIBRARY_FPATH)

# Some OpenBlas constants
CblasRowMajor = 101
CblasNoTrans = 111
CblasTrans = 112
incX = incY = 1

class WordVecSpace(object):
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
