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
        '''
        cblas_sgemv is used to multiply vector and matrix.

        CblasRowMajor                   -> Multiiply in a row major order
        CblasNoTrans                    -> Whether to transpose matix or not
        nvecs, dim                      -> Rows, columns of Matrix
        c_float(1.0)                    -> Scaling factor for the product of matrix and vector
        mat.ctypes.data_as(c_void_p)    -> matrix
        dim                             -> Columns of matirx
        v.ctypes.data_as(c_void_p)      -> vector
        incX                            -> Stride within X. For example, if incX is 7, every 7th element is used.
        c_float(0.0)                    -> Scaling factor for vector.
        vec_out.ctypes.data_as(c_void_p)-> result vector
        incY                            -> Stride within Y. For example, if incY is 7, every 7th element is used

        Read more                       -> https://developer.apple.com/documentation/accelerate/1513065-cblas_sgemv?language=objc
        '''
        cblas.cblas_sgemv(CblasRowMajor,
                          CblasNoTrans,
                          nvecs,
                          dim,
                          c_float(1.0),
                          mat.ctypes.data_as(c_void_p),
                          dim,
                          v.ctypes.data_as(c_void_p),
                          incX,
                          c_float(0.0),
                          vec_out.ctypes.data_as(c_void_p),
                          incY)
        return vec_out

    @abstractmethod
    def _perform_sgemm(self, mat_a, mat_b, mat_out):
        '''
        cblas_sgemm is for multiplying matrix and matrix

        CblasRowMajor                   -> Specifies row-major (C)
        CblasNoTrans                    -> Specifies whether to transpose matrix mat_a
        CblasTrans                      -> Specifies whether to transpose matrix mat_b
        len(mat_a)                      -> Rows of result(Rows of mat_out)
        len(mat_b)                      -> Columns of result(Columns of mat_out)
        self.dim                        -> Common dimension in mat_a and mat_b
        c_float(1.0)                    -> Scaling factor for the product of matrices mat_a and mat_b
        mat_a.ctypes.data_as(c_void_p)  -> matrix mat_a
        self.dim                        -> Columns of mat_a
        mat_b.ctypes.data_as(c_void_p)  -> matrix mat_b
        self.dim                        -> Columns of mat_b
        c_float(0.0)                    -> Scaling factor for matrix mat_out
        mat_out.ctypes.data_as(c_void_p)-> matirx mat_out
        len(mat_b)                      -> Columns of mat_out

        Read more                       -> https://developer.apple.com/documentation/accelerate/1513264-cblas_sgemm?language=objc
        '''
        cblas.cblas_sgemm(CblasRowMajor,
                          CblasNoTrans,
                          CblasTrans,
                          len(mat_a),
                          len(mat_b),
                          self.dim,
                          c_float(1.0),
                          mat_a.ctypes.data_as(c_void_p),
                          self.dim,
                          mat_b.ctypes.data_as(c_void_p),
                          self.dim,
                          c_float(0.0),
                          mat_out.ctypes.data_as(c_void_p),
                          len(mat_b))
        return mat_out

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
