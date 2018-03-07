import numpy as np
import pycuda.autoinit
from pycuda.gpuarray import GPUArray, dot, to_gpu
import skcuda.cublas as cublas

from .mem import WordVecSpaceMem

class CudaWordVecSpaceMem(WordVecSpaceMem):

    def __init__(self):
        super(CudaWordVecSpace, self).__init__()

        vectors_gpu = to_gpu(self.vectors)
        self.vectors = vectors_gpu

    def _make_array(self, shape, dtype):
        return GPUArray(shape, dtype)

    def _perform_dot(self, v1, v2):
        return dot(v1, v2)

    def _perform_sgemv(self, mat, v, vec_out, nvecs, dim):
	'''
        NOTES: cuBLAS uses Fortran layout
        cublas_sgemv is used to multiply matrix and vector (LEVEl 2 BLAS)

        cublas_handle   -> handle to the cuBLAS library context
        t               -> transpose
        dim             -> number of columns of matrix
        nvecs           -> number of rows of matrix
        alpha           -> scalar used for multiplication of mat
        mat.gpudata     -> matrix mat
        dim             -> columns of matrix
        v.gpudata       -> vector v
        incX            -> Stride within X. For example, if incX is 7, every 7th element is used.
        beta            -> scalar used for multiplication of v
        v_out.gpudata   -> result
        incY            -> Stride within Y. For example, if incx is 7, every 7th element is used

        Readmore        -> http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemv
        '''
        alpha = np.float32(1.0)
        beta = np.float32(0.0)

        incx = 1
        incy = 1

        cublas_handle = cublas.cublasCreate()

        cublas.cublasSgemv(cublas_handle,
                            't',
                            dim,
                            nvecs,
                            alpha,
                            mat.gpudata,
                            dim,
                            v.gpudata,
                            incx,
                            beta,
                            vec_out.gpudata,
                            incy)

        cublas.cublasDestroy(cublas_handle)

        return vec_out

    def _perform_sgemm(self, mat_a, mat_b, mat_out):
        nvecs_a, dim = mat_a.shape
        nvecs_b, dim = mat_b.shape

        alpha = np.float32(1.0)
        beta = np.float32(0.0)

        '''
        cublas_sgemm is used to  multiply matrix and matrix(LEVEL 3 BLAS)

        cublas_handle   -> handle to the cuBLAS-library context
        t               -> transpose mat_b
        n               -> notranspose mat_a
        nvecs_b         -> rows of mat_b
        nvecs_a         -> rows of mat_a
        dim             -> Common dimensions in mat_a and mat_b
        alpha           -> scaling factor for multiplication of mat_a and mat_b
        mat_b.gpudata   -> matrix mat_b
        dim             -> columns of mat_b
        mat_a.gpudata   -> matrix mat_a
        dim             -> columns of mat_a
        beta            -> scaling factor for r_gpu
        mat_out.gpudata -> matirx mat_out
        nvecs_b         -> rows of mat_b

        Read more       -> http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
        '''

        cublas_handle = cublas.cublasCreate()

        cublas.cublasSgemm(cublas_handle,
                            't',
                            'n',
                            nvecs_b,
                            nvecs_a,
                            dim,
                            alpha,
                            mat_b.gpudata,
                            dim,
                            mat_a.gpudata,
                            dim,
                            beta,
                            mat_out.gpudata,
                            nvecs_b)
        cublas.cublasDestroy(cublas_handle)

        return mat_out

    def get_distances(self, row_words, col_words=None, raise_exc=False):
        dvec = super(CudaWordVecSpace, self).get_distances(row_words, col_words, raise_exc)
        return dvec.get()

WordVecSpaceMem = CudaWordVecSpaceMem
