import os
from math import sqrt

import numpy as np

from .fileformat import WordVecSpaceFile
from .base import WordVecSpace

# export data file path for test cases
# export WORDVECSPACE_DATAFILE=/path/to/data
DATAFILE_ENV_VAR = os.environ.get('WORDVECSPACE_DATAFILE', ' ')

class WordVecSpaceDisk(WordVecSpace):
    def __init__(self, input_file):
        self._f = WordVecSpaceFile(input_file, 'r')

        self.dim = int(self._f.dim)
        self.nvecs = len(self._f)

    def _make_array(self, shape, dtype):
        return np.ndarray(shape, dtype)

    def does_word_exist(self, word):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.does_word_exist("india"))
        True
        >>> print(wv.does_word_exist("inidia"))
        False
        '''

        if isinstance(word, int) or not self._f.get_word_index(word):
            return False
        else:
            return True

    def get_word_index(self, word, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_index("india"))
        509
        >>> print(wv.get_word_index("inidia"))
        None
        '''

        return self._f.get_word_index(word, raise_exc=raise_exc)

    def get_word_indices(self, words, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_indices(['the', 'deepcompute', 'india']))
        [1, None, 509]
        '''

        indices = []

        if isinstance(words, (list, tuple)):
            for word in words:
                index = self._f.get_word_index(word, raise_exc=raise_exc)
                indices.append(index)

        return indices

    def get_word_at_index(self, index, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_at_index(509))
        india
        >>> print(wv.get_word_at_index(72000))
        None
        '''

        return self._f.get_word(index, raise_exc=raise_exc)

    def get_word_at_indices(self, indices, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> wv.get_word_at_indices([1,509,71190])
        ['the', 'india', 'reka']
        >>> wv.get_word_at_indices([1,509,71190,72000])
        ['the', 'india', 'reka', None]
        '''

        words = []

        if isinstance(indices, (list, tuple)):
            for i in indices:
                word = self._f.get_word(i, raise_exc=raise_exc)
                words.append(word)

        return words

    def get_vector_magnitude(self, word_or_index, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_vector_magnitude("hi"))
        9.36555047958608
        '''

        vec = self.get_word_vector(word_or_index, raise_exc=raise_exc)

        res = 0
        for val in vec:
            res += (val * val)

        return sqrt(res)

    def get_vector_magnitudes(self, words_or_indices, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_vector_magnitudes(["hi", "india"]))
        [9.36555047958608, 10.141716028484925]
        '''

        mag = []
        if isinstance(words_or_indices, (list, tuple)):
            for w in words_or_indices:
                mag.append(self.get_vector_magnitude(w, raise_exc=raise_exc))

        return mag

    def get_word_vector(self, word_or_index, normalized=False, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_vector('india'))
        [-8.4037  4.2569  2.7932  0.6523 -2.4258]
        >>> wv.get_word_vector('inidia')
        array([[ 0.,  0.,  0.,  0.,  0.]], dtype=float32)
        '''
        if normalized:
            return self._f.get_word_vector(word_or_index, raise_exc=raise_exc) \
                                / self.get_vector_magnitude(word_or_index)

        return self._f.get_word_vector(word_or_index, raise_exc=raise_exc)

    def get_word_vectors(self, words_or_indices, normalized=False, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_vectors(["hi", "india"]))
        [[ 2.94   -3.3523 -6.4059 -2.1225 -4.7214]
         [-8.4037  4.2569  2.7932  0.6523 -2.4258]]
        >>> print(wv.get_word_vectors(["hi", "inidia"]))
        [[ 2.94   -3.3523 -6.4059 -2.1225 -4.7214]
         [ 0.      0.      0.      0.      0.    ]]
        '''

        wmat = []
        if isinstance(words_or_indices, (list, tuple)):
            n = len(words_or_indices)
            wmat = self._make_array(dtype=np.float32, shape=(n, self.dim))

            for i, w in enumerate(words_or_indices):
                wmat[i] = self.get_word_vector(w, normalized=normalized, raise_exc=raise_exc)

        return wmat

    def get_word_occurrence(self, word_or_index, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_occurrence('india'))
        3242
        >>> print(wv.get_word_occurrence('inidia'))
        None
        '''

        return self._f.get_word_occurrence(word_or_index, raise_exc=raise_exc)

    def get_word_occurrences(self, words_or_indices, raise_exc=False):
        '''
        >>> wv = WordVecSpaceDisk(DATAFILE_ENV_VAR)
        >>> print(wv.get_word_occurrences(['hi', 'india', 'bye']))
        [210, 3242, 82]
        >>> print(wv.get_word_occurrences(('hi', 'Deepcompute')))
        [210, None]
        '''

        word_occur = []

        if isinstance(words_or_indices, (list, tuple)):
            for word in words_or_indices:
                occur = self._f.get_word_occurrence(word, raise_exc=raise_exc)
                word_occur.append(occur)

        return word_occur
