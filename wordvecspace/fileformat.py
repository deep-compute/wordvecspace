import os
import sys

import numpy as np
from diskarray import DiskArray
from diskdict import DiskDict
from deeputil import Dummy

from .exception import UnknownIndex, UnknownWord

DUMMY_LOG = Dummy()

# export data directory path for test cases
# $export WORDVECSPACE_DATADIR=/path/to/data/
DATAFILE_ENV_VAR = os.environ.get('WORDVECSPACE_DATADIR', '')

class WordVecSpaceFile(object):
    DEFAULT_MODE = 'w'
    VECTOR = 1 << 0
    WORD = 1 << 1
    OCCURRENCE = 1 << 2
    ALL = VECTOR | WORD | OCCURRENCE

    GROWBY = 1000

    def __init__(self, dirpath, dim=None, mode=DEFAULT_MODE, growby=GROWBY, log=DUMMY_LOG):
        self.mode = mode
        self.dim = dim
        self.dirpath = dirpath
        self.log = log
        self._growby = growby

        if self.mode == 'w':
            self._meta, (self._vectors, self._occurrences, self._wordtoindex, self._indextoword) = self._init_disk()

        if self.mode == 'r':
            self._meta, (self._vectors, self._occurrences, self._wordtoindex, self._indextoword) = self._read_from_disk()
            self.dim = self._meta['dim']

    def _init_disk(self):
        def J(x): return os.path.join(self.dirpath, x)

        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

        meta = DiskDict(J('meta'))
        meta['dim'] = self.dim

        return meta, self._prepare_word_index_wvspace(self.dim, initialize=True)

    def _read_from_disk(self):
        m = DiskDict(os.path.join(self.dirpath, 'meta'))

        return m, self._prepare_word_index_wvspace(m['dim'])

    def _prepare_word_index_wvspace(self, dim, initialize=False):
        v_dtype, o_dtype = self._make_dtype(dim)

        def J(x): return os.path.join(self.dirpath, x)

        def S(f): return os.stat(f).st_size

        v_path = J('vectors')
        o_path = J('occurrences')

        nvecs = noccurs = 0
        if not initialize:
            nvecs = S(v_path) / np.dtype(v_dtype).itemsize
            noccurs = S(o_path) / np.dtype(o_dtype).itemsize

        v_array = DiskArray(v_path, shape=(int(nvecs),), dtype=v_dtype, growby=self._growby, log=self.log)
        o_array = DiskArray(o_path, shape=(int(noccurs),), dtype=o_dtype, growby=self._growby, log=self.log)

        w_index = DiskDict(J('wordtoindex'))
        i_word = DiskDict(J('indextoword'))

        return v_array, o_array, w_index, i_word

    def _make_dtype(self, dim):

        v_dtype = [('vector', np.float32, dim)]

        o_dtype = [('occurrence', np.uint64)]

        return v_dtype, o_dtype

    def _make_array(self, shape, dtype):
        return np.ndarray(shape, dtype)

    def __len__(self):
        '''
        >>> wv = WordVecSpaceFile(DATAFILE_ENV_VAR, mode='r')
        >>> wv.__len__()
        71291
        '''

        return len(self._vectors)

    def add(self, word, occurrence, vector):
        self._vectors.append((vector))
        self._occurrences.append((occurrence,))

        pos = len(self._vectors) - 1
        self._wordtoindex[word] = pos
        self._indextoword[pos] = word

    def get_word(self, index, raise_exc=False):
        '''
        >>> wv = WordVecSpaceFile(DATAFILE_ENV_VAR, mode='r')
        >>> wv.get_word(1)
        'the'
        '''

        try:
            word = self._indextoword[index]
        except KeyError:
            if raise_exc:
                raise UnknownIndex(index)
            else:
                return None

        return word

    def get(self, index, flags=ALL):
        '''
        >>> wv = WordVecSpaceFile(DATAFILE_ENV_VAR, mode='r')
        >>> sorted(wv.get(1).items())
        [('occurrence', 1061396), ('vector', array([-0.5927,  0.0707,  0.2333, -0.5614, -0.5236], dtype=float32)), ('word', 'the')]
        >>> wv.get(1, wv.VECTOR)
        array([-0.5927,  0.0707,  0.2333, -0.5614, -0.5236], dtype=float32)
        >>> wv.get(1, wv.OCCURRENCE)
        1061396
        >>> wv.get(1, wv.WORD)
        'the'
        '''

        if not isinstance(index, int) or index >= len(self):
            raise IndexError

        vector = word = occurrence = None

        if flags & self.VECTOR:
            vector = self._vectors[index]['vector']

        if flags & self.WORD:
            word = self.get_word(index)

        if flags & self.OCCURRENCE:
            occurrence = self._occurrences[index]['occurrence']

        d = (('vector', vector), ('word', word), ('occurrence', occurrence))
        d = [(k, v) for k, v in d if v is not None]
        # only one attribute requested
        if len(d) == 1:
            return d[0][-1]  # return single value
        else:
            return dict(d)

    def get_word_index(self, word, raise_exc=False):
        '''
        >>> wv = WordVecSpaceFile(DATAFILE_ENV_VAR, mode='r')
        >>> wv.get_word_index('the')
        1
        '''

        if isinstance(word, int):
            if word < len(self._vectors):
                return word
            else:
                if raise_exc:
                    raise UnknownIndex(word)
                return None
        try:
            index = self._wordtoindex[word]
            get_word = self.get(index, self.WORD)
            if word == get_word:
                return index
        except KeyError:
            if raise_exc == True:
                raise UnknownWord(word)
            return None

    def get_word_vector(self, word, raise_exc=False):
        '''
        >>> wv = WordVecSpaceFile(DATAFILE_ENV_VAR, mode='r')
        >>> wv.get_word_vector('the')
        array([-0.5927,  0.0707,  0.2333, -0.5614, -0.5236], dtype=float32)
        '''

        index = self.get_word_index(word, raise_exc=raise_exc)
        if index:
            vector = self.get(index, self.VECTOR)
        else:
            vector = self._make_array(shape=(self.dim,), dtype=np.float32)
            vector.fill(0.0)

        return vector

    def getmany(self, index, num, type=VECTOR):
        '''
        This function returns vectors, occurrences, words
        based on the type in the range of indices
        >>> wv = WordVecSpaceFile(DATAFILE_ENV_VAR, mode='r')
        >>> wv.getmany(0, 2)
        memmap([[ 0.5049,  0.5575, -0.4832, -0.4135,  0.1724],
                [-0.5927,  0.0707,  0.2333, -0.5614, -0.5236]], dtype=float32)
        >>> wv.getmany(0, 2, wv.WORD)
        ['</s>', 'the']
        >>> wv.getmany(0, 2, wv.OCCURRENCE)
        memmap([      0, 1061396], dtype=uint64)
        '''

        if index > len(self) or num > len(self):
            raise IndexError

        s, e = index, num
        if type == self.VECTOR:
            return self._vectors[s:e]['vector']
        elif type == self.OCCURRENCE:
            return self._occurrences[s:e]['occurrence']
        else:
            words = []
            for i in range(s, e):
                word = self.get(i, self.WORD)
                words.append(word)
            return words

    def get_word_occurrence(self, word, raise_exc=False):
        '''
        >>> wv = WordVecSpaceFile(DATAFILE_ENV_VAR, mode='r')
        >>> wv.get_word_occurrence('the')
        1061396
        '''

        index = self.get_word_index(word, raise_exc=raise_exc)
        try:
            occurrence = self.get(index, self.OCCURRENCE)
        except IndexError:
            occurrence = None

        return occurrence

    def close(self):
        self._vectors.flush()
        self._occurrences.flush()
        self._wordtoindex.close()
        self._indextoword.close()
