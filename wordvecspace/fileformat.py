import os
import sys

import numpy as np
from diskarray import DiskArray
from diskarray import DiskStringArray
from diskdict import DiskDict, StaticStringIndexDict
from deeputil import Dummy

from .exception import UnknownIndex, UnknownWord

DUMMY_LOG = Dummy()

# export data directory path for test cases
# $export WORDVECSPACE_DATADIR=/path/to/data/
DATAFILE_ENV_VAR = os.environ.get('WORDVECSPACE_DATADIR', '')

class WordVecSpaceFile(object):
    DEFAULT_MODE = 'w'
    WORD = 1 << 1

    GROWBY = 1000

    def __init__(self, dirpath, dim=None, sharding=False, mode=DEFAULT_MODE, growby=GROWBY, log=DUMMY_LOG):
        self.mode = mode
        self.dim = dim
        self.sharding = sharding
        self.dirpath = dirpath
        self.log = log
        self._growby = growby

        self.words = []

        if self.mode == 'w':
            self._meta, (self._vectors, self._occurrences, self._magnitudes, self._wordtoindex, self._indextoword) = self._init_disk()

        if self.mode == 'r':
            self._meta, (self._vectors, self._occurrences, self._magnitudes, self._wordtoindex, self._indextoword) = self._read_from_disk()
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

        return m, self._prepare_word_index_wvspace(m['dim'], mode='r')

    def _prepare_word_index_wvspace(self, dim, initialize=False, mode='r'):
        v_dtype, o_dtype, m_dtype = self._make_dtype(dim)

        def J(x): return os.path.join(self.dirpath, x)

        def S(f): return os.stat(f).st_size

        v_path = J('vectors')
        m_path = J('magnitudes')
        o_path = J('occurrences')

        nvecs = noccurs = nmags = 0
        if not initialize:
            nvecs = S(v_path) / np.dtype(np.float32).itemsize
            nmags = S(m_path) / np.dtype(np.float32).itemsize
            noccurs = S(o_path) / np.dtype(o_dtype).itemsize

        v_array = DiskArray(v_path, shape=(int(nvecs),), dtype=v_dtype, growby=self._growby, log=self.log)
        m_array = DiskArray(m_path, shape=(int(nmags),), dtype=m_dtype, growby=self._growby, log=self.log)
        o_array = DiskArray(o_path, shape=(int(noccurs),), dtype=o_dtype, growby=self._growby, log=self.log)

        w_index = i_word = None
        if not self.sharding:
            if mode == 'r' and not os.path.isdir(J('wordtoindex')):
                return v_array, o_array, m_array, w_index, i_word

            if mode == 'r': w_index = StaticStringIndexDict(J('wordtoindex'))
            i_word = DiskStringArray(J('indextoword'))

            #o_path = J('occurrences')
            #if not initialize: noccurs = S(o_path) / np.dtype(np.uint64).itemsize
            #o_array = DiskArray(o_path, shape=(int(noccurs),), dtype=np.uint64, growby=self._growby, log=self.log)

        return v_array, o_array, m_array, w_index, i_word

    def _make_dtype(self, dim):
        v_dtype = [('vector', np.float32, dim)]
        o_dtype = [('occurrence', np.uint64)]
        m_dtype = [('magnitude', np.float32)]

        return v_dtype, o_dtype, m_dtype

    def _make_array(self, shape, dtype):
        return np.ndarray(shape, dtype)

    def __len__(self):
        '''
        >>> wv = WordVecSpaceFile(DATAFILE_ENV_VAR, mode='r')
        >>> wv.__len__()
        71291
        '''

        return len(self._vectors)

    def add(self, word, occurrence, vector, mag):
        self._vectors.append((vector,))
        self._magnitudes.append((mag, ))
        self._occurrences.append((occurrence,))

        if not self.sharding:
            self.words.append(word.encode('utf-8'))

    def close(self):
        self._vectors.flush()
        self._vectors.close()

        self._occurrences.flush()
        self._occurrences.close()

        if not self.sharding:
            if self.words:
                self._wordtoindex = StaticStringIndexDict(J('wordtoindex'), keys=self.words)
                self._indextoword.extend(self.words)

            self._wordtoindex.flush()
            self._wordtoindex.close()

            self._indextoword.flush()
            self._indextoword.close()

        self._magnitudes.flush()
        self._magnitudes.close()
