import os
import sys

import numpy as np
from diskarray import DiskArray
from diskdict import DiskDict
from deeputil import Dummy

from .exception import UnknownIndex, UnknownWord

DUMMY_LOG = Dummy()

class WordVecSpaceFile(object):
    DEFAULT_MODE = 'w'
    GROWBY = 1000

    def __init__(self, dirpath, dim=None, sharding=False, mode=DEFAULT_MODE, growby=GROWBY, log=DUMMY_LOG):
        self.mode = mode
        self.dim = dim
        self.sharding = sharding
        self.dirpath = dirpath
        self.log = log
        self._growby = growby

        if self.mode == 'w':
            self._meta, (self.vecs, self.occurs, self.mags, self.wtoi, self.itow) = self._init_disk()

        if self.mode == 'r':
            self._meta, (self.vecs, self.occurs, self.mags, self.wtoi, self.itow) = self._read_from_disk()
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

    def _prepare_word_index_wvspace(self, dim, initialize=False, mode='r+'):
        def J(x): return os.path.join(self.dirpath, x)

        v_path = J('vectors')
        m_path = J('magnitudes')
        o_path = J('occurrences')

        # FIXME: Support taking memmap array from diskarray
        m_array = DiskArray(m_path, dtype='float32', mode=mode,
                            growby=self._growby, log=self.log)
        o_array = DiskArray(o_path, dtype='uint64', mode=mode,
                            growby=self._growby, log=self.log)

        if not initialize:
            v_array = DiskArray(v_path, dtype='float32', mode=mode,
                                growby=self._growby, log=self.log)
            vec_l = int(len(v_array)/dim)
            v_array = v_array[:].reshape(vec_l, dim)
            m_array = m_array[:]
            o_array = o_array[:]
        else:
            v_array = DiskArray(v_path, shape=(0, dim), dtype='float32', mode=mode,
                            growby=self._growby, log=self.log)

        wtoi = itow = None
        if not self.sharding:
            wtoi = DiskDict(J('wordtoindex'))
            itow = DiskDict(J('indextoword'))

        return v_array, o_array, m_array, wtoi, itow

    def __len__(self):
        return len(self.vecs)

    def add(self, vec, word, index, mag, occur):
        self.vecs.append(vec)
        self.mags.append(mag)
        self.occurs.append(occur)

        if not self.sharding:
            self.wtoi[word] = index
            self.itow[index] = word

    def close(self):
        self.vecs.flush()
        self.vecs.close()

        if not self.sharding:
            self.wtoi.close()
            self.itow.close()

        self.occurs.flush()
        self.occurs.close()

        self.mags.flush()
        self.mags.close()
