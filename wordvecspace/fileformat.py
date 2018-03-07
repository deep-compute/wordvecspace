import os
import sys
import tempfile

import numpy as np
import tables
import cmph

from .exception import UnknownIndex, UnknownWord

# export data file path for test cases
# $export WVSPACEFILE_DATADIR=/path/to/data/
DATAFILE_ENV_VAR = os.environ.get('WORDVECSPACE_DATAFILE', '')
DIM = 5

class WordVecSpaceFile(object):
    DEFAULT_MODE = 'r'
    VECTOR       = 1 << 0
    WORD         = 1 << 1
    OCCURRENCE   = 1 << 2
    ALL          = VECTOR | WORD | OCCURRENCE

    VAL_TO_TABLE_NAME = {VECTOR: 'vectors', WORD: 'words', OCCURRENCE: 'occurrences'}

    def __init__(self, input_file, dim=None, mode=DEFAULT_MODE):
        self.mode = mode
        self.dim = dim
        self._fobj = tables.open_file(input_file, self.mode)

        if self.mode == 'w':
            self._create_new()
            self._fobj.root._v_attrs.dim = self.dim

        if self.mode == 'r':
            self.dim = self._fobj.root._v_attrs.dim

            d = self._fobj.root
            t = self._create_tmpfile()

            if (sys.version_info > (3, 0)):
                t.write((bytes(d.cmph_data[:])))
            else:
                cmph_data = [chr(x) for x in d.cmph_data]
                for value in cmph_data:
                    t.write(value)

            t.close()

            with open(t.name, 'r') as inpf:
                self.mph = cmph.load_hash(inpf)

            if os.path.exists(t.name):
                os.remove(t.name)

    def _make_array(self, shape, dtype):
        return np.ndarray(shape, dtype)

    def _create_tmpfile(self):
        _, f_obj = tempfile.mkstemp()

        return open(f_obj, 'wb')

    def _create_new(self):
        _float = tables.Float32Atom()
        _char = tables.UInt8Atom()
        _int = tables.UInt64Atom()

        f = self._fobj
        shape = (0, self.dim)
        f.create_earray(f.root, 'vectors', _float, shape)
        f.create_earray(f.root, 'words', _char, (0,))
        f.create_earray(f.root, 'words_index', _int, (0,))
        f.create_earray(f.root, 'occurrences', _int, (0,))

        self._tmp_f = self._create_tmpfile()

    def __len__(self):
        '''
        >>> wv = WordVecSpaceFile(DATAFILE_ENV_VAR, DIM, 'r')
        >>> len(wv)
        71291
        '''

        return len(self._fobj.root.vectors)

    def get_word(self, index, raise_exc=False):
        d = self._fobj.root
        try:
            s = d.words_index[index]
        except IndexError:
            if raise_exc:
                raise UnknownIndex(index)
            else:
                return None

        try:
            e = d.words_index[index+1]
        except IndexError:
            e = None

        data = d.words[s:e]

        # FIXME: is there a way to avoid this list comprehension
        # based iteration in python level (so we can inc perf)
        chars = [chr(x) for x in data]
        word = ''.join(chars[:-1])

        return word

    def get(self, index, flags=ALL):
        '''
        When we have vectors loaded into the hdf5 file we can retrive
        word, vector, occurrence together or individually based on index
        >>> wv = WordVecSpaceFile(DATAFILE_ENV_VAR, DIM , 'r')
        >>> from pprint import pprint
        >>> pprint(wv.get(1))
        {'occurrence': 1061396,
         'vector': array([-0.8461,  0.8698,  1.0971, -0.8056,  0.7051], dtype=float32),
         'word': 'the'}
        >>> wv.get(1, wv.VECTOR)
        array([-0.8461,  0.8698,  1.0971, -0.8056,  0.7051], dtype=float32)
        >>> wv.get(1, wv.OCCURRENCE)
        1061396
        >>> wv.get(1, wv.WORD)
        'the'
        '''

        if not isinstance(index, int) or index >= len(self):
            raise IndexError

        vector = word = occurrence = None
        d = self._fobj.root

        if flags & self.VECTOR:
            vector = d.vectors[index]

        if flags & self.WORD:
            word = self.get_word(index)

        if flags & self.OCCURRENCE:
            occurrence = d.occurrences[index]

        d = (('vector', vector), ('word', word), ('occurrence', occurrence))
        d = [(k, v) for k, v in d if v is not None]
        # only one attribute requested
        if len(d) == 1:
            return d[0][-1] # return single value
        else:
            return dict(d)

    def _get_index(self, word):
        d = self._fobj.root
        c_index = self.mph(word)
        w_index = d.cmph_index[c_index].item()

        return w_index

    def get_word_vector(self, word, raise_exc=False):
        '''
        >>> wv = WordVecSpaceFile(DATAFILE_ENV_VAR, DIM , 'r')
        >>> wv.get_word_vector('the')
        array([-0.8461,  0.8698,  1.0971, -0.8056,  0.7051], dtype=float32)
        '''

        index = self.get_word_index(word, raise_exc=raise_exc)
        try:
            vector = self.get(index, self.VECTOR)
        except IndexError:
            vector = self._make_array(dtype=np.float32, shape=(1, self.dim))
            vector.fill(0.0)

        return vector

    def get_word_occurrence(self, word, raise_exc=False):
        '''
        >>> wv = WordVecSpaceFile(DATAFILE_ENV_VAR, DIM , 'r')
        >>> wv.get_word_occurrence('the')
        1061396
        '''

        index = self.get_word_index(word, raise_exc=raise_exc)
        try:
            occurrence = self.get(index, self.OCCURRENCE)
        except IndexError:
            occurrence = None

        return occurrence

    def get_word_index(self, word, raise_exc=False):
        '''
        >>> wv = WordVecSpaceFile(DATAFILE_ENV_VAR, DIM, 'r')
        >>> if (sys.version_info > (3, 0)):
        ...    wv.get_word_index('the')
        1
        '''
        if isinstance(word, int):
            if word < len(self._fobj.root.vectors):
                return word
            else:
                if raise_exc:
                    raise UnknownIndex(word)
                return None

        index = self._get_index(word)
        get_word = self.get(index, self.WORD)
        if word == get_word:
            return index
        else:
            if raise_exc == True:
                raise UnknownWord(word)

    def getmany(self, index, num, type=VECTOR):
        '''
        This function returns vectors, occurrences, words
        based on the type in the range of indices
        >>> wv = WordVecSpaceFile(DATAFILE_ENV_VAR, DIM, 'r')
        >>> wv.getmany(0, 2)
        array([[ 0.0801,  0.0884, -0.0766, -0.0656,  0.0273],
               [-0.8461,  0.8698,  1.0971, -0.8056,  0.7051]], dtype=float32)

        >>> wv.getmany(0, 2, wv.WORD)
        ['</s>', 'the']
        >>> wv.getmany(0, 2, wv.OCCURRENCE)
        array([      0, 1061396], dtype=uint64)
        '''

        if index > len(self):
            raise IndexError

        d = self._fobj.root
        s, e = index, num
        if type == self.VECTOR or type == self.OCCURRENCE:
            return getattr(d, self.VAL_TO_TABLE_NAME[type])[s:e]

        if type == self.WORD:
            s = d.words_index[index]
            try:
                e = d.words_index[num]
            except IndexError:
                e = None

            data = d.words[s:e]
            chars = ''.join([chr(x) for x in data])
            words_fo = filter(None, chars.split('\x00'))

            if not isinstance(words_fo, list):
                words = []
                for word in words_fo:
                    words.append(word)

                return words

            return words_fo

    def add(self, word, vector, occurrence=0):
        d = self._fobj.root
        #word = word.decode('utf8').encode('utf8', 'ignore')
        #We are storing words as stream of bytes, so we are using null character as a seperater b/w the words.
        data = [ord(x) for x in word] + [0]
        index = d.words.nrows

        #Adding data to tables
        d.vectors.append(vector.reshape(1, self.dim))
        d.words_index.append([index])
        d.words.append(data)
        d.occurrences.append([occurrence])

        if (sys.version_info > (3, 0)):
            self._tmp_f.write(bytes((word + '\n').encode('utf-8')))
        else:
            self._tmp_f.write(word + '\n')

    def _create_words_index(self):
        with open(self._tmp_f.name, 'r') as inpf:
            mph = cmph.generate_hash(inpf)

        # Find the range of values given by mph
        # min is 0. we need to find the max index
        max_index = 0
        tmp_obj = open(self._tmp_f.name, 'r')

        for i, word in enumerate(tmp_obj):
            word = word.strip()
            if not word: continue
            max_index = max(mph(word), max_index)

        f = self._fobj
        d = f.create_carray(f.root, 'cmph_index', tables.UInt64Atom(), (max_index+1,))

        #FIXME Make it better avoid reading file more than once
        tmp_obj = open(self._tmp_f.name, 'r')

        for i, word in enumerate(tmp_obj):
            word = word.strip()
            if not word: continue
            index = mph(word)
            d[index] = [i]

        _, tmp_f = tempfile.mkstemp()
        with open(tmp_f, 'w') as outf:
            mph.save(outf)

        data = open(tmp_f, 'rb').read()
        c = f.create_carray(f.root, 'cmph_data', tables.UInt8Atom(), (len(data),))
        if (sys.version_info > (3, 0)):
            c[:] = [x for x in data]
        else:
            c[:] = [ord(x) for x in data]

        if os.path.exists(tmp_f):
            os.remove(tmp_f)

    def close(self):
        self._tmp_f.close()
        if self.mode == 'w':
            self._create_words_index()

        if os.path.exists(self._tmp_f.name):
            os.remove(self._tmp_f.name)

        self._fobj.close()
