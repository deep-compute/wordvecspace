import os
import tempfile

import tables
import cmph

# $export WVSPACEFILE_DATADIR=/path/to/data/
DATADIR_ENV_VAR = os.environ.get('WVSPACEFILE_DATADIR', '')

class WordVecSpaceFile(object):
    DEFAULT_MODE = 'r'
    VECTOR       = 1 << 0
    WORD         = 1 << 1
    OCCURRENCE   = 1 << 2
    ALL          = VECTOR | WORD | OCCURRENCE

    VAL_TO_TABLE_NAME = {VECTOR: 'vectors', WORD: 'words', OCCURRENCE: 'occurrences'}

    def __init__(self, fpath, dim, mode=DEFAULT_MODE):
        self.mode = mode
        self.dim = dim
        self._fobj = tables.open_file(fpath, self.mode)

        if self.mode == 'w':
            self._create_new()

    def _create_tmpfile(self):
        _, f_obj = tempfile.mkstemp()

        return open(f_obj, 'w')

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
        >>> ws = WordVecSpaceFile(DATADIR_ENV_VAR, 5 , 'r')
        >>> len(ws)
        10
        '''

        return len(self._fobj.root.vectors)

    def _get_word(self, index):
        d = self._fobj.root
        s = d.words_index[index]

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
        >>> ws = WordVecSpaceFile(DATADIR_ENV_VAR, 5 , 'r')
        >>> ws.get(1)
        {'vector': array([ -2.40367788e-36,   4.56501001e-41,  -2.40367788e-36,
                 4.56501001e-41,   4.48415509e-44], dtype=float32), 'word': 'the', 'occurrence': 761757}

        >>> ws.get(1, ws.VECTOR)
        array([ -2.40367788e-36,   4.56501001e-41,  -2.40367788e-36,
                 4.56501001e-41,   4.48415509e-44], dtype=float32)

        >>> ws.get(1, ws.OCCURRENCE)
        761757

        >>> ws.get(1, ws.WORD)
        'the'
        '''

        if index >= len(self):
            raise IndexError

        vector = word = occurrence = None
        d = self._fobj.root

        if flags & self.VECTOR:
            vector = d.vectors[index]

        if flags & self.WORD:
            word = self._get_word(index)

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
        t = self._create_tmpfile()
        cmph_data = [chr(x) for x in d.cmph_data]
        for value in cmph_data:
            t.write(value)

        t.close()

        with open(t.name, 'r') as inpf:
            mph = cmph.load_hash(inpf)

        if os.path.exists(t.name):
            os.remove(t.name)

        c_index = mph(word)
        w_index = d.cmph_index[c_index].item()

        return w_index

    def get_word_vector(self, word):
        '''
        >>> ws = WordVecSpaceFile(DATADIR_ENV_VAR, 5 , 'r')
        >>> ws.get_word_vector('the')
        array([ -2.40367788e-36,   4.56501001e-41,  -2.40367788e-36,
                 4.56501001e-41,   4.48415509e-44], dtype=float32)
        '''

        index = self._get_index(word)
        try:
            vector = self.get(index, self.VECTOR)
        except IndexError:
            vector = "Given word doesn't exists"

        return vector

    def get_word_occurrence(self, word):
        '''
        >>> ws = WordVecSpaceFile(DATADIR_ENV_VAR, 5 , 'r')
        >>> ws.get_word_occurrence('the')
        761757
        '''

        index = self._get_index(word)
        try:
            occurrence = self.get(index, self.OCCURRENCE)
        except IndexError:
            occurrence = "Given word doesn't exists"

        return occurrence

    def get_word_index(self, word):
        '''
        >>> ws = WordVecSpaceFile(DATADIR_ENV_VAR, 5 , 'r')
        >>> ws.get_word_index('the')
        1L
        '''

        index = self._get_index(word)
        get_word = self.get(index, self.WORD)
        if word == get_word:
            return index
        else:
            return "Given word doesn't exists"

    def getmany(self, index, num, type=VECTOR):
        '''
        This function returns vectors, occurrences, words
        based on the type in the range of indices
        >>> ws = WordVecSpaceFile(DATADIR_ENV_VAR, 5 , 'r')
        >>> ws.getmany(0, 2)
	array([[ -2.40367788e-36,   4.56501001e-41,  -2.40367788e-36,
          	  4.56501001e-41,   0.00000000e+00],
       	       [ -2.40367788e-36,   4.56501001e-41,  -2.40367788e-36,
          	  4.56501001e-41,   4.48415509e-44]], dtype=float32)

        >>> ws.getmany(0, 2, ws.WORD)
        ['</s>', 'the']

        >>> ws.getmany(0, 2, ws.OCCURRENCE)
        array([2558858,  761757], dtype=uint64)
        '''

        if index >= len(self):
            raise IndexError

        d = self._fobj.root
        s, e = index, num
        if type == self.VECTOR or type == self.OCCURRENCE:
            return getattr(d, self.VAL_TO_TABLE_NAME[type])[s:e]

        if type == self.WORD:
            s, e = d.words_index[index], d.words_index[num]
            data = d.words[s:e]
            chars = ''.join([chr(x) for x in data])
            words = filter(None, chars.split('\x00'))

            return words

    def add(self, word, vector, occurrence=0):
        d = self._fobj.root
        word = word.decode('utf8').encode('utf8', 'ignore')
        #We are storing words as stream of bytes, so we are using null character as a seperater b/w the words.
        data = [ord(x) for x in word] + [0]
        index = d.words.nrows

        #Adding data to tables
        d.vectors.append(vector.reshape(1, self.dim))
        d.words_index.append([index])
        d.words.append(data)
        d.occurrences.append([occurrence])

        self._tmp_f.write(word + '\n')

    def _create_words_index(self):
        f = self._fobj
        tmp_obj = open(self._tmp_f.name, 'r')
        with open(self._tmp_f.name, 'r') as inpf:
            mph = cmph.generate_hash(inpf)

        max_index = 0
        for i, word in enumerate(tmp_obj):
            word = word.strip()
            if not word: continue
            max_index = max(mph(word), max_index)

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
