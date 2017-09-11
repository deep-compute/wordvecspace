import os
import pprint
import struct

class GWVecBinReader(object):
    '''
    Abstraction that handles the reading of binary vector data
    from the Google Word2vec's vector bin file (usually named
    vectors.bin)
    '''

    FLOAT_SIZE = 4

    def __init__(self, w2v_bin_file):
        self.w2v_bin_file = w2v_bin_file

        first_line = self.w2v_bin_file.readline().strip()
        self.nvecs, self.dim = [int(k) for k in first_line.split()]

        self._vec_nbytes = self.dim * self.FLOAT_SIZE

    def get_vectors(self):
        f = self.w2v_bin_file

        for i in xrange(self.nvecs):
            token = []

            while 1:
                ch = f.read(1)
                if ch == ' ': break
                token.append(ch)

            token = ''.join(token)
            vec = f.read(self._vec_nbytes)

            f.read(1) # read and discard newline

            yield token, vec

class WordVecSpaceBinWriter(object):
    '''
    Abstraction that handles the writing of vector data
    into WordVecSpace binary format (numpy format).
    '''

    def __init__(self, out_vecs, dim, nvecs):
        self.out_vecs = out_vecs

        header = self._compute_header(dim, nvecs)
        self.out_vecs.write(header)

    def _compute_header(self, dim, nvecs):
        # https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html
        # <93>NUMPY^A^@F^@{'descr': '<f4', 'fortran_order': False, 'shape': (18000000000,), }
        # header = MAGIC_STRING + MAJOR_V + MINOR_V + HEADER_DATA_LENGTH + HEADER_DATA + Spaces + '\n'

        MAGIC_STRING = '\x93NUMPY'
        MAJOR_V = '\x01'
        MINOR_V = '\x00'

        HEADER_DATA = {'descr': '<f4', 'fortran_order': False, 'shape': (nvecs, dim)}
        HEADER_DATA = pprint.pformat(HEADER_DATA)

        total_length = len(MAGIC_STRING) + len(MAJOR_V) + len(MINOR_V) + len(HEADER_DATA) + 2 + len('\n') # 2 is len(HEADER_DATA_LENGTH)

        spaces = ' ' * (16 - (total_length % 16))

        HEADER_DATA = HEADER_DATA + spaces

        HEADER_DATA_LENGTH = struct.pack('h', len(HEADER_DATA) + len('\n'))

        header = '%(MAGIC_STRING)s%(MAJOR_V)s%(MINOR_V)s%(HEADER_DATA_LENGTH)s%(HEADER_DATA)s\n' % locals()

        return header

    def write_vector(self, vec):
        self.out_vecs.write(vec)

class GWVec2WordVecSpace(object):
    '''
    Abstraction that helps in converting word vector space data
    (vectors and vocabulary) from Google Word2Vec format to
    WordVecSpace format.

    vectors.bin => vectors.npy (google word2vec to wordvecspace/numpy format)
    vocab.txt => vocab.txt (no change in format)

    It also supports horizontally paritioning the vector space
    into many shards.
    '''

    def __init__(self, in_dir, out_dir, vecs_per_shard=0):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.vecs_per_shard = vecs_per_shard

    def _create_shard(self, index, shard_count):
        shard_dir = os.path.join(self.out_dir, 'shard_%d' % shard_count)
        if not os.path.exists(shard_dir):
            os.makedirs(shard_dir)

        remaining_vecs = self.nvecs - index

        if remaining_vecs > self.vecs_per_shard:
            shard_vecs = self.vecs_per_shard

        else:
            shard_vecs = remaining_vecs

        return shard_dir, shard_vecs

    def start(self):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        w2v_bin_file = open(os.path.join(self.in_dir, 'vectors.bin'))
        w2v_vocab_file = open(os.path.join(self.in_dir, 'vocab.txt'))

        reader = GWVecBinReader(w2v_bin_file)
        self.dim = reader.dim
        self.nvecs = reader.nvecs

        if not self.vecs_per_shard:
            self.vecs_per_shard = self.nvecs

        shard_count = -1
        for index, (token, vec) in enumerate(reader.get_vectors()):
            if index % self.vecs_per_shard == 0:
                shard_count += 1
                shard_dir, shard_vecs  = self._create_shard(index, shard_count)

                out_vecs = open(os.path.join(shard_dir, 'vectors.npy'), 'wb')
                out_vocab = open(os.path.join(shard_dir, 'vocab.txt'), 'w')

                writer = WordVecSpaceBinWriter(out_vecs, self.dim, shard_vecs)

            writer.write_vector(vec)
            out_vocab.write(w2v_vocab_file.next())
