import os

import numpy as np

from .fileformat import WordVecSpaceFile

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

    def iter_vectors(self):
        f = self.w2v_bin_file

        for i in range(self.nvecs):
            token = []

            while True:
                try:
                    ch = f.read(1)
                    ch = ch.decode('utf-8')
                except UnicodeDecodeError:
                    ch = ch.decode('unicode-escape')

                if ch == ' ':
                    break
                token.append(ch)

            token = ''.join(token)
            vec = f.read(self._vec_nbytes)

            f.read(1)  # read and discard newline

            yield token, vec


class GWVecBinWriter(object):
    def __init__(self, outdir, dim):
        self.out = WordVecSpaceFile(outdir, dim, mode="w")

    def write(self, token, occur, vec):
        self.out.add(token, occur, vec)

    def close(self):
        self.out.close()

class GW2VectoWordVecSpaceFile(object):
    '''
    Abstraction that helps in converting word vector space data
    (vectors and vocabulary) from Google Word2Vec format to
    WordVecSpaceFile format.
    '''

    def __init__(self, in_dir, outdir):
        self.in_dir = in_dir
        self.outdir = outdir

    def start(self):
        inp_vec_f = open(os.path.join(self.in_dir, 'vectors.bin'), 'rb')
        inp_vecs = GWVecBinReader(inp_vec_f)

        vocab_file = open(os.path.join(self.in_dir, 'vocab.txt'), 'r', encoding="ISO-8859-1")

        wr_vecs = GWVecBinWriter(self.outdir, inp_vecs.dim)

        for index, (token, vec) in enumerate(inp_vecs.iter_vectors()):
            vec = np.fromstring(vec, dtype='float32')
            occur = int(vocab_file.readline().split(' ')[1])

            wr_vecs.write(token, occur, vec)

        wr_vecs.close()
