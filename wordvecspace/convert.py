import os
import json
from datetime import datetime

import numpy as np
from diskdict import DiskDict

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
        # FIXME: don't hardcode growby value here
        self.outdir = outdir
        self.out = WordVecSpaceFile(self.outdir, dim, growby=100000, mode="w")

    def write(self, token, occur, vec, mag):
        self.out.add(token, occur, vec, mag)

    def close(self):
        self.out.close()

class GW2VectoWordVecSpaceFile(object):
    '''
    Abstraction that helps in converting word vector space data
    (vectors and vocabulary) from Google Word2Vec format to
    WordVecSpaceFile format.
    '''

    def __init__(self, in_dir, outdir,
            nvecs_per_shard=0, shard_name='shard', full_name='full'):
        self.in_dir = in_dir
        self.outdir = outdir
        self.nvecs_per_shard = nvecs_per_shard
        self.shard_name = shard_name
        self.full_file = None

        self.do_sharding = bool(self.nvecs_per_shard)
        if self.do_sharding:
            self.full_fpath = os.path.join(self.outdir, full_name)
            os.makedirs(self.full_fpath)
            map_fpath = os.path.join(self.full_fpath, 'wordtoindex')
            self.full_file = DiskDict(map_fpath)

    def _iter_vecs(self, vfile, vocabfile):
        for token, vec in vfile.iter_vectors():
            vec = np.fromstring(vec, dtype='float32')
            mag = np.linalg.norm(vec)
            vec = vec / mag
            occur = int(vocabfile.readline().split(' ')[1])
            yield vec, token, mag, occur

    def _build_writer(self, vidx, dim):
        if self.do_sharding:
            shard_num = int(vidx / self.nvecs_per_shard)
            shard_name = '{}{}'.format(self.shard_name, shard_num)
            fpath = os.path.join(self.outdir, shard_name)
            return GWVecBinWriter(fpath, dim)
        else:
            return GWVecBinWriter(self.outdir, dim)

    def _create_manifest(self, out_fpath, nvecs, dim, N, t_occur, in_fpath, m_info={}):
        mfc = dict(num_shards=N, num_vectors=nvecs,
                dimension=dim, num_words=t_occur,
                dt_creation=datetime.utcnow().isoformat(),
                input_path=in_fpath, manifest_info=m_info)

        fp = open(os.path.join(out_fpath, 'manifest.json'), 'w')
        fp.write(json.dumps(mfc))
        fp.close()

    def _find_manifest_info(self, fpath):
        m_file = os.path.join(fpath, 'manifest.json')
        c = {}
        if os.path.isfile(m_file):
            fp = open(m_file, 'r')
            c = json.loads(fp.read())
        return c

    def start(self):
        inp_vec_f = open(os.path.join(self.in_dir, 'vectors.bin'), 'rb')
        inp_vecs = GWVecBinReader(inp_vec_f)
        dim = inp_vecs.dim
        nvecs = inp_vecs.nvecs

        vocab_file = open(os.path.join(self.in_dir, 'vocab.txt'), 'r', encoding="ISO-8859-1")
        m_info = self._find_manifest_info(self.in_dir)

        w = None
        vecs = self._iter_vecs(inp_vecs, vocab_file)
        N = self.nvecs_per_shard
        num_shards = int(nvecs / N) + 1

        t_occur = 0
        count = -1
        for index, (vec, token, mag, occur) in enumerate(vecs):
            if self.do_sharding and index % N == 0:
                if w:
                    count += 1
                    t_occur += s_occur
                    self._create_manifest(w.outdir, (index-count*N)+1, dim, N, s_occur,
                            self.in_dir, m_info)
                    w.close()
                    w = None

            if not w:
                s_occur = 0
                w = self._build_writer(index, dim)

            if self.do_sharding:
                self.full_file[token] = index

            w.write(token, occur, vec, mag)

            s_occur += occur

        if w:
            w.close()
            count += 1
            t_occur += s_occur
            self._create_manifest(w.outdir, (index-count*N)+1, dim, num_shards, s_occur,
                    self.in_dir, m_info)

        if self.do_sharding:
            self._create_manifest(self.full_fpath, nvecs, dim, num_shards, t_occur,
                    self.in_dir, m_info)
