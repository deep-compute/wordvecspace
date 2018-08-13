import code

from basescript import BaseScript

from .convert import GW2VectoWordVecSpaceFile
from .mem import WordVecSpaceMem
from .disk import WordVecSpaceDisk
from .annoy import WordVecSpaceAnnoy
from .exception import UnknownType

class WordVecSpaceCommand(BaseScript):
    DESC = 'Word Vector Space command-line tool'

    DEFAULT_VECS_PER_SHARD = 0
    DEFAULT_SHARD_NAME = 'shard'
    DEFAULT_FULL_NAME = 'full'

    N_TREES = 1
    METRIC = 'angular'
    DEFAULT_PORT = 8900
    EXTRA_ARGS = None

    def convert(self):
        # FIXME: track issue to send logger
        convertor = GW2VectoWordVecSpaceFile(
                        self.args.input_dir,
                        self.args.output_dir,
                        self.args.vecs_per_shard,
                        self.args.shard_name,
                        self.args.full_name
                        )
        convertor.start()

    def _interact_console(self, interact, dim, _type):
        print("%s console (vectors=%s dims=%s)" % (_type, interact.nvecs, dim))

        namespace = {}
        namespace['wv'] = interact

        code.interact("", local=namespace)

    def _get_extra_args(self):
        eargs = self.args.eargs
        if eargs:
            eargs = dict(a.split('=', 1) for a in eargs.split(':'))

            if eargs.get('n_trees'):
                eargs['n_trees'] = int(eargs['n_trees'])

            return eargs

        return {}

    def interact(self):
        if self.args.type == 'mem':
            interact = WordVecSpaceMem(self.args.input_dir, self.args.metric)

            self._interact_console(interact, interact.dim, 'WordVecSpaceMem')

        elif self.args.type == 'annoy':
            eargs = self._get_extra_args()
            interact = WordVecSpaceAnnoy(self.args.input_dir, metric=self.args.metric, **eargs)

            self._interact_console(interact, interact.dim, 'WordVecSpaceAnnoy')

        elif self.args.type == 'disk':
            interact = WordVecSpaceDisk(self.args.input_dir, self.args.metric)

            self._interact_console(interact, interact.dim, 'WordVecSpaceDisk')

        else:
            raise UnknownType(self.args.type)

    def runserver(self):
    # Installing service feature is optional.
    # Service feature depends on kwikapi.

    # We imported here because it should not distrub
    # interact fuctionality even though kwikapi is not installed.

        from .server import WordVecSpaceServer

        if self.args.type == 'mem':
            server = WordVecSpaceServer(self.args.type,
                                        self.args.input_dir,
                                        metric=self.args.metric,
                                        port=self.args.port)
            server.start()

        elif self.args.type == 'annoy':
            eargs = self._get_extra_args()
            server = WordVecSpaceServer(self.args.type,
                                        self.args.input_dir,
                                        metric=self.args.metric,
                                        port=self.args.port,
                                        **eargs)
            server.start()

        elif self.args.type == 'disk':
            server = WordVecSpaceServer(self.args.type,
                                        self.args.input_dir,
                                        metric=self.args.metric,
                                        port=self.args.port)
            server.start()

        else:
            raise UnknownType(self.args.type)

    def define_subcommands(self, subcommands):
        super(WordVecSpaceCommand, self).define_subcommands(subcommands)

        convert_cmd = subcommands.add_parser('convert',
                help='Convert data in Google\'s Word2Vec format to WordVecSpace format')
        convert_cmd.set_defaults(func=self.convert)
        convert_cmd.add_argument('input_dir',
                help='Input directory containing Google Word2Vec format files'
                    '(vocab.txt, vectors.bin)')
        convert_cmd.add_argument('output_dir',
                help='Output directory where WordVecSpace format files are produced')
        convert_cmd.add_argument('-v', '--vecs_per_shard',
                default=self.DEFAULT_VECS_PER_SHARD, type=int,
                help='Number of vectors per each shard while splitting the full vector space')
        convert_cmd.add_argument('-s', '--shard_name',
                default=self.DEFAULT_SHARD_NAME, type=str,
                help='Shard name of splitting vector spaces')
        convert_cmd.add_argument('-f', '--full_name',
                default=self.DEFAULT_FULL_NAME, type=str,
                help='Full name to map the tokens')

        interact_cmd = subcommands.add_parser('interact',
                help='WordVecSpace Console')
        interact_cmd.set_defaults(func=self.interact)
        interact_cmd.add_argument('type',
                help='wordvecspace feature mem or annoy or disk')
        interact_cmd.add_argument('input_dir',
                help='wordvecspace input dir')
        interact_cmd.add_argument('-m', '--metric',
                default=self.METRIC, type=str,
                help='wordvecspace metric for type of distance calculation')
        interact_cmd.add_argument('-e', '--eargs',
                default=self.EXTRA_ARGS, type=str,
                help='wordvecspace extra arguments (n_trees and index_fpath) for annoy.\
                        Eg: --eargs n_trees=1:index_fpath=/tmp\
                        (This is considered only when the type is annoy)')

        runserver_cmd = subcommands.add_parser('runserver',
                help='WordVecSpace Service')
        runserver_cmd.set_defaults(func=self.runserver)
        runserver_cmd.add_argument('type',
                help='wordvecspace feature mem or annoy or disk')
        runserver_cmd.add_argument('input_dir',
                help='wordvecspace input directory')
        runserver_cmd.add_argument('-m', '--metric',
                default=self.METRIC, type=str,
                help='wordvecspace metric for type of distance calculation')
        runserver_cmd.add_argument('-p', '--port',
                default=self.DEFAULT_PORT, type=int,
                help='port is to run wordvecspace in that port.')
        runserver_cmd.add_argument('-e', '--eargs',
                default=self.EXTRA_ARGS, type=str,
                help='wordvecspace extra arguments (n_trees and index_fpath) for annoy.\
                        Eg: --eargs n_trees=1:index_fpath=/tmp\
                        (This is considered only when the type is annoy)')

def main():
    WordVecSpaceCommand().start()

if __name__ == '__main__':
    main()
