import code

from basescript import BaseScript

from .convert import GW2VectoWordVecSpaceFile
from .mem import WordVecSpaceMem
from .annoy import WordVecSpaceAnnoy
from .exception import UnknownType

class WordVecSpaceCommand(BaseScript):
    DESC = 'Word Vector Space command-line tool'

    N_TREES = 1
    METRIC = 'angular'
    DEFAULT_PORT = 8900
    EXTRA_ARGS = None

    def convert(self):
        #FIXME: track issue to send logger
        convertor = GW2VectoWordVecSpaceFile(
                        self.args.input_dir,
                        self.args.output_file
                        )
        convertor.start()

    def _interact_console(self, interact, dim, _type):
        print("%s console (vectors=%s dims=%s)" %(_type, interact.nvecs, dim))

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
            interact = WordVecSpaceMem(self.args.input_file, self.args.metric)

            self._interact_console(interact, interact.dim, 'WordVecSpaceMem')

        elif self.args.type == 'annoy':
            eargs = self._get_extra_args()
            interact = WordVecSpaceAnnoy(self.args.input_file, metric=self.args.metric, **eargs)

            self._interact_console(interact, interact.dim, 'WordVecSpaceAnnoy')

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
                    self.args.input_file,
                    metric=self.args.metric,
                    port=self.args.port)
            server.start()

        elif self.args.type == 'annoy':
            eargs = self._get_extra_args()
            server = WordVecSpaceServer(self.args.type,
                    self.args.input_file,
                    metric=self.args.metric,
                    port=self.args.port,
                    **eargs)
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
                 ' (vocab.txt, vectors.bin)')
        convert_cmd.add_argument('output_file',
            help='Output file where WordVecSpace format files are produced')

        interact_cmd = subcommands.add_parser('interact',
                help='WordVecSpace Console')
        interact_cmd.set_defaults(func=self.interact)
        interact_cmd.add_argument('type',
                help='wordvecspace feature mem or annoy')
        interact_cmd.add_argument('input_file',
                help='wordvecspace input file')
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
                help='wordvecspace feature mem or annoy')
        runserver_cmd.add_argument('input_file',
                help='wordvecspace input file')
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
