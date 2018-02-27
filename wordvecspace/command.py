import code

from basescript import BaseScript
from .convert import GW2VectoWordVecSpaceFile
from .mem import WordVecSpaceMem
from .annoy import WordVecSpaceAnnoy
from .server import WordVecSpaceServer

class BaseException(Exception):
    pass

class UnknownType(BaseException):
    def __init__(self, _type):
        self._type = _type

    def __str__(self):
        return '"%s"' % self._type

class WordVecSpaceCommand(BaseScript):
    DESC = 'Word Vector Space command-line tool and service'

    def convert(self):
        #FIXME: track issue to send logger
        convertor = GW2VectoWordVecSpaceFile(
                        self.args.input_dir,
                        self.args.output_file
                        )
        convertor.start()

    def _interact_common_code(self, interact, dim, _type):
        print("Total number of vectors and dimensions in wvspace file\
                (%s, %s)" %(interact.nvecs, dim))
        print("\nhelp")
        print("%s" %(dir(interact)))

        namespace = {}
        namespace[_type] = interact

        code.interact(_type + " Console", local=namespace)

    def interact(self):
        wvargs = self.args.wvargs
        wvargs = dict(a.split('=', 1) for a in wvargs.split(':'))
        wvargs['dim'] = int(wvargs['dim'])

        if wvargs.get('n_trees'):
            wvargs['n_trees'] = int(wvargs['n_trees'])

        if self.args.type == 'mem':
            interact = WordVecSpaceMem(**wvargs)

            self._interact_common_code(interact, wvargs['dim'], 'WordVecSpaceMem')

        elif self.args.type == 'annoy':
            interact = WordVecSpaceAnnoy(**wvargs)

            self._interact_common_code(interact, wvargs['dim'], 'WordVecSpaceAnnoy')

        else:
            raise UnknownType(self.args.type)

    def runserver(self):
        wvargs = self.args.wvargs
        wvargs = dict(a.split('=', 1) for a in wvargs.split(':'))
        wvargs['dim'] = int(wvargs['dim'])
        wvargs['_type'] = self.args.type

        if wvargs.get('port'):
            wvargs['port'] = int(wvargs['port'])
        if wvargs.get('n_trees'):
            wvargs['n_trees'] = int(wvargs['n_trees'])

        if self.args.type == 'mem':
            server = WordVecSpaceServer(**wvargs)
            server.start()

        elif self.args.type == 'annoy':
            server = WordVecSpaceServer(**wvargs)
            server.start()

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
        interact_cmd.add_argument('wvargs',
                help='wordvecspace arguments.\
                        for mem <input_file=/home/user:dim=5>\
                        for annoy <input_file=/home/user:dim=5:n_trees=1:metric=angular>\
                        ntrees and metric are optional in annoy')

        runserver_cmd = subcommands.add_parser('runserver',
                help='WordVecSpace Service')
        runserver_cmd.set_defaults(func=self.runserver)
        runserver_cmd.add_argument('type',
                help='wordvecspace feature mem or annoy')
        runserver_cmd.add_argument('wvargs',
                help='wordvecspace arguments.\
                        for mem <input_file=/home/user:dim=5:port=8000>\
                        for annoy <input_file=/home/user:dim=5:n_trees=1:metric=angular:port=8000>\
                        ntrees and metric are optional in annoy\
                        port is optional in both mem and annoy')

def main():
    WordVecSpaceCommand().start()

if __name__ == '__main__':
    main()
