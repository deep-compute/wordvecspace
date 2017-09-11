from basescript import BaseScript
from convert import GWVec2WordVecSpace

class WordVecSpaceCommand(BaseScript):
    DESC = 'Word Vector Space command-line tool'

    def convert(self):
        convertor = GWVec2WordVecSpace(
                        self.args.input_dir,
                        self.args.output_dir,
                        self.args.num_vecs_per_shard)
        convertor.start()

    DEFAULT_NUM_VECS_PER_SHARD = 0

    def define_subcommands(self, subcommands):
        super(WordVecSpaceCommand, self).define_subcommands(subcommands)

        convert_cmd = subcommands.add_parser('convert',
            help='Convert data in Google\'s Word2Vec format to WordVecSpace format')
        convert_cmd.set_defaults(func=self.convert)
        convert_cmd.add_argument('input_dir',
            help='Input directory containing Google Word2Vec format files'
                 ' (vocab.txt, w2v_vectors.bin)')
        convert_cmd.add_argument('output_dir',
            help='Output directory where WordVecSpace format files are produced')
        convert_cmd.add_argument('-n', '--num-vecs-per-shard',
            default=self.DEFAULT_NUM_VECS_PER_SHARD, type=int,
            help='Number of vectors per shard. 0 value ensures all vecs in one shard.')

def main():
    WordVecSpaceCommand().start()

if __name__ == '__main__':
    main()
