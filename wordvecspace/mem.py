from .wvspace import WordVecSpace

class WordVecSpaceMem(WordVecSpace):
    METRIC = WordVecSpace.METRIC

    def __init__(self, input_dir, metric=METRIC):
        super().__init__(input_dir, metric)
        self.wtoi = dict(self.wtoi.items())
        self.itow = dict(self.itow.items())
