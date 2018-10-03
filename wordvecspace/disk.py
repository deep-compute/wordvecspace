from .wvspace import WordVecSpace

class WordVecSpaceDisk(WordVecSpace):
    def does_word_exist(self, word: str):
        i = self.wtoi[word]

        if i is not None:
            return True
        else:
            return False
