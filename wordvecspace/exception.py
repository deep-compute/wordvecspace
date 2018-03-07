class BaseException(Exception):
    pass

class UnknownWord(BaseException):
    def __init__(self, word):
        self.word = word

    def __str__(self):
        return '"%s"' % self.word

class UnknownIndex(BaseException):
    def __init__(self, index):
        self.index = index

    def __int__(self):
        return '"%s"' % self.index

class UnknownType(BaseException):
    def __init__(self, _type):
        self._type = _type

    def __str__(self):
        return '"%s"' % self._type
