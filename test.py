#!/usr/bin/env python

import doctest

from wordvecspace import wordvecspace

suite = doctest.DocTestSuite(wordvecspace)

if __name__ == "__main__":
    doctest.testmod(wordvecspace)
