#!/usr/bin/env python

import doctest
import unittest

from wordvecspace import wordvecspace, server

def suitefn():
    suite = unittest.TestSuite()

    suite.addTests(doctest.DocTestSuite(wordvecspace))
    suite.addTests(doctest.DocTestSuite(server))

    return suite

if __name__ == "__main__":
    doctest.testmod(wordvecspace)
    doctest.testmod(server)
