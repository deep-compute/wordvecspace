#!/usr/bin/env python

import doctest
import unittest

from wordvecspace import wordvecspace, fileformat

def suite_maker():
    suite = unittest.TestSuite()
    suite.addTests(doctest.DocTestSuite(wordvecspace))
    suite.addTests(doctest.DocTestSuite(fileformat))
    return suite

if __name__ == "__main__":
    doctest.testmod(wordvecspace)
    doctest.testmod(fileformat)
