#!/usr/bin/env python

import doctest
import unittest

from wordvecspace import fileformat
from wordvecspace import mem
from wordvecspace import annoy
from wordvecspace import disk

def suite_test():
    suite = unittest.TestSuite()

    suite.addTests(doctest.DocTestSuite(fileformat))
    suite.addTests(doctest.DocTestSuite(mem))
    suite.addTests(doctest.DocTestSuite(annoy))
    suite.addTests(doctest.DocTestSuite(disk))

    return suite

if __name__ == "__main__":
    doctest.testmod(fileformat)
    doctest.testmod(mem)
    doctest.testmod(annoy)
    doctest.testmod(disk)
