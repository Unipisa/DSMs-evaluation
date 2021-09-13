from importlib.resources import path
from exeval.util import invert_index
import itertools
import json
import gzip
import numpy
import nltk


def read_file(filename):
    with path(__name__, filename) as file, gzip.open(file, 'rt') as lines:
        for line in lines:
            yield line.split()


def get(type):
    pos = '{}_pos.txt.gz'.format(type)
    neg = '{}_neg.txt.gz'.format(type)
    yield from itertools.chain(
        ((words, 1) for words in read_file(pos)),
        ((words, 0) for words in read_file(neg))
    )

