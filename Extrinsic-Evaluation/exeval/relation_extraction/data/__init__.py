from importlib.resources import path
from exeval.util import invert_index
import itertools
import json
import gzip
import numpy





def read_file(filename):
    with path(__name__, filename) as file, gzip.open(file, 'rt') as lines:
        sentences = []
        labels = []

        for line in lines:
            splits = line.strip().split('\t')
            label = LABELMAPPING = splits[0]
            pos0 = int(splits[1])
            pos1 = int(splits[2])
            words = [word.lower() for word in splits[3].split()]
            yield (label, pos0, pos1, words)


def get(type):
    filename = '{}.txt.gz'.format(type)
    yield from read_file(filename)



