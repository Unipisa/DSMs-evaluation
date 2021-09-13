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
            splits = line.strip().split()
            label = int(splits[0])
            words = splits[1:]
            yield (words, label)



def get(type):
    filename = '{}.txt.gz'.format(type)
    yield from read_file(filename)

