from importlib.resources import path
import itertools
import json
import gzip
import numpy


def read_file(filename, lower=False):
    with path(__name__, filename) as file, gzip.open(file, 'rt') as lines:
        sentences = []
        labels = []

        for line in lines:
            splits = line.split()
            label = int(splits[0])
            if lower:
                words = [word.lower() for word in splits[1:]]
            else:
                words = splits[1:]

            yield (words, label)

def get(type, lower=False):
    filename = '{}.txt.gz'.format(type)
    words, labels = list(zip(*read_file(filename, lower)))

    return words, labels

