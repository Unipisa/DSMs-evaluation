from importlib.resources import path
import itertools
import json
import gzip
import numpy

LABELS = {
    'contradiction': 0,
    'neutral': 1,
    'entailment': 2}


TYPES = ['train', 'test', 'dev']


def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()


def read_file(filename, skip_no_majority=True):
    with path(__name__, filename) as file, gzip.open(file, 'rt') as lines:
        for line in lines:
            data = json.loads(line)
            label = data['gold_label']
            s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
            s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
            if skip_no_majority and label == '-':
                continue
            yield (s1, s2, label)

def get_data(filename, limit=None):
    lefts, rights, labels = zip(*itertools.islice(read_file(filename), limit))
    from keras.utils import np_utils
    Y = numpy.array([LABELS[l] for l in labels])
    Y = np_utils.to_categorical(Y, len(LABELS))
    return lefts, rights, Y


def get(type):
    return get_data('snli_1.0_{}.jsonl.gz'.format(type))
