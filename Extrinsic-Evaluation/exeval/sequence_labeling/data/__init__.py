from importlib.resources import path
import gzip
import itertools
from exeval.util import open_gzipped

TASK2POS = {
    'pos': 1,
    'chunk': 2,
    'ner': -1
}


def read_file(task, filename, words, tags):
    tag_position = TASK2POS[task]

    def read_sentence(lines):
        ws = []
        ts = []
        for line in lines:
            wt = line.strip().split()
            word = wt[0].lower()
            tag = wt[tag_position]
            words.add(word)
            tags.add(tag)
            ws.append(word)
            ts.append(tag)

        return ws, ts
    with path(__name__, filename) as fn, gzip.open(fn, 'rt') as lines:
        for empty, sentence in itertools.groupby(lines, key=lambda line: line.strip() == ""):
            if not empty:
                yield read_sentence(sentence)

def load(task):
    out = {}
    words = set()
    tags = set()
    for type in ["train", "test", "valid"] :
        filename = '{}.txt.gz'.format(type)
        out[type] = list(read_file(task, filename, words, tags))

    return out["train"], out["valid"], out["test"], words, tags
