from importlib.resources import path
import gzip

def invert_index(xs):
    return {x: i for i, x in enumerate(xs)}


def open_gzipped(base, filename, mode='rt'):
    with path(base, filename) as file:
        yield gzip.open(file, mode)
