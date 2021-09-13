from exeval import DSM
from .data import get
import itertools
import logging
import numpy

def to_ixs(sentence, dsm):
    [dsm.get_ix(word) for word in sentence]

def mk_parser(parser):
    parser.set_defaults(go=run)


def create_tensor(data, dsm, maxlen):
    data = list(data)
    N = len(data)
    labels = numpy.zeros(N, dtype=numpy.int32)
    tokens = numpy.zeros((N, maxlen), dtype=numpy.int32)

    for batch, (sentence, label) in enumerate(data):
        for ix, word in zip(range(maxlen), sentence):
            tokens[batch, ix] = dsm.get_ix(word)
        labels[batch] = label

    return tokens, labels

def run(args):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Embedding
    from keras.layers import LSTM

    maxlen = 80  # cut texts after this number of words (among top max_features most common words)
    batch_size = 128

    logging.info('Loading train & test data')
    train = list(get('train'))
    test = list(get('test'))

    logging.info('Constructing vocabulary')
    words = set()
    for (sentence, _) in itertools.chain(train, test):
        words.update(sentence)

    logging.info('Loading DSM')
    dsm = DSM.read(args.vector_path, restrict=words)

    logging.info('Embedding-data vocab ratio: {}'.format(len(dsm) / len(words)))

    logging.info('Transforming data to ix')
    train_x, train_y = create_tensor(train, dsm, maxlen)
    test_x, test_y = create_tensor(test, dsm, maxlen)

    logging.info('Building model')

    model = Sequential()
    model.add(Embedding(dsm.shape[0], dsm.shape[1], weights=[dsm.m],trainable=False))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary(print_fn=logging.info)
    model.fit(train_x, train_y,
              batch_size=batch_size,
              shuffle=True,
              verbose=0,
              epochs=20)


    loss, acc = model.evaluate(test_x, test_y,
                               verbose=False,
                               batch_size=batch_size)

    return {
        'loss': loss,
        'accuracy': acc
    }
