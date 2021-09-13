"""
The file preprocesses the data/train.txt, data/dev.txt and data/test.txt from sentiment classification task (English)

"""

from exeval import DSM
from .data import get
import logging
import itertools
import numpy

def mk_parser(parser):
    parser.set_defaults(go=run)

def mk_BoV(sentences, dsm):
    N = len(sentences)
    D = dsm.shape[1]

    matrix = numpy.zeros((N, D), dtype=numpy.float32)

    for ix, sentence in enumerate(sentences):
        matrix[ix] = dsm.get(sentence).sum(0)

    return matrix




def run(args):
    logging.info('Loading training data')
    train_x, train_y = zip(*get('train'))
    logging.info('Loading testing data')
    test_x, test_y = zip(*get('test'))
    logging.info('Loading dev data')
    dev_x, dev_y = zip(*get('dev'))

    logging.info('Constructing Vocabulary')

    words = set()

    for sentence in itertools.chain(train_x, test_x, dev_x):
        words.update(sentence)

    logging.info('Loading DSM')
    dsm = DSM.read(args.vector_path, restrict=words)

    logging.info('Embedding/Data vocabulary ratio: {:.2f}'.format(len(dsm) / len(words)))


    """
    This implementation is a model for sentence classification.
    Adapted from https://github.com/UKPLab/deeplearning4nlp-tutorial
    """

    import keras
    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate




    train_y = numpy.array(train_y)
    dev_y = numpy.array(dev_y)
    test_y = numpy.array(test_y)

    train_x = mk_BoV(train_x, dsm)
    dev_x = mk_BoV(dev_x, dsm)
    test_x = mk_BoV(test_x, dsm)





    #  :: Create the network ::

    logging.info('Building model')

    # set parameters:
    batch_size = 64
    nb_epoch = 25



    BoV = Input(shape=(dsm.shape[1],), dtype='float32', name='BoV')

    # We project onto a single unit output layer, and squash it with a sigmoid:
    output = Dense(1, activation='sigmoid')(BoV)

    model = Model(inputs=[BoV], outputs=[output])

    dev_acc = []
    test_acc = []

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary(print_fn=logging.info)
    model.fit(train_x, train_y, batch_size=batch_size, shuffle=True, epochs=nb_epoch, verbose=0,validation_data=(dev_x, dev_y))

    #Use Keras to compute the loss and the accuracy
    dev_loss, dev_accuracy = model.evaluate(dev_x, dev_y, verbose=False)
    test_loss, test_accuracy = model.evaluate(test_x, test_y, verbose=False)

    return {
        'loss': test_loss,
        'accuracy': test_accuracy,
    }
