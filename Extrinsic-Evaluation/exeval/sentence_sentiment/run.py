"""
The file preprocesses the data/train.txt, data/dev.txt and data/test.txt from sentiment classification task (English)
"""
import os
from .data import get
from exeval import DSM
import numpy
import logging

def createMatrices(sentences, dsm):
    xMatrix = []
    unknownWordCount = 0
    wordCount = 0

    for sentence in sentences:
        sentenceWordIdx = []
        for word in sentence:
            wordCount += 1
            sentenceWordIdx.append(dsm.get_ix(word))
            if word not in dsm:
                unknownWordCount += 1

        xMatrix.append(sentenceWordIdx)


    logging.info("Unknown tokens: {:.2f}%".format((unknownWordCount/wordCount)*100))
    return xMatrix


def mk_parser(parser):
    parser.add_argument('--preserve_case', action='store_true', help='if set preserves case, otherwise lowercase')
    parser.set_defaults(go = run)


def run(args):
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    #      Start of the preprocessing
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #

    logging.info("Loading train, dev, and test set")
    train_x, train_y = get('train', lower = not args.preserve_case)
    dev_x, dev_y = get('dev', lower = not args.preserve_case)
    test_x, test_y = get('test', lower = not args.preserve_case)


    # :: Compute which words are needed for the train/dev/test set ::
    words = set()
    max_len = 0;
    for sentences in [train_x, dev_x, test_x]:
        for sentence in sentences:
            max_len = max(len(sentence), max_len)
            for token in sentence:
                words.add(token)



    # :: Load the pre-trained embeddings file ::
    logging.info('Loading embeddings')
    dsm = DSM.read(args.vector_path, restrict=words)

    logging.info("Embeddings shape: {}".format(dsm.shape))
    logging.info("Len words: {}".format(len(words)))



    # :: Create matrices ::
    train_x = createMatrices(train_x, dsm)
    dev_x = createMatrices(dev_x, dsm)
    test_x = createMatrices(test_x, dsm)


    logging.info("Longest sentence: {}".format(max_len))

    import keras
    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate
    from keras.layers import Embedding
    from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
    from keras.regularizers import Regularizer
    from keras.preprocessing import sequence

    train_y = numpy.array(train_y)
    dev_y = numpy.array(dev_y)
    test_y = numpy.array(test_y)

    train_x = sequence.pad_sequences(train_x, maxlen=max_len)
    dev_x = sequence.pad_sequences(dev_x, maxlen=max_len)
    test_x = sequence.pad_sequences(test_x, maxlen=max_len)

    logging.info('train shape: {}'.format(train_x.shape))
    logging.info('dev shape:   {}'.format(dev_x.shape))
    logging.info('test shape:  {}'.format(test_x.shape))


    # :: MAKE MODEL ::
    # set parameters:
    batch_size = 64

    nb_filter = 50
    filter_lengths = [1,2,3]
    hidden_dims = 100
    nb_epoch = 25



    words_input = Input(shape=(max_len,), dtype='int32', name='words_input')

    #word embedding layer
    wordsEmbeddingLayer = Embedding(
        dsm.shape[0],
        dsm.shape[1],
        weights=[dsm.m],
        trainable=False)

    words = wordsEmbeddingLayer(words_input)

    #add a variable number of convolutions
    words_convolutions = []
    for filter_length in filter_lengths:
        words_conv = Convolution1D(filters=nb_filter,
                                kernel_size=filter_length,
                                padding='same',
                                activation='relu',
                                strides=1)(words)

        words_conv = GlobalMaxPooling1D()(words_conv)

        words_convolutions.append(words_conv)

    output = concatenate(words_convolutions)



    # vanilla hidden layer together with dropout layer:
    output = Dense(hidden_dims, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(output)
    output = Dropout(0.25)(output)


    #project onto a single unit output layer, and squash it with a sigmoid:
    output = Dense(1, activation='sigmoid',  kernel_regularizer=keras.regularizers.l2(0.01))(output)

    model = Model(inputs=[words_input], outputs=[output])

    model.summary(print_fn=logging.info)


    dev_acc = []
    test_acc = []

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=batch_size, shuffle=True, epochs=nb_epoch, verbose=0, validation_data=(dev_x, dev_y))

    #compute the loss and the accuracy
    dev_loss, dev_accuracy = model.evaluate(dev_x, dev_y, verbose=False)
    test_loss, test_accuracy = model.evaluate(test_x, test_y, verbose=False)


    #print("Dev-Accuracy: %.2f" % (dev_accuracy*100))
    #print("Test-Accuracy: %.2f)" % (test_accuracy*100))

    return {
        'loss': test_loss,
        'accuracy': test_accuracy
    }
