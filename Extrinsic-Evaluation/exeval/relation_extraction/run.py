from exeval import DSM
from exeval.util import invert_index
from .data import get
import numpy
import itertools
import logging


LABEL_TYPES = [
    "Other",
    "Message-Topic",
    "Product-Producer",
    "Instrument-Agency",
    "Entity-Destination",
    "Cause-Effect",
    "Component-Whole",
    "Entity-Origin",
    "Member-Collection",
    "Content-Container",
]

LABELS = []
LABELS.append(LABEL_TYPES[0])
for l in LABEL_TYPES[1:]:
    LABELS.append('{}(e1,e2)'.format(l))
    LABELS.append('{}(e2,e1)'.format(l))

#Mapping of the labels to integers
LABELMAPPING = invert_index(LABELS)

LOWERMIN = 1
UPPERMAX = 2

def distance_mapping(distance, window):
    if distance < -window:
        return LOWERMIN
    elif distance > window:
        return UPPERMAX
    else:
        return distance + window + 3


def create_tensor(data, dsm, maxlen, window):
    data = list(data)
    N = len(data)
    labels = numpy.zeros(N, dtype=numpy.int32)
    tokens = numpy.zeros((N, maxlen), dtype=numpy.int32)
    relpos1 = numpy.zeros((N, maxlen), dtype=numpy.int32)
    relpos2 = numpy.zeros((N, maxlen), dtype=numpy.int32)

    for batch, (label, pos1, pos2, sentence) in enumerate(data):
        for ix, word in enumerate(sentence):
            tokens[batch, ix] = dsm.get_ix(word)
            relpos1[batch, ix] = distance_mapping(ix - pos1, window)
            relpos2[batch, ix] = distance_mapping(ix - pos2, window)

        labels[batch] = LABELMAPPING[label]

    return labels, tokens, relpos1, relpos2




def mk_parser(parser):
    parser.add_argument('--window', type=int, default=30, help='window size for relative positions')
    parser.set_defaults(go = run)

def run(args):
    batch_size = 16
    nb_filter = 100
    filter_length = 3
    hidden_dims = 100
    nb_epoch = 10
    position_dims = 50

    words = set()
    maxlen = 0
    for (_, _, _, sentence) in itertools.chain(get('train'), get('test')):
        maxlen = max(maxlen, len(sentence))
        words.update(sentence)

    logging.info('maximum length: {}'.format(maxlen))
    logging.info('vocabulary size: {}'.format(len(words)))
    logging.info('reading embedding')
    dsm = DSM.read(args.vector_path, words)
    logging.info("Embeddings shape: {}".format(dsm.shape))

    logging.info("reading train data")
    train = create_tensor(get('train'), dsm, maxlen, args.window)
    logging.info("reading test data")
    test = create_tensor(get('test'), dsm, maxlen, args.window)

    max_position = args.window * 2 + 4 #max(np.max(positionTrain1), np.max(positionTrain2))+1
    n_out = len(LABELS)

    import keras
    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate
    from keras.layers import Embedding
    from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
    from keras.regularizers import Regularizer
    from keras.preprocessing import sequence

    words_input = Input(shape=(maxlen,), dtype='int32', name='words_input')
    words = dsm.to_embedding_layer()(words_input)#Embedding(dsm.shape[0], dsm.shape[1], weights=[dsm.m], trainable=False)(words_input)

    distance1_input = Input(shape=(maxlen,), dtype='int32', name='distance1_input')
    distance1 = Embedding(max_position, position_dims)(distance1_input)

    distance2_input = Input(shape=(maxlen,), dtype='int32', name='distance2_input')
    distance2 = Embedding(max_position, position_dims)(distance2_input)


    output = concatenate([words, distance1, distance2], -1)


    output = Convolution1D(filters=nb_filter,
                            kernel_size=filter_length,
                            padding='same',
                            activation='tanh',
                            strides=1)(output)

    # we use standard max over time pooling
    output = GlobalMaxPooling1D()(output)

    output = Dropout(0.25)(output)
    output = Dense(n_out, activation='softmax')(output)

    #create the model
    model = Model(inputs=[words_input, distance1_input, distance2_input], outputs=[output])
    model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
    model.summary(print_fn=logging.info)

    train_x = [train[1], train[2], train[3]]
    train_y = train[0]

    test_x = [test[1], test[2], test[3]]
    test_y = test[0]

    logging.info("Start training")
    hist = model.fit(train_x, train_y, batch_size=batch_size, shuffle=True, verbose=0, epochs=nb_epoch)

    test_loss, test_accuracy = model.evaluate(test_x, test_y, verbose=False)
    #print("Train loss: {:.4f}, Train accuracy: {:.4f}".format(hist.history['loss'][-1], hist.history['acc'][-1]))

    return {
        'loss': test_loss,
        'accuracy': test_accuracy,
    }
