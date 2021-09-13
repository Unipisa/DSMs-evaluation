from .data import get, LABELS

import tempfile
import numpy as np
import logging
np.random.seed(1337)  # for reproducibility

'''
Adapted from https://github.com/Smerity/keras_snli repo
300D Model - Train / Test (epochs)
'''

from exeval import DSM, UNK, PAD


def mk_parser(parser):
    parser.add_argument('--preserve_case', action='store_true', help='preserve case during preprocessing, default: lowercase')
    parser.set_defaults(go = run)


def run(args):
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.layers import concatenate, recurrent, Dense, Input, Dropout, TimeDistributed
    from keras.layers.embeddings import Embedding
    from keras.models import Model
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from keras.regularizers import l2

    logging.info('Loading train dev and test sets')
    training = get('train')
    validation = get('dev')
    test = get('test')

    logging.info('Constructing tokenizer')
    tokenizer = Tokenizer(lower=not args.preserve_case, filters='', oov_token=UNK)
    tokenizer.fit_on_texts(training[0] + training[1])

    logging.info('Loading embeddings')
    dsm = DSM.read(args.vector_path, restrict=tokenizer.word_counts)

    logging.info('Constructing model')
    # Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
    VOCAB = len(tokenizer.word_counts) + 2
    RNN = recurrent.LSTM
    RNN = lambda *args, **kwargs: recurrent.LSTM(*args, **kwargs)
    #RNN = recurrent.GRU
    #RNN = lambda *args, **kwargs: Bidirectional(recurrent.GRU(*args, **kwargs))
    # Summation of word embeddings
    #RNN = None
    LAYERS = 1
    USE_PRETRAIN_EMED = True
    TRAIN_EMBED = False
    EMBED_HIDDEN_SIZE = dsm.shape[1]
    SENT_HIDDEN_SIZE = 250
    BATCH_SIZE = 512
    PATIENCE = 4 # 8
    MAX_EPOCHS = 15
    MAX_LEN = 42
    DP = 0.5
    L2 = 4e-6
    ACTIVATION = 'relu'
    OPTIMIZER = 'rmsprop'
    logging.info('RNN / Embed / Sent = {}, {}, {}'.format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE))
    logging.info('GloVe / Trainable Word Embeddings = {}, {}'.format(USE_PRETRAIN_EMED, TRAIN_EMBED))


    def prepare_data(data):
        return (
            pad_sequences(tokenizer.texts_to_sequences(data[0]), maxlen=MAX_LEN),
            pad_sequences(tokenizer.texts_to_sequences(data[1]), maxlen=MAX_LEN),
            data[2]
        )

    training = prepare_data(training)
    validation = prepare_data(validation)
    test = prepare_data(test)

    logging.info('Build model...')
    logging.info('Vocab size: {}'.format(VOCAB))



    embedding_matrix = np.zeros((VOCAB, EMBED_HIDDEN_SIZE))
    nulls = 0
    for word, i in tokenizer.word_index.items():
        if word.lower() in dsm:
            embedding_matrix[i] = dsm[word.lower()]
        else:
            embedding_matrix[i] = dsm[PAD]
            nulls += 1

    logging.info('Total number of null word embeddings: {}'.format(nulls))

    embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=TRAIN_EMBED)

    rnn_kwargs = dict(output_dim=SENT_HIDDEN_SIZE, dropout_W=DP, dropout_U=DP)

    premise = Input(shape=(MAX_LEN,), dtype='int32')
    hypothesis = Input(shape=(MAX_LEN,), dtype='int32')

    prem = embed(premise)
    hypo = embed(hypothesis)

    rnn_prem = RNN(return_sequences=False, **rnn_kwargs)
    rnn_hypo = RNN(return_sequences=False, **rnn_kwargs)
    prem = rnn_prem(prem)
    prem = Dropout(DP)(prem)
    hypo = rnn_hypo(hypo)
    hypo = Dropout(DP)(hypo)


    joint = concatenate([prem, hypo], axis=-1)
    joint = Dense(output_dim=50, activation='tanh', W_regularizer=l2(0.01))(joint)
    pred = Dense(len(LABELS), activation='softmax', W_regularizer=l2(0.01))(joint)

    model = Model(input=[premise, hypothesis], output=pred)
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary(print_fn=logging.info)

    logging.info('Training')
    _, tmpfn = tempfile.mkstemp()
    # Save the best model during validation and bail out of training early if we're not improving
    callbacks = [EarlyStopping(patience=PATIENCE), ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)]
    model.fit([training[0], training[1]], training[2], shuffle=True, batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCHS, verbose=0, validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks)

    # Restore the best found model during validation
    model.load_weights(tmpfn)

    loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE, verbose=False)

    return {
        'loss': loss,
        'accuracy': acc
    }
