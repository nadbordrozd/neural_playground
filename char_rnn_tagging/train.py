#!/usr/bin/env python
import argparse
import os
from glob import glob
from random import choice
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, TimeDistributed
from keras.models import Sequential
from text_encoding import char2vec, n_chars


def chars_from_files(list_of_files):
    while True:
        filename = choice(list_of_files)
        with open(filename, 'rb') as f:
            chars = f.read()
            for c in chars:
                yield c


def splice_texts(files_a, jump_size_a, files_b, jump_size_b):
    a_chars = chars_from_files(files_a)
    b_chars = chars_from_files(files_b)
    generators = [a_chars, b_chars]

    a_range = range(jump_size_a[0], jump_size_a[1])
    b_range = range(jump_size_b[0], jump_size_b[1])
    ranges = [a_range, b_range]

    source_ind = choice([0, 1])
    while True:
        jump_size = choice(ranges[source_ind])
        gen = generators[source_ind]
        for _ in range(jump_size):
            yield (gen.next(), source_ind)
        source_ind = 1 - source_ind


def generate_batches(files_a, jump_size_a, files_b, jump_size_b, batch_size, sample_len, return_text=False):
    gens = [splice_texts(files_a, jump_size_a, files_b, jump_size_b) for _ in range(batch_size)]
    while True:
        X = []
        y = []
        texts = []
        for g in gens:
            chars = []
            vecs = []
            labels = []
            for _ in range(sample_len):
                c, l = g.next()
                vecs.append(char2vec[c])
                labels.append([l])
                chars.append(c)
            X.append(vecs)
            y.append(labels)
            if return_text:
                texts.append(''.join(chars))

        if return_text:
            yield (np.array(X), np.array(y), texts)
        else:
            yield (np.array(X), np.array(y))


def main(model_path, dir_a, dir_b, min_jump_size_a, max_jump_size_a,
         min_jump_size_b, max_jump_size_b, seq_len, batch_size, rnn_size, lstm_layers, dropout_rate,
         bidirectional, steps_per_epoch, epochs):
        train_a = glob(os.path.join(dir_a, "train/*"))
        train_b = glob(os.path.join(dir_b, "train/*"))

        juma = [min_jump_size_a, max_jump_size_a]
        jumb = [min_jump_size_b, max_jump_size_b]
        batch_shape = (batch_size, seq_len, n_chars)
        from keras.layers import Bidirectional
        model = Sequential()
        for _ in range(lstm_layers):
            if bidirectional:
                model.add(Bidirectional(LSTM(rnn_size, return_sequences=True,
                                             batch_input_shape=batch_shape, stateful=True),
                                        batch_input_shape=batch_shape))
            else:
                model.add(LSTM(rnn_size, return_sequences=True, batch_input_shape=batch_shape,
                               stateful=True))

            model.add(Dropout(dropout_rate))

        model.add(TimeDistributed(Dense(units=1, activation='sigmoid')))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'binary_crossentropy'])

        train_gen = generate_batches(train_a, juma, train_b, jumb, batch_size, seq_len)
        # doesn't work anyway
        # validation_gen = generate_batches(val_a, juma, val_b, jumb, batch_size, seq_len)
        checkpointer = ModelCheckpoint(model_path)
        for _ in range(epochs):
            model.fit_generator(train_gen,
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs,
                                callbacks=[checkpointer])
            model.reset_states()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("train tagger and save trained model")
    parser.add_argument("model_path", help="where to put trained model")
    parser.add_argument("dir_a", help="directory with first source of input files. It should "
                                      "contain 'train' and 'test' subdirectories that contain "
                                      "actual files")
    parser.add_argument("dir_b", help="directory with second source of input files. It should "
                                      "contain 'train' and 'test' subdirectories that contain "
                                      "actual files")
    parser.add_argument("--min_jump_a", type=int, default=20, help="snippets from source A will "
                                                                   "be at least this long")
    parser.add_argument("--max_jump_a", type=int, default=200, help="snippets from source B will "
                                                                    "be at most this long")
    parser.add_argument("--min_jump_b", type=int, default=20, help="snippets from source B will "
                                                                   "be at least this long")
    parser.add_argument("--max_jump_b", type=int, default=200, help="snippets from source B will "
                                                                    "be at most this long")
    parser.add_argument("--sequence_length", type=int, default=100, help="how many characters in "
                                                                         "single sequence")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--rnn_size", type=int, default=128, help="how many LSTM units per layr")
    parser.add_argument("--lstm_layers", type=int, default=3, help="how many LSTM layers")
    parser.add_argument("--dropout_rate", type=int, default=0.2, help="dropout rate for a "
                                                                      "droupout layer inserted "
                                                                      "after every LSTM layer")
    parser.add_argument("--bidirectional", action="store_true",
                        help="Whether to use bidirectional LSTM. If true, inserts a backwards LSTM"
                        " layer after every normal layer.")
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1000)

    args = parser.parse_args()

    main(
        args.model_path,
        args.dir_a,
        args.dir_b,
        args.min_jump_a,
        args.max_jump_a,
        args.min_jump_b,
        args.max_jump_b,
        args.sequence_length,
        args.batch_size,
        args.rnn_size,
        args.lstm_layers,
        args.dropout_rate,
        args.bidirectional,
        args.steps_per_epoch,
        args.epochs)
