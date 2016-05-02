'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
import os

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
from unidecode import unidecode
from utils import save_model

#path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
def make_lstm_trainset(path, chars, maxlen=40, step=3, ascify=True):
    batch_size = 128
    while True:
        with open(path) as f:
            text = f.read().decode("utf-8")

        if ascify:
            text = unidecode(text)
        print('corpus length:', len(text))

        print('total chars:', len(chars))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        # cut the text in semi-redundant sequences of maxlen characters
        batch_start = 0
        while batch_start < len(text) - maxlen:
            # add sentences that fall on the boundary between batches
            sentences = []
            next_chars = []
            for i in range(max(0, batch_start - maxlen), min(batch_start + batch_size, len(text) - maxlen), step):
                sentences.append(text[i: i + maxlen])
                next_chars.append(text[i + maxlen])
            print('nb sequences:', len(sentences))

            print('Vectorization...')
            X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
            y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
            for i, sentence in enumerate(sentences):
                for t, char in enumerate(sentence):
                    X[i, t, char_indices[char]] = 1
                y[i, char_indices[next_chars[i]]] = 1
            yield X, y
            batch_start += batch_size



def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def generate(model, seed, diversity, gen_len, print_shit=True):
    maxlen = model.additional_config['maxlen']
    chars = model.additional_config['chars']
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    assert len(seed) >= maxlen
    generated = seed
    sentence = seed[len(seed)-maxlen: len(seed)]
    print('----- Generating with seed: "' + seed + '"')
    sys.stdout.write(generated)

    for i in range(gen_len):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
        if print_shit:
            sys.stdout.write(next_char.encode("utf-8"))
            sys.stdout.flush()
    if print_shit:
        print()
    return generated

def train_lstm(model, input_path, save_dir, iters=100, save_every=5, ascify=True):
    chars = model.additional_config['chars']
    if type(chars) != unicode:
        chars = chars.decode("utf-8")
        model.additional_config['chars'] = chars
    maxlen = model.additional_config['maxlen']
    with open(input_path, "rb") as f:
        seed = f.read()[:maxlen]
    #X, y = make_lstm_trainset(input_path, chars, maxlen, ascify)
    # train the model, output generated text after each iteration
    for iteration in range(1, iters + 1):
        

        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit_generator(make_lstm_trainset(input_path, chars, maxlen, ascify), samples_per_epoch=10**7, nb_epoch=1)
        if iteration % save_every == 0:
            save_model(model, os.path.join(save_dir, "epoch_%s" % iteration))
        #start_index = random.randint(0, len(text) - maxlen - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)
            print('----- Generating with seed: "' + seed + '"')
            generate(model, seed, diversity, 100)

if __name__ == "__main__":
    # build the model: 2 stacked LSTM
    print('Build model...')
    chars = '\n! #"%$\'&)(+*-,/.1032547698;:=<?>@[]\\_^a`cbedgfihkjmlonqpsrutwvyx{z}|~'
    maxlen = 40
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    model.additional_config = dict(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        maxlen=40,
        chars=chars
    )
    train_lstm(model, "data/some_of_the_java.java", "models/test", chars)