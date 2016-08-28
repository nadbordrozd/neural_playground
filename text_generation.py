'''Based on Keras text generation example
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
import os

import numpy as np
import sys
from unidecode import unidecode
from utils import save_model, logger

# we limit ourselves to the following chars.
# Uppercase letters will be represented by prefixing them with a U
chars = '\n !"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~U'
charset = set(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


def fix_char(c):
    if c.isupper():
        return 'U' + c.lower()
    elif c in charset:
        return c
    elif c == '\t':
        return '    '
    else:
        return ''


def encode(text):
    return ''.join(fix_char(c) for c in unidecode(text))


def decode(chars):
    upper = False
    for c in chars:
        if c == 'U':
            upper = True
        elif upper:
            upper = False
            yield c.upper()
        else:
            yield c


def make_lstm_trainset(path, seqlen=40, step=3, batch_size=1024):
    while True:
        with open(path) as f:
            text = f.read().decode("utf-8")

        # limit the charset, encode uppercase etc
        text = encode(text)
        # yield seed
        yield text[:seqlen]
        logger.info('corpus length:', len(text))

        # cut the text in semi-redundant sequences of maxlen characters
        batch_start = 0
        while batch_start < len(text) - seqlen:
            # add sentences that fall on the boundary between batches
            sentences = []
            next_chars = []
            for i in range(max(0, batch_start - seqlen), min(batch_start + batch_size, len(text) - maxlen), step):
                sentences.append(text[i: i + seqlen])
                next_chars.append(text[i + seqlen])

            logger.info('nb sequences: %s. Vectorization' % len(sentences))

            X = np.zeros((len(sentences), seqlen, len(chars)), dtype=np.bool)
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
    # this is stupid but np.random.multinomial throws an error if the probabilities
    # sum to > 1 - which they do due to finite precision
    while sum(a) > 1:
        a /= 1.000001
    return np.argmax(np.random.multinomial(1, a, 1))


def generate(model, seed, diversity):
    _, maxlen, _ = model.input_shape
    assert len(seed) >= maxlen
    sentence = seed[len(seed)-maxlen: len(seed)]
    while True:
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        yield next_char
        sentence = sentence[1:] + next_char


def generate_and_print(model, seed, diversity, n):
    generator = decode(generate(model, seed, diversity))
    sys.stdout.write(''.join(decode(seed)))
    sys.stdout.write('\ngenerated:\n')

    full_text = []
    for _ in range(n):
        next_char = generator.next()
        sys.stdout.write(next_char.encode("utf-8"))
        sys.stdout.flush()
        full_text.append(next_char)

    return full_text


def train_lstm(model, input_path, save_dir, iters=1000, save_every=5):
    _, seqlen, _ = model.input_shape
    gen = make_lstm_trainset(input_path, seqlen=seqlen)
    seed = gen.next()
    for epoch in range(1, iters + 1):
        model.fit_generator(gen, samples_per_epoch=10**6, nb_epoch=1)
        if epoch % save_every == 0:
            save_path = os.path.join(save_dir, ('epoch_%s' % ('%s' % epoch).zfill(5)))
            logger.info("done fitting epoch %s  Now saving mode to %s" % (epoch, save_path))
            save_model(model, save_path)
            logger.info("saved model, now generating a sample")
        generate_and_print(model, seed, 0.5, 1000)
