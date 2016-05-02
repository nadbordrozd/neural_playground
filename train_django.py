from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

from text_generation import make_lstm_trainset, generate

print('Build model...')
chars = '\n !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
maxlen = 40
model = Sequential()
model.add(LSTM(1024, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.3))
model.add(LSTM(1024, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(1024, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.additional_config = dict(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    maxlen=40,
    chars=chars
)


seed = """# -*- coding: utf-8 -*-
#
# Django documentation build configuration file, created by
# sphinx-quickstart on Thu Mar 27 09:06:53 2008."""


gen = make_lstm_trainset('data/all_the_django.py', chars)
for iteration in range(1000):
    model.fit_generator(gen, samples_per_epoch=10**6, nb_epoch=1)
    save_model(model, "data/django/")
    generate(model, seed, 0.5, 500)
#'/home/ubuntu/.keras/datasets/nietzsche.txt'
#'data/all_the_django.py'
