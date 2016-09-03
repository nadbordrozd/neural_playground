from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

from text_generation import chars, train_lstm


maxlen = 80
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


train_lstm(model=model,
           input_path='data/flat_earth/train.txt',
           validation_path='data/flat_earth/test.txt',
           from_checkpoint=True,
           save_dir='models/flat_earth/',
           save_every=1,
           batch_size=512)

