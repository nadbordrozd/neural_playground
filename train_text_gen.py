from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

from text_generation import chars, train_lstm


maxlen = 40
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


train_lstm(model=model,
           input_path='data/hadoop/train.java',
           validation_path='data/hadoop/test.java',
           from_checkpoint=True,
           save_dir='models/hadoop/',
           save_every=1,
           batch_size=1024)

