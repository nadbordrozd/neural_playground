from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

from text_generation import train_lstm

print('Build model...')
chars = u'\n !"\'(),-.0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz\xc6\xe4\xe6\xe9\xeb'
maxlen = 40
model = Sequential()
model.add(LSTM(1024, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(1024, return_sequences=False))
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

train_lstm(model, '/home/ubuntu/.keras/datasets/nietzsche.txt', "models/nietzsche", save_every=2)