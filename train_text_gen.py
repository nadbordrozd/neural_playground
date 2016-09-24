#!/usr/bin/python
import argparse
import os

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

from text_generation import chars, train_lstm
from utils import load_latest_model

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", help="path to the file to train on")
parser.add_argument("--test_path", help="path to the file to test on")
parser.add_argument("--train_test_path",
                    help="directory that contains both a 'train' and a 'test' file. "
                         "If this parameter is given --train_path and --test_path "
                         "are ignored")
parser.add_argument("--model_dir", help="directory to save the model to/load from")
parser.add_argument("--maxlen", type=int, default=80,
                    help="cut the texts into sequences of this many chars")
parser.add_argument("--step", type=int, default=1,
                    help="move the window by this many chars. default 1. "
                         "step > 1 means sampling the data")
parser.add_argument("--lstm_size", type=int, default=256, help="lstm units per layer")
parser.add_argument("--dropout", type=float, default=0.3,
                    help='fraction of units to drop in every dropout layer')
parser.add_argument("--batch_size", type=int, default=1024, help='batch size')
parser.add_argument("--max_epochs", type=int, default=10000,
                    help='Train for how many epochs. Default 10000 (which is like '
                         'infinity)')
parser.add_argument('--layers', type=int, default=2, help='how many lstm layers to stack, '
                                                          'must be positive')
args = parser.parse_args()


model, epoch = load_latest_model(args.model_dir)
if model is None:
    layers = args.layers
    maxlen = args.maxlen
    model = Sequential()

    model.add(LSTM(args.lstm_size,
                   return_sequences=(layers != 1),
                   input_shape=(maxlen, len(chars))))
    model.add(Dropout(args.dropout))
    if layers > 1:
        for _ in range(layers - 2):
            model.add(LSTM(args.lstm_size, return_sequences=True))
            model.add(Dropout(args.dropout))

        model.add(LSTM(args.lstm_size, return_sequences=False))
        model.add(Dropout(args.dropout))

    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    epoch = 0
    model.metadata = {'epoch': 0, 'loss': [], 'val_loss': []}

if args.train_test_path is None:
    train_path = args.train_path
    test_path = args.test_path
else:
    train_path = os.path.join(args.train_test_path, 'train')
    test_path = os.path.join(args.train_test_path, 'test')

train_lstm(model=model,
           input_path=train_path,
           validation_path=test_path,
           save_dir=args.model_dir,
           save_every=1,
           step=args.step,
           batch_size=args.batch_size,
           iters=args.max_epochs)
