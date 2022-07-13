# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 01:46:20 2022

@author: user
"""


import pandas as pd
import numpy as np
import seaborn as sns
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, Bidirectional, SimpleRNN, RNN
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import gensim.downloader as api


def replace_pun(str):
    return str.replace('.', 'PUN')


word_vec = api.load('word2vec-google-news-300')


#   Main driver code ----------------------------------------------------------

data = pd.read_csv('dataset.csv')

#data['y'] = data['y'].apply(replace_pun)

X = data['x']
Y = data['y']

word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(X)

X_encoded = word_tokenizer.texts_to_sequences(X)

tag_tokenizer = Tokenizer()
tag_tokenizer.fit_on_texts(Y)
Y_encoded = tag_tokenizer.texts_to_sequences(Y)

MAX_SEQ_LENGTH = 100
X_padded = pad_sequences(X_encoded, maxlen=MAX_SEQ_LENGTH,
                         padding='pre', truncating='post')
Y_padded = pad_sequences(Y_encoded, maxlen=MAX_SEQ_LENGTH,
                         padding='pre', truncating='post')

EMBEDDING_SIZE = 300
VOCABULARY_SIZE = len(word_tokenizer.word_index) + 1

#   word2vec pretained weights for RNN ----------------------------------------

embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))

word2id = word_tokenizer.word_index

for word, index in word2id.items():
    try:
        embedding_weights[index, :] = word_vec[word]
    except KeyError:
        pass

# ------------------------------------------------------------------------------

X, Y = X_padded, Y_padded


Y = to_categorical(Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=4)

NUM_CLASSES = Y.shape[2]

bi_lstm_model = Sequential()


bi_lstm_model.add(Embedding(input_dim=VOCABULARY_SIZE,
                            weights=[embedding_weights],
                            output_dim=EMBEDDING_SIZE,
                            input_length=MAX_SEQ_LENGTH,
                            trainable=True
                            ))

bi_lstm_model.add(GRU(64, return_sequences=True))


bi_lstm_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))

bi_lstm_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

# check summary of the model
bi_lstm_model.summary()

bilstm_training = bi_lstm_model.fit(X_train, Y_train, batch_size=128, epochs=3)

loss, accuracy = bi_lstm_model.evaluate(X_test, Y_test)
print("Loss: {0},\nAccuracy: {1}".format(loss, accuracy))


bi_lstm_model.save('gru_trained.h5')
print('Model Saved!')
