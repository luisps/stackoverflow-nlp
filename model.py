from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

import os
import pickle
from global_variables import *

embedding_dim = 64
hidden_dim = 256
num_layers = 2
batch_size = 128
epochs = 1

with open(os.path.join(data_dir, in_file[:-4] + '-ready.pkl'), 'rb') as f:
    data = pickle.load(f)

x, y = data
del data

model = Sequential()
#defining input_length - would change if we go with a bucket like solution to have different input lengths based on sentence size
model.add(Embedding(vocab_size, embedding_dim, input_length=max_seq_len))

for _ in range(num_layers-1):
    model.add(LSTM(hidden_dim, return_sequences=True))
model.add(LSTM(hidden_dim))

model.add(Dense(total_keep_tags, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

#model.fit(x, y, batch_size=batch_size, epochs=epochs)
