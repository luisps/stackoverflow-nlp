from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras import backend as K

import os
import pickle
from global_variables import *


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

    # transform back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                       logits=output)

embedding_dim = 512
hidden_dim = 512
num_layers = 3
batch_size = 256
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

model.summary()

#model.fit(x, y, batch_size=batch_size, epochs=epochs)
