from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import tensorflow as tf

import os
import pickle
from global_variables import *


def weighted_binary_crossentropy(y_true, y_pred):
    return K.mean(tf_weighted_binary_crossentropy(y_true, y_pred), axis=-1)

def tf_weighted_binary_crossentropy(target, output, from_logits=False, pos_weight=5):

    #transform back to logits
    if not from_logits:
        _epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                       logits=output, pos_weight=pos_weight)

embedding_dim = 512
hidden_dim = 512
num_layers = 1

batch_size = 128
epochs = 100

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
              loss=weighted_binary_crossentropy,
              metrics=['accuracy'])

#model.summary()

file_path = 'model_layers-1_hidden-512.h5'
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False)
#early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=6)

#model.load(file_path)

history = model.fit(x, y,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2,
                    verbose=2,
                    callbacks=[checkpoint]
                   )

