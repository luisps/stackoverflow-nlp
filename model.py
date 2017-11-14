from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import tensorflow as tf

import os
import pickle
import yaml


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


with open('config.yml', 'r') as f:
    config = yaml.load(f)

data_dir = config['dir_name']['data']
dataset_name = config['dataset']['name']

embedding_dim = config['model']['embedding_dim']
hidden_dim = config['model']['hidden_dim']
num_layers = config['model']['num_layers']
bidirectional = config['model']['bidirectional']

input_dropout = config['model']['input_dropout']
recurrent_dropout = config['model']['recurrent_dropout']

batch_size = config['model']['batch_size']
epochs = config['model']['epochs']

num_keep_tags = config['vocabularies']['keep_tags']
num_keep_words = config['vocabularies']['keep_words']
skip_top = config['vocabularies']['skip_top']

word_vocab_size = num_keep_words - skip_top + 3
tag_vocab_size = num_keep_tags

#load posts dataset
with open(os.path.join(data_dir, dataset_name + '.pkl'), 'rb') as f:
    data = pickle.load(f)

(x_train, y_train), (x_val, y_val), (x_test, y_test) = data
del data

exit()
model = Sequential()
#defining input_length - would change if we go with a bucket like solution to have different input lengths based on sentence size
model.add(Embedding(vocab_size, embedding_dim, input_length=max_seq_len))

for _ in range(num_layers-1):
    model.add(Bidirectional(LSTM(hidden_dim, dropout=input_dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))

model.add(Bidirectional(LSTM(hidden_dim, dropout=input_dropout, recurrent_dropout=recurrent_dropout)))

model.add(Dense(total_keep_tags, activation='sigmoid'))

model.compile(optimizer='adam',
              loss=weighted_binary_crossentropy,
              metrics=['accuracy'])

#model.summary()

file_path = 'model_layers-1_hidden-512.h5'
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False)
#early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=6)

#model = load_model(file_path)

print('Started training')
history = model.fit(x, y,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2,
                    verbose=2#,
#                    callbacks=[checkpoint]
                   )
