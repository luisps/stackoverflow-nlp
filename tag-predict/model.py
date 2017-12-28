import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import tensorflow as tf

import os
import pickle
import yaml
from matplotlib import pyplot as plt

class PlotLosses(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        #self.fig = plt.figure()
        #plt.ion()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        with open('loss.pkl', 'wb') as f:
            pickle.dump((self.x, self.losses, self.val_losses), f)

        #plt.plot(self.x, self.losses, label="loss")
        #plt.plot(self.x, self.val_losses, label="val_loss")
        #plt.legend()

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
models_dir = config['dir_name']['models']
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
max_seq_len = config['keras_format']['max_seq_len']

word_vocab_size = num_keep_words - skip_top + 3
tag_vocab_size = num_keep_tags

#load posts dataset
with open(os.path.join(data_dir, dataset_name + '.pkl'), 'rb') as f:
    data = pickle.load(f)

(x_train, y_train), (x_val, y_val), (x_test, y_test) = data
del data

#build the model
model = Sequential()
model.add(Embedding(word_vocab_size, embedding_dim, input_length=max_seq_len))

for _ in range(num_layers-1):
    lstm_layer = LSTM(hidden_dim, dropout=input_dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)
    lstm_layer = Bidirectional(lstm_layer) if bidirectional else lstm_layer
    model.add(lstm_layer)

lstm_layer = LSTM(hidden_dim, dropout=input_dropout, recurrent_dropout=recurrent_dropout)
lstm_layer = Bidirectional(lstm_layer) if bidirectional else lstm_layer
model.add(lstm_layer)

model.add(Dense(tag_vocab_size, activation='sigmoid'))

model.compile(optimizer='adam',
              loss=weighted_binary_crossentropy,
              metrics=['accuracy'])

model.summary()

#create models dir if it doesn't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

#callbacks
bidirectional_str = '_bidirectional' if bidirectional else ''
model_path = os.path.join(models_dir, dataset_name + '_embedding-%d_hidden-%d_layers-%d%s.h5' % (embedding_dim, hidden_dim, num_layers, bidirectional_str))

checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False)
#early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=6)
        
plot_losses = PlotLosses()

print('Started training')
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[checkpoint, plot_losses],
                    validation_data=(x_val, y_val)
                   )
