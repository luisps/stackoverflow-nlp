from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.wrappers import TimeDistributed

import numpy as np
import os
import yaml
import sys
from math import ceil
import time
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Code adapted from https://github.com/asturkmani/Keras-char-rnn
"""
class CharRNN:

    def __init__(self, corpus, max_len, step, batch_size):
        self.corpus = corpus
        self.max_len = max_len
        self.step = step
        self.batch_size = batch_size

        self.create_vocab()
        self.split_text()
        self.steps_per_epoch = ceil(self.num_sent / self.batch_size)

    def create_vocab(self):
        """create vocab and char index mappings"""

        chars = sorted(list(set(self.corpus)))
        self.num_chars = len(chars)
        self.char_to_idx = dict((c, i) for i, c in enumerate(chars))
        self.idx_to_char = dict((i, c) for i, c in enumerate(chars))

    def split_text(self):
        """split text into overlapping sentences"""

        sentence_len = self.max_len + 1
        sentences = []

        for i in range(0, len(self.corpus) - sentence_len, self.step):
            sentences.append(self.corpus[i:i + sentence_len])

        self.sentences = sentences
        self.num_sent = len(self.sentences)

    def vectorize(self, batch_sentences):
        """create a batch of examples encoded as a one hot vector"""

        num_sent_batch = len(batch_sentences)
        X = np.zeros((num_sent_batch, self.max_len, self.num_chars), dtype=np.bool)
        Y = np.zeros((num_sent_batch, self.max_len, self.num_chars), dtype=np.bool)
        
        for i, sentence in enumerate(batch_sentences):
            for t, char in enumerate(sentence):
                if t == 0:
                    X[i, t, self.char_to_idx[char]] = 1
                elif t == max_len:
                    Y[i, t-1, self.char_to_idx[char]] = 1
                else:
                    X[i, t, self.char_to_idx[char]] = 1
                    Y[i, t-1, self.char_to_idx[char]] = 1

        return X, Y

    def generator(self):
        """retrieve training batches"""

        #fetch a batch of sentences, apply vectorization to it and yield that batch
        #to be used for training, applying vectorization for each batch is much
        #more memory friendly than vectorizing the entire data set at once
        while True:
            for j in range(0, self.num_sent, self.batch_size):
                batch_sentences = self.sentences[j:j + self.batch_size]
                X, Y = self.vectorize(batch_sentences)
                yield (X, Y)


class LossHistory:

    def __init__(self, history_file, plot_file, epoch_slices, load_from_file):
        self.history_file = history_file
        self.plot_file = plot_file
        self.epoch_slices = epoch_slices

        if load_from_file:
            with open(self.history_file, 'rb') as f:
                self.x, self.loss = pickle.load(f)

        else:
            self.x, self.loss = [], []

    def update(self, curr_epoch, curr_slice, curr_loss):
        curr_x = curr_epoch + curr_slice / self.epoch_slices

        self.x.append(curr_x)
        self.loss.append(curr_loss)

        #save loss history to file
        with open(self.history_file, 'wb') as f:
            pickle.dump((self.x, self.loss), f)

        self.plot()

    def plot(self):
        if len(self.loss) < 2:
            return

        plt.plot(self.x, self.loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        #save loss plot to file
        plt.savefig(self.plot_file)
        plt.close()


def sample_chars(inference_model, charRNN, sample_size, initial_char='S'):
    """
    sample an amount of sample_size characters starting with the
    character initial_char, at each timestep the inference_model
    is used to predict a probability distribution over the next
    character conditioned on all the characters sampled before
    """

    inference_model.reset_states()

    sampled_text = ''
    curr_char = np.zeros((1, 1, charRNN.num_chars))

    curr_char_idx = charRNN.char_to_idx[initial_char]
    curr_char[0, 0, curr_char_idx] = 1
    sampled_text += initial_char

    for _ in range(sample_size):

        #retrieve a probability distribution over the next char in the sequence
        next_char_probs = inference_model.predict(curr_char)

        #softmax output is float32, must convert to float64 first as expected by np.random.multinomial
        next_char_probs = np.asarray(next_char_probs).astype('float64') 
        next_char_probs = next_char_probs / next_char_probs.sum()

        #sample a char from the distribution
        next_char_idx = np.random.multinomial(1, next_char_probs.squeeze(), 1).argmax()
        sampled_text += charRNN.idx_to_char[next_char_idx]

        curr_char[0, 0, curr_char_idx] = 0
        curr_char[0, 0, next_char_idx] = 1
        curr_char_idx = next_char_idx

    return sampled_text


with open('config.yml', 'r') as f:
    config = yaml.load(f)

region = config['region']
data_dir = config['dir_name']['data']
models_dir = config['dir_name']['models']
samples_dir = config['dir_name']['samples']

mode = config['charRNNmodel']['mode']
file_type = config['charRNNmodel']['file_type']

hidden_dim = config['charRNNmodel']['hidden_dim']
num_layers = config['charRNNmodel']['num_layers']

input_dropout = config['charRNNmodel']['input_dropout']
recurrent_dropout = config['charRNNmodel']['recurrent_dropout']

epochs = config['charRNNmodel']['epochs']
epoch_slices = config['charRNNmodel']['epoch_slices']
resume_training = config['charRNNmodel']['resume_training']

train_sample_size = config['charRNNmodel']['train_sample_size']
test_sample_size = config['charRNNmodel']['test_sample_size']

max_len = config['charRNNmodel']['max_len']
step = config['charRNNmodel']['step']
batch_size = config['charRNNmodel']['batch_size']

model_file = os.path.join(models_dir, 'char_rnn_{}_{}.h5'.format(file_type, region))

loss_history_file = os.path.join(models_dir, 'char_rnn_{}_{}.loss'.format(file_type, region))
loss_plot_file = os.path.join(models_dir, 'char_rnn_{}_{}.png'.format(file_type, region))

text_file = os.path.join(data_dir, '{}_{}.txt'.format(file_type, region))
sample_file = os.path.join(samples_dir, '{}_{}.txt'.format(file_type, region))

#create models dir if it doesn't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

#create samples dir if it doesn't exist
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

#read text file - used as training data for the CharRNN model
if not os.path.isfile(text_file):
    sys.exit('Text file {} does not exist'.format(text_file))

with open(text_file, 'r') as f:
    corpus = f.read()

#create instance of CharRNN with the chosen parameters
#the generator is passed when training to Keras model.fit_generator
charRNN = CharRNN(corpus, max_len, step, batch_size)
gen = charRNN.generator()

#build training model
model = Sequential()

model.add(LSTM(hidden_dim, input_shape=(max_len, charRNN.num_chars), dropout=input_dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))

for _ in range(num_layers-1):
    model.add(LSTM(hidden_dim, dropout=input_dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))

model.add(TimeDistributed(Dense(charRNN.num_chars, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='Adam')

#build inference model
#the inference model is essentialy the same model as the training model
#and so the number of parameters should match for both models
#differently from the training model however this model is stateful
#and only receives 1 sample at a time with time window 1
inference_model = Sequential()

inference_model.add(LSTM(hidden_dim, batch_input_shape=(1, 1, charRNN.num_chars), stateful=True, dropout=input_dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))

for _ in range(num_layers-1):
    inference_model.add(LSTM(hidden_dim, stateful=True, dropout=input_dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))

inference_model.add(TimeDistributed(Dense(charRNN.num_chars, activation='softmax')))
inference_model.compile(loss='categorical_crossentropy', optimizer='Adam')


if mode == 'train':

    #load saved model if it exists
    if resume_training and os.path.isfile(model_file):
        model.load_weights(model_file)

    if resume_training and os.path.isfile(loss_history_file):
        loss = LossHistory(loss_history_file, loss_plot_file, epoch_slices, True)
        sample_f = open(sample_file, 'a', encoding='utf-8')

        initial_epoch = (len(loss.loss) // epoch_slices) + 1
        initial_slice = (len(loss.loss) % epoch_slices) + 1
    else:
        loss = LossHistory(loss_history_file, loss_plot_file, epoch_slices, False)
        sample_f = open(sample_file, 'w', encoding='utf-8')

        initial_epoch = 1
        initial_slice = 1

    steps_per_slice = charRNN.steps_per_epoch // epoch_slices
    remaining_steps = charRNN.steps_per_epoch % epoch_slices

    print('Started training')
    for curr_epoch in range(initial_epoch, epochs+1):

        start_epoch = time.time()

        for curr_slice in range(initial_slice, epoch_slices+1):

            start_slice = time.time()
            history = model.fit_generator(gen, steps_per_epoch=steps_per_slice, epochs=1, verbose=0)
            #history = model.fit_generator(gen, steps_per_epoch=1, epochs=1, verbose=0)
            model.save_weights(model_file)

            curr_loss = history.history['loss'][0]
            loss.update(curr_epoch, curr_slice, curr_loss)

            end_slice = time.time()
            elapsed_secs = round(end_slice - start_slice)
            print('{}s - Epoch {} - {}/{} - Loss {:.4f}'.format(elapsed_secs, curr_epoch, curr_slice, epoch_slices, curr_loss))

            #sample characters and save them to a file
            inference_model.load_weights(model_file)
            sampled_text = sample_chars(inference_model, charRNN, train_sample_size)

            sample_f.write('Epoch {} - {}/{}\n'.format(curr_epoch, curr_slice, epoch_slices))
            sample_f.write(sampled_text)
            sample_f.write('\n\n')
            sample_f.flush()

        if remaining_steps != 0:
            model.fit_generator(gen, steps_per_epoch=remaining_steps, epochs=1, verbose=0)

        initial_slice = 1
        end_epoch = time.time()
        elapsed_secs = round(end_epoch - start_epoch)
        print('{}s - Epoch {}'.format(elapsed_secs, curr_epoch))

    sample_f.close()
    print('Finished training')

elif mode == 'test':
    
    if not os.path.isfile(model_file):
        sys.exit('Model weights {} do not exist'.format(model_file))

    inference_model.load_weights(model_file)
    sampled_text = sample_chars(inference_model, charRNN, test_sample_size)

    print('\nSampled text')
    print(sampled_text)

else:
    sys.exit('Mode {} not understood'.format(mode))
