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


def sample_chars(inference_model, charRNN, sample_size, initial_char='S'):
    """
        sample an amount of sample_size characters starting with the
        character initial_char, at each timestep the inference_model
        is used to predict a probability distribution over the next
        character conditioned on all the characters sampled before
    """

    inference_model.reset_states()

    text = ''
    currChar = np.zeros((1, 1, charRNN.num_chars))

    currCharIdx = charRNN.char_to_idx[initial_char]
    currChar[0, 0, currCharIdx] = 1
    text += initial_char

    for _ in range(sample_size):

        #retrieve a probability distribution over the next char in the sequence
        nextCharProbs = inference_model.predict(currChar)

        #softmax output is float32, must convert to float64 first as expected by np.random.multinomial
        nextCharProbs = np.asarray(nextCharProbs).astype('float64') 
        nextCharProbs = nextCharProbs / nextCharProbs.sum()

        #sample a char from the distribution
        nextCharIdx = np.random.multinomial(1, nextCharProbs.squeeze(), 1).argmax()
        text += charRNN.idx_to_char[nextCharIdx]

        currChar[0, 0, currCharIdx] = 0
        currChar[0, 0, nextCharIdx] = 1
        currCharIdx = nextCharIdx

    return text


with open('config.yml', 'r') as f:
    config = yaml.load(f)

region = config['region']
data_dir = config['dir_name']['data']
models_dir = config['dir_name']['models']
samples_dir = config['dir_name']['samples']

#create models dir if it doesn't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

#create samples dir if it doesn't exist
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

mode = config['charRNNmodel']['mode']
file_type = config['charRNNmodel']['file_type']

hidden_dim = config['charRNNmodel']['hidden_dim']
num_layers = config['charRNNmodel']['num_layers']

input_dropout = config['model']['input_dropout']
recurrent_dropout = config['model']['recurrent_dropout']

epochs = config['charRNNmodel']['epochs']
resume_training = config['charRNNmodel']['resume_training']
sample_size = config['charRNNmodel']['sample_size']

max_len = config['charRNNmodel']['max_len']
step = config['charRNNmodel']['step']
batch_size = config['charRNNmodel']['batch_size']

model_file = os.path.join(models_dir, 'char_rnn_{}_{}.h5'.format(file_type, region))
losses_file = os.path.join(models_dir, 'char_rnn_{}_{}.loss'.format(file_type, region))

#read text file - used as training data for the CharRNN model
text_file = os.path.join(data_dir, '{}_{}.txt'.format(file_type, region))
if not os.path.isfile(text_file):
    sys.exit(text_file, "doesn't exist")

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

    #load saved models if they exist and train on them
    if resume_training and os.path.exists(model_file):
        model.load_weights(model_file)

    if resume_training and os.path.exists(losses_file):
        with open(losses_file, 'rb') as f:
            losses = pickle.load(f)
            initial_epoch = len(losses) + 1
    else:
        losses = []
        initial_epoch = 1


    for epoch in range(initial_epoch, initial_epoch + epochs):

        startEpoch = time.time()
        history = model.fit_generator(gen, steps_per_epoch=charRNN.steps_per_epoch, epochs=1, verbose=0)
        model.save_weights(model_file)

        #save losses to a file
        curr_loss = history.history['loss'][0]
        losses.append(curr_loss)
        with open(losses_file, 'wb') as f:
            pickle.dump(losses, f)

        endEpoch = time.time()
        elapsedSecs = round(endEpoch - startEpoch)
        print('%ds - Epoch %d - Loss %.4f' % (elapsedSecs, epoch, curr_loss))

        #sample characters and save them to a file
        inference_model.load_weights(model_file)
        sampled_text = sample_chars(inference_model, charRNN, sample_size)

        sample_file = os.path.join(samples_dir, '{}_{}_epoch_{}.txt'.format(file_type, region, epoch))
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sampled_text)

        print ('Created', sample_file)

elif mode == 'test':
    
    if os.path.exists(model_file):
        inference_model.load_weights(model_file)
    else:
        print('Model does not exist')
        sys.exit('Exiting')

    sampled_text = sample_chars(inference_model, charRNN, sample_size)

    print('\nSampled text')
    print(text)
    print()

else:
    sys.exit('Mode %s not understood' % mode)
