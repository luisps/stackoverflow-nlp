import pickle
from collections import defaultdict
import os
from global_variables import *
import nltk
import json

def text_preprocess(body):

    words = nltk.word_tokenize(body)
    return words


with open(os.path.join(data_dir, in_file[:-4] + '-tagged.pkl'), 'rb') as f:
    data = pickle.load(f)

post_body, tags, creation_date = data
num_posts = len(post_body)
del data

word_count = defaultdict(int)
num_words = []

for post_idx in range(num_posts):

    words = text_preprocess(post_body[post_idx])
    num_words.append(len(words))
    for word in words:
        word_count[word] += 1


#delete some words which shouldn't be useful
min_word_count = 2
max_word_len = 30
total_keep_words = 1000
save_mapping = True

word_count = {word:count for word, count in word_count.items() if count >= min_word_count and len(word) < max_word_len}


z = sorted(word_count, key=word_count.get)[::-1]
keep_words = z[:total_keep_words]
word_to_index = {word:i for i, word in enumerate(keep_words)}

if save_mapping:
    with open(os.path.join(mappings_dir, in_file[:-4] + '-word-to-index.json'), 'w') as f:
        json.dump(word_to_index, f, indent=2)

removed_words = []

for post_idx in range(num_posts):

    words = text_preprocess(post_body[post_idx])
    new_words = [word_to_index[word] for word in words if word in keep_words]
    if len(new_words) == 0:
        print('Post without any words after removal. Oh oh')

    removed_words.append(num_words[post_idx] - len(new_words))
    post_body[post_idx] = new_words


data = (post_body, tags)
out_file = in_file[:-4] + '-body-processed.pkl'

with open(os.path.join(data_dir, out_file), 'wb') as f:
    pickle.dump(data, f)
