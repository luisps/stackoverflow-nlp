import pickle
from collections import defaultdict
import os
from global_variables import *
import json
import yaml
import random
import sys
import nltk

with open('config.yml', 'r') as f:
    config = yaml.load(f)

data_dir = config['dir_name']['data']

training_size = config['dataset_size']['training_set']
validation_size = config['dataset_size']['validation_set']
test_size = config['dataset_size']['test_set']

num_keep_tags = config['data_preprocess']['keep_tags']
num_keep_words = config['data_preprocess']['keep_words']
skip_top = config['data_preprocess']['skip_top']

min_word_freq = config['data_preprocess']['min_word_freq']
max_word_len = config['data_preprocess']['max_word_len']

with open(os.path.join(data_dir, in_file), 'rb') as f:
    data = pickle.load(f)

#before proceeding we must shuffle the data
#not doing so would mean our data was time dependent, that the test set would
#always be in the future compared to the training set, similar to a time series dataset
#while that scenario is indeed realistic it would further complicate our problem unnecessarily
random.seed(119)
random.shuffle(data)
num_posts = len(data)

tag_count = defaultdict(int)
num_tags = defaultdict(int)

word_count = defaultdict(int)
num_words = []

#iterate all posts of the training set
#and accumulate both word and tag frequency
for post_idx in range(training_size):

    post_body, tags = data[post_idx]

    #word frequency
    words = nltk.word_tokenize(post_body)
    num_words.append(len(words))
    for word in words:
        word_count[word] += 1

    #tag frequency
    num_tags[len(tags)] += 1

    for tag in tags:
        tag_count[tag] += 1

#we keep just the top tags(most frequent) in the training set
#and discard all other tags from the posts
sorted_tag_freq = sorted(tag_count, key=tag_count.get)[::-1]
keep_tags = sorted_tag_freq[:num_keep_tags]

tag_to_index = {tag:i for i, tag in enumerate(keep_tags)}
index_to_tag = {i:tag for i, tag in enumerate(keep_tags)}

#simple word preprocess to reduce training set vocab size
word_count = {word:count for word, count in word_count.items() if count >= min_word_freq and len(word) < max_word_len}

#similar to tags, we keep just a portion of the words with most counts
#however for words the words with most counts are likely to be either
#punctuation or stop words, which depending on the task might not be
#so relevant and so we have an option to skip the top words as well
sorted_word_freq = sorted(word_count, key=word_count.get)[::-1]
keep_words = sorted_word_freq[skip_top:num_keep_words]

#add special tokens to word list
pad_idx = 0
start_idx = 1
oov_idx = 2
keep_words = ['_pad', '_start', '_oov'] + keep_words

word_to_index = {word:i for i, word in enumerate(keep_words)}
index_to_word = {i:word for i, word in enumerate(keep_words)}
training_set_words = set(sorted_word_freq)

#delete unneeded variables with a big memory footprint
#del tag_count, sorted_tag_freq
#del word_count, sorted_word_freq

training_set = []
validation_set = []
test_set = []

not_done = True
seen_posts = 0
usable_posts = 0

train_plus_val = training_size + validation_size
train_plus_val_plus_test = train_plus_val + test_size

while not_done:

    if seen_posts == num_posts:
        sys.exit("We have seen all posts but still couldn't fill up the training, validation or test sets, this is because too many 'non usable' posts got discarded. Try increasing the read_extra variable. Exiting ...")

    post_body, tags = data[seen_posts]
    seen_posts += 1

    new_tags = [tag_to_index[tag] for tag in tags if tag in keep_tags]
    new_tags = tuple(new_tags)

    #if the resulting post has no tags, then it doesn't have value
    #for our purposes and so we discard it
    if len(new_tags) == 0:
        continue

    words = nltk.word_tokenize(post_body)
    new_words = []
    for word in words:
        if word in keep_words:
            new_words.append(word_to_index[word])
        elif word in training_set_words:
            new_words.append(oov_idx)
        else:
            #ignore these words
            pass

    #new_words = [word_to_index[word] for word in words if word in keep_words]

    if usable_posts < training_size:
        training_set.append((new_words, new_tags))
    elif usable_posts < train_plus_val:
        validation_set.append((new_words, new_tags))
    elif usable_posts < train_plus_val_plus_test:
        test_set.append((new_words, new_tags))
    else:
        #we're done now, we simply discard the remaining posts
        not_done = False

    usable_posts += 1




"""
print('There are in total', num_posts, 'posts')
print('There are in total', len(tag_count), 'different tags')


most_common = 20
least_common = 20

#print('\nThe', most_common, 'most common tags')
#print(', '.join(z[:most_common]))

#print('\nThe', least_common, 'least common tags')
#print(u', '.join(z[-least_common:]))


keep_tags = z[:total_keep_tags]
tag_to_index = {tag:i for i, tag in enumerate(keep_tags)}
save_mapping = True

if save_mapping:
    with open(os.path.join(mappings_dir, in_file[:-4] + '-tag-to-index.json'), 'w') as f:
        json.dump(tag_to_index, f, indent=2)

removing_posts_idxs = []

for post_idx in range(num_posts):

    new_tags = [tag_to_index[tag] for tag in tags[post_idx] if tag in keep_tags]
    new_tags = tuple(new_tags)

    if len(new_tags) == 0:
        removing_posts_idxs.append(post_idx)

    tags[post_idx] = new_tags

#must remove in reverse order
for post_idx in removing_posts_idxs[::-1]:
    del post_body[post_idx]
    del tags[post_idx]
    del creation_date[post_idx]


data = (post_body, tags)
out_file = in_file[:-4] + '-tag-processed.pkl'

with open(os.path.join(data_dir, out_file), 'wb') as f:
    pickle.dump(data, f)

"""
