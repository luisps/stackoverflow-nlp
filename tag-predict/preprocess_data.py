from lxml import etree
from collections import defaultdict
import os
import json
import yaml
import random
import sys
import nltk
from tqdm import tqdm
import re
import numpy as np
import pickle

"""
Part 1 - Extract posts from XML file

The file Posts.xml contains all user posts from the beginning of StackOverflow's operation
until the latest data release for a specific region. Since StackOverflow is a Q&A website, a post
in this context can be either a user question or answer.
StackOverflow operates in 5 different regions(or languages) which are English, Portuguese, Spanish,
Russian and Japanese(often abbreviated by their ISO codes, EN, PT, ES, RU and JA).
English has by far the biggest community, the file Posts.xml for English is ~60GBs so it can take a
while to extract posts from that file, for all other regions the file Posts.xml is smaller than 1GB
and it should run this part relatively fast.
Posts are read sequentially starting on a specific date(e.g: read 10000 posts starting on January 2015),
more complex filtering operations could be implemented if need be.
"""

def fast_iter(posts_file, row_filter_func, row_process_func, use_end_posts, end_posts):

    data = []
    seen_posts = 0
    context = etree.iterparse(posts_file, events=('end',), tag='row')

    for event, elem in context:

        if row_filter_func(elem):
            seen_posts += 1
            
            if use_end_posts and seen_posts > end_posts:
                break

            data.append(row_process_func(elem))

        #resource cleaning - contributes to small memory footprint
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    del context
    return data

def row_filter(elem):

    #only user questions have tags and so 
    #for our purposes we discard the answers
    if elem.attrib['PostTypeId'] != '1':
        return False

    #perform post filtering based on the creation date
    creation_date = elem.attrib['CreationDate'][:10]

    #filter out all posts created before date_year and date_month
    date_year = int(creation_date[:4])
    if date_year < begin_year:
        return False

    date_month = int(creation_date[5:7])
    if date_month < begin_month:
        return False

    return True
    
def row_process(elem):

    body = elem.attrib['Body']
    tags = elem.attrib['Tags']

    return (body, tags)


#read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

data_dir = config['dir_name']['data']
mappings_dir = config['dir_name']['mappings']
posts_dir = config['dir_name']['posts']

region = config['xml_extract']['region']
begin_year = config['xml_extract']['begin_year']
begin_month = config['xml_extract']['begin_month']
read_extra = config['xml_extract']['read_extra']

dataset_name = config['dataset']['name']
training_size = config['dataset']['training_size']
validation_size = config['dataset']['validation_size']
test_size = config['dataset']['test_size']

#Further down on the data pipeline when we apply preprocessing to each post,
#we may discard posts that don't match certain criteria. Since we don't know
#beforehand how many posts will be discarded downstream, a simple solution is
#to read more posts than actually needed and only use the posts that we have to
total_posts = training_size + validation_size + test_size
total_posts *= (1. + read_extra)
total_posts = int(total_posts)

use_end_posts = True
posts_file = os.path.join(posts_dir, 'Posts_' + region + '.xml')
if not os.path.isfile(posts_file):
    print('The file', posts_file, "doesn't exist. Download it first by running the script on", posts_dir, 'directory.')
    sys.exit('Exiting')

data = fast_iter(posts_file, row_filter, row_process, use_end_posts, total_posts)
print('Successfully extracted posts from', posts_file)

"""
Part 2 - Create word and tag vocabularies and mappings

Calculate word and tag frequency for posts in the training set.
These frequencies will be used to create a vocabulary for words and
a vocabulary for tags. In order to restrict the problem size we consider
just subsets for all the tags and all the words. After choosing a vocabulary
we create mappings to map from word/tag to indexes. These mappings can be
optionally saved to a file to be analysed or used for online inference.
Tokenizing a post's text into words is done using NLTK, this is a suboptimal
solution since it's targeted for natural language and not for programming languages.
A typical question in StackOverflow mixes natural language and programming languages
and so ideally we should use a different tokenizer for each.
Using programming language specific lexical analysers for tokenization should
be the right way to go, however here we simply go with a simpler approach of using
NLTK's word tokenizer for tokenizing all post's contents.
"""

num_keep_tags = config['vocabularies']['keep_tags']
num_keep_words = config['vocabularies']['keep_words']
skip_top = config['vocabularies']['skip_top']

min_word_freq = config['vocabularies']['min_word_freq']
max_word_len = config['vocabularies']['max_word_len']

save_word_mapping = config['mappings']['save_word_mapping']
save_tag_mapping = config['mappings']['save_tag_mapping']

#The data is currently ordered by creation time, before proceeding with the train/val/test split
#we must shuffle the data. Failing to do so would mean our data was time dependent,
#that the test set would always be in the future compared to the training set, similar
#to a time series dataset. While that scenario is indeed realistic it would further complicate
#our problem unnecessarily by introducing time dependencies.
random.seed(119)
random.shuffle(data)
num_posts = len(data)

tag_count = defaultdict(int)
num_tags = defaultdict(int)

word_count = defaultdict(int)
num_words = []

#compile regex outside the loop for efficiency
tag_re = re.compile('<(.*?)>')

#iterate all posts of the training set and accumulate both word and tag frequency
print('Creating word and tag dictionaries for the training set')
for post_idx in tqdm(range(training_size)):

    post_body, tag_str = data[post_idx]

    #word frequency
    words = nltk.word_tokenize(post_body)
    num_words.append(len(words))

    for word in words:
        word_count[word] += 1

    #tag frequency
    tags = tuple(tag_re.findall(tag_str))
    num_tags[len(tags)] += 1

    for tag in tags:
        tag_count[tag] += 1

#we keep just the top tags(most frequent) in the training set
#and discard all other tags from the posts
sorted_tag_freq = sorted(tag_count, key=tag_count.get)[::-1]
keep_tags = sorted_tag_freq[:num_keep_tags]

tag_to_index = {tag:i for i, tag in enumerate(keep_tags)}
index_to_tag = {i:tag for i, tag in enumerate(keep_tags)}

print('\nTag statistics')
print('Unique tags:', len(tag_count))
print('Number of tags per post:', ', '.join(['%s - %s' % (tag, count) for (tag, count) in sorted(num_tags.items())]))
print('Most common 10 tags:', ', '.join(sorted_tag_freq[:10]))
print('Least common 10 tags:', ', '.join(sorted_tag_freq[-10:]))

#simple word preprocess to reduce training set vocab size
word_count = {word:count for word, count in word_count.items() if count >= min_word_freq and len(word) < max_word_len}

#Similar to tags, we keep just a portion of the words with most counts.
#Differently for words, the words with most counts are likely to be either
#punctuation or stop words, which depending on the task might not be
#so relevant and so we can skip the top words as well. This behaviour
#is consistent with the way Keras handles NLP pre-processing.
sorted_word_freq = sorted(word_count, key=word_count.get)[::-1]
keep_words = sorted_word_freq[skip_top:num_keep_words]

#add special tokens to the word list
pad_idx = 0
start_idx = 1
oov_idx = 2
keep_words = ['_pad', '_start', '_oov'] + keep_words

word_to_index = {word:i for i, word in enumerate(keep_words)}
index_to_word = {i:word for i, word in enumerate(keep_words)}
training_set_words = set(sorted_word_freq)

num_words = np.asarray(num_words)
print('\nWord statistics')
print('Unique words:', len(word_count))
print('Number of words per post(post length): Avg - %0.1f, Std - %0.1f, Max - %d' %
      (num_words.mean(), num_words.std(), num_words.max()))
print('Most common 10 words:', ', '.join(sorted_word_freq[:10]))
print('Least common 10 words:', ', '.join(sorted_word_freq[-10:]))

#delete unnecessary variables with a big memory footprint
del tag_count, sorted_tag_freq
del word_count, sorted_word_freq

#optionally create mappings dir if it doesn't exist and save word/tag mappings
if (save_tag_mapping or save_word_mapping) and not os.path.exists(mappings_dir):
    os.makedirs(mappings_dir)

if save_tag_mapping:
    with open(os.path.join(mappings_dir, dataset_name + '_tag_to_index.json'), 'w') as f:
        json.dump(tag_to_index, f, indent=2)

if save_word_mapping:
    with open(os.path.join(mappings_dir, dataset_name + '_word_to_index.json'), 'w') as f:
        json.dump(word_to_index, f, indent=2)


"""
Part 3 - Create train/val/test sets

Posts are read sequentially and assigned either to train, val or test sets. Tags inside the tag vocabulary
are mapped to tag indexes, otherwise they are dropped. Words inside the word vocabulary are also mapped to
word indexes, otherwise we first check if that word ocurred on the training set and if so we assign it the
out of vocabulary(OOV) token, if the word isn't on the word vocabulary nor training set we simply drop it.
"""

training_set = []
validation_set = []
test_set = []

not_done = True
seen_posts = 0

curr_mode = 'training'
usable_posts = 0
print('\nCreating training set')
pbar = tqdm(total=training_size)

while not_done:

    if seen_posts == num_posts:
        pbar.close()
        print("\nWe have seen all the posts but still couldn't fill up the training, validation or test sets, this is because too many 'non usable' posts got discarded. Try increasing the read_extra variable.")
        sys.exit('Exiting')

    post_body, tag_str = data[seen_posts]
    seen_posts += 1

    #convert the tags to indexes and keep just the selected tags
    tags = tuple(tag_re.findall(tag_str))
    new_tags = [tag_to_index[tag] for tag in tags if tag in keep_tags]
    new_tags = tuple(new_tags)

    #if the resulting post has no tags, then it doesn't have value
    #for our purposes and so we discard it
    if len(new_tags) == 0:
        continue

    #Convert the words to indexes. If the word occured on the training set
    #but is not on keep_words we assign it oov. If the word didn't occur on the training
    #set we discard that word. This behavior is consistent with the way keras
    #handles NLP pre-processing.
    words = nltk.word_tokenize(post_body)
    new_words = [start_idx]
    for word in words:
        if word in keep_words:
            new_words.append(word_to_index[word])
        elif word in training_set_words:
            new_words.append(oov_idx)
        else:
            #ignore these words
            pass

    usable_posts += 1
    pbar.update(1)

    if curr_mode == 'training':
        training_set.append((new_words, new_tags))

        if usable_posts == training_size:
            curr_mode = 'validation'
            usable_posts = 0

            pbar.close()
            print('Creating validation set')
            pbar = tqdm(total=validation_size)

    elif curr_mode == 'validation':
        validation_set.append((new_words, new_tags))

        if usable_posts == validation_size:
            curr_mode = 'test'
            usable_posts = 0

            pbar.close()
            print('Creating test set')
            pbar = tqdm(total=test_size)

    elif curr_mode == 'test':
        test_set.append((new_words, new_tags))

        if usable_posts == test_size:
            #we're done now, we simply discard the remaining posts
            not_done = False
            pbar.close()

    else:
        sys.exit('This should not occur')


"""
Part 4 - Convert train/val/test sets to keras expected format

Convert variable length sequences(each post can have arbitrary length
which differs from post to post) to a fixed size length sequence.
In order to convert posts to a fixed size, posts above a threshold are
truncated and posts below the same threshold are zero padded. Both
truncation and padding can occur either at the beginning or end of
sequence. All sequences begin with a start sequence token.
The train/val/test sets are converted to NumPy matrices which is the
format expected by Keras. The final dataset is saved to a file.
"""

max_seq_len = config['keras_format']['max_seq_len']
truncating = config['keras_format']['truncating']
padding = config['keras_format']['padding']

datasets = [(training_set, training_size),
            (validation_set, validation_size),
            (test_set, test_size)]

final_dataset = []

for d in datasets:
    dataset = d[0]
    dataset_size = d[1]

    x = np.zeros((dataset_size, max_seq_len), dtype=np.int32)
    y = np.zeros((dataset_size, num_keep_tags), dtype=np.float32)

    for post_idx in range(dataset_size):

        #words and tags are lists of word/tag indexes
        words, tags = dataset[post_idx][0], dataset[post_idx][1]

        if truncating == 'pre':
            trunc = words[-max_seq_len:]
        elif truncating == 'post':
            trunc = words[:max_seq_len]
        else:
            sys.exit('Truncating type "%s" not understood' % truncating)

        if padding == 'pre':
            x[post_idx, -len(trunc):] = trunc
        elif padding == 'post':
            x[post_idx, :len(trunc)] = trunc
        else:
            sys.exit('Padding type "%s" not understood' % padding)

        for tag in tags:
            y[post_idx, tag] = 1

    final_dataset.append((x, y))

#create data dir if it doesn't exist and save the dataset to a file
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

with open(os.path.join(data_dir, dataset_name + '.pkl'), 'wb') as f:
    pickle.dump(final_dataset, f)

print('Done')
