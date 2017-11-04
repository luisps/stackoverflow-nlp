import pickle
from collections import defaultdict
import os
from global_variables import *

in_file = 'first-500-tagged.pkl'
#in_file = 'whole-pt.pkl'

with open(os.path.join(data_dir, in_file), 'rb') as f:
    data = pickle.load(f)

post_body, tags, creation_date = data
num_posts = len(post_body)
del data

word_count = defaultdict(int)

"""
for post_idx in range(num_posts):

    for tag in tags[post_idx]:
        tag_count[tag] += 1

"""




