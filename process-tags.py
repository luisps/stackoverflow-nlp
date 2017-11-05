import pickle
from collections import defaultdict
import os
from global_variables import *
import json

with open(os.path.join(data_dir, in_file), 'rb') as f:
    data = pickle.load(f)

post_body, tags, creation_date = data
num_posts = len(post_body)
del data

tag_count = defaultdict(int)
num_tags = defaultdict(int)

for post_idx in range(num_posts):

    #tags
    num_tags[len(tags[post_idx])] += 1

    for tag in tags[post_idx]:
        tag_count[tag] += 1


print('There are in total', num_posts, 'posts')
print('There are in total', len(tag_count), 'different tags')

z = sorted(tag_count, key=tag_count.get)[::-1]

most_common = 20
least_common = 20

print('\nThe', most_common, 'most common tags')
print(', '.join(z[:most_common]))

print('\nThe', least_common, 'least common tags')
print(', '.join(z[-least_common:]))


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


data = (post_body, tags, creation_date)
out_file = in_file[:-4] + '-tag-processed.pkl'

with open(os.path.join(data_dir, out_file), 'wb') as f:
    pickle.dump(data, f)


