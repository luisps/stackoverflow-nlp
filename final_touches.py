import pickle
import os
from global_variables import *
import numpy as np

with open(os.path.join(data_dir, in_file[:-4] + '-body-processed.pkl'), 'rb') as f:
    data = pickle.load(f)

post_body, tags = data
num_posts = len(post_body)
del data

x = np.zeros((num_posts, max_seq_len), dtype=np.int32)
y = np.zeros((num_posts, total_keep_tags), dtype=np.int32)

for post_idx in range(num_posts):

    words = [start_char] + [word + index_from for word in post_body[post_idx]]

    if truncating == 'pre':
        trunc = words[-max_seq_len:]
    elif truncating == 'post':
        trunc = s[:max_seq_len]
    else:
        raise ValueError('Truncating type "%s" not understood' % truncating)

    if padding == 'pre':
        x[post_idx, -len(trunc):] = trunc
    elif padding == 'post':
        x[post_idx, :len(trunc)] = trunc
    else:
        raise ValueError('Padding type "%s" not understood' % padding)

    for tag in tags[post_idx]:
        y[post_idx, tag] = 1

data = (x, y)
out_file = in_file[:-4] + '-ready.pkl'

with open(os.path.join(data_dir, out_file), 'wb') as f:
    pickle.dump(data, f)

