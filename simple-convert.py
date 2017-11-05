from lxml import etree
import pickle
import re
import os
from global_variables import *

def fast_iter(posts_file, row_filter_func, row_process_func, use_end_posts=True, end_posts=500):

    seen_posts = 0
    context = etree.iterparse(posts_file, events=('end',), tag='row')

    for event, elem in context:

        if row_filter_func(elem):
            seen_posts += 1
            
            if use_end_posts and seen_posts > end_posts:
                break

            row_process_func(elem)

        #resource cleaning - contributes to small memory footprint
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context

def row_filter(elem):
    return elem.attrib['PostTypeId'] == '1'
    
def row_process(elem):

    body = elem.attrib['Body']
    tag_str = elem.attrib['Tags']
    creation_date = elem.attrib['CreationDate'][:10]

    tags = tuple(tag_re.findall(tag_str))
    date_year = int(creation_date[:4])
    date_month = int(creation_date[5:7])

    body_list.append(body)
    tags_list.append(tags)
    creation_date_list.append((date_year, date_month))


region = 'pt'
posts_file = os.path.join('..', region + '.stackoverflow.com', 'Posts.xml')

use_end_posts = True
total_posts = 500


tag_re = re.compile('<(.*?)>')

body_list = []
tags_list = []
creation_date_list = []

fast_iter(posts_file, row_filter, row_process, use_end_posts, total_posts)

data = (body_list, tags_list, creation_date_list)

with open(os.path.join(data_dir, in_file), 'wb') as f:
    pickle.dump(data, f)
