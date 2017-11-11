from lxml import etree
import pickle
import re
import os
from global_variables import *
import yaml

def fast_iter(posts_file, row_filter_func, row_process_func, use_end_posts, end_posts):

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

    if elem.attrib['PostTypeId'] != '1':
        return False

    creation_date = elem.attrib['CreationDate'][:10]

    date_year = int(creation_date[:4])
    if date_year < begin_year:
        return False

    date_month = int(creation_date[5:7])
    if date_month < begin_month:
        return False

    return True
    
def row_process(elem):

    body = elem.attrib['Body']
    tag_str = elem.attrib['Tags']

    tags = tuple(tag_re.findall(tag_str))

    body_list.append(body)
    tags_list.append(tags)


with open('config.yml', 'r') as f:
    config = yaml.load(f)

data_dir = config['dir_name']['data']
posts_dir = config['dir_name']['posts']

region = config['xml_extract']['region']
begin_year = config['xml_extract']['begin_year']
begin_month = config['xml_extract']['begin_month']

training_size = config['dataset_size']['training_set']
validation_size = config['dataset_size']['validation_set']
test_size = config['dataset_size']['test_set']

posts_file = os.path.join(posts_dir, 'Posts_' + region + '.xml')

read_extra = 0.2
total_posts = training_size + validation_size + test_size
total_posts = int((1. + read_extra) * total_posts)


tag_re = re.compile('<(.*?)>')

body_list = []
tags_list = []
"""
use_end_posts = True
fast_iter(posts_file, row_filter, row_process, use_end_posts, total_posts)

data = (body_list, tags_list)

with open(os.path.join(data_dir, in_file), 'wb') as f:
    pickle.dump(data, f)
"""
