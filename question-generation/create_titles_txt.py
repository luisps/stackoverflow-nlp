import sqlite3
import os
import yaml
import sys

#read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

region = config['region']
data_dir = config['dir_name']['data']
posts_dir = config['dir_name']['posts']

db_file = os.path.join(posts_dir, 'Posts_' + region + '.db')

if not os.path.isfile(db_file):
    sys.exit(db_file + " doesn't exist")

#create data dir if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

#create connection
conn = sqlite3.connect(db_file)
cur = conn.cursor()

select_iter = cur.execute('SELECT title FROM Posts')

titles_text = ''
for row in select_iter:
    titles_text += row[0] + '\n\n'

#remove extra \n at the end
titles_text = titles_text[:-2]

titles_file = os.path.join(data_dir, 'titles_' + region + '.txt')
with open(titles_file, 'w', encoding='utf-8') as f:
    f.write(titles_text)

print ('Created', titles_file)

conn.close()
