import sqlite3
import os
import yaml
import sys

#read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

data_dir = config['dir_name']['data']
posts_dir = config['dir_name']['posts']
region = config['xml_extract']['region']

db_file = os.path.join(posts_dir, 'Posts_' + region + '.db')

if not os.path.isfile(db_file):
    print(db_file, "doesn't exist")
    sys.exit('Exiting')

#create connection
conn = sqlite3.connect(db_file)
cur = conn.cursor()

select_iter = cur.execute('SELECT title FROM Posts')

titles_text = ''
for row in select_iter:
    titles_text += row[0] + '\n\n'

#remove extra \n at the end
titles_text = titles_text[:-2]

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

titles_file = os.path.join(data_dir, 'titles_' + region + '.txt')
with open(titles_file, 'w') as f:
    f.write(titles_text)

conn.close()
