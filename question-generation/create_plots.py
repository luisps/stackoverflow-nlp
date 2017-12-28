import sqlite3
import os
import yaml
import sys
import re
from collections import defaultdict
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

region = config['region']
images_dir = config['dir_name']['images']
posts_dir = config['dir_name']['posts']

db_file = os.path.join(posts_dir, 'Posts_{}.db'.format(region))

if not os.path.isfile(db_file):
    sys.exit('SQLite database {} does not exist'.format(db_file))

#create images dir if it doesn't exist
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

#create connection
conn = sqlite3.connect(db_file)
cur = conn.cursor()


#plot number of questions through time
questions_per_day_query = '''
SELECT CreationDate, count(Id)
FROM Posts
GROUP BY CreationDate
'''
select_iter = cur.execute(questions_per_day_query)

date = []
num_questions = []
for row in select_iter:
    date.append(row[0])
    num_questions.append(row[1])

s = pd.Series(num_questions, index=pd.to_datetime(date))

#variables used by this and other timeseries plots
start_date, end_date = date[0], date[-1]
last_year, last_month = int(end_date[:4]), int(end_date[5:7])

#remove last month as it is not complete and so it can not be used
#when aggregating data in months
s = s[(s.index.year != last_year) | (s.index.month != last_month)]

s.resample('M').sum().plot(title='Number of questions')

plot_file = os.path.join(images_dir, 'timeseries_num_questions_{}.png'.format(region))
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot number of tags per question
select_iter = cur.execute('SELECT tags FROM Posts')

tag_count = defaultdict(int)
num_tags = defaultdict(int)

#compile regex outside the loop for efficiency
tag_re = re.compile('<(.*?)>')

for row in select_iter:

    #tag frequency
    tags = tuple(tag_re.findall(row[0]))
    num_tags[len(tags)] += 1

    for tag in tags:
        tag_count[tag] += 1

s = pd.Series(num_tags)
s.plot.barh(title='Number of tags per question')
plt.gca().invert_yaxis()

plot_file = os.path.join(images_dir, 'num_tags_{}.png'.format(region))
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot most popular tags
num_popular_tags = 15
sorted_tag_freq = sorted(tag_count, key=tag_count.get)

popular_tags = sorted_tag_freq[-num_popular_tags:][::-1]
popular_tags_count = [tag_count[tag] for tag in popular_tags]

s = pd.Series(popular_tags_count, index=popular_tags)
s.plot.bar(title='Most popular {} tags'.format(num_popular_tags))

plot_file = os.path.join(images_dir, 'popular_tags_{}.png'.format(region))
plt.tight_layout()
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot number of questions through time for most popular tags
tag_questions_per_day_query = '''
SELECT CreationDate, count(Id)
FROM Posts
WHERE tags like '%<{}>%'
GROUP BY CreationDate
'''

#reduce number of popular tags to avoid information overload on the plots
num_popular_tags = 5
popular_tags = popular_tags[:num_popular_tags]

df_index = pd.date_range(start_date, end_date, freq='D')
df = pd.DataFrame(0, index=df_index, columns=popular_tags)

for tag in popular_tags:
    select_iter = cur.execute(tag_questions_per_day_query.format(tag))

    for row in select_iter:
        date, num_questions = row[0], row[1]
        df.loc[date, tag] = num_questions


#remove last month as it is not complete
df = df[(df.index.year != last_year) | (df.index.month != last_month)]

df.resample('M').sum().plot(title='Number of questions for popular tags')

plot_file = os.path.join(images_dir, 'timeseries_popular_tags_{}.png'.format(region))
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


conn.close()
