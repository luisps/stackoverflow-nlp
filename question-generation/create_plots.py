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


#plot number of questions, answers and active users through time

#find date range(the date for the very first post and for the most recent one)
date_range_query = '''
SELECT MIN(CreationDate) AS startDate, MAX(CreationDate) AS endDate
FROM Questions
'''
select_iter = cur.execute(date_range_query)

#variables used by this and other timeseries plots
start_date, end_date = select_iter.fetchone()
last_year, last_month = int(end_date[:4]), int(end_date[5:7])

df_index = pd.date_range(start_date, end_date, freq='D')
df = pd.DataFrame(0, index=df_index, columns=['Questions', 'Answers', 'Active Users'])

questions_per_day_query = '''
SELECT CreationDate, COUNT(Id)
FROM Questions
GROUP BY CreationDate
'''
select_iter = cur.execute(questions_per_day_query)

for row in select_iter:
    date, num_questions = row[0], row[1]
    df.loc[date, 'Questions'] = num_questions

answers_per_day_query = '''
SELECT CreationDate, COUNT(Id)
FROM Answers
WHERE CreationDate >= "{}"
GROUP BY CreationDate
'''
select_iter = cur.execute(answers_per_day_query.format(start_date))

for row in select_iter:
    date, num_answers = row[0], row[1]
    df.loc[date, 'Answers'] = num_answers

active_users_per_day_query = '''
SELECT CreationDate, SUM(activeUsers)
FROM (
SELECT CreationDate, COUNT(DISTINCT UserId) AS activeUsers
FROM Questions
GROUP BY CreationDate
UNION ALL
SELECT CreationDate, COUNT(DISTINCT UserId) AS activeUsers
FROM Answers
WHERE CreationDate >= "{}"
GROUP BY CreationDate
)
GROUP BY CreationDate
'''
select_iter = cur.execute(active_users_per_day_query.format(start_date))

for row in select_iter:
    date, num_active_users = row[0], row[1]
    df.loc[date, 'Active Users'] = num_active_users

#remove last month as it is not complete and so it can not be used
#when aggregating data in months
df = df[(df.index.year != last_year) | (df.index.month != last_month)]

df.resample('M').mean().plot(title='StackOverflow evolution through time')

plot_file = os.path.join(images_dir, 'stackoverflow_evolution_{}.png'.format(region))
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot number of tags per question
select_iter = cur.execute('SELECT Tags FROM Questions')

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
num_popular_tags = 10
sorted_tag_freq = sorted(tag_count, key=tag_count.get)

popular_tags = sorted_tag_freq[-num_popular_tags:][::-1]
popular_tags_count = [tag_count[tag] for tag in popular_tags]

s = pd.Series(popular_tags_count, index=popular_tags)
s.plot.bar(title='Number of questions for popular tags')

plot_file = os.path.join(images_dir, 'popular_tags_total_{}.png'.format(region))
plt.tight_layout()
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot number of questions through time for most popular tags
tag_questions_per_day_query = '''
SELECT CreationDate, COUNT(Id)
FROM Questions
WHERE tags like '%<{}>%'
GROUP BY CreationDate
'''

df_index = pd.date_range(start_date, end_date, freq='D')
df = pd.DataFrame(0, index=df_index, columns=popular_tags)

for tag in popular_tags:
    select_iter = cur.execute(tag_questions_per_day_query.format(tag))

    for row in select_iter:
        date, num_questions = row[0], row[1]
        df.loc[date, tag] = num_questions


#remove last month as it is not complete
df = df[(df.index.year != last_year) | (df.index.month != last_month)]

df.resample('M').mean().plot.area(title='Popular tags evolution through time')

plot_file = os.path.join(images_dir, 'popular_tags_evolution_{}.png'.format(region))
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


conn.close()
