import sqlite3
import os
import yaml
import sys
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
questions_per_day_query = '''
SELECT CreationDate, COUNT(Id) AS Questions
FROM Questions
GROUP BY CreationDate
'''
questions_per_day = pd.read_sql_query(questions_per_day_query, conn, 'CreationDate', parse_dates=['CreationDate'])

answers_per_day_query = '''
SELECT CreationDate, COUNT(Id) AS Answers
FROM Answers
GROUP BY CreationDate
'''
answers_per_day = pd.read_sql_query(answers_per_day_query, conn, 'CreationDate', parse_dates=['CreationDate'])

active_users_per_day_query = '''
SELECT CreationDate, SUM(activeUsers) AS ActiveUsers
FROM (
    SELECT CreationDate, COUNT(DISTINCT UserId) AS ActiveUsers
    FROM Questions
    GROUP BY CreationDate
    UNION ALL
    SELECT CreationDate, COUNT(DISTINCT UserId) AS ActiveUsers
    FROM Answers
    GROUP BY CreationDate
) GROUP BY CreationDate
'''
active_users_per_day = pd.read_sql_query(active_users_per_day_query, conn, 'CreationDate', parse_dates=['CreationDate'])
active_users_per_day.rename(columns={'ActiveUsers': 'Active Users'}, inplace=True)  # add space character on the column name for clarity

#concatenate all metrics into a single dataframe
df = pd.concat([questions_per_day, answers_per_day, active_users_per_day], axis=1, join='inner')

#reindex dataframe so that missing days with counts 0 are also on the index
start_date, end_date = df.index[0], df.index[-1]
df_index = pd.date_range(start_date, end_date, freq='D')
df = df.reindex(df_index)

df.fillna(0, inplace=True)
df = df.astype('int64')

#remove last month if it is not complete, otherwise when aggregating data 
#in months the last month would be way lower than expected
if not end_date.is_month_end:
    df = df[(df.index.year != end_date.year) | (df.index.month != end_date.month)]

df.resample('M').mean().plot(title='StackOverflow evolution through time')

plot_file = os.path.join(images_dir, 'stackoverflow_evolution_{}.png'.format(region))
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot number of tags per question
num_tags_query = '''
SELECT numTags, count(numTags) AS Count
FROM (
    SELECT count(Tag) AS numTags
    FROM Questions, Tags
    WHERE Questions.Id = Tags.QuestionId
    GROUP BY Id
) GROUP BY numTags
LIMIT 5
'''
df = pd.read_sql_query(num_tags_query, conn, 'numTags')

del df.index.name  # otherwise the index name would appear on the plot
s = df['Count']

s.plot.barh(title='Number of tags per question')
plt.gca().invert_yaxis()

plot_file = os.path.join(images_dir, 'num_tags_{}.png'.format(region))
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot most popular tags
popular_tags_query = '''
SELECT Tag, count(Id) AS Count
FROM Questions, Tags
WHERE Questions.Id = Tags.QuestionId
GROUP BY Tag
ORDER BY count(Id) DESC
LIMIT {}
'''
num_popular_tags = 10
df = pd.read_sql_query(popular_tags_query.format(num_popular_tags), conn, 'Tag')
popular_tags = tuple(df.index)

del df.index.name
s = df['Count']

s.plot.bar(title='Number of questions for popular tags')
plt.tight_layout()

plot_file = os.path.join(images_dir, 'popular_tags_total_{}.png'.format(region))
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot number of questions through time for most popular tags
tag_questions_per_day_query = '''
SELECT Tag, CreationDate, count(Id) AS Count
FROM Questions, Tags
WHERE Questions.Id = Tags.QuestionId AND Tag in {}
GROUP BY Tag, CreationDate
'''
df = pd.read_sql_query(tag_questions_per_day_query.format(popular_tags), conn, parse_dates=['CreationDate'])

#apply pivoting so that each tag of the popular tags becomes its own column
df = df.pivot(index='CreationDate', columns='Tag', values='Count')
df = df[list(popular_tags)]
del df.columns.name

#reindex dataframe so that missing days with counts 0 are also on the index
start_date, end_date = df.index[0], df.index[-1]
df_index = pd.date_range(start_date, end_date, freq='D')
df = df.reindex(df_index)

df.fillna(0, inplace=True)
df = df.astype('int64')

#remove last month if it is not complete
if not end_date.is_month_end:
    df = df[(df.index.year != end_date.year) | (df.index.month != end_date.month)]

df.resample('M').mean().plot.area(title='Popular tags evolution through time')

plot_file = os.path.join(images_dir, 'popular_tags_evolution_{}.png'.format(region))
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)

conn.close()
