import sqlite3
import os
import yaml
import sys
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

#use a categorical repeating color palette
max_n = 20
sns.set_palette('deep', max_n)

def tick_formatter(val, pos, K=1000, M=1000**2):
    """
    format axis ticks displaying K for thousands and M for millions
    """
    if val < K:
        return '{0:g}'.format(val)
    elif K <= val < M:
        return '{0:g}K'.format(val / K)
    elif val >= M:
        return '{0:g}M'.format(val / M)


#read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

region = config['region']
posts_dir = config['dir_name']['posts']
images_dir = config['dir_name']['images']

images_dir = os.path.join(images_dir, region)
db_file = os.path.join(posts_dir, 'Posts_{}.db'.format(region))

if not os.path.isfile(db_file):
    sys.exit('SQLite database {} does not exist'.format(db_file))

#create images dir if it doesn't exist
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

#create connection
conn = sqlite3.connect(db_file)
cur = conn.cursor()


#plot number of questions, answers and new users through time
questions_per_day_query = '''
SELECT CreationDate, COUNT(QuestionId) AS Questions
FROM Questions
GROUP BY CreationDate
'''
questions_per_day = pd.read_sql_query(questions_per_day_query, conn, 'CreationDate', parse_dates=['CreationDate'])

answers_per_day_query = '''
SELECT CreationDate, COUNT(AnswerId) AS Answers
FROM Answers
GROUP BY CreationDate
'''
answers_per_day = pd.read_sql_query(answers_per_day_query, conn, 'CreationDate', parse_dates=['CreationDate'])

new_users_per_day_query = '''
SELECT CreationDate, COUNT(UserId) AS NewUsers
FROM Users
GROUP BY CreationDate
'''
new_users_per_day = pd.read_sql_query(new_users_per_day_query, conn, 'CreationDate', parse_dates=['CreationDate'])
new_users_per_day.rename(columns={'NewUsers': 'New Users'}, inplace=True)  # add space character on the column name for clarity

#concatenate all metrics into a single dataframe
df = pd.concat([questions_per_day, answers_per_day, new_users_per_day], axis=1, join='inner')

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

with sns.color_palette('muted'):
    ax = df.resample('M').mean().plot(title='StackOverflow evolution through time')

ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(tick_formatter))

plot_file = os.path.join(images_dir, 'stackoverflow_evolution.png')
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot number of tags per question
num_tags_query = '''
SELECT numTags, COUNT(numTags) AS Count
FROM (
    SELECT COUNT(Tag) AS numTags
    FROM Tags
    GROUP BY QuestionId
) GROUP BY numTags
LIMIT 5
'''
df = pd.read_sql_query(num_tags_query, conn, 'numTags')

del df.index.name  # otherwise the index name would appear on the plot
s = df['Count']

ax = s.plot.barh(title='Number of tags per question')
ax.invert_yaxis()
ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(tick_formatter))

plot_file = os.path.join(images_dir, 'num_tags.png')
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot most popular tags
popular_tags_query = '''
SELECT Tag, COUNT(QuestionId) AS Count
FROM Tags
GROUP BY Tag
ORDER BY COUNT(QuestionId) DESC
LIMIT {}
'''
num_popular_tags = 10
df = pd.read_sql_query(popular_tags_query.format(num_popular_tags), conn, 'Tag')
popular_tags = tuple(df.index)

del df.index.name
s = df['Count']

ax = s.plot.bar(title='Number of questions for popular tags')
plt.tight_layout()
ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(tick_formatter))

plot_file = os.path.join(images_dir, 'popular_tags.png')
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot number of questions through time for most popular tags
tag_questions_per_day_query = '''
SELECT Tag, CreationDate, COUNT(*) AS Count
FROM Questions, Tags
WHERE Questions.QuestionId = Tags.QuestionId AND Tag IN {}
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

ax = df.resample('M').mean().plot.area(title='Popular tags evolution through time')
ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(tick_formatter))

plot_file = os.path.join(images_dir, 'popular_tags_evolution.png')
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot user reputation through time based on when the user account was created
user_reputation_per_day_query = '''
SELECT CreationDate, AVG(Reputation) AS Reputation
FROM Users
GROUP BY CreationDate
'''
df = pd.read_sql_query(user_reputation_per_day_query, conn, 'CreationDate', parse_dates=['CreationDate'])

del df.index.name
s = df['Reputation']

s.resample('M').mean().plot(title='User Reputation based on account creation date')

plot_file = os.path.join(images_dir, 'user_reputation.png')
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot tags with the highest average score
highest_tag_score_query = '''
SELECT Tag, AVG(Score) AS Score
FROM Questions, Tags
WHERE Questions.QuestionId = Tags.QuestionId
GROUP BY Tag
HAVING COUNT(Tag) > {}
ORDER BY AVG(Score) DESC
LIMIT {}
'''
min_tag_freq = 20
max_tags = 15
df = pd.read_sql_query(highest_tag_score_query.format(min_tag_freq, max_tags), conn, 'Tag')

del df.index.name
s = df['Score']

ax = s.plot.barh(title='Tags with the highest average score')
ax.invert_yaxis()
plt.tight_layout()

plot_file = os.path.join(images_dir, 'highest_tag_score.png')
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot tags with the most comments
most_tag_comments_query = '''
SELECT Tag, AVG(CommentCount) AS Comments
FROM Answers, Tags
WHERE Answers.QuestionId = Tags.QuestionId
GROUP BY Tag
HAVING COUNT(Tag) > {}
ORDER BY AVG(CommentCount) DESC
LIMIT {}
'''
min_tag_freq = 25
max_tags = 15
df = pd.read_sql_query(most_tag_comments_query.format(min_tag_freq, max_tags), conn, 'Tag')

del df.index.name
s = df['Comments']

ax = s.plot.barh(title='Tags with the most comments')
ax.invert_yaxis()
plt.tight_layout()

plot_file = os.path.join(images_dir, 'most_tag_comments.png')
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot tags with highest and lowest ratio of questions with accepted answers to total questions
accepted_answers_ratio_per_tag_query = '''
SELECT Tag, COUNT(AcceptedAnswerId) * 1.0 / COUNT(Questions.QuestionId) AS Ratio
FROM Questions, Tags
WHERE Questions.QuestionId = Tags.QuestionId
GROUP BY Tag
HAVING COUNT(Tag) > {}
ORDER BY Ratio
'''
min_tag_freq = 50
max_tags = 9
df = pd.read_sql_query(accepted_answers_ratio_per_tag_query.format(min_tag_freq), conn, 'Tag')

del df.index.name
s = df['Ratio']

lowest_ratio_tags = s.iloc[:max_tags]
highest_ratio_tags = s.iloc[-max_tags:]

fig, axes = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(8.5, 4.0)

sns.stripplot(x=lowest_ratio_tags, y=['']*max_tags, hue=lowest_ratio_tags.index, ax=axes[0])
sns.stripplot(x=highest_ratio_tags, y=['']*max_tags, hue=highest_ratio_tags.index, ax=axes[1])

for ax in axes:
    ax.set_xlabel('')
    ax.set_ylim(0.5, -2.5)
    ax.legend(loc='upper center', ncol=3)


plot_file = os.path.join(images_dir, 'accepted_answer_ratio.png')
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


#plot tags with highest and lowest ratio of questions having at least one answer to total questions
any_answers_ratio_per_tag_query = '''
SELECT Tag, COUNT(Answers.QuestionId) * 1.0 / COUNT(Tags.QuestionId) AS Ratio
FROM Tags LEFT JOIN Answers
ON Tags.QuestionId = Answers.QuestionId
GROUP BY Tag
HAVING COUNT(Tag) > {}
ORDER BY Ratio
'''
min_tag_freq = 50
max_tags = 9
df = pd.read_sql_query(any_answers_ratio_per_tag_query.format(min_tag_freq), conn, 'Tag')

del df.index.name
s = df['Ratio']

lowest_ratio_tags = s.iloc[:max_tags]
highest_ratio_tags = s.iloc[-max_tags:]

fig, axes = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(8.5, 4.0)

sns.stripplot(x=lowest_ratio_tags, y=['']*max_tags, hue=lowest_ratio_tags.index, ax=axes[0])
sns.stripplot(x=highest_ratio_tags, y=['']*max_tags, hue=highest_ratio_tags.index, ax=axes[1])

for ax in axes:
    ax.set_xlabel('')
    ax.set_ylim(0.5, -2.5)
    ax.legend(loc='upper center', ncol=3)


plot_file = os.path.join(images_dir, 'any_answers_ratio.png')
plt.savefig(plot_file)
plt.close()
print ('Created', plot_file)


conn.close()
