import sqlite3
import os
import yaml
import sys
import numpy as np
import pandas as pd
from timeit import default_timer as timer

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

#use a categorical repeating color palette
max_n = 20
sns.set_palette('deep', max_n)

def tick_formatter(val, pos, K=1000, M=1000**2):
    '''
    format axis ticks displaying K for thousands and M for millions
    '''
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

images_root_dir = config['dir_name']['images']
images_dir = os.path.join(images_root_dir, region)

with open(os.path.join(posts_dir, 'config.yml'), 'r') as f:
    posts_config = yaml.load(f)

available_regions = posts_config['available_regions']
if region not in available_regions:
    sys.exit('Region must be one of the available regions: ' + ', '.join(available_regions))

db_file = os.path.join(posts_dir, 'Posts_{}.db'.format(region))
if not os.path.isfile(db_file):
    sys.exit('SQLite database {} does not exist'.format(db_file))

#create images dir if it doesn't exist
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

#font that supports Japanese characters
if region == 'ja':
    matplotlib.rc('font', family='TakaoPGothic')
else:
    matplotlib.rc('font', family='DejaVu Sans')

#create connection
conn = sqlite3.connect(db_file)
cur = conn.cursor()


#plot number of questions, answers and new users through time
start = timer()
questions_per_day_query = '''
SELECT date(CreationDateTime) AS CreationDate, COUNT(QuestionId) AS Questions
FROM Questions
GROUP BY CreationDate
'''
questions_per_day = pd.read_sql_query(questions_per_day_query, conn, 'CreationDate', parse_dates=['CreationDate'])

answers_per_day_query = '''
SELECT date(CreationDateTime) AS CreationDate, COUNT(AnswerId) AS Answers
FROM Answers
GROUP BY CreationDate
'''
answers_per_day = pd.read_sql_query(answers_per_day_query, conn, 'CreationDate', parse_dates=['CreationDate'])

new_users_per_day_query = '''
SELECT date(CreationDateTime) AS CreationDate, COUNT(UserId) AS NewUsers
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
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot most popular tags - tags with the most number of questions
start = timer()
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

plot_file = os.path.join(images_dir, 'popular_tags_num_questions.png')
plt.savefig(plot_file)
plt.close()
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot number of questions through time for most popular tags
start = timer()
tag_questions_per_day_query = '''
SELECT Tag, date(CreationDateTime) AS CreationDate, COUNT(*) AS Count
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
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot tags that most co-occur in questions with popular tags
start = timer()
tag_co_occurrence_query = '''
SELECT t1.Tag AS Tag1, t2.Tag AS Tag2, COUNT(*) AS Count
FROM Tags t1, Tags t2
WHERE t1.QuestionId = t2.QuestionId AND Tag1 != Tag2 AND Tag1 IN {}
GROUP BY Tag1, Tag2
HAVING Count >= {}
ORDER BY Tag1, Count DESC
'''
num_keep_tags = 5
min_tag_freq = 25
df = pd.read_sql_query(tag_co_occurrence_query.format(popular_tags, min_tag_freq), conn, ['Tag1', 'Tag2'])

s = df['Count']
s = s.groupby(level=0).head(num_keep_tags)

fig, axes = plt.subplots(nrows=len(popular_tags), ncols=1, sharex=True)
palette = sns.color_palette().as_hex()

for idx, tag1 in enumerate(popular_tags):
    tag2_counts = s.loc[[tag1]]
    tag2_counts = tag2_counts.unstack(level=0).sort_values(tag1)

    del tag2_counts.index.name
    tag2_counts.plot(kind='barh', colormap=ListedColormap(palette[idx]), ax=axes[idx])

    axes[idx].get_xaxis().set_major_formatter(ticker.FuncFormatter(tick_formatter))
    axes[idx].legend(loc='lower right')

axes[0].set_title('Tags that most co-occur with popular tags')
fig.set_size_inches(8, 15)
plt.tight_layout()

plot_file = os.path.join(images_dir, 'popular_tags_co_occurrence.png')
plt.savefig(plot_file)
plt.close()
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot number of answers for most popular tags
start = timer()
num_answers_per_tag_query = '''
SELECT Tag, numAnswers, COUNT(numAnswers) AS Count FROM (
    SELECT Tag, COUNT(Answers.QuestionId) as numAnswers
    FROM Tags LEFT JOIN Answers
    ON Tags.QuestionId = Answers.QuestionId
    WHERE Tag IN {}
    GROUP BY Tag, Tags.QuestionId
) GROUP BY Tag, numAnswers
'''
df = pd.read_sql_query(num_answers_per_tag_query.format(popular_tags), conn, ['Tag', 'numAnswers'])

def aggregate_num_answers(group, max_answers=5):

    if group.index.nlevels == 2:
        group = group.reset_index(level=0, drop=True)

    s_keep = group.iloc[:max_answers+1]
    count_others = group.iloc[max_answers+1:].sum()

    s_others = pd.Series([count_others], index=[str(max_answers) + '+'])
    s_keep = s_keep.append(s_others)

    return s_keep

s_num_answers = df['Count']  # variable to be used on KDE plot
s = s_num_answers.groupby(level=0).apply(aggregate_num_answers)

df = s.unstack(level=0)
del df.index.name
df = df[list(popular_tags)]

axes = df.plot(kind='barh', subplots=True, figsize=(8,15), title=['']*len(popular_tags))
axes[0].set_title('Number of answers for popular tags')
plt.tight_layout()

for ax in axes:
    ax.invert_yaxis()
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(tick_formatter))
    ax.legend(loc='lower right')

plot_file = os.path.join(images_dir, 'popular_tags_num_answers.png')
plt.savefig(plot_file)
plt.close()
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot KDE density estimation of number of answers for most popular tags
start = timer()
bw = 3.5 if region == 'en' else 0.5
xlim = (-1.5, 10) if region == 'en' else (-0.5, 4.5)

for tag in popular_tags:
    counts = s_num_answers.loc[tag]
    num_answers_unrolled = np.repeat(counts.index.values.astype(np.uint16), counts.values)
    num_answers_unrolled = pd.Series(num_answers_unrolled, name=tag)
    sns.kdeplot(num_answers_unrolled, bw=bw)

plt.title('KDE estimation of number of answers for popular tags')
plt.xlabel('Number of answers')
plt.ylabel('Density')

plt.xlim(*xlim)
plt.ylim(ymin=0.0)

plot_file = os.path.join(images_dir, 'popular_tags_num_answers_kde.png')
plt.savefig(plot_file)
plt.close()
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot number of answers per question
start = timer()
num_answers_query = '''
SELECT 0 AS numAnswers, COUNT(*) AS Count FROM (
    SELECT DISTINCT QuestionId FROM Tags
    EXCEPT SELECT QuestionId FROM Answers
) UNION ALL
SELECT numAnswers, COUNT(numAnswers) AS Count FROM (
    SELECT COUNT(QuestionId) AS numAnswers
    FROM Answers
    GROUP BY QuestionId
) GROUP BY numAnswers
'''
df = pd.read_sql_query(num_answers_query, conn, 'numAnswers')

df = df.apply(aggregate_num_answers)
s = df['Count']

ax = s.plot.barh(title='Number of answers per question')
ax.invert_yaxis()
ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(tick_formatter))

plot_file = os.path.join(images_dir, 'num_answers.png')
plt.savefig(plot_file)
plt.close()
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot number of tags per question
start = timer()
num_tags_query = '''
SELECT numTags, COUNT(numTags) AS Count FROM (
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
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot elapsed time until a question gets answered
start = timer()
days_to_first_answer_query = '''
SELECT MIN(ElapsedDays) AS ElapsedDays
FROM AnswerFreshness
GROUP BY QuestionId
'''
max_days = 7
max_quantile = 0.995
df_first_answer = pd.read_sql_query(days_to_first_answer_query, conn)
df_first_answer.rename(columns={'ElapsedDays': 'Days to first answer'}, inplace=True)

days_to_accepted_answer_query = '''
SELECT ElapsedDays
FROM AnswerFreshness
WHERE IsAcceptedAnswer = 1
'''
df_accepted_answer = pd.read_sql_query(days_to_accepted_answer_query, conn)
df_accepted_answer.rename(columns={'ElapsedDays': 'Days to accepted answer'}, inplace=True)

df_time_to_answer = pd.concat([df_first_answer, df_accepted_answer], axis=1)

#simple outlier detection and removal based on quantiles
quantiles = df_time_to_answer.quantile(max_quantile)
df_time_to_answer = df_time_to_answer[(df_time_to_answer['Days to first answer'] < quantiles.loc['Days to first answer']) &
                                      (df_time_to_answer['Days to accepted answer'] < quantiles.loc['Days to accepted answer'])]

ax = df_time_to_answer.plot.kde()
ax.set_title('Time until a question gets answered')
ax.set_xlim(xmin=0.0, xmax=max_days)
ax.set_xlabel('Days')

plot_file = os.path.join(images_dir, 'time_to_answer.png')
plt.savefig(plot_file)
plt.close()
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot elapsed time until a question gets answered for most popular tags
start = timer()
days_to_first_answer_per_tag_query = '''
SELECT Tag, MIN(ElapsedDays) AS ElapsedDays
FROM Tags, AnswerFreshness
WHERE Tags.QuestionId = AnswerFreshness.QuestionId AND Tag IN {}
GROUP BY AnswerFreshness.QuestionId, Tag
'''
df_first_answer = pd.read_sql_query(days_to_first_answer_per_tag_query.format(popular_tags), conn)

#converting column Tag to Categorical datatype can save a considerable amount of RAM
df_first_answer = df_first_answer.astype({'Tag': 'category'})
df_first_answer.set_index('Tag', inplace=True)
df_first_answer.rename(columns={'ElapsedDays': 'Days to first answer'}, inplace=True)

days_to_accepted_answer_per_tag_query = '''
SELECT Tag, ElapsedDays
FROM Tags, AnswerFreshness
WHERE Tags.QuestionId = AnswerFreshness.QuestionId AND Tag IN {}
AND IsAcceptedAnswer = 1
'''
df_accepted_answer = pd.read_sql_query(days_to_accepted_answer_per_tag_query.format(popular_tags), conn)

df_accepted_answer = df_accepted_answer.astype({'Tag': 'category'})
df_accepted_answer.set_index('Tag', inplace=True)
df_accepted_answer.rename(columns={'ElapsedDays': 'Days to accepted answer'}, inplace=True)

df_time_to_answer = pd.concat([df_first_answer, df_accepted_answer], axis=1)
fig, axes = plt.subplots(nrows=len(popular_tags), ncols=1)

for idx, tag in enumerate(popular_tags):
    df = df_time_to_answer.loc[tag]

    #outlier detection and removal
    quantiles = df.quantile(max_quantile)
    df = df[(df['Days to first answer'] < quantiles.loc['Days to first answer']) &
                                          (df['Days to accepted answer'] < quantiles.loc['Days to accepted answer'])]

    df.plot.kde(ax=axes[idx])
    axes[idx].set_xlim(xmin=0.0, xmax=max_days)
    axes[idx].set_ylabel(tag)
    axes[idx].legend().set_visible(False)

axes[0].set_title('Time until a question gets answered for popular tags')
axes[0].legend().set_visible(True)
axes[-1].set_xlabel('Days')

fig.set_size_inches(8, 15)
plt.tight_layout()

plot_file = os.path.join(images_dir, 'popular_tags_time_to_answer.png')
plt.savefig(plot_file)
plt.close()
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot user reputation through time based on when the user account was created
start = timer()
user_reputation_per_day_query = '''
SELECT date(CreationDateTime) AS CreationDate, AVG(Reputation) AS Reputation
FROM Users
GROUP BY CreationDate
'''
df = pd.read_sql_query(user_reputation_per_day_query, conn, 'CreationDate', parse_dates=['CreationDate'])

del df.index.name
s = df['Reputation']
s_mean = s.resample('M').mean()

fig, axes = plt.subplots(nrows=2, ncols=1)
s_mean.plot(title='User reputation based on account creation date', ax=axes[0])
s_mean.plot(logy=True, ax=axes[1])

axes[0].get_yaxis().set_major_formatter(ticker.FuncFormatter(tick_formatter))

plot_file = os.path.join(images_dir, 'user_reputation.png')
plt.savefig(plot_file)
plt.close()
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot the tags for which users mostly ask questions on their early days
start = timer()
tags_user_early_days_query = '''
SELECT ElapsedDays, Tag, COUNT(*) AS Count
FROM Tags, QuestionFreshness
WHERE Tags.QuestionId = QuestionFreshness.QuestionId
AND ElapsedDays <= {}
GROUP BY ElapsedDays, Tag
ORDER BY ElapsedDays, COUNT(*) DESC
'''
num_keep_tags = 20
max_days = 7
df = pd.read_sql_query(tags_user_early_days_query.format(max_days), conn, ['ElapsedDays', 'Tag'])

def aggregate_others(group, keep_tags):
    '''
    aggregate counts for all tags that aren't keep_tags
    into a single tag called others, the result series
    becomes keep_tags + others
    '''

    group = group.reset_index(level=0, drop=True)
    s_keep = group.reindex(keep_tags, fill_value=0.0)
    count_others = group.sum() - s_keep.sum()

    s_others = pd.Series([count_others], index=['Others'])
    s_keep = s_keep.append(s_others)

    return s_keep


#keep_tags is set to be the tags with the highest counts on day 0
s = df['Count']
keep_tags = list(s.loc[0].head(num_keep_tags).index)

s_keep = s.groupby(level=0).apply(aggregate_others, keep_tags)
df = s_keep.unstack(level=1)

old_palette = sns.color_palette()
curr_palette = old_palette.copy()

#add color of gray for others
rgb_gray = (0.7, 0.7, 0.7)
curr_palette.insert(num_keep_tags, rgb_gray)
sns.set_palette(curr_palette)

fig, axes = plt.subplots(nrows=2, ncols=1)
axes[0].set_title('Tags that users ask about on their early days')
#fig.set_size_inches(8.5, 4.0)

df.plot(kind='area', ax=axes[0])
axes[0].get_yaxis().set_major_formatter(ticker.FuncFormatter(tick_formatter))
axes[0].legend(loc='upper right', ncol=3)
axes[0].set_xlabel('')

#normalize tag counts for each day
df = df.div(df.sum(axis=1), axis=0)

df.plot(kind='area', ax=axes[1])
axes[1].legend().set_visible(False)

#revert back to previous color palette
sns.set_palette(old_palette)

plot_file = os.path.join(images_dir, 'user_early_days.png')
plt.savefig(plot_file)
plt.close()
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot tags with the highest average score
start = timer()
highest_tag_score_query = '''
SELECT Tag, AVG(Score) AS Score FROM (
    SELECT Tag, Score
    FROM Questions, Tags
    WHERE Questions.QuestionId = Tags.QuestionId
    UNION ALL
    SELECT Tag, Score
    FROM Answers, Tags
    WHERE Answers.QuestionId = Tags.QuestionId
) GROUP BY Tag
HAVING COUNT(Tag) > {}
ORDER BY AVG(Score) DESC
LIMIT {}
'''
df = pd.read_sql_query('SELECT COUNT(*) AS Count FROM Tags GROUP BY Tag', conn)
tag_count = df['Count']

region_std_coeff = 0.5 if region == 'en' else 2.5
min_tag_freq = int(tag_count.mean() + tag_count.std() / region_std_coeff)

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
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot tags with the most comments
start = timer()
most_tag_comments_query = '''
SELECT Tag, AVG(CommentCount) AS Comments FROM (
    SELECT Tag, CommentCount
    FROM Questions, Tags
    WHERE Questions.QuestionId = Tags.QuestionId
    UNION ALL
    SELECT Tag, CommentCount
    FROM Answers, Tags
    WHERE Answers.QuestionId = Tags.QuestionId
) GROUP BY Tag
HAVING COUNT(Tag) > {}
ORDER BY AVG(CommentCount) DESC
LIMIT {}
'''
region_std_coeff = 0.5 if region == 'en' else 5.0
min_tag_freq = int(tag_count.mean() + tag_count.std() / region_std_coeff)

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
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot tags with highest and lowest ratio of questions with accepted answers to total questions
start = timer()
accepted_answers_ratio_per_tag_query = '''
SELECT Tag, COUNT(AcceptedAnswerId) * 1.0 / COUNT(Questions.QuestionId) AS Ratio
FROM Questions, Tags
WHERE Questions.QuestionId = Tags.QuestionId
GROUP BY Tag
HAVING COUNT(Tag) > {}
ORDER BY Ratio
'''
df = pd.read_sql_query('SELECT COUNT(*) AS Count FROM Tags GROUP BY Tag', conn)
tag_count = df['Count']

region_std_coeff = 2.5 if region == 'en' else 10.0
min_tag_freq = int(tag_count.mean() + tag_count.std() / region_std_coeff)

max_tags = 9
df = pd.read_sql_query(accepted_answers_ratio_per_tag_query.format(min_tag_freq), conn, 'Tag')

del df.index.name
s = df['Ratio']

lowest_ratio_tags = s.iloc[:max_tags]
highest_ratio_tags = s.iloc[-max_tags:]

fig, axes = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(8.5, 4.0)
axes[0].set_title('Least/Most likely tags to have an accepted answer')

sns.stripplot(x=lowest_ratio_tags, y=['']*max_tags, hue=lowest_ratio_tags.index, ax=axes[0])
sns.stripplot(x=highest_ratio_tags, y=['']*max_tags, hue=highest_ratio_tags.index, ax=axes[1])

for ax in axes:
    ax.set_xlabel('')
    ax.set_ylim(0.5, -2.5)
    ax.legend(loc='upper center', ncol=3)

plot_file = os.path.join(images_dir, 'accepted_answer_ratio.png')
plt.savefig(plot_file)
plt.close()
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot tags with highest and lowest ratio of questions having at least one answer to total questions
start = timer()
any_answers_ratio_per_tag_query = '''
SELECT Tag, COUNT(Answers.QuestionId) * 1.0 / COUNT(Tags.QuestionId) AS Ratio
FROM Tags LEFT JOIN Answers
ON Tags.QuestionId = Answers.QuestionId
GROUP BY Tag
HAVING COUNT(Tag) > {}
ORDER BY Ratio
'''
region_std_coeff = 5.0 if region == 'en' else 10.0
min_tag_freq = int(tag_count.mean() + tag_count.std() / region_std_coeff)

max_tags = 9
df = pd.read_sql_query(any_answers_ratio_per_tag_query.format(min_tag_freq), conn, 'Tag')

del df.index.name
s = df['Ratio']

lowest_ratio_tags = s.iloc[:max_tags]
highest_ratio_tags = s.iloc[-max_tags:]

fig, axes = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(8.5, 4.0)
axes[0].set_title('Least/Most likely tags to have at least one answer')

sns.stripplot(x=lowest_ratio_tags, y=['']*max_tags, hue=lowest_ratio_tags.index, ax=axes[0])
sns.stripplot(x=highest_ratio_tags, y=['']*max_tags, hue=highest_ratio_tags.index, ax=axes[1])

for ax in axes:
    ax.set_xlabel('')
    ax.set_ylim(0.5, -2.5)
    ax.legend(loc='upper center', ncol=3)

plot_file = os.path.join(images_dir, 'any_answers_ratio.png')
plt.savefig(plot_file)
plt.close()
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))


#plot total number of questions, answers and users for each region
start = timer()
region_metrics = {}

for curr_region in available_regions:

    db_file = os.path.join(posts_dir, 'Posts_{}.db'.format(curr_region))
    if not os.path.isfile(db_file):
        print('SQLite database {} does not exist'.format(db_file))
        print('Skipping region {}'.format(curr_region))
        continue

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    total_questions_query = 'SELECT COUNT(*) FROM Questions'
    total_questions = cur.execute(total_questions_query).fetchone()[0]

    total_answers_query = 'SELECT COUNT(*) FROM Answers'
    total_answers = cur.execute(total_answers_query).fetchone()[0]

    total_users_query = 'SELECT COUNT(*) FROM Users'
    total_users = cur.execute(total_users_query).fetchone()[0]

    region_metrics[curr_region] = {
        'Questions': total_questions,
        'Answers': total_answers,
        'Users': total_users
    }

region_metrics = {region.capitalize() : metrics for region, metrics in region_metrics.items()}
df = pd.DataFrame.from_dict(region_metrics, orient='index')
df.sort_values('Questions', ascending=False, inplace=True)

fig, axes = plt.subplots(nrows=2, ncols=1)

df.plot.bar(title='Language comparison', ax=axes[0], rot=0)
axes[0].get_yaxis().set_major_formatter(ticker.FuncFormatter(tick_formatter))

df = df[df.index != 'En']
df.plot.bar(ax=axes[1], rot=0)
axes[1].get_yaxis().set_major_formatter(ticker.FuncFormatter(tick_formatter))

plot_file = os.path.join(images_root_dir, 'language_comparison.png')
plt.savefig(plot_file)
plt.close()
print ('{:.2f}s - Created {}'.format(timer() - start, plot_file))

conn.close()
