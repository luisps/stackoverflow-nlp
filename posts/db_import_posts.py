from timeit import default_timer as timer
from lxml import etree
import sqlite3
import os
import yaml
import subprocess
import sys
import re

def create_tables(conn, cur):

    #create table for questions
    create_table_query = '''
CREATE TABLE Questions (
    Id INTEGER,
    UserId INTEGER,
    Title TEXT,
    Tags TEXT,
    CreationDate TEXT,
    Body TEXT,
    PRIMARY KEY (Id)
)
    '''
    cur.execute('DROP TABLE IF EXISTS Questions')
    cur.execute(create_table_query)
    
    #create view of Questions without body(makes select statements faster to type)
    cur.execute('DROP VIEW IF EXISTS q')
    cur.execute('CREATE VIEW q AS SELECT Id, UserId, Title, Tags, CreationDate FROM Questions')

    #create table for tags
    create_table_query = '''
CREATE TABLE Tags (
    QuestionId INTEGER,
    Tag TEXT,
    PRIMARY KEY (QuestionId, Tag)
)
'''
    cur.execute('DROP TABLE IF EXISTS Tags')
    cur.execute(create_table_query)

    #create table for answers
    create_table_query = '''
CREATE TABLE Answers (
    Id INTEGER,
    UserId INTEGER,
    CreationDate TEXT,
    Body TEXT,
    PRIMARY KEY (Id)
)
    '''
    cur.execute('DROP TABLE IF EXISTS Answers')
    cur.execute(create_table_query)
    
    #create view of Answers without body(makes select statements faster to type)
    cur.execute('DROP VIEW IF EXISTS a')
    cur.execute('CREATE VIEW a AS SELECT Id, UserId, CreationDate FROM Answers')

    #create table for users
    create_table_query = '''
CREATE TABLE Users (
    Id INTEGER,
    Reputation INTEGER,
    CreationDate TEXT,
    DisplayName TEXT,
    UpVotes INTEGER,
    DownVotes INTEGER,
    PRIMARY KEY (Id)
)
'''
    cur.execute('DROP TABLE IF EXISTS Users')
    cur.execute(create_table_query)

    #empty otherwise unused space and commit table creation
    cur.execute('vacuum')
    conn.commit()

def import_posts(posts_file, conn, cur, params):

    use_bulk_insert = params['use_bulk_insert']
    bulk_size = params['bulk_size']

    context = etree.iterparse(posts_file, events=('end',), tag='row')
    bulk_seen = 0

    questions_to_insert = []
    answers_to_insert = []
    params['currentMonth'] = ''

    for event, elem in context:

        if row_filter(elem):

            row_process(questions_to_insert, answers_to_insert, params, elem)
            bulk_seen += 1

            if use_bulk_insert and bulk_seen >= bulk_size:
                bulk_insert(questions_to_insert, answers_to_insert, conn, cur)
                bulk_seen = 0

        #resource cleaning - contributes to small memory footprint
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    if bulk_seen != 0:
        bulk_insert(questions_to_insert, answers_to_insert, conn, cur)

def row_filter(elem):

    #discard rows that are not questions or answers
    postTypeId = int(elem.attrib['PostTypeId'])
    if postTypeId != 1 and postTypeId != 2:
        return False

    #discard posts with a deleted user
    if 'OwnerUserId' not in elem.attrib:
        return False

    return True

def row_process(questions_to_insert, answers_to_insert, params, elem):

    postId = int(elem.attrib['Id'])
    postTypeId = int(elem.attrib['PostTypeId'])

    userId = int(elem.attrib['OwnerUserId'])
    creationDate = elem.attrib['CreationDate'][:10]

    if postTypeId == 1 or params['insert_answers_body']:
        body = elem.attrib['Body']
        body = body.replace('\r\n', '\n')  # convert <CR><LF> DOS format to <LF> Unix format
    else:
        body = None

    currentMonth = creationDate[:7]
    if params['verbose'] and currentMonth > params['currentMonth']:
        print('Importing posts from {}'.format(currentMonth))
        params['currentMonth'] = currentMonth

    if postTypeId == 1:
        question = (postId, userId, elem.attrib['Title'],
                    elem.attrib['Tags'], creationDate, body)

        questions_to_insert.append(question)
    else:
        answer = (postId, userId, creationDate, body)

        answers_to_insert.append(answer)

def bulk_insert(questions_to_insert, answers_to_insert, conn, cur):

    cur.executemany('INSERT INTO Questions VALUES (?,?,?,?,?,?)', questions_to_insert)
    questions_to_insert.clear()

    cur.executemany('INSERT INTO Answers VALUES (?,?,?,?)', answers_to_insert)
    answers_to_insert.clear()
    conn.commit()

def import_tags(conn, cur, params):

    tags_to_insert = []
    use_bulk_insert = params['use_bulk_insert']
    bulk_size = params['bulk_size']

    #compile regex outside the loop for efficiency
    tag_re = re.compile('<(.*?)>')

    select_iter = cur.execute('SELECT Id, Tags FROM Questions')
    bulk_seen = 0
    insert_cur = conn.cursor()

    for row in select_iter:
        question_id, tag_str = row
        tag_list = tag_re.findall(tag_str) if tag_str else []

        tags_to_insert += [(question_id, tag) for tag in tag_list]
        bulk_seen += len(tags_to_insert)

        if use_bulk_insert and bulk_seen >= bulk_size:
            insert_cur.executemany('INSERT INTO Tags VALUES (?,?)', tags_to_insert)
            tags_to_insert.clear()

            bulk_seen = 0
            conn.commit()

    if bulk_seen != 0:
        insert_cur.executemany('INSERT INTO Tags VALUES (?,?)', tags_to_insert)
        tags_to_insert.clear()
        conn.commit()

def import_users(users_file, conn, cur, params):

    users_to_insert = []
    use_bulk_insert = params['use_bulk_insert']
    bulk_size = params['bulk_size']

    context = etree.iterparse(users_file, events=('end',), tag='row')
    bulk_seen = 0

    for event, elem in context:

        user = (int(elem.attrib['Id']), int(elem.attrib['Reputation']), elem.attrib['CreationDate'][:10],
                elem.attrib['DisplayName'], int(elem.attrib['UpVotes']), int(elem.attrib['DownVotes']))

        users_to_insert.append(user)
        bulk_seen += 1

        if use_bulk_insert and bulk_seen >= bulk_size:
            cur.executemany('INSERT INTO Users VALUES (?,?,?,?,?,?)', users_to_insert)
            users_to_insert.clear()

            bulk_seen = 0
            conn.commit()

        #resource cleaning - contributes to small memory footprint
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    if bulk_seen != 0:
        cur.executemany('INSERT INTO Users VALUES (?,?,?,?,?,?)', users_to_insert)
        users_to_insert.clear()
        conn.commit()

def post_preprocess(conn, cur):

    #convert tab characters in Title to spaces
    #cur.execute("UPDATE Questions SET Title = REPLACE(Title, char(9), ' ')")

    #delete user with Id = -1 since it corresponds to a bot
    cur.execute('DELETE FROM Users WHERE Id = -1')

    conn.commit()


#change cwd to the directory that holds the script
os.chdir(sys.path[0])

#read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

region = config['region']
available_regions = config['available_regions']
params = config['params']

if region not in available_regions:
    sys.exit('Region must be one of the available regions: ' + ', '.join(available_regions))

posts_file = 'Posts_{}.xml'.format(region)
users_file = 'Users_{}.xml'.format(region)
db_file = 'Posts_{}.db'.format(region)

#download XML files if not available locally
if not os.path.isfile(posts_file) or not os.path.isfile(users_file):
    print("Downloading XML files for region {}".format(region))
    subprocess.call(['./get_posts.sh', region])

#create connection
conn = sqlite3.connect(db_file)
cur = conn.cursor()

start = timer()
create_tables(conn, cur)
print('Creating tables took {:.2f}s'.format(timer() - start))

print('Started importing posts')
start = timer()
import_posts(posts_file, conn, cur, params)
print('Importing posts took {:.2f}s'.format(timer() - start))

start = timer()
import_tags(conn, cur, params)
print('Importing tags took {:.2f}s'.format(timer() - start))

start = timer()
import_users(users_file, conn, cur, params)
print('Importing users took {:.2f}s'.format(timer() - start))

start = timer()
post_preprocess(conn, cur)
print('Preprocessing posts took {:.2f}s'.format(timer() - start))

print('Created DB file', db_file)

conn.close()
