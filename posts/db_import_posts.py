from timeit import default_timer as timer
from lxml import etree
import sqlite3
import os
import yaml
import subprocess
import sys

def import_posts(posts_file, conn, cur, params):

    use_bulk_insert = params['use_bulk_insert']
    bulk_size = params['bulk_size']
    params['currentMonth'] = ''

    insert_questions_query = 'INSERT INTO Questions VALUES (?,?,?,?,?,?,?,?,?)'
    insert_answers_query = 'INSERT INTO Answers VALUES (?,?,?,?,?,?,?)'

    questions_to_insert = []
    answers_to_insert = []

    context = etree.iterparse(posts_file, events=('end',), tag='row')
    bulk_seen = 0

    for event, elem in context:

        if row_filter(elem):

            row_process(questions_to_insert, answers_to_insert, params, elem)
            bulk_seen += 1

            if use_bulk_insert and bulk_seen >= bulk_size:
                cur.executemany(insert_questions_query, questions_to_insert)
                questions_to_insert.clear()

                cur.executemany(insert_answers_query, answers_to_insert)
                answers_to_insert.clear()

                bulk_seen = 0
                conn.commit()

        #resource cleaning - contributes to small memory footprint
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    if bulk_seen != 0:
        cur.executemany(insert_questions_query, questions_to_insert)
        questions_to_insert.clear()

        cur.executemany(insert_answers_query, answers_to_insert)
        answers_to_insert.clear()
        conn.commit()

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
    score = int(elem.attrib['Score'])
    commentCount = int(elem.attrib['CommentCount'])

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
        acceptedAnswerId = int(elem.attrib['AcceptedAnswerId']) if 'AcceptedAnswerId' in elem.attrib else None
        question = (postId, userId, acceptedAnswerId, creationDate, score, commentCount,
                    elem.attrib['Title'], elem.attrib['Tags'], body)

        questions_to_insert.append(question)
    else:
        answer = (postId, userId, int(elem.attrib['ParentId']),
                  creationDate, score, commentCount, body)

        answers_to_insert.append(answer)

def import_users(users_file, conn, cur, params):

    use_bulk_insert = params['use_bulk_insert']
    bulk_size = params['bulk_size']

    insert_users_query = 'INSERT INTO Users VALUES (?,?,?,?)'
    users_to_insert = []

    context = etree.iterparse(users_file, events=('end',), tag='row')
    bulk_seen = 0

    for event, elem in context:

        user = (int(elem.attrib['Id']), elem.attrib['CreationDate'][:10],
                elem.attrib['DisplayName'], int(elem.attrib['Reputation']))

        users_to_insert.append(user)
        bulk_seen += 1

        if use_bulk_insert and bulk_seen >= bulk_size:
            cur.executemany(insert_users_query, users_to_insert)
            users_to_insert.clear()

            bulk_seen = 0
            conn.commit()

        #resource cleaning - contributes to small memory footprint
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    if bulk_seen != 0:
        cur.executemany(insert_users_query, users_to_insert)
        users_to_insert.clear()
        conn.commit()

    #delete user with Id = -1 since it corresponds to a bot
    cur.execute('DELETE FROM Users WHERE UserId = -1')
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
with open('create_tables.sql') as f:
    cur.executescript(f.read())

conn.commit()
print('Creating tables took {:.2f}s'.format(timer() - start))

start = timer()
import_posts(posts_file, conn, cur, params)
print('Importing posts took {:.2f}s'.format(timer() - start))

start = timer()
import_users(users_file, conn, cur, params)
print('Importing users took {:.2f}s'.format(timer() - start))

start = timer()
with open('populate_tags_table.sql') as f:
    cur.executescript(f.read())

conn.commit()
print('Importing tags took {:.2f}s'.format(timer() - start))

start = timer()
with open('calc_derived_measures.sql') as f:
    cur.executescript(f.read())

conn.commit()
print('Calculating derived measures took {:.2f}s'.format(timer() - start))

print('Created DB file', db_file)

conn.close()
