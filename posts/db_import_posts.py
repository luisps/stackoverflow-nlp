from lxml import etree
import sqlite3
import os
import yaml
import subprocess
import sys
import time
import re

def recreate_table(conn, cur):

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

    #commit table creation
    conn.commit()

def import_posts(posts_file, params, cur):

    posts = {}
    use_bulk_insert = params['use_bulk_insert']
    bulk_size = params['bulk_size']

    params['currentMonth'] = ''
    params['questions'] = []
    params['answers'] = []

    context = etree.iterparse(posts_file, events=('end',), tag='row')
    bulk_seen = 0

    for event, elem in context:

        if row_filter(elem):
            row_process(posts, params, elem)
            bulk_seen += 1

            if use_bulk_insert and bulk_seen >= bulk_size and bulk_ready(elem):
                bulk_insert(posts, params, cur)
                bulk_seen = 0

        #resource cleaning - contributes to small memory footprint
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    if bulk_seen != 0:
        bulk_insert(posts, params, cur)

def bulk_insert(posts, params, cur):

    questions_to_insert = []
    answers_to_insert = []
    insert_answers_body = params['insert_answers_body']
    cur.execute('BEGIN TRANSACTION')

    for postId, post in posts.items():

        if postId in params['questions']:
            is_update = True
            is_question = True
        elif postId in params['answers']:
            is_update = True
            is_question = False
        else:
            is_update = False

        #updating a previously inserted post
        if is_update:
            row = []
            table_to_update = 'Questions' if is_question else 'Answers'

            update_query = 'UPDATE {} SET UserId=?,CreationDate=?,'.format(table_to_update)
            row.append(post['userId'])
            row.append(post['creationDate'])

            if post['title'] is not None:
                update_query += 'Title=?,'
                row.append(post['title'])

            if post['tags'] is not None:
                update_query += 'Tags=?,'
                row.append(post['tags'])

            if post['body'] is not None:
                if is_question or insert_answers_body:
                    update_query += 'Body=?,'
                    row.append(post['body'])

            update_query = update_query[:-1]  # remove last comma
            update_query += ' WHERE Id=?'
            row.append(postId)

            cur.execute(update_query, tuple(row))

        #inserting a new post
        else:

            #post is a question
            if post['title'] is not None or post['tags'] is not None:
                row = (postId, post['userId'], post['title'], post['tags'],
                       post['creationDate'], post['body']
                      )

                questions_to_insert.append(row)
                params['questions'].append(postId)

            #post is an answer
            else:
                #if we don't need the answer's body we can save a lot of space by omitting it
                body = post['body'] if insert_answers_body else None
                row = (postId, post['userId'], post['creationDate'], body)

                answers_to_insert.append(row)
                params['answers'].append(postId)

    #perform inserts all at once for extra performance
    cur.executemany('INSERT INTO Questions VALUES (?,?,?,?,?,?)', questions_to_insert)
    cur.executemany('INSERT INTO Answers VALUES (?,?,?,?)', answers_to_insert)

    cur.execute('COMMIT TRANSACTION')

    #since we have inserted all posts on the DB we can now clear the posts dict
    #clearing the dict is crucial to make sure that RAM doesn't grow unbounded
    posts.clear()

def bulk_ready(elem):

    #make sure the last row is not body, otherwise we couldn't tell
    #whether that row is a question or an answer without looking at the
    #next rows on the XML file
    postTypeId = int(elem.attrib['PostHistoryTypeId'])
    return postTypeId != 2

def row_filter(elem):

    #discard rows that are not title, tags or body
    postTypeId = int(elem.attrib['PostHistoryTypeId'])
    if postTypeId > 9:
        return False

    #discard rows without a text attribute
    if 'Text' not in elem.attrib:
        return False

    #discard posts with a deleted user
    if 'UserId' in elem.attrib and elem.attrib['UserId'] == '-1':
        return False

    return True

def row_process(posts, params, elem):

    postTypeId = int(elem.attrib['PostHistoryTypeId'])
    postId = int(elem.attrib['PostId'])
    userId = int(elem.attrib['UserId']) if 'UserId' in elem.attrib else None

    creationDate = elem.attrib['CreationDate'][:10]
    text = elem.attrib['Text']

    currentMonth = creationDate[:7]
    if params['verbose'] and currentMonth > params['currentMonth']:
        print('Importing posts from {}'.format(currentMonth))
        params['currentMonth'] = currentMonth

    posts[postId] = posts[postId] if postId in posts else {'title': None, 'tags': None, 'body': None}
    posts[postId]['creationDate'] = creationDate
    posts[postId]['userId'] = userId

    if is_title(postTypeId):
        posts[postId]['title'] = text

    elif is_tags(postTypeId):
        posts[postId]['tags'] = text

    elif is_body(postTypeId):
        posts[postId]['body'] = text

    else:
        sys.exit('This should not occur')


def is_title(postTypeId):
    return postTypeId == 1 or postTypeId == 4 or postTypeId == 7

def is_body(postTypeId):
    return postTypeId == 2 or postTypeId == 5 or postTypeId == 8

def is_tags(postTypeId):
    return postTypeId == 3 or postTypeId == 6 or postTypeId == 9

def populate_tags_table(conn, cur):

    #compile regex outside the loop for efficiency
    tag_re = re.compile('<(.*?)>')
    tags_to_insert = []

    select_iter = cur.execute('SELECT Id, Tags FROM Questions')
    for row in select_iter:
        question_id, tag_str = row
        tag_list = tag_re.findall(tag_str) if tag_str else []

        tags_to_insert += [(question_id, tag) for tag in tag_list]

    cur.executemany('INSERT INTO Tags VALUES (?,?)', tags_to_insert)
    conn.commit()


#change cwd to the directory that holds the script
os.chdir(sys.path[0])

#read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

region = config['region']
available_regions = config['available_regions']
params_name = config['params']['regions'][region]
params = config['params'][params_name]

if region not in available_regions:
    sys.exit('Region must be one of the available regions: ' + ', '.join(available_regions))

posts_file = 'Posts_{}.xml'.format(region)
db_file = 'Posts_{}.db'.format(region)

#download XML file if not available locally
if not os.path.isfile(posts_file):
    print("The file {} doesn't exist. Downloading it now to {}".format(posts_file, os.getcwd()))
    subprocess.call(['./get_posts.sh', region])

#create connection
conn = sqlite3.connect(db_file)
cur = conn.cursor()

start_import = time.time()

recreate_table(conn, cur)
import_posts(posts_file, params, cur)
populate_tags_table(conn, cur)

end_import = time.time()
elapsed_secs = round(end_import - start_import)

conn.close()

print('Importing posts took {}s'.format(elapsed_secs))
print('Successfully extracted posts from XML file', posts_file)
print('Created DB file', db_file)
