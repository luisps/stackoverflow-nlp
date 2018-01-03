from lxml import etree
import sqlite3
import os
import subprocess
import sys
import time

def recreate_table(cur):

    #create table for questions
    create_table_query = '''\
CREATE TABLE Questions (
    Id UNSIGNED BIG INT,
    UserId UNSIGNED BIG INT,
    Title text,
    Tags text,
    CreationDate TEXT,
    Body text,
    PRIMARY KEY (Id)
);\
    '''

    cur.execute('DROP TABLE IF EXISTS Questions')
    cur.execute(create_table_query)
    
    #create view of Questions without body(makes select statements faster to type)
    cur.execute('DROP VIEW IF EXISTS q')
    cur.execute('CREATE VIEW q AS SELECT Id, UserId, Title, Tags, CreationDate FROM Questions')

    #create table for answers
    create_table_query = '''\
CREATE TABLE Answers (
    Id UNSIGNED BIG INT,
    UserId UNSIGNED BIG INT,
    CreationDate TEXT,
    Body text,
    PRIMARY KEY (Id)
);\
    '''

    cur.execute('DROP TABLE IF EXISTS Answers')
    cur.execute(create_table_query)
    
    #create view of Answers without body(makes select statements faster to type)
    cur.execute('DROP VIEW IF EXISTS a')
    cur.execute('CREATE VIEW a AS SELECT Id, UserId, CreationDate FROM Answers')

def import_posts(posts_file, row_filter_func, row_process_func, bulk_ready_func, cur, use_bulk_insert=True, bulk_size=4096):

    posts = {}
    params = {}
    params['currentMonth'] = ''
    params['questions'] = []
    params['answers'] = []

    context = etree.iterparse(posts_file, events=('end',), tag='row')
    bulk_seen = 0

    for event, elem in context:

        if row_filter_func(elem):
            row_process_func(posts, params, elem)
            bulk_seen += 1

            if use_bulk_insert and bulk_seen >= bulk_size and bulk_ready_func(elem):
                bulk_insert(posts, params, cur)
                bulk_seen = 0

        #resource cleaning - contributes to small memory footprint
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    if bulk_seen != 0:
        bulk_insert(posts, params, cur)

def bulk_insert(posts, params, cur, insert_answers_body=False):

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

                params['questions'].append(postId)
                cur.execute('INSERT INTO Questions VALUES (?,?,?,?,?,?)', row)

            #post is an answer
            else:
                #if we don't need the answer's body we can save a lot of space by omitting it
                body = post['body'] if insert_answers_body else None
                row = (postId, post['userId'], post['creationDate'], body)

                params['answers'].append(postId)
                cur.execute('INSERT INTO Answers VALUES (?,?,?,?)', row)


    #since we have inserted all posts on the DB we can now clear the posts dict
    #clearing the dict is crucial to make sure that RAM doesn't grow unbounded
    posts.clear()

    cur.execute('COMMIT TRANSACTION')

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

    #discard posts with a deleted user
    if 'UserId' in elem.attrib and elem.attrib['UserId'] == '-1':
        return False

    #discard titles without a text attribute
    if postTypeId == 1 and 'Text' not in elem.attrib:
        return False

    return True

def row_filter_en(elem):

    #creationDate = elem.attrib['CreationDate'][:10]
    postTypeId = int(elem.attrib['PostHistoryTypeId'])
    if is_body(postTypeId):
        return False

    if is_tags(postTypeId) and 'Text' not in elem.attrib:
        return False

    return row_filter(elem)
    
def row_process(posts, params, elem):

    postTypeId = int(elem.attrib['PostHistoryTypeId'])
    postId = int(elem.attrib['PostId'])
    userId = int(elem.attrib['UserId']) if 'UserId' in elem.attrib else None

    creationDate = elem.attrib['CreationDate'][:10]
    text = elem.attrib['Text']

    currentMonth = creationDate[:7]
    if currentMonth > params['currentMonth']:
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


available_regions = ['en', 'pt', 'es', 'ru', 'ja']

#Selecting region
if len(sys.argv) < 2:
    region = 'pt'
    print ('No region passed as argument. Using default region: ' + region)
else:
    region = sys.argv[1]

    if region not in available_regions:
        sys.exit('Region must be one of the available regions: ' + ', '.join(available_regions))


#change cwd to the directory that holds the script
os.chdir(sys.path[0])

posts_file = 'Posts_{}.xml'.format(region)
db_file = 'Posts_{}.db'.format(region)

#create connection
conn = sqlite3.connect(db_file)
cur = conn.cursor()

recreate_table(cur)
conn.commit()

#download XML file if not available locally
if not os.path.isfile(posts_file):
    print('The file', posts_file, "doesn't exist. Downloading it now to posts directory.")
    subprocess.call(['./get_posts.sh', region])

start_import = time.time()

if region == 'en':
    import_posts(posts_file, row_filter_en, row_process, bulk_ready, cur)
else:
    import_posts(posts_file, row_filter, row_process, bulk_ready, cur)

end_import = time.time()
elapsed_secs = round(end_import - start_import)

conn.close()

print('Importing posts took {}s'.format(elapsed_secs))
print('Successfully extracted posts from XML file', posts_file)
print('Created DB file', db_file)
