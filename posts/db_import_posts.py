from lxml import etree
import sqlite3
import os
import subprocess
import sys

def recreate_table(cur):

    create_table_query = '''\
CREATE TABLE Posts (
    Id UNSIGNED BIG INT,
    Title text,
    Tags text,
    CreationDate TEXT,
    Body text,
    PRIMARY KEY (Id)
);\
    '''

    cur.execute('DROP TABLE IF EXISTS Posts')
    cur.execute(create_table_query)
    
    #create view of Posts without post body(makes select statements faster to type)
    cur.execute('DROP VIEW IF EXISTS p')
    cur.execute('CREATE VIEW p AS SELECT Id, Title, Tags, CreationDate FROM Posts')

def read_posts(posts_file, row_filter_func, row_process_func):

    posts = {}
    context = etree.iterparse(posts_file, events=('end',), tag='row')

    for event, elem in context:

        if row_filter_func(posts, elem):
            row_process_func(posts, elem)

        #resource cleaning - contributes to small memory footprint
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    del context
    return posts

def row_filter(posts, elem):

    postTypeId = int(elem.attrib['PostHistoryTypeId'])
    postId = int(elem.attrib['PostId'])

	#discard rows that are not titles, tags or body
    if postTypeId > 9:
        return False

	#discard answers
    if is_body(postTypeId) and postId not in posts:
        return False

    if postTypeId == 1 and 'Text' not in elem.attrib:
        return False

    return True
    
def row_process(posts, elem):

    postTypeId = int(elem.attrib['PostHistoryTypeId'])
    postId = int(elem.attrib['PostId'])

    creationDate = elem.attrib['CreationDate'][:10]
    text = elem.attrib['Text']

    posts[postId] = posts[postId] if postId in posts else {'title': '', 'tags': '', 'body': ''}
    posts[postId]['creationDate'] = creationDate

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

posts = read_posts(posts_file, row_filter, row_process)

#insert posts
cur.execute('BEGIN TRANSACTION');
for postId, question in posts.items():
	row = (postId, question['title'], question['tags'],
           question['creationDate'], question['body']
		  )

	cur.execute('INSERT INTO Posts VALUES (?,?,?,?,?)', row)

cur.execute('COMMIT TRANSACTION');

conn.commit()
conn.close()

print('Successfully extracted posts from XML file', posts_file)
print('Created DB file', db_file)
