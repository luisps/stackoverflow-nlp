
DROP TABLE IF EXISTS Questions;
CREATE TABLE Questions (
    Id INTEGER,
    UserId INTEGER,
    Title TEXT,
    Tags TEXT,
    CreationDate TEXT,
    Body TEXT,
    PRIMARY KEY (Id)
);
    
-- create view of Questions without body(makes select statements faster to type)
DROP VIEW IF EXISTS q;
CREATE VIEW q AS SELECT Id, UserId, Title, Tags, CreationDate FROM Questions;

DROP TABLE IF EXISTS Answers;
CREATE TABLE Answers (
    Id INTEGER,
    UserId INTEGER,
    CreationDate TEXT,
    Body TEXT,
    PRIMARY KEY (Id)
);

-- create view of Answers without body(makes select statements faster to type)
DROP VIEW IF EXISTS a;
CREATE VIEW a AS SELECT Id, UserId, CreationDate FROM Answers;

DROP TABLE IF EXISTS Tags;
CREATE TABLE Tags (
    QuestionId INTEGER,
    Tag TEXT,
    PRIMARY KEY (QuestionId, Tag)
);

DROP TABLE IF EXISTS Users;
CREATE TABLE Users (
    Id INTEGER,
    Reputation INTEGER,
    CreationDate TEXT,
    DisplayName TEXT,
    UpVotes INTEGER,
    DownVotes INTEGER,
    PRIMARY KEY (Id)
);

-- empty otherwise unused space - useful when recreating the database often
VACUUM
