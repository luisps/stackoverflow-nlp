
DROP TABLE IF EXISTS Questions;
CREATE TABLE Questions (
    QuestionId INTEGER,
    UserId INTEGER,
    AcceptedAnswerId INTEGER,
    CreationDateTime TEXT,
    Score INTEGER,
    CommentCount INTEGER,
    Title TEXT,
    Tags TEXT,
    Body TEXT,
    PRIMARY KEY (QuestionId),
    FOREIGN KEY (UserId) REFERENCES Users(UserId),
    FOREIGN KEY (AcceptedAnswerId) REFERENCES Answers(AnswerId)
);

DROP TABLE IF EXISTS Answers;
CREATE TABLE Answers (
    AnswerId INTEGER,
    UserId INTEGER,
    QuestionId INTEGER,
    CreationDateTime TEXT,
    Score INTEGER,
    CommentCount INTEGER,
    Body TEXT,
    PRIMARY KEY (AnswerId),
    FOREIGN KEY (UserId) REFERENCES Users(UserId),
    FOREIGN KEY (QuestionId) REFERENCES Questions(QuestionId)
);

DROP TABLE IF EXISTS Tags;
CREATE TABLE Tags (
    QuestionId INTEGER,
    Tag TEXT,
    PRIMARY KEY (QuestionId, Tag)
);

DROP TABLE IF EXISTS Users;
CREATE TABLE Users (
    UserId INTEGER,
    CreationDateTime TEXT,
    DisplayName TEXT,
    Reputation INTEGER,
    PRIMARY KEY (UserId)
);

-- empty unused space - useful when recreating the database from a larger one
VACUUM
