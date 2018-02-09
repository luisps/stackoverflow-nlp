
-- indexes for fast access when manipulating dates
CREATE INDEX Questions_CreationDate_Idx ON Questions(CreationDate);
CREATE INDEX Answers_CreationDate_Idx ON Answers(CreationDate);
