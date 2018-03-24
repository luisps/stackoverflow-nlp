
-- filtering or aggregating based on date is a common use case
CREATE INDEX Questions_CreationDateTime_Idx ON Questions(CreationDateTime);
CREATE INDEX Answers_CreationDateTime_Idx ON Answers(CreationDateTime);

-- searching for answers based on their QuestionId is a common use case
CREATE INDEX Answers_QuestionId_Idx ON Answers(QuestionId);

-- run analyze to create summary statistics that aid the query optimizer
ANALYZE;
