
-- question freshness is defined as the difference in days
-- between a question's creation date and the creation date
-- of the user who posted that question
-- it is a measure of the "newness" of a user for each question
DROP TABLE IF EXISTS QuestionFreshness;
CREATE TABLE QuestionFreshness (
    QuestionId INTEGER,
    ElapsedDays INTEGER,
    PRIMARY KEY (QuestionId),
    FOREIGN KEY (QuestionId) REFERENCES Questions(QuestionId)
);

INSERT INTO QuestionFreshness
SELECT QuestionId, CAST(julianday(Questions.CreationDateTime) - julianday(Users.CreationDateTime) AS INTEGER) AS ElapsedDays
FROM Questions, Users
WHERE Questions.UserId = Users.UserId
AND ElapsedDays >= 0;

-- answer freshness is defined as the difference in days
-- between an answer's creation date and the creation date
-- of the question that answer belongs to
-- it is a measure of the time it takes between a question
-- being posted and that same question being answered
DROP TABLE IF EXISTS AnswerFreshness;
CREATE TABLE AnswerFreshness (
    QuestionId INTEGER,
    IsAcceptedAnswer INTEGER,
    ElapsedDays REAL,
    FOREIGN KEY (QuestionId) REFERENCES Questions(QuestionId)
);

INSERT INTO AnswerFreshness
SELECT Questions.QuestionId, Questions.AcceptedAnswerId = Answers.AnswerId,
julianday(Answers.CreationDateTime) - julianday(Questions.CreationDateTime)
FROM Questions, Answers
WHERE Questions.QuestionId = Answers.QuestionId
AND Questions.AcceptedAnswerId IS NOT NULL
ORDER BY Questions.QuestionId;

DELETE FROM AnswerFreshness
WHERE QuestionId IN (
    SELECT QuestionId
    FROM AnswerFreshness
    GROUP BY QuestionId
    HAVING SUM(IsAcceptedAnswer) != 1 OR MIN(ElapsedDays) < 0.0
);

