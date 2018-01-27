
-- user freshness is defined as the difference in days
-- between the question creation date and the user creation date
INSERT INTO UserFreshness
SELECT QuestionId, CAST(julianday(Questions.CreationDate) - julianday(Users.CreationDate) AS INTEGER) AS Days
FROM Questions, Users
WHERE Questions.UserId = Users.UserId
AND Days >= 0;

