
WITH RECURSIVE tag_list(Id, Tag, RemainingTags ) AS (
    SELECT Id, NULL AS Tag , SUBSTR(Tags, 2, LENGTH(Tags) - 2) AS RemainingTags FROM Questions
        UNION ALL
    SELECT
        Id,
        CASE
            WHEN INSTR(RemainingTags, '>') > 0 THEN
                SUBSTR(RemainingTags, 0, INSTR(RemainingTags, '>'))
            ELSE
                RemainingTags
        END AS Tag,
        CASE
            WHEN INSTR(RemainingTags, '>') > 0 THEN
                SUBSTR(RemainingTags, INSTR(RemainingTags, '>') + 2)
            ELSE
                NULL
        END AS RemainingTags
    FROM tag_list
    WHERE RemainingTags IS NOT NULL
)
INSERT INTO Tags (QuestionId, Tag)
SELECT Id, Tag FROM tag_list WHERE Tag IS NOT NULL ORDER BY Id;
