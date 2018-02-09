
WITH RECURSIVE tag_list(QuestionId, Tag, RemainingTags ) AS (
    SELECT QuestionId, NULL AS Tag , SUBSTR(Tags, 2, LENGTH(Tags) - 2) AS RemainingTags FROM Questions
        UNION ALL
    SELECT
        QuestionId,
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
SELECT QuestionId, Tag FROM tag_list WHERE Tag IS NOT NULL ORDER BY QuestionId;
