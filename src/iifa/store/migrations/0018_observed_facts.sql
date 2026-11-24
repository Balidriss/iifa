-- Schema version 18: what was observed is true at a date.
--
-- An observed fact (the state of something that changes on its own, seen at
-- one moment: a folder's content, a service's status) is flagged `observed`
-- and keeps the day it was last observed. A later observation of the same
-- value moves that day; recall ranks it lower as it ages.

ALTER TABLE facts ADD COLUMN observed_at TEXT;
