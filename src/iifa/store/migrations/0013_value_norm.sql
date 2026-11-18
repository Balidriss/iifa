-- Schema version 13: a value's comparison form (SPEC 5.4).
--
-- With [extraction] value_norm on, an extracted fact keeps the canonical form
-- the model gave for its value. It is read only to compare two values across
-- languages: never shown, never indexed, never sent to a model. It is not part
-- of the fact record, and forget deletes it with the row (P5).

ALTER TABLE facts ADD COLUMN value_norm TEXT;
