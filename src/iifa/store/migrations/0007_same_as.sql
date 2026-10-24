-- Schema version 7: equivalence groups of facts (SPEC 5.4).
--
-- A fact stating the same value as another one in another form or language
-- ("Lyon" and "Лион", "mars 2026" and "March 2026") points at the group's first
-- fact with same_as; the first fact's own same_as is NULL. Recall injects one
-- fact per group. forget of the first fact re-roots the group on the next one,
-- so no reference to a forgotten fact remains (P5).

ALTER TABLE facts ADD COLUMN same_as TEXT;
CREATE INDEX facts_same_as ON facts (same_as);
