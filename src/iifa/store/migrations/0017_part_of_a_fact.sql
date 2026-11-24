-- Schema version 17: where an external fact was read (P6).
--
-- Beside facts.provenance, an extracted fact from an external part keeps that
-- part's kind (tool | web | document | other), the short form of its ref (a
-- host or a file name, never a whole URL) and the day it was read, so that a
-- read route can say where the fact came from without reading its source. A
-- session note keeps the same three, for the fact its close may write.

ALTER TABLE facts ADD COLUMN part_kind TEXT;
ALTER TABLE facts ADD COLUMN part_ref TEXT;
ALTER TABLE facts ADD COLUMN read_at TEXT;

ALTER TABLE session_facts ADD COLUMN part_kind TEXT;
ALTER TABLE session_facts ADD COLUMN part_ref TEXT;
ALTER TABLE session_facts ADD COLUMN read_at TEXT;
