-- Schema version 16: who started a dream, and an edge's decision revision
-- (SPEC 4.4, 8.6).
--
-- * dream_runs.started_by: the name of the token that started a run, so that
--   a list of the runs says it; null when unknown. The index serves that list,
--   newest first.
-- * edges.decision_rev: grows only when what a person decides about an edge
--   changes - status, valid_to, archived_at, src, dst, type, its scope set -
--   never on weight, use counters or evidence. One trigger keeps both counters
--   (a second trigger's own UPDATE would move rev once more); rev still grows
--   on every change.

ALTER TABLE dream_runs ADD COLUMN started_by TEXT;
CREATE INDEX IF NOT EXISTS dream_runs_created ON dream_runs (created_at, id);

ALTER TABLE edges ADD COLUMN decision_rev INTEGER NOT NULL DEFAULT 1;

DROP TRIGGER IF EXISTS edges_rev;
CREATE TRIGGER edges_rev AFTER UPDATE ON edges WHEN NEW.rev = OLD.rev
BEGIN
    UPDATE edges SET
        rev = OLD.rev + 1,
        decision_rev = OLD.decision_rev + (
            NEW.status IS NOT OLD.status
            OR NEW.valid_to IS NOT OLD.valid_to
            OR NEW.archived_at IS NOT OLD.archived_at
            OR NEW.src IS NOT OLD.src
            OR NEW.dst IS NOT OLD.dst
            OR NEW.type IS NOT OLD.type
        )
    WHERE id = NEW.id;
END;

DROP TRIGGER IF EXISTS record_scopes_rev_insert;
CREATE TRIGGER record_scopes_rev_insert AFTER INSERT ON record_scopes
BEGIN
    UPDATE nodes SET rev = rev + 1 WHERE id = NEW.record_id;
    UPDATE facts SET rev = rev + 1 WHERE id = NEW.record_id;
    UPDATE edges SET rev = rev + 1, decision_rev = decision_rev + 1 WHERE id = NEW.record_id;
END;

DROP TRIGGER IF EXISTS record_scopes_rev_delete;
CREATE TRIGGER record_scopes_rev_delete AFTER DELETE ON record_scopes
BEGIN
    UPDATE nodes SET rev = rev + 1 WHERE id = OLD.record_id;
    UPDATE facts SET rev = rev + 1 WHERE id = OLD.record_id;
    UPDATE edges SET rev = rev + 1, decision_rev = decision_rev + 1 WHERE id = OLD.record_id;
END;
