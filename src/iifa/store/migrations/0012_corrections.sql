-- Schema version 12: the corrections journal (SPEC 5.7), what a person changed
-- on a node - a rename, aliases, a kind, a merge - with who did it and how to
-- revert it.
--
-- * corrections: one entry per correction that changed something: the
--   operation, the name of the token that made it, what was asked (payload),
--   how to revert it (changes), the revisions of what it changed right before
--   and right after it (before_revs, after_revs), its status (applied |
--   undone). An entry holds content (names, aliases), so
--   forget wipes every entry that mentions the forgotten record (P5).
-- * correction_records: the records an entry touches, so forget finds it by
--   record and an undo knows whom to ask about.

CREATE TABLE IF NOT EXISTS corrections (
    id          TEXT PRIMARY KEY,
    op          TEXT NOT NULL,
    status      TEXT NOT NULL,
    token       TEXT NOT NULL,
    payload     TEXT,
    changes     TEXT,
    before_revs TEXT,
    after_revs  TEXT,
    forgotten   INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    undone_at   TEXT,
    undone_by   TEXT
);
CREATE INDEX IF NOT EXISTS corrections_created ON corrections (created_at, id);

CREATE TABLE IF NOT EXISTS correction_records (
    correction_id TEXT NOT NULL REFERENCES corrections (id) ON DELETE CASCADE,
    record_id     TEXT NOT NULL,
    PRIMARY KEY (correction_id, record_id)
);
CREATE INDEX IF NOT EXISTS correction_records_record ON correction_records (record_id);
