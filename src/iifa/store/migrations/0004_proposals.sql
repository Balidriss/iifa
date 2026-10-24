-- Schema version 4: proposals (SPEC 4.4b), what the write path would not decide alone.
--
-- A proposal carries text taken from, or attached to, a source, so it is a record
-- like the others: its scope set (the source's) lives in record_scopes and every
-- read goes through the view (P9); it is hard-deleted with its node or its source
-- (P5). Only `alias` proposals exist for now: an alias the model attached to a
-- node without it appearing in the source (SPEC 5.1). Resolution and recall never
-- read this table.

CREATE TABLE proposals (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    node_id       TEXT NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    value         TEXT NOT NULL,
    value_norm    TEXT NOT NULL,
    source_id     TEXT REFERENCES sources (id) ON DELETE CASCADE,
    excerpt_start INTEGER,
    excerpt_end   INTEGER,
    confidence    REAL,
    status        TEXT NOT NULL DEFAULT 'pending',
    created_at    TEXT NOT NULL,
    decided_at    TEXT
);
CREATE INDEX proposals_node ON proposals (node_id, status);
CREATE INDEX proposals_source ON proposals (source_id);
