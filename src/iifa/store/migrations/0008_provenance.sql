-- Schema version 8: who said it (P6) and fact proposals (SPEC 4.4b).
--
-- facts.provenance is the part of the source a fact's quote lies in: 'owner',
-- 'assistant' or 'external' (a tool result, a web page, a document); NULL for
-- remember, imports and every fact written before parts existed. A fact
-- accepted from a fact proposal points at it with proposal_id.
--
-- A fact proposal is a fact about the owner the write path would not write (not
-- the owner's own words). It may have no node (a free speaker fact), so
-- proposals.node_id becomes optional; the rest of the fact rides in payload
-- (JSON), and applied_id names the fact an accepted proposal wrote. SQLite
-- cannot drop a NOT NULL, so the table is rebuilt; every row is kept.

ALTER TABLE facts ADD COLUMN provenance TEXT;
ALTER TABLE facts ADD COLUMN proposal_id TEXT;

CREATE TABLE proposals_v8 (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    node_id       TEXT REFERENCES nodes (id) ON DELETE CASCADE,
    value         TEXT NOT NULL,
    value_norm    TEXT NOT NULL,
    source_id     TEXT REFERENCES sources (id) ON DELETE CASCADE,
    excerpt_start INTEGER,
    excerpt_end   INTEGER,
    confidence    REAL,
    status        TEXT NOT NULL DEFAULT 'pending',
    created_at    TEXT NOT NULL,
    decided_at    TEXT,
    payload       TEXT,
    applied_id    TEXT
);
INSERT INTO proposals_v8 (id, kind, node_id, value, value_norm, source_id, excerpt_start,
                          excerpt_end, confidence, status, created_at, decided_at)
    SELECT id, kind, node_id, value, value_norm, source_id, excerpt_start, excerpt_end,
           confidence, status, created_at, decided_at
    FROM proposals;
DROP TABLE proposals;
ALTER TABLE proposals_v8 RENAME TO proposals;
CREATE INDEX proposals_node ON proposals (node_id, status);
CREATE INDEX proposals_source ON proposals (source_id);
CREATE INDEX proposals_pending ON proposals (status, kind, created_at);
