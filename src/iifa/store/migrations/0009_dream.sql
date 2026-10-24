-- Schema version 9: what the dream needs (SPEC 4.4, 5.6).
--
-- * rev on nodes, facts and edges: a counter that grows with every change to the
--   record, kept by triggers so that no write path can forget it - an UPDATE of
--   the row (content, life, flags, links), an alias added or removed, a scope
--   added to or removed from its set. A dream compares revisions to know that
--   what it read is still there (rule 8) and that nothing written since depends
--   on what it would undo (rule 4). The triggers only fire when the UPDATE left
--   rev alone, so their own UPDATE does not fire them again.
-- * nodes.merged_into: the node a dream merged it into (it stays, archived).
-- * facts.consolidated_into: the fact a dream consolidated it into (not
--   superseded_by: a consolidation is not a change of value).
-- * dream_runs, dream_ops, dream_op_records: the runs and their journal. An entry
--   holds content (statements, summaries, the model's reason): forget wipes every
--   entry that mentions the forgotten record (P5). dream_op_records lists the
--   records an entry touches, so forget and the review queue find it by record.
-- * dream_cursors: where the next run of a scope starts.

ALTER TABLE nodes ADD COLUMN rev INTEGER NOT NULL DEFAULT 1;
ALTER TABLE facts ADD COLUMN rev INTEGER NOT NULL DEFAULT 1;
ALTER TABLE edges ADD COLUMN rev INTEGER NOT NULL DEFAULT 1;
ALTER TABLE nodes ADD COLUMN merged_into TEXT;
ALTER TABLE facts ADD COLUMN consolidated_into TEXT;
CREATE INDEX nodes_merged_into ON nodes (merged_into);
CREATE INDEX facts_consolidated_into ON facts (consolidated_into);

CREATE TRIGGER nodes_rev AFTER UPDATE ON nodes WHEN NEW.rev = OLD.rev
BEGIN
    UPDATE nodes SET rev = OLD.rev + 1 WHERE id = NEW.id;
END;
CREATE TRIGGER facts_rev AFTER UPDATE ON facts WHEN NEW.rev = OLD.rev
BEGIN
    UPDATE facts SET rev = OLD.rev + 1 WHERE id = NEW.id;
END;
CREATE TRIGGER edges_rev AFTER UPDATE ON edges WHEN NEW.rev = OLD.rev
BEGIN
    UPDATE edges SET rev = OLD.rev + 1 WHERE id = NEW.id;
END;
CREATE TRIGGER node_aliases_rev_insert AFTER INSERT ON node_aliases
BEGIN
    UPDATE nodes SET rev = rev + 1 WHERE id = NEW.node_id;
END;
CREATE TRIGGER node_aliases_rev_delete AFTER DELETE ON node_aliases
BEGIN
    UPDATE nodes SET rev = rev + 1 WHERE id = OLD.node_id;
END;
CREATE TRIGGER record_scopes_rev_insert AFTER INSERT ON record_scopes
BEGIN
    UPDATE nodes SET rev = rev + 1 WHERE id = NEW.record_id;
    UPDATE facts SET rev = rev + 1 WHERE id = NEW.record_id;
    UPDATE edges SET rev = rev + 1 WHERE id = NEW.record_id;
END;
CREATE TRIGGER record_scopes_rev_delete AFTER DELETE ON record_scopes
BEGIN
    UPDATE nodes SET rev = rev + 1 WHERE id = OLD.record_id;
    UPDATE facts SET rev = rev + 1 WHERE id = OLD.record_id;
    UPDATE edges SET rev = rev + 1 WHERE id = OLD.record_id;
END;

CREATE TABLE dream_runs (
    id               TEXT PRIMARY KEY,
    scope            TEXT NOT NULL,
    status           TEXT NOT NULL,
    dry_run          INTEGER NOT NULL DEFAULT 0,
    operations       TEXT NOT NULL,
    budget           TEXT NOT NULL,
    policy           TEXT NOT NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    job_id           TEXT,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT,
    probes_before    TEXT,
    state            TEXT,
    report           TEXT
);
CREATE INDEX dream_runs_status ON dream_runs (status, scope);

CREATE TABLE dream_ops (
    id           TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES dream_runs (id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,
    op           TEXT NOT NULL,
    risk         TEXT NOT NULL,
    route        TEXT,
    status       TEXT NOT NULL,
    reason       TEXT,
    payload      TEXT,
    reason_given TEXT,
    snapshot     TEXT,
    changes      TEXT,
    after_revs   TEXT,
    forgotten    INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    decided_at   TEXT,
    undone_at    TEXT
);
CREATE INDEX dream_ops_run ON dream_ops (run_id, seq);
CREATE INDEX dream_ops_status ON dream_ops (status, created_at);

CREATE TABLE dream_op_records (
    op_id     TEXT NOT NULL REFERENCES dream_ops (id) ON DELETE CASCADE,
    record_id TEXT NOT NULL,
    PRIMARY KEY (op_id, record_id)
);
CREATE INDEX dream_op_records_record ON dream_op_records (record_id);

CREATE TABLE dream_cursors (
    scope      TEXT PRIMARY KEY,
    after      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
