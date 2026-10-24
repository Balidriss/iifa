-- Schema version 6: probe sets (SPEC 4.7, 11.2), what the dream's rollback rule
-- measures recall with.
--
-- A set belongs to exactly one scope and holds questions with the ids of the
-- records that answer them. Questions are content: they live in the encrypted
-- file and never reach a log (P5). A set is not memory: recall, expansion and
-- dreams never read these tables. `forget` deletes a record's rows from
-- probe_expected (P5: no trace); deleting a set deletes its probes.

CREATE TABLE probe_sets (
    id            TEXT PRIMARY KEY,
    scope         TEXT NOT NULL,
    name          TEXT NOT NULL,
    budget_tokens INTEGER NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX probe_sets_scope ON probe_sets (scope, id);

CREATE TABLE probes (
    id       TEXT PRIMARY KEY,
    set_id   TEXT NOT NULL REFERENCES probe_sets (id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    query    TEXT NOT NULL
);
CREATE INDEX probes_set ON probes (set_id, position);

CREATE TABLE probe_expected (
    probe_id  TEXT NOT NULL REFERENCES probes (id) ON DELETE CASCADE,
    record_id TEXT NOT NULL,
    position  INTEGER NOT NULL,
    PRIMARY KEY (probe_id, record_id)
);
CREATE INDEX probe_expected_record ON probe_expected (record_id);
