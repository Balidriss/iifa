-- Schema version 10: the dream's merge adjudications (SPEC 5.6).
--
-- * dream_adjudications: one row per pair a dream run asked the model about
--   (never in a dry run). The pair is stored ordered (a < b). `fingerprint`
--   is the revisions of both nodes and their live facts when the model read
--   them: the pair is not asked again until one of them moves. `reason` is the
--   model's words, shown to the reviewer beside the pair; it may quote what it
--   read, so forget deletes every adjudication that read a forgotten record
--   (P5). `proposal_id` is the merge proposal an `unsure` verdict created
--   (undoing the run deletes it while it is pending); `op_id` the merge_nodes
--   operation a `same` verdict submitted.
-- * dream_adjudication_records: the records an adjudication read (both nodes
--   and their live facts), so forget finds it by record.

CREATE TABLE dream_adjudications (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES dream_runs (id) ON DELETE CASCADE,
    a           TEXT NOT NULL,
    b           TEXT NOT NULL,
    verdict     TEXT NOT NULL,
    reason      TEXT,
    fingerprint TEXT NOT NULL,
    proposal_id TEXT,
    op_id       TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX dream_adjudications_pair ON dream_adjudications (a, b, created_at);
CREATE INDEX dream_adjudications_run ON dream_adjudications (run_id);

CREATE TABLE dream_adjudication_records (
    adjudication_id TEXT NOT NULL REFERENCES dream_adjudications (id) ON DELETE CASCADE,
    record_id       TEXT NOT NULL,
    PRIMARY KEY (adjudication_id, record_id)
);
CREATE INDEX dream_adjudication_records_record ON dream_adjudication_records (record_id);
