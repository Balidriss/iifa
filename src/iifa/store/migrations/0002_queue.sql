-- Schema version 2: the persisted ingestion queue and the operational state that
-- must survive a restart (SPEC 7.2, P11). A job is a row before the request that
-- created it answers; a row found `running` at startup was interrupted by a
-- crash and goes back to `pending`. Payloads live in the same encrypted file as
-- everything else and are cleared as soon as a job is done (P5).

CREATE TABLE queue (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    id          TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL,
    payload     TEXT,
    scopes      TEXT NOT NULL,
    session_id  TEXT,
    status      TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    created_at  TEXT NOT NULL,
    not_before  TEXT,
    started_at  TEXT,
    finished_at TEXT
);
CREATE INDEX queue_status ON queue (status, seq);
CREATE INDEX queue_session ON queue (session_id);

-- Key/value state the service itself writes: the background mode set at runtime,
-- so a paused service comes back paused.
CREATE TABLE service_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
