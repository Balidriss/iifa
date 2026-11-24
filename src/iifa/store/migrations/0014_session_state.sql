-- Schema version 14: session state. What the service keeps for one
-- session while it runs: the frame an earlier turn opened and the
-- facts that hold only for the work in progress. Never recalled, never
-- exported; consolidated, then deleted, when the session ends or goes idle.

CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    session_key   TEXT NOT NULL,              -- the host's session id; '' when it sent none
    scopes        TEXT NOT NULL,              -- the scope set, sorted, as a JSON list
    frame         TEXT,                       -- the frame an earlier turn opened, or NULL
    frame_episode INTEGER NOT NULL DEFAULT 0, -- that frame's episode is written
    rev           INTEGER NOT NULL DEFAULT 0, -- grows with every change of the row
    started_at    TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL,
    closing_job   TEXT,                       -- the job queued to close it, once queued
    UNIQUE (session_key, scopes)
);

CREATE TABLE IF NOT EXISTS session_facts (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    source_id     TEXT REFERENCES sources (id) ON DELETE CASCADE,
    duration      TEXT NOT NULL,              -- task | unsure
    statement     TEXT NOT NULL,
    subject_norm  TEXT,                       -- the subject, normalised (forget by name)
    provenance    TEXT,                       -- owner | assistant | external (P6)
    excerpt_start INTEGER,
    excerpt_end   INTEGER,
    fact          TEXT NOT NULL,              -- the proposed fact, as JSON
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS session_facts_by_session ON session_facts (session_id, created_at, id);
CREATE INDEX IF NOT EXISTS session_facts_by_subject ON session_facts (subject_norm);
CREATE INDEX IF NOT EXISTS sessions_by_last_seen ON sessions (last_seen_at);
