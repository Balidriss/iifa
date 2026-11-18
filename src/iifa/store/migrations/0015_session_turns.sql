-- Schema version 15: the turn trace of a session (SPEC 5.8). For every turn
-- of a named session read as sincere, the person's own words, bounded, so
-- that the close can say what the session was about. Never shown to the
-- extraction, never recalled, never exported; deleted at the close.

CREATE TABLE IF NOT EXISTS session_turns (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    source_id   TEXT REFERENCES sources (id) ON DELETE CASCADE,
    excerpt     TEXT NOT NULL,              -- the owner parts of the turn, bounded
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS session_turns_by_session ON session_turns (session_id, created_at, id);
CREATE INDEX IF NOT EXISTS session_turns_by_source ON session_turns (source_id);
