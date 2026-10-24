-- Schema version 1: nodes, facts, edges, sources, their scope sets, and the lexical index.
-- Timestamps are ISO 8601 in UTC. A record belongs to one or more scope paths
-- (record_scopes); it is visible when any of them is visible to the reader.

CREATE TABLE nodes (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    name        TEXT NOT NULL,
    name_norm   TEXT NOT NULL,
    summary     TEXT NOT NULL DEFAULT '',
    external_id TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    archived_at TEXT
);
CREATE INDEX nodes_name ON nodes (name_norm);
CREATE INDEX nodes_external ON nodes (external_id);

CREATE TABLE node_aliases (
    node_id    TEXT NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    alias      TEXT NOT NULL,
    alias_norm TEXT NOT NULL,
    PRIMARY KEY (node_id, alias_norm)
);
CREATE INDEX node_aliases_norm ON node_aliases (alias_norm);

CREATE TABLE sources (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    text        TEXT,
    text_sha256 TEXT NOT NULL,
    reference   TEXT,
    external_id TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX sources_external ON sources (external_id);

CREATE TABLE facts (
    id            TEXT PRIMARY KEY,
    node_id       TEXT REFERENCES nodes (id) ON DELETE CASCADE,
    statement     TEXT NOT NULL,
    attribute     TEXT,
    value         TEXT,
    valid_from    TEXT,
    valid_to      TEXT,
    origin        TEXT NOT NULL,
    confidence    REAL NOT NULL DEFAULT 1.0,
    source_id     TEXT REFERENCES sources (id) ON DELETE CASCADE,
    excerpt_start INTEGER,
    excerpt_end   INTEGER,
    superseded_by TEXT,
    pinned        INTEGER NOT NULL DEFAULT 0,
    flags         TEXT NOT NULL DEFAULT '',
    external_id   TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    archived_at   TEXT
);
CREATE INDEX facts_node ON facts (node_id);
CREATE INDEX facts_source ON facts (source_id);
CREATE INDEX facts_external ON facts (external_id);

CREATE TABLE edges (
    id           TEXT PRIMARY KEY,
    src          TEXT NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    dst          TEXT NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    type         TEXT NOT NULL,
    weight       REAL NOT NULL DEFAULT 1.0,
    status       TEXT NOT NULL,
    origin       TEXT NOT NULL,
    source_id    TEXT REFERENCES sources (id) ON DELETE SET NULL,
    last_used_at TEXT,
    use_count    INTEGER NOT NULL DEFAULT 0,
    valid_from   TEXT,
    valid_to     TEXT,
    external_id  TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    archived_at  TEXT
);
CREATE INDEX edges_src ON edges (src);
CREATE INDEX edges_dst ON edges (dst);
CREATE INDEX edges_external ON edges (external_id);

-- The scope set of every record (node, fact, edge or source), at least one row each.
CREATE TABLE record_scopes (
    record_id TEXT NOT NULL,
    scope     TEXT NOT NULL,
    PRIMARY KEY (record_id, scope)
);
CREATE INDEX record_scopes_scope ON record_scopes (scope, record_id);

-- The lexical index. lex_docs owns every FTS row: which record it describes,
-- its length in tokens, and whether it is currently hidden from recall
-- (archived, closed, or belonging to an archived node). lex_scopes mirrors the
-- record's scope set keyed by the integer rowid, and lex_scope_stats keeps the
-- per-scope document count and token total the BM25 normalisation needs, so a
-- recall never scans the corpus for statistics (P9 and SPEC 7.4).
CREATE TABLE lex_docs (
    rowid  INTEGER PRIMARY KEY,
    doc_id TEXT NOT NULL UNIQUE,
    kind   TEXT NOT NULL,
    ntok   INTEGER NOT NULL,
    hidden INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE lex_scopes (
    doc   INTEGER NOT NULL REFERENCES lex_docs (rowid) ON DELETE CASCADE,
    scope TEXT NOT NULL,
    PRIMARY KEY (doc, scope)
);
CREATE INDEX lex_scopes_scope ON lex_scopes (scope, doc);

CREATE TABLE lex_scope_stats (
    scope  TEXT PRIMARY KEY,
    docs   INTEGER NOT NULL DEFAULT 0,
    tokens INTEGER NOT NULL DEFAULT 0
);

CREATE VIRTUAL TABLE lex USING fts5 (text, tokenize = 'unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE lex_vocab USING fts5vocab ('lex', 'instance');
