-- Schema version 3: what the model-driven write path needs (SPEC 5.1-5.4, P6, P7, P12).
--
-- * facts.language: the language the statement is written in, detected from the
--   text (P12); NULL when undetermined. facts.conflicts_with: the explicit fact an
--   extracted fact contradicts without overriding it (P6): both stay, the new one
--   is flagged `conflict`, and this column says with what.
-- * sources.session_id lets a session end find what the session touched;
--   sources.language is the dominant language of the source text.
-- * nodes.language is the language of the maintained summary; nodes.flags carries
--   `language_mismatch` when a summary could not be written in that language.
-- * edge_evidence: the distinct sources an edge was observed in (SPEC 4.3
--   `evidence`); an edge proposed by extraction is confirmed once it has been
--   re-observed in enough independent sources (SPEC 5.2).
-- * edges.decayed_at: when the stored weight was last decayed to, so that decay
--   (SPEC 5.3) is a function of time and running it twice changes nothing.

ALTER TABLE facts ADD COLUMN language TEXT;
ALTER TABLE facts ADD COLUMN conflicts_with TEXT;
ALTER TABLE sources ADD COLUMN session_id TEXT;
ALTER TABLE sources ADD COLUMN language TEXT;
ALTER TABLE nodes ADD COLUMN language TEXT;
ALTER TABLE nodes ADD COLUMN flags TEXT NOT NULL DEFAULT '';
ALTER TABLE edges ADD COLUMN decayed_at TEXT;

CREATE INDEX sources_session ON sources (session_id);
CREATE INDEX facts_attribute ON facts (node_id, attribute);

CREATE TABLE edge_evidence (
    edge_id   TEXT NOT NULL REFERENCES edges (id) ON DELETE CASCADE,
    source_id TEXT NOT NULL REFERENCES sources (id) ON DELETE CASCADE,
    PRIMARY KEY (edge_id, source_id)
);
CREATE INDEX edge_evidence_source ON edge_evidence (source_id);

-- Every existing edge with a source is evidenced by that source.
INSERT OR IGNORE INTO edge_evidence (edge_id, source_id)
    SELECT id, source_id FROM edges WHERE source_id IS NOT NULL;
