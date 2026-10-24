-- Schema version 11: a node's lexical document holds its name and aliases only
-- (SPEC 6.1: a summary is model-written text, shown in handles, never ranked by
-- recall). The document lengths come from iifa's tokenizer, which SQL cannot
-- run: this marks the nodes for a reindex that the store does once, at open,
-- in one transaction, then clears the mark.

INSERT INTO service_state (key, value) VALUES ('reindex_nodes', 'names_only')
    ON CONFLICT (key) DO UPDATE SET value = excluded.value;
