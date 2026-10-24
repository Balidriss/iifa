-- Schema version 5: vectors (SPEC 6, 6.1), tier 3.
--
-- One row per embedded facet of a record: a fact's `text` (its statement), a
-- node's `name` (name and aliases) and `summary`. The vector is stored
-- unit-normalised as float32 bytes, inside the encrypted file like everything
-- else, stamped with the model, the dimension and the SHA-256 of the embedded
-- text. A vector whose model differs from the configured one is stale: never
-- compared, recomputed in the background. A change of the embedded text deletes
-- the row (it is computed again). `forget` deletes a record's rows (P5).

CREATE TABLE vectors (
    record_id   TEXT NOT NULL,
    facet       TEXT NOT NULL,
    model       TEXT NOT NULL,
    dimension   INTEGER NOT NULL,
    text_sha256 TEXT NOT NULL,
    vector      BLOB NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (record_id, facet)
);
CREATE INDEX vectors_model ON vectors (model, facet);
