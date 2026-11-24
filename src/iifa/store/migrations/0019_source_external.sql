-- Schema version 19: what content from outside the conversation a source
-- held (P6): none, only parts the host vouched for, or at least one
-- it did not; and, in the agent perspective, the activity a
-- rule of method was learnt during and the activity a session is doing.
--
-- The parts' offsets are not kept, yet a session's close writes from a
-- source later; the agent perspective's rule needs to know what
-- the turn held.
--
-- An activity is a link beside a fact's subject, never its subject: a `task`
-- node a turn named in the owner's words. A rule has one, a session's
-- episode one per activity of its session. A session keeps the activities
-- its turns named, the last named being the current one, so that a rule read
-- in a later turn is linked to what the owner asked. Forgetting the fact,
-- the node or the session deletes the link (P5).

ALTER TABLE sources ADD COLUMN external TEXT NOT NULL DEFAULT 'none';

CREATE TABLE fact_activities (
    fact_id     TEXT NOT NULL REFERENCES facts (id) ON DELETE CASCADE,
    activity_id TEXT NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, activity_id)
) WITHOUT ROWID;
CREATE INDEX fact_activities_activity ON fact_activities (activity_id);

CREATE TABLE session_activities (
    session_id  TEXT NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    activity_id TEXT NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    PRIMARY KEY (session_id, activity_id)
) WITHOUT ROWID;
