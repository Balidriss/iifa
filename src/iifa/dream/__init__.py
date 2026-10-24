"""The dream (SPEC 5.6): an offline job in which a model reorganises a scope
while nobody waits on it. The model proposes operations from a closed
catalogue (``catalogue``); the service checks, applies, journals and can undo
every one of them (``apply``, ``journal``), under a review policy by risk
(``policy``), unit by unit, within a budget (``run``)."""
