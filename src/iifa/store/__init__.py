"""One encrypted SQLite file (SPEC section 7.1)."""

from iifa.store.connection import WrongKeyError, open_database
from iifa.store.migrate import SCHEMA_VERSION, migrate
from iifa.store.repo import NotFound, ScopeMismatch, Store

__all__ = [
    "SCHEMA_VERSION",
    "NotFound",
    "ScopeMismatch",
    "Store",
    "WrongKeyError",
    "migrate",
    "open_database",
]
