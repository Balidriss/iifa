"""Fixtures the security tests share with other folders: the dream app and its
client (SPEC 5.6), defined in ``tests/dream/conftest.py``."""

from tests.dream.conftest import dream_app, dream_client, dream_review

__all__ = ["dream_app", "dream_client", "dream_review"]
