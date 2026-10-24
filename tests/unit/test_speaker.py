"""Speaker words (SPEC 4.2): a subject that, whole, names the person speaking -
"User", "l'utilisateur", "Пользователь" - and nothing that merely contains one."""

from __future__ import annotations

import pytest

from iifa.ingest.speaker import is_speaker, speaker_words


@pytest.mark.parametrize(
    "subject",
    [
        "User",
        "user",
        "the user",
        "The User",
        "Utilisateur",
        "l'utilisateur",
        "L’utilisateur",
        "l'utilisatrice",
        "el usuario",
        "o usuário",
        "der Benutzer",
        "Nutzer",
        "l'utente",
        "Пользователь",
        "пользователь",
        "المستخدم",
        "用户",
        "使用者",
        "ユーザー",
        "사용자",
        "  User  ",
    ],
)
def test_a_whole_speaker_word_names_the_speaker(subject: str) -> None:
    assert is_speaker(subject)


@pytest.mark.parametrize(
    "subject",
    [
        "User Group Lyon",
        "Karim",
        "the users",
        "I",
        "me",
        "moi",
        "я",
        "",
        "the",
        "Userland",
        "utilisateur principal du serveur",
    ],
)
def test_anything_else_is_an_ordinary_subject(subject: str) -> None:
    assert not is_speaker(subject)


def test_the_words_cover_every_script_the_statements_may_be_written_in() -> None:
    words = speaker_words()
    for expected in ("user", "utilisateur", "пользователь", "المستخدم", "用户", "ユーザー"):
        assert expected in words
