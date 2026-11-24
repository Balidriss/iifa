"""The service's own reading of secrets (P5): what it finds, what it
leaves alone, and how a quote, a trace or an answer loses a value. The
reading by wording is ``detect_secrets_by_wording`` (off by default): the
``collect`` and ``detect`` of this file turn it on; the tests of the default
call ``credentials`` directly."""

from __future__ import annotations

import pytest

from iifa.ingest import credentials
from iifa.ingest.schema import (
    MAX_SECRETS,
    SECRET_CHARS,
    Answer,
    ProposedActivity,
    ProposedFact,
    parse_answer,
)


def collect(text: str, answers: list[Answer], **extra: object) -> tuple[str, ...]:
    return credentials.collect(text, answers, wording=True, **extra)  # type: ignore[arg-type]


def detect(text: str) -> list[str]:
    return credentials.detect(text, wording=True)


def test_by_default_the_service_reads_shapes_only() -> None:
    """``detect_secrets_by_wording`` is off: no wording is read, so no
    ordinary word is ever taken for a secret; the shapes of keys still are."""
    assert credentials.collect("my password is Hunter22, I moved in 2024", []) == ()
    assert credentials.collect("Boarding pass: AF1234", []) == ()
    key = "sk-abcdefghijklmnopqrstuv"
    assert credentials.collect(f"the key is {key}", []) == (key,)
    assert credentials.collect("Authorization: Bearer abcdefghijklmnop1234", []) == (
        "abcdefghijklmnop1234",
    )
    assert credentials.collect("clone https://ana:s3cr3tpass@git.example.org/r.git", []) == (
        "s3cr3tpass",
    )
    # the model's reading is always on
    said = Answer(secrets=["sunflower"])
    assert credentials.collect("the wifi takes sunflower", [said]) == ("sunflower",)


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("my password there changed, the new one is `Tr0ub4dor&3`.", "Tr0ub4dor&3"),
        ("Password: hunter22", "hunter22"),
        ("the wifi password is now Gr33nTea, tell nobody", "Gr33nTea"),
        ('the passphrase is "correct horse battery"', "correct horse battery"),
        ("Your verification code 482913 expires soon", "482913"),
        ("my PIN is 4071", "4071"),
        ("mon mot de passe est désormais Soleil2024", "Soleil2024"),
        ("le mdp : x9!kQ", "x9!kQ"),
        ("das Passwort ist Wint3rzeit", "Wint3rzeit"),
        ("la contraseña es Gato#77", "Gato#77"),
        ("мой пароль это Kot2024", "Kot2024"),
        ("the API key is sk-abcdefghijklmnopqrstuv", "sk-abcdefghijklmnopqrstuv"),
        (
            "use ghp_abcdefghijklmnopqrstuvwxyz0123 for the repo",
            "ghp_abcdefghijklmnopqrstuvwxyz0123",
        ),
        ("key AKIAABCDEFGHIJKLMNOP here", "AKIAABCDEFGHIJKLMNOP"),
        ("Authorization: Bearer abcdefghijklmnop1234", "abcdefghijklmnop1234"),
        ("clone https://ana:s3cr3tpass@git.example.org/r.git", "s3cr3tpass"),
        ("the door code is 4071", "4071"),
        ("the alarm code is 482913, do not share it", "482913"),
        ("le code est 4071", "4071"),
        ("pw: Hunter22", "Hunter22"),
        ("the 2FA code was 123 456", "123 456"),
        ("password -> Hunter22x", "Hunter22x"),
        ("the API token is 12ab34cd56", "12ab34cd56"),
        ("DB_PASSWORD=Xk9pq2zz", "Xk9pq2zz"),
        ("I changed my shop password to Xy7#kq9", "Xy7#kq9"),
        ("set the password to S3cret#1", "S3cret#1"),
        ("my new password will be Hunter22", "Hunter22"),
        ("j'ai changé mon mot de passe en Soleil2024", "Soleil2024"),
        ("export OPENAI_API_KEY=abc123def456", "abc123def456"),
        ("The door code is 12345678.", "12345678"),
        ("verification code 12345678", "12345678"),
        ("My iPhone code is 482913.", "482913"),
        ("my password changed, the new one is Hunter22x", "Hunter22x"),
    ],
)
def test_the_service_finds_a_secret_by_its_wording_or_its_shape(text: str, secret: str) -> None:
    assert secret in detect(text)
    assert collect(text, []) == (secret,)


@pytest.mark.parametrize(
    "text",
    [
        "the password is wrong, I will reset it tomorrow",
        "I changed my password yesterday.",
        "the token was refused.",
        "the password is well-known to the team",
        "my zip code 75001 and the area code 0212",
        "the code review is at 1400",
        "Sam likes passwords. Later: 1234 people came",
        "my password manager is KeePassXC",
        "my password manager is 1Password",
        "the token limit is 8192 for this model",
        "the tokens 2024 summit, I moved in 2024",
        "mon gestionnaire de mots de passe est Bitwarden2",
        "the zip code is 75001",
        "le code postal est 75001",
        "my bus pass is valid until 2025-03",
        "the max tokens is 4096 for this model",
        "Usage tokens: 12000 today",
        "Meet at the pin: 10am",
        "I changed my password and my flight is AF1234",
        "the token is 98765",
        "The code is 1500 lines and the budget is 1500 euros",
        "I forgot my password: IKEA account. I bought a sofa at IKEA.",
        "the password_changed flag is Set2",
        "TOKEN_LIMIT=4096",
        "SESSION_TOKEN_TTL=3600",
        "PASSWORD_MAX_LENGTH: 1024",
        "the code: 2024-11-05",
        "Project code 2024 kicks off; Alice joined in 2024.",
        "my password to the bank changed",
    ],
)
def test_ordinary_wording_is_no_secret(text: str) -> None:
    assert collect(text, []) == ()


def test_a_value_ending_in_punctuation_is_masked_whole() -> None:
    text = "my password is Hunter22! Do not lose it."
    secrets = collect(text, [])
    assert secrets == ("Hunter22!", "Hunter22")
    assert credentials.distinct(secrets) == 1
    assert collect("the password is Hunter22, I think", []) == ("Hunter22",)
    assert credentials.mask(text, secrets) == "my password is [secret] Do not lose it."


@pytest.mark.parametrize(
    "text", ["My password is **Hunter22**.", "my password is (Hunter22)", "password: <Hunter22>"]
)
def test_markdown_or_brackets_around_a_value_are_not_part_of_it(text: str) -> None:
    secrets = collect(text, [])
    assert "Hunter22" in secrets
    assert "Hunter22" not in credentials.mask(text, secrets)


def test_every_value_of_a_list_is_found() -> None:
    assert set(collect("recovery codes: a1b2c-3d4e5 f6g7h-8i9j0", [])) == {
        "a1b2c-3d4e5",
        "f6g7h-8i9j0",
    }
    assert set(collect("my passwords are Hunter22 and Tiger33", [])) == {
        "Hunter22",
        "Tiger33",
    }
    assert collect("the password is Hunter22 for the shop", []) == ("Hunter22",)


def test_more_credential_keys() -> None:
    for key in ("access_code", "door_code", "license_key", "session_token", "pin_number"):
        assert credentials.credential_key(key), key
    assert not credentials.credential_key("zip_code")


def test_a_weak_secret_still_leaves_every_summary_alias_and_quote() -> None:
    answer = parse_answer(
        """{"nodes": [{"name": "Server", "summary": "Its admin password is Xk9#pq22.",
                       "aliases": ["Xk9#pq22", "srv"]}],
            "edges": [{"src": "Alice", "dst": "Server", "type": "uses",
                       "quote": "Alice uses the server, admin password Xk9#pq22"}]}"""
    )
    assert credentials.scrub(answer, ("Xk9#pq22",), weak=("Xk9#pq22",)) == 0
    # an owned fact keeps its words, without the weak value in its quote
    owned = parse_answer(
        '{"facts": [{"subject": "Alice", "statement": "Alice moved to Lisbon.",'
        ' "quote": "I moved to Lisbon last week"}]}'
    )
    assert credentials.scrub(owned, ("Lisbon",), weak=("Lisbon",), owned=lambda f: True) == 0
    assert owned.facts[0].quote == "I moved to"
    assert [(n.summary, n.aliases) for n in answer.nodes] == [("", ["srv"])]
    assert answer.edges[0].quote == "Alice uses the server, admin password"


def test_a_weak_secret_never_empties_the_owners_activity() -> None:
    answer = parse_answer('{"activity": {"name": "planning a trip", "quote": "Plan Lisbon"}}')
    assert credentials.scrub(answer, ("Lisbon",), weak=("Lisbon",)) == 0
    assert answer.activity is not None and answer.activity.quote == "Plan Lisbon"


def test_a_passphrase_under_a_credential_key_is_masked_whole() -> None:
    said = Answer(
        facts=[
            ProposedFact(
                statement="A s.", attribute="passphrase", value="correct horse battery staple"
            )
        ]
    )
    text = "my new passphrase is correct horse battery staple"
    assert collect(text, [said]) == ("correct horse battery staple",)
    assert credentials.scrub(said, collect(text, [said])) == 1


@pytest.mark.parametrize(
    "text",
    [
        "I emailed the password to alice@example.com",
        "Send the API key to https://hooks.example.com/abc",
        "J'ai changé mon mot de passe en 2019",
        "Mot de passe changé, Alice a 2024 euros",
        "my API tokens are 20000 per month",
        "Access tokens: 3600 per hour max",
        "reset code is 9876 lines",
    ],
)
def test_a_recipient_a_year_or_a_count_is_no_secret(text: str) -> None:
    assert collect(text, []) == ()


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("The password to iCloud is Abc123x and I use it daily.", "Abc123x"),
        ("the password to GitHub is hunter2!x", "hunter2!x"),
    ],
)
def test_a_preposition_names_where_the_credential_goes(text: str, secret: str) -> None:
    assert collect(text, []) == (secret,)


@pytest.mark.parametrize(
    "text",
    [
        "Run pwd: /home/alice/projects/app",
        "pin the release to 2024",
        "My code review is 2024 based",
    ],
)
def test_a_path_or_a_year_is_no_secret(text: str) -> None:
    assert collect(text, []) == ()


@pytest.mark.parametrize("value", ["too weak", "same as email", "new one", "expires soon"])
def test_a_description_under_a_credential_key_is_the_event(value: str) -> None:
    said = Answer(facts=[ProposedFact(statement="A d.", attribute="password", value=value)])
    assert collect(f"her password was {value}", [said]) == ()
    assert credentials.scrub(said, ()) == 0 and said.facts[0].attribute is None


@pytest.mark.parametrize(
    ("text", "codes"),
    [
        ("recovery codes: 11111111, 22222222", {"11111111", "22222222"}),
        ("backup codes 11111111 22222222 33333333", {"11111111", "22222222", "33333333"}),
        ("door codes: 1234, 5678", {"1234", "5678"}),
    ],
)
def test_a_list_of_digit_codes_is_read_whole(text: str, codes: set[str]) -> None:
    assert set(collect(text, [])) == codes


@pytest.mark.parametrize(
    "text",
    [
        "I'm doing Advent of Code 2024 this December",
        "the code 1984 by Orwell",
        "the password file is /etc/shadow",
        "the password is ~/secrets/key.txt",
    ],
)
def test_a_year_after_a_code_or_a_path_is_no_secret(text: str) -> None:
    assert collect(text, []) == ()


def test_a_connection_string_gives_its_field() -> None:
    secrets = collect("password=Hunter22;user=bob, I typed Hunter22 again", [])
    assert "Hunter22" in secrets
    masked = credentials.mask("password=Hunter22;user=bob, I typed Hunter22 again", secrets)
    assert "Hunter22" not in masked


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("the password is _Hunter22_", "Hunter22"),
        ("password: Hunter22_", "Hunter22"),
        ("pwd: Hunter22", "Hunter22"),
        ("DB_PASS=Hunter22", "Hunter22"),
        ("secret: Hunter22", "Hunter22"),
    ],
)
def test_underscores_and_short_names(text: str, secret: str) -> None:
    secrets = collect(text, [])
    assert secret in secrets
    assert secret not in credentials.mask(text, secrets)


@pytest.mark.parametrize(
    "text",
    [
        "Send the reset code to 0612345678 please",
        "The token endpoint is https://auth.example.com/oauth/token",
        "I reset my password yesterday. Send the invoice to 0612345678",
    ],
)
def test_a_number_or_a_link_where_a_credential_goes_is_no_secret(text: str) -> None:
    assert collect(text, []) == ()


def test_a_long_passphrase_opening_like_a_description_is_still_one() -> None:
    value = "the old oak tree by the river"
    said = Answer(facts=[ProposedFact(statement="A o.", attribute="wifi_password", value=value)])
    text = f"my wifi password is {value}"
    assert collect(text, [said]) == (value,)


def test_a_short_description_in_its_fact_drops_the_fact() -> None:
    said = Answer(
        facts=[
            ProposedFact(
                statement="Alice's password is too weak.", attribute="password", value="too weak"
            )
        ]
    )
    assert collect("Alice said her password was too weak", [said]) == ()
    assert credentials.scrub(said, ()) == 1


def test_a_value_in_brackets_is_masked_with_its_brackets_left() -> None:
    text = "my password is (Hunter22)"
    assert credentials.mask(text, collect(text, [])) == "my password is ([secret])"


@pytest.mark.parametrize(
    "text",
    [
        "Boarding pass: AF1234, gate 12",
        "Ski pass: Val-Thorens2025",
        "the secret: Paris2024 is the venue",
    ],
)
def test_a_ticket_or_a_venue_is_no_secret(text: str) -> None:
    assert collect(text, []) == ()


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("my pass: Hunter22", "Hunter22"),
        ("pass: Hunter22", "Hunter22"),
        ("ACCESS_CODE=482913", "482913"),
        ("door_code: 4071", "4071"),
        ("VERIFICATION_CODE=482913", "482913"),
        ("recovery_code: 48291375", "48291375"),
    ],
)
def test_a_short_word_or_a_code_key_names_its_value(text: str, secret: str) -> None:
    assert secret in collect(text, [])


def test_a_secret_with_a_symbol_at_its_edge_is_masked_against_a_letter() -> None:
    assert credentials.mask("#abc1 x#abc1", ["#abc1"]) == "[secret] x[secret]"
    secrets = collect("password: _Secret1", [])
    assert credentials.mask("password: _Secret1", secrets) == "password: [secret]"


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("家里无线网的密码是向日葵。", "向日葵"),
        ("Wi-Fiのパスワードはひまわりです。", "ひまわり"),
        ("我的新密码是Yk2#pm9Wt。", "Yk2#pm9Wt"),
        ("新しいパスワードはQs8&ln3Vxです。", "Qs8&ln3Vx"),
        ("비밀번호는하늘색입니다", "하늘색"),
        ("รหัสผ่านคือทานตะวัน", "ทานตะวัน"),
    ],
)
def test_a_secret_in_a_script_without_spaces_is_kept_and_masked(text: str, secret: str) -> None:
    """The model names it; written against its neighbours, it is still
    written whole there: kept, and masked."""
    said = Answer(secrets=[secret])
    assert credentials.collect(text, [said]) == (secret,)
    assert secret not in credentials.mask(text, [secret])
    assert "[secret]" in credentials.mask(text, [secret])


def test_a_spaced_script_still_needs_a_whole_word() -> None:
    assert credentials.mask("Karim and Karima", ["Karim"]) == "[secret] and Karima"
    assert credentials.mask("call 12345 now", ["1234"]) == "call 12345 now"
    assert credentials.mask("Карим и Карима", ["Карим"]) == "[secret] и Карима"


def test_a_mask_is_never_masked_again() -> None:
    assert credentials.mask("secret [secret] x", ["secret"]) == "[secret] [secret] x"


def test_only_a_plural_word_reads_a_list() -> None:
    assert collect("my password is Hunter22 Paris2024 was the old one", []) == ("Hunter22",)


def test_named_values_are_the_list_and_the_credential_words() -> None:
    answer = Answer(
        secrets=["marigold"],
        facts=[ProposedFact(statement="A p.", attribute="password", value="changed to Xy7#kq9")],
    )
    assert credentials.named_values(answer) == ["marigold", "Xy7#kq9"]


def test_a_quoted_value_with_an_apostrophe_inside_is_read_as_a_run() -> None:
    assert collect("the password is 'Don't4get!'", []) == ("Don't4get!",)
    assert collect("my password is 'Hunter's2day'", []) == ("Hunter's2day",)


def test_a_private_key_block_is_found_whole() -> None:
    block = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        + "QUJD" * 300
        + "\n-----END OPENSSH PRIVATE KEY-----"
    )
    assert collect(f"here it is:\n{block}\nthanks", []) == (block,)


def test_a_credential_value_holding_an_event_word_still_gives_its_secret() -> None:
    said = Answer(
        facts=[ProposedFact(statement="A c.", attribute="password", value="changed to Xy7#kq9")]
    )
    assert collect("I changed my password to Xy7#kq9", [said]) == ("Xy7#kq9",)
    assert credentials.scrub(said, ("Xy7#kq9",)) == 1


def test_a_line_break_ends_the_window_and_an_env_name_may_go_on() -> None:
    assert collect("what is the wifi password\nThe router is TP-Link_5G", []) == ()
    assert collect("DB_PASSWORD_PROD=Hunter22x", []) == ("Hunter22x",)


@pytest.mark.parametrize(
    ("attribute", "credential"),
    [
        ("password", True),
        ("shop_password", True),
        ("API key", True),
        ("api.key", True),
        ("card_pin", True),
        ("refresh-token", True),
        ("password_changed", False),
        ("pinned_note", False),
        ("tokenizer", False),
        ("map_pin", False),
        ("favorite_token", False),
        ("access_token", True),
        (None, False),
        ("", False),
    ],
)
def test_the_last_word_of_an_attribute_says_whether_it_holds_a_credential(
    attribute: str | None, credential: bool
) -> None:
    assert credentials.credential_key(attribute) is credential


def test_a_secret_matches_only_as_a_whole_word() -> None:
    assert credentials.written_in("code 1234 sent", "1234")
    assert not credentials.written_in("call 12345 now", "1234")
    assert not credentials.written_in("Karima came", "Karim")
    assert credentials.mask("Karim and Karima", ("Karim",)) == "[secret] and Karima"


def test_collect_keeps_only_what_the_source_writes_longest_first() -> None:
    text = "the pin is 4071 and the router takes plum-tart-4071"
    answer = Answer(
        secrets=["plum-tart-4071", "nowhere", "abc"],
        facts=[
            ProposedFact(statement="A x.", attribute="pin", value="4071"),
            ProposedFact(statement="B y.", attribute="password", value="not in the text"),
        ],
    )
    assert collect(text, [answer]) == ("plum-tart-4071", "4071")
    # an event under a credential attribute is no value: the fact is dropped, the word stays
    event = Answer(facts=[ProposedFact(statement="A z.", attribute="password", value="changed")])
    assert collect("I changed my password yesterday.", [event]) == ()
    # a plain word under a credential attribute drops its fact, but masks
    # nothing: it may be an ordinary word of other facts ("temporary")
    plain = Answer(
        facts=[ProposedFact(statement="A w.", attribute="wifi_password", value="temporary")]
    )
    assert collect("my wifi password is temporary for now", [plain]) == ()
    assert credentials.scrub(plain, ()) == 1 and plain.facts == []
    # an event in several words or with its punctuation is no value either
    for value in ("changed today", "reset.", "changed!"):
        said = Answer(facts=[ProposedFact(statement="A v.", attribute="password", value=value)])
        assert collect(f"my shop password was {value}", [said]) == ()
    masked = credentials.mask(text, collect(text, [answer]))
    assert masked == "the pin is [secret] and the router takes [secret]"


def test_a_quote_keeps_its_longest_piece_without_the_secret() -> None:
    secrets = ("Tr0ub4dor&3",)
    quote = "I changed my password yesterday, the new one is Tr0ub4dor&3. Log in."
    assert (
        credentials.trim_quote(quote, secrets) == "I changed my password yesterday, the new one is"
    )
    assert credentials.trim_quote("it is Tr0ub4dor&3", secrets) is None
    assert credentials.trim_quote("nothing here", secrets) == "nothing here"
    assert credentials.trim_quote(None, secrets) is None


def test_scrub_drops_what_carries_a_secret_and_keeps_the_rest() -> None:
    answer = parse_answer(
        """{"nodes": [{"name": "Shop", "summary": "pass x9!kQ", "aliases": ["x9!kQ", "Le Shop"]},
                     {"name": "x9!kQ"}],
            "facts": [{"subject": "Alice", "statement": "Alice changed her password.",
                       "quote": "changed it, the new one is x9!kQ"},
                      {"subject": "Alice", "statement": "Alice's password is x9!kQ."},
                      {"subject": "Alice", "statement": "Alice has a code.", "attribute": "otp",
                       "value": "renewed today"}],
            "edges": [{"src": "Alice", "dst": "x9!kQ", "type": "knows"},
                      {"src": "Alice", "dst": "Shop", "type": "uses", "quote": "x9!kQ"}],
            "activity": {"name": "logging in", "quote": "log in with x9!kQ please"}}"""
    )
    dropped = credentials.scrub(answer, ("x9!kQ",))
    assert dropped == 3  # a fact, a node, an edge
    assert [f.statement for f in answer.facts] == [
        "Alice changed her password.",
        "Alice has a code.",  # an event under a credential key: kept, without the key
    ]
    assert (answer.facts[1].attribute, answer.facts[1].value) == (None, None)
    assert answer.facts[0].quote == "changed it, the new one is"
    assert [(n.name, n.aliases, n.summary) for n in answer.nodes] == [("Shop", ["Le Shop"], "")]
    assert [(e.dst, e.quote) for e in answer.edges] == [("Shop", None)]
    assert answer.activity == ProposedActivity(name="logging in", quote="log in with")


def test_scrub_drops_an_activity_left_without_a_quote() -> None:
    answer = Answer(activity=ProposedActivity(name="logging in", quote="x9!kQ"))
    assert credentials.scrub(answer, ("x9!kQ",)) == 1
    assert answer.activity is None


def test_the_answer_lists_secrets_in_a_closed_shape() -> None:
    assert parse_answer('{"facts": [], "secrets": ["a1b2c3"]}').secrets == ["a1b2c3"]
    for none in ("none", "null", "N/A", ""):
        assert parse_answer('{"facts": [], "secrets": "' + none + '"}').secrets == []
    alone = parse_answer('{"facts": [], "secrets": "a1b2c3"}')  # one, without its list
    assert alone.secrets == ["a1b2c3"] and alone.rejected.keys == 0
    wrong = parse_answer('{"facts": [], "secrets": {"a": 1}}')
    assert wrong.secrets == [] and wrong.rejected.keys == 1
    mixed = parse_answer(
        '{"facts": [], "secrets": ["ok-1234", 42, true, "", "' + "x" * (SECRET_CHARS + 1) + '"]}'
    )
    # a number is a code; "" is the list's placeholder, skipped
    assert mixed.secrets == ["ok-1234", "42"] and mixed.rejected.keys == 2
    listed = parse_answer('{"facts": [], "secrets": ["none", "NULL", "aucun", "a1b2c3"]}')
    assert listed.secrets == ["a1b2c3"] and listed.rejected.keys == 0
    count = MAX_SECRETS + 5
    many = parse_answer('{"secrets": [' + ", ".join(f'"s{i:04}"' for i in range(count)) + "]}")
    assert len(many.secrets) == MAX_SECRETS and many.rejected.keys == 5
    assert parse_answer('{"facts": []}').secrets == []
    block = "-----BEGIN PRIVATE KEY-----" + "Q" * 3000 + "-----END PRIVATE KEY-----"
    assert parse_answer('{"secrets": ["' + block + '"]}').secrets == [block]


def test_a_weak_secret_never_drops_what_the_owner_said() -> None:
    """A value only a page writes, by the service's reading, drops no fact
    the owner's words hold, and no node."""
    answer = parse_answer(
        """{"nodes": [{"name": "Paris"}],
            "facts": [{"subject": "Alice", "statement": "Alice moved to Paris.",
                       "quote": "I moved to Paris"},
                      {"subject": "Guide", "statement": "The guide's password is Paris.",
                       "quote": "password: Paris"}]}"""
    )
    dropped = credentials.scrub(
        answer, ("Paris",), weak=("Paris",), owned=lambda f: f.quote == "I moved to Paris"
    )
    assert dropped == 1
    assert [f.statement for f in answer.facts] == ["Alice moved to Paris."]
    assert [n.name for n in answer.nodes] == ["Paris"]


def test_a_fact_whose_quote_was_only_the_secret_is_dropped() -> None:
    answer = parse_answer(
        '{"facts": [{"subject": "Bob", "statement": "Bob sent a code.", "quote": "482913"}]}'
    )
    assert credentials.scrub(answer, ("482913",)) == 1
    assert answer.facts == []


def test_shapes_and_claims() -> None:
    assert credentials.shaped("sk-abcdefghijklmnopqrstuv")
    assert not credentials.shaped("Paris")
    answer = Answer(
        secrets=[" Tok3n "],
        facts=[ProposedFact(statement="A u.", attribute="pin", value="4071")],
    )
    assert credentials.claimed([answer]) == {"tok3n", "4071"}
