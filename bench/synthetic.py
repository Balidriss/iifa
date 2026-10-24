"""The synthetic multilingual corpus of the benchmark (SPEC 11.1).

People whose names exist in several scripts live in three scopes (one holds a
homonym: another Karim), and their attributes change over the months: the city
they live in (moves), their role, the drink they prefer (changes of mind). Each
change is told in a turn written in English, French, Russian, Arabic, Chinese or
Japanese, the name spelt in the turn's script; chatter turns with nothing to
remember sit in between. Every question is generated from the ground truth and
asked through one scope's view, in that scope's language.

Everything comes from ``random.Random(seed)``: the same seed gives the same
corpus, byte for byte. Each turn states at most one fact, so a recalled record
supports the event of the turn it came from.

For every event turn the corpus also holds the **gold extraction answer**: what
an ideal model would answer to the extraction prompt (the known name reused,
the source's spelling in ``subject_as_written``, attribute and value set). The
harness's oracle hands it to the real pipeline.

Options make other corpora, each compared only with runs that used it
(SPEC 11.1): ``rephrase`` adds, after some event turns, a turn of the same
month restating the same value in another of the person's languages (not a
change); ``speaker`` adds the owner of ``user:alice``, Alice, who speaks in the
first person in her own turns and is asked about in the first person;
``name_traps`` adds three people whose names are near another's of the same
scope (Karima beside Karim, Ivana beside Ivan, Nadine beside Nadia).
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

LANGUAGES = ("en", "fr", "ru", "ar", "zh", "ja")
ALICE, SOLAR, BOB = "user:alice", "user:alice/project:solar", "user:bob"
ASK_LANGUAGE = {ALICE: "fr", SOLAR: "en", BOB: "en"}
ATTRIBUTES = ("city", "role", "drink")
# --traps: where each person works, a city other than where they live (SPEC 5.4:
# a key lumping both together closes a home that is still true)
TRAP = "office"
# --traps: a team, then values ADDED in the last month to what already holds -
# a second role, a second team, a second favourite drink (SPEC 5.4: a
# "replaces" that closes one of them closes a value still true)
TEAM = "team"
# the keys of SPEC 5.4's base vocabulary an ideal model gives them
GOLD_KEYS = {
    "city": "home",
    "role": "role",
    "drink": "preference.drink",
    TRAP: "workplace",
    TEAM: "team",
}
YEAR = 2026
# --speaker: the declared owner of user:alice (SPEC 4.6), who says "I"
OWNER = "alice@user:alice"
OWNER_NAME = "Alice"
OWNER_LANGUAGES = ("fr", "en", "ru")
# what an ideal model writes as the subject of a Russian statement about the
# owner: the Latin name is not in the statement's script, a speaker word is
RU_SPEAKER = "Пользователь"


# --------------------------------------------------------------------------- people

NAMES: dict[str, dict[str, str]] = {
    "karim": {
        "en": "Karim",
        "fr": "Karim",
        "ru": "Карим",
        "ar": "كريم",
        "zh": "卡里姆",
        "ja": "カリム",
    },
    "emilie": {
        "en": "Émilie",
        "fr": "Émilie",
        "ru": "Эмили",
        "ar": "إميلي",
        "zh": "埃米莉",
        "ja": "エミリー",
    },
    "ivan": {"en": "Ivan", "fr": "Ivan", "ru": "Иван", "ar": "إيفان", "zh": "伊万", "ja": "イワン"},
    "meiling": {
        "en": "Meiling",
        "fr": "Meiling",
        "ru": "Мэйлин",
        "ar": "ميلينغ",
        "zh": "美玲",
        "ja": "メイリン",
    },
    "youssef": {
        "en": "Youssef",
        "fr": "Youssef",
        "ru": "Юсеф",
        "ar": "يوسف",
        "zh": "优素福",
        "ja": "ユセフ",
    },
    "nadia": {
        "en": "Nadia",
        "fr": "Nadia",
        "ru": "Надия",
        "ar": "نادية",
        "zh": "娜迪亚",
        "ja": "ナディア",
    },
    "kenji": {
        "en": "Kenji",
        "fr": "Kenji",
        "ru": "Кэндзи",
        "ar": "كينجي",
        "zh": "健二",
        "ja": "健二",
    },
    "alice": {
        "en": "Alice",
        "fr": "Alice",
        "ru": "Алиса",
        "ar": "أليس",
        "zh": "爱丽丝",
        "ja": "アリス",
    },
    "sofia": {
        "en": "Sofia",
        "fr": "Sofia",
        "ru": "София",
        "ar": "صوفيا",
        "zh": "索菲亚",
        "ja": "ソフィア",
    },
}


@dataclass(frozen=True)
class Person:
    key: str  # unique across scopes: "karim@user:bob" is not "karim@user:alice"
    name_key: str
    scope: str
    feminine: bool
    languages: tuple[str, ...]  # the languages their turns are told in

    def name(self, language: str) -> str:
        return (NAMES.get(self.name_key) or NEAR_NAMES[self.name_key])[language]


def _person(name_key: str, scope: str, feminine: bool, *languages: str) -> Person:
    return Person(f"{name_key}@{scope}", name_key, scope, feminine, languages)


PEOPLE: tuple[Person, ...] = (
    _person("karim", ALICE, False, "fr", "ru", "ar"),
    _person("emilie", ALICE, True, "fr", "en"),
    _person("ivan", ALICE, False, "ru", "fr"),
    _person("meiling", ALICE, True, "zh", "en"),
    _person("youssef", ALICE, False, "ar", "fr"),
    _person("nadia", SOLAR, True, "en", "fr"),
    _person("kenji", SOLAR, False, "ja", "en"),
    _person("karim", BOB, False, "en", "ru"),  # the homonym: P9 in the bench
    _person("sofia", BOB, True, "en", "ru"),
)
OWNER_PERSON = _person("alice", ALICE, True, *OWNER_LANGUAGES)
# --name-traps: near another person's name of the same scope, never the same person
NEAR_NAMES: dict[str, dict[str, str]] = {
    "karima": {
        "en": "Karima",
        "fr": "Karima",
        "ru": "Карима",
        "ar": "كريمة",
        "zh": "卡里玛",
        "ja": "カリマ",
    },
    "ivana": {
        "en": "Ivana",
        "fr": "Ivana",
        "ru": "Ивана",
        "ar": "إيفانا",
        "zh": "伊万娜",
        "ja": "イヴァナ",
    },
    "nadine": {
        "en": "Nadine",
        "fr": "Nadine",
        "ru": "Надин",
        "ar": "نادين",
        "zh": "娜丁",
        "ja": "ナディーン",
    },
}
NEAR_PEOPLE: tuple[Person, ...] = (
    _person("karima", ALICE, True, "fr", "ar", "ru"),  # beside Karim
    _person("ivana", ALICE, True, "ru", "fr"),  # beside Ivan
    _person("nadine", SOLAR, True, "en", "fr"),  # beside Nadia
)


# --------------------------------------------------------------------------- values

CITIES: dict[str, dict[str, str]] = {
    "lyon": {"en": "Lyon", "fr": "Lyon", "ru": "Лион", "ar": "ليون", "zh": "里昂", "ja": "リヨン"},
    "paris": {
        "en": "Paris",
        "fr": "Paris",
        "ru": "Париж",
        "ar": "باريس",
        "zh": "巴黎",
        "ja": "パリ",
    },
    "berlin": {
        "en": "Berlin",
        "fr": "Berlin",
        "ru": "Берлин",
        "ar": "برلين",
        "zh": "柏林",
        "ja": "ベルリン",
    },
    "kazan": {
        "en": "Kazan",
        "fr": "Kazan",
        "ru": "Казань",
        "ar": "قازان",
        "zh": "喀山",
        "ja": "カザン",
    },
    "casablanca": {
        "en": "Casablanca",
        "fr": "Casablanca",
        "ru": "Касабланка",
        "ar": "الدار البيضاء",
        "zh": "卡萨布兰卡",
        "ja": "カサブランカ",
    },
    "osaka": {
        "en": "Osaka",
        "fr": "Osaka",
        "ru": "Осака",
        "ar": "أوساكا",
        "zh": "大阪",
        "ja": "大阪",
    },
    "montreal": {
        "en": "Montreal",
        "fr": "Montréal",
        "ru": "Монреаль",
        "ar": "مونتريال",
        "zh": "蒙特利尔",
        "ja": "モントリオール",
    },
    "tunis": {
        "en": "Tunis",
        "fr": "Tunis",
        "ru": "Тунис",
        "ar": "تونس",
        "zh": "突尼斯",
        "ja": "チュニス",
    },
}
# Russian needs the city in the prepositional ("живёт в Лионе") and accusative ("переехал в Казань")
RU_IN: dict[str, str] = {
    "lyon": "Лионе",
    "paris": "Париже",
    "berlin": "Берлине",
    "kazan": "Казани",
    "casablanca": "Касабланке",
    "osaka": "Осаке",
    "montreal": "Монреале",
    "tunis": "Тунисе",
}
RU_TO: dict[str, str] = {
    "lyon": "Лион",
    "paris": "Париж",
    "berlin": "Берлин",
    "kazan": "Казань",
    "casablanca": "Касабланку",
    "osaka": "Осаку",
    "montreal": "Монреаль",
    "tunis": "Тунис",
}

ROLES: dict[str, dict[str, str]] = {
    "architect": {
        "en": "an architect",
        "fr": "architecte",
        "ru": "архитектором",
        "zh": "建筑师",
        "ja": "建築家",
    },
    "accountant": {
        "en": "an accountant",
        "fr": "comptable",
        "ru": "бухгалтером",
        "zh": "会计",
        "ja": "会計士",
    },
    "photographer": {
        "en": "a photographer",
        "fr": "photographe",
        "ru": "фотографом",
        "zh": "摄影师",
        "ja": "写真家",
    },
    "journalist": {
        "en": "a journalist",
        "fr": "journaliste",
        "ru": "журналистом",
        "zh": "记者",
        "ja": "記者",
    },
}
AR_ROLES: dict[str, tuple[str, str]] = {  # (masculine, feminine), after the preposition ك
    "architect": ("مهندس معماري", "مهندسة معمارية"),
    "accountant": ("محاسب", "محاسبة"),
    "photographer": ("مصور", "مصورة"),
    "journalist": ("صحفي", "صحفية"),
}

DRINKS: dict[str, dict[str, str]] = {
    "tea": {"en": "tea", "fr": "le thé", "ru": "чай", "ar": "الشاي", "zh": "茶", "ja": "お茶"},
    "coffee": {
        "en": "coffee",
        "fr": "le café",
        "ru": "кофе",
        "ar": "القهوة",
        "zh": "咖啡",
        "ja": "コーヒー",
    },
}

TEAMS: dict[str, dict[str, str]] = {
    "atlas": {
        "en": "Atlas",
        "fr": "Atlas",
        "ru": "Атлас",
        "ar": "أطلس",
        "zh": "阿特拉斯",
        "ja": "アトラス",
    },
    "orion": {
        "en": "Orion",
        "fr": "Orion",
        "ru": "Орион",
        "ar": "أوريون",
        "zh": "猎户座",
        "ja": "オリオン",
    },
}

CHATTER: dict[str, tuple[str, ...]] = {
    "en": (
        "The weather is lovely today.",
        "Let's talk about it later.",
        "Thanks, that was helpful.",
    ),
    "fr": ("Il pleut encore ce matin.", "On en reparle demain.", "Merci, c'était utile."),
    "ru": ("Сегодня очень холодно.", "Давай обсудим это позже.", "Спасибо, это помогло."),
    "ar": ("الطقس جميل اليوم.", "لنتحدث عن ذلك لاحقا.", "شكرا، كان ذلك مفيدا."),
    "zh": ("今天天气很好。", "我们以后再谈吧。", "谢谢，这很有帮助。"),
    "ja": ("今日はとても暑い。", "また後で話しましょう。", "ありがとう、助かりました。"),
}


def value_forms() -> dict[str, str]:
    """Every written form of every value (any language, both Arabic genders) ->
    the value it names, e.g. "Лион" -> "city:lyon": the truth an oracle answers
    the value adjudication of SPEC 5.4 with."""
    forms: dict[str, str] = {}
    for city, names in CITIES.items():
        forms.update({name.lower(): f"city:{city}" for name in names.values()})
        forms[RU_IN[city].lower()] = forms[RU_TO[city].lower()] = f"city:{city}"
    for role, names in ROLES.items():
        forms.update({name.lower(): f"role:{role}" for name in names.values()})
        forms.update({name.lower(): f"role:{role}" for name in AR_ROLES[role]})
    for drink, names in DRINKS.items():
        forms.update({name.lower(): f"drink:{drink}" for name in names.values()})
    for team, names in TEAMS.items():
        forms.update({name.lower(): f"team:{team}" for name in names.values()})
    return forms


def value_text(attribute: str, value: str, language: str, *, feminine: bool = False) -> str:
    """The value as the turn writes it (the gold ``value`` of the fact)."""
    if attribute in ("city", TRAP):
        return CITIES[value][language]
    if attribute == "role":
        if language == "ar":
            return AR_ROLES[value][int(feminine)]
        return ROLES[value][language]
    if attribute == TEAM:
        return TEAMS[value][language]
    return DRINKS[value][language]


def sentence(person: Person, attribute: str, value: str, language: str, *, change: bool) -> str:
    """One complete sentence naming its subject in the turn's script (SPEC 4.2)."""
    n, f = person.name(language), person.feminine
    v = value_text(attribute, value, language, feminine=f)
    if attribute == "city":
        return {
            "en": f"{n} moved to {v}." if change else f"{n} lives in {v}.",
            "fr": f"{n} a déménagé à {v}." if change else f"{n} habite à {v}.",
            "ru": (f"{n} переехала в {RU_TO[value]}." if f else f"{n} переехал в {RU_TO[value]}.")
            if change
            else f"{n} живёт в {RU_IN[value]}.",
            "ar": (f"{n} انتقلت إلى {v}." if f else f"{n} انتقل إلى {v}.")
            if change
            else (f"{n} تسكن في {v}." if f else f"{n} يسكن في {v}."),
            "zh": f"{n}搬到了{v}。" if change else f"{n}住在{v}。",
            "ja": f"{n}は{v}に引っ越した。" if change else f"{n}は{v}に住んでいる。",
        }[language]
    if attribute == "role":
        return {
            "en": f"{n} now works as {v}." if change else f"{n} works as {v}.",
            "fr": f"{n} travaille maintenant comme {v}." if change else f"{n} travaille comme {v}.",
            "ru": f"{n} теперь работает {v}." if change else f"{n} работает {v}.",
            "ar": (f"{n} تعمل الآن ك{v}." if f else f"{n} يعمل الآن ك{v}.")
            if change
            else (f"{n} تعمل ك{v}." if f else f"{n} يعمل ك{v}."),
            "zh": f"{n}现在是一名{v}。" if change else f"{n}是一名{v}。",
            "ja": f"{n}は今{v}として働いている。" if change else f"{n}は{v}として働いている。",
        }[language]
    if attribute == TEAM:  # the first team someone joins (--traps)
        return {
            "en": f"{n} joined the {v} team.",
            "fr": f"{n} a rejoint l'équipe {v}.",
            "ru": f"{n} присоединилась к команде {v}."
            if f
            else f"{n} присоединился к команде {v}.",
            "ar": f"انضمت {n} إلى فريق {v}." if f else f"انضم {n} إلى فريق {v}.",
            "zh": f"{n}加入了{v}团队。",
            "ja": f"{n}は{v}チームに参加した。",
        }[language]
    if attribute == TRAP:  # where they work: a city that is not where they live
        return {
            "en": f"{n} works at the {v} office.",
            "fr": f"{n} travaille au bureau de {v}.",
            "ru": f"{n} работает в офисе в {RU_IN[value]}.",
            "ar": f"{n} تعمل في مكتب {v}." if f else f"{n} يعمل في مكتب {v}.",
            "zh": f"{n}在{v}的办公室工作。",
            "ja": f"{n}は{v}のオフィスで働いている。",
        }[language]
    return {
        "en": f"{n} changed their mind and now prefers {v}." if change else f"{n} prefers {v}.",
        "fr": f"Finalement, {n} préfère {v}." if change else f"{n} préfère {v}.",
        "ru": (
            f"{n} передумала и теперь предпочитает {v}."
            if f
            else f"{n} передумал и теперь предпочитает {v}."
        )
        if change
        else f"{n} предпочитает {v}.",
        "ar": (f"{n} غيرت رأيها وأصبحت تفضل {v}." if f else f"{n} غير رأيه وأصبح يفضل {v}.")
        if change
        else (f"{n} تفضل {v}." if f else f"{n} يفضل {v}."),
        "zh": f"{n}改变了主意，现在更喜欢喝{v}。" if change else f"{n}更喜欢喝{v}。",
        "ja": f"{n}は考えを変えて、今は{v}の方が好きだ。" if change else f"{n}は{v}が好きだ。",
    }[language]


def first_person(attribute: str, value: str, language: str, *, change: bool) -> str:
    """What the owner says about herself (``--speaker``): "I", never her name."""
    v = value_text(attribute, value, language, feminine=True)
    if attribute == "city":
        return {
            "en": f"I moved to {v}." if change else f"I live in {v}.",
            "fr": f"J'ai déménagé à {v}." if change else f"J'habite à {v}.",
            "ru": f"Я переехала в {RU_TO[value]}." if change else f"Я живу в {RU_IN[value]}.",
        }[language]
    if attribute == "role":
        return {
            "en": f"I now work as {v}." if change else f"I work as {v}.",
            "fr": f"Je travaille maintenant comme {v}." if change else f"Je travaille comme {v}.",
            "ru": f"Я теперь работаю {v}." if change else f"Я работаю {v}.",
        }[language]
    return {
        "en": f"I changed my mind and now prefer {v}." if change else f"I prefer {v}.",
        "fr": f"Finalement, je préfère {v}." if change else f"Je préfère {v}.",
        "ru": f"Я передумала и теперь предпочитаю {v}." if change else f"Я предпочитаю {v}.",
    }[language]


def added_sentence(person: Person, attribute: str, value: str, language: str) -> str:
    """A value added to what already holds (--traps): "also", never "now"."""
    n, f = person.name(language), person.feminine
    v = value_text(attribute, value, language, feminine=f)
    if attribute == "role":
        return {
            "en": f"{n} also works as {v}.",
            "fr": f"{n} travaille aussi comme {v}.",
            "ru": f"{n} также работает {v}.",
            "ar": f"{n} تعمل أيضا ك{v}." if f else f"{n} يعمل أيضا ك{v}.",
            "zh": f"{n}也是一名{v}。",
            "ja": f"{n}は{v}としても働いている。",
        }[language]
    if attribute == TEAM:
        return {
            "en": f"{n} is also in the {v} team.",
            "fr": f"{n} fait aussi partie de l'équipe {v}.",
            "ru": f"{n} также входит в команду {v}.",
            "ar": f"{n} عضوة أيضا في فريق {v}." if f else f"{n} عضو أيضا في فريق {v}.",
            "zh": f"{n}也在{v}团队。",
            "ja": f"{n}は{v}チームにも所属している。",
        }[language]
    return {
        "en": f"{n} also likes {v}.",
        "fr": f"{n} aime aussi {v}.",
        "ru": f"{n} также любит {v}.",
        "ar": f"{n} تحب أيضا {v}." if f else f"{n} يحب أيضا {v}.",
        "zh": f"{n}也喜欢喝{v}。",
        "ja": f"{n}は{v}も好きだ。",
    }[language]


QUESTIONS_TO_SELF: dict[str, dict[str, str]] = {
    "city": {"en": "Where do I live now?", "fr": "Où est-ce que j'habite maintenant ?"},
    "role": {"en": "What do I work as?", "fr": "Comme quoi est-ce que je travaille ?"},
    "drink": {"en": "Which drink do I prefer?", "fr": "Qu'est-ce que je préfère boire ?"},
}

QUESTIONS: dict[str, dict[str, str]] = {
    "city": {"en": "Where does {name} live now?", "fr": "Où habite {name} maintenant ?"},
    "role": {"en": "What does {name} work as?", "fr": "Comme quoi travaille {name} ?"},
    "drink": {"en": "Which drink does {name} prefer?", "fr": "Que préfère boire {name} ?"},
    "office": {"en": "Where is {name}'s office?", "fr": "Où se trouve le bureau de {name} ?"},
    TEAM: {"en": "Which team is {name} in?", "fr": "Dans quelle équipe est {name} ?"},
}


# --------------------------------------------------------------------------- the corpus


@dataclass(frozen=True)
class Event:
    key: str
    person: str
    scope: str
    attribute: str
    value: str
    language: str
    month: int
    change: bool
    previous: str | None  # the event this one replaces (same person and attribute)
    added: bool = False  # --traps: a value added to what holds, replacing nothing


@dataclass(frozen=True)
class Turn:
    key: str
    scope: str
    session: str
    language: str
    month: int
    text: str
    event: str | None = None  # None: chatter, or a restatement
    restates: str | None = None  # --rephrase: the event this turn tells again
    speaker: bool = False  # --speaker: the owner speaking about herself


@dataclass(frozen=True)
class Question:
    key: str
    scope: str  # asked through this scope's view (scope + ancestors)
    language: str
    query: str
    person: str
    attribute: str
    expected: tuple[str, ...]  # event keys; any one supports the answer
    stale: tuple[str, ...]  # earlier values of the same attribute
    same_language_change: bool | None  # None when the attribute never changed

    @property
    def changed(self) -> bool:
        return bool(self.stale)


@dataclass
class Corpus:
    seed: int
    months: int
    traps: bool = False
    rephrase: bool = False
    speaker: bool = False
    name_traps: bool = False
    people: tuple[Person, ...] = PEOPLE
    turns: list[Turn] = field(default_factory=list)
    events: dict[str, Event] = field(default_factory=dict)
    questions: list[Question] = field(default_factory=list)
    gold: dict[str, dict[str, Any]] = field(default_factory=dict)  # turn text -> answer

    @property
    def scopes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(t.scope for t in self.turns))

    def turn_of_event(self) -> dict[str, str]:
        return {t.event: t.key for t in self.turns if t.event is not None}

    def turns_of_event(self) -> dict[str, tuple[str, ...]]:
        """Every turn that supports an event: its own, and its restatements."""
        out: dict[str, list[str]] = {}
        for turn in self.turns:
            told = turn.event or turn.restates
            if told is not None:
                out.setdefault(told, []).append(turn.key)
        return {event: tuple(keys) for event, keys in out.items()}

    def is_chatter(self, turn: Turn) -> bool:
        return turn.event is None and turn.restates is None

    def summary(self) -> dict[str, Any]:
        languages: dict[str, int] = {}
        for turn in self.turns:
            languages[turn.language] = languages.get(turn.language, 0) + 1
        changed = [q for q in self.questions if q.changed]
        return {
            "seed": self.seed,
            "months": self.months,
            "traps": self.traps,
            "rephrase": self.rephrase,
            "speaker": self.speaker,
            "name_traps": self.name_traps,
            "turns": len(self.turns),
            "event_turns": sum(1 for t in self.turns if t.event),
            "chatter_turns": sum(1 for t in self.turns if self.is_chatter(t)),
            "rephrase_turns": sum(1 for t in self.turns if t.restates),
            "added_events": sum(1 for e in self.events.values() if e.added),
            "speaker_turns": sum(1 for t in self.turns if t.speaker),
            "people": len({e.person for e in self.events.values()}),
            "scopes": list(self.scopes),
            "languages": dict(sorted(languages.items())),
            "questions": len(self.questions),
            "changed_questions": len(changed),
            "same_language_changes": sum(1 for q in changed if q.same_language_change),
            "cross_language_changes": sum(1 for q in changed if q.same_language_change is False),
        }


def generate(
    seed: int = 7,
    *,
    months: int = 12,
    chatter: float = 3.0,
    traps: bool = False,
    rephrase: bool = False,
    speaker: bool = False,
    name_traps: bool = False,
) -> Corpus:
    """The corpus for ``seed``: ``months`` months of events (at least 2), about
    ``chatter`` chatter turns per event turn. ``traps`` adds, for every person,
    where they work (a city other than their first home), drawn from a random
    generator of its own: without traps the corpus is unchanged. ``rephrase``
    ``speaker`` and ``name_traps`` (see the module) draw from generators of
    their own too."""
    if months < 2:
        raise ValueError("months must be at least 2: a change needs a later month")
    rng = random.Random(seed)
    people = (*PEOPLE, *(NEAR_PEOPLE if name_traps else ()))
    people = (*people, OWNER_PERSON) if speaker else people
    corpus = Corpus(
        seed=seed,
        months=months,
        traps=traps,
        rephrase=rephrase,
        speaker=speaker,
        name_traps=name_traps,
        people=people,
    )
    planned: list[Event] = []
    for person in PEOPLE:
        for attribute in ATTRIBUTES:
            planned.extend(_timeline(rng, person, attribute, months, len(planned)))
    if traps:
        planned.extend(_offices(random.Random(f"traps:{seed}"), planned, months))
        planned.extend(_additions(random.Random(f"added:{seed}"), planned, months))
    if name_traps:
        near = random.Random(f"names:{seed}")
        for person in NEAR_PEOPLE:
            for attribute in ATTRIBUTES:
                planned.extend(_timeline(near, person, attribute, months, len(planned)))
    if speaker:
        own = random.Random(f"speaker:{seed}")
        for attribute in ATTRIBUTES:
            planned.extend(_timeline(own, OWNER_PERSON, attribute, months, len(planned)))
    planned.sort(key=lambda e: (e.month, e.added, rng.random()))  # additions last
    restate = random.Random(f"rephrase:{seed}")
    known: dict[str, str] = {}  # person -> the name the first turn spelt (the node's name)
    for event in planned:
        person = next(p for p in people if p.key == event.person)
        corpus.events[event.key] = event
        _tell(corpus, person, event, known)
        if rephrase and len(person.languages) > 1 and restate.random() < 0.5:
            other = restate.choice([lang for lang in person.languages if lang != event.language])
            _tell(corpus, person, dataclasses.replace(event, language=other), known, again=True)
        extra = int(chatter) + (1 if rng.random() < chatter - int(chatter) else 0)
        for _ in range(extra):
            language = rng.choice(person.languages)
            line = rng.choice(CHATTER[language])
            corpus.turns.append(_turn(corpus, event.scope, language, event.month, line, None))
    corpus.questions = _questions(corpus)
    return corpus


def _tell(
    corpus: Corpus, person: Person, event: Event, known: dict[str, str], *, again: bool = False
) -> None:
    """The turn telling ``event`` in ``event.language`` and its gold answer; with
    ``again``, a restatement of an event already told (``--rephrase``)."""
    owner = person.key == OWNER
    if owner:
        text = first_person(event.attribute, event.value, event.language, change=event.change)
    elif event.added:
        text = added_sentence(person, event.attribute, event.value, event.language)
    else:
        text = sentence(person, event.attribute, event.value, event.language, change=event.change)
    turn = _turn(
        corpus, event.scope, event.language, event.month, text, None if again else event.key
    )
    corpus.turns.append(
        dataclasses.replace(turn, restates=event.key if again else None, speaker=owner)
    )
    if owner:
        corpus.gold[text] = owner_gold_answer(event, text)
        return
    written = person.name(event.language)
    name = known.setdefault(person.key, written)
    corpus.gold[text] = gold_answer(person, event, text, name, written)


def _timeline(
    rng: random.Random, person: Person, attribute: str, months: int, start: int
) -> list[Event]:
    """The first value of ``attribute`` in month 1, then up to two changes later."""
    pool = {"city": sorted(CITIES), "role": sorted(ROLES), "drink": sorted(DRINKS)}[attribute]
    changes = (
        rng.choices((0, 1, 2), weights=(3, 5, 2))[0] if attribute != "drink" else rng.choice((0, 1))
    )
    months_of_change = sorted(rng.sample(range(2, months + 1), k=min(changes, months - 1)))
    values = [rng.choice(pool)]
    for _ in months_of_change:
        values.append(rng.choice([v for v in pool if v != values[-1]]))
    events: list[Event] = []
    for index, (month, value) in enumerate(zip([1, *months_of_change], values, strict=True)):
        events.append(
            Event(
                key=f"e{start + index:03d}",
                person=person.key,
                scope=person.scope,
                attribute=attribute,
                value=value,
                language=rng.choice(person.languages),
                month=month,
                change=index > 0,
                previous=events[-1].key if events else None,
            )
        )
    return events


def _offices(rng: random.Random, planned: list[Event], months: int) -> list[Event]:
    """One office per person, in a city other than their first home, in a month
    where they already live somewhere (a later home may coincide: still a trap)."""
    events: list[Event] = []
    for person in PEOPLE:
        home = next(e.value for e in planned if e.person == person.key and e.attribute == "city")
        events.append(
            Event(
                key=f"e{len(planned) + len(events):03d}",
                person=person.key,
                scope=person.scope,
                attribute=TRAP,
                value=rng.choice([c for c in sorted(CITIES) if c != home]),
                language=rng.choice(person.languages),
                month=rng.randint(1, months),
                change=False,
                previous=None,
            )
        )
    return events


def _additions(rng: random.Random, planned: list[Event], months: int) -> list[Event]:
    """--traps: for every person, a team in a month before the last, then, in the
    last month, a second team, a second role and a second favourite drink -
    values added to what already holds, replacing nothing."""
    events: list[Event] = []

    def event(person: Person, attribute: str, value: str, month: int, added: bool) -> None:
        events.append(
            Event(
                key=f"e{len(planned) + len(events):03d}",
                person=person.key,
                scope=person.scope,
                attribute=attribute,
                value=value,
                language=rng.choice(person.languages),
                month=month,
                change=False,
                previous=None,
                added=added,
            )
        )

    for person in PEOPLE:
        first, second = rng.sample(sorted(TEAMS), k=2)
        event(person, TEAM, first, rng.randint(1, months - 1), False)
        event(person, TEAM, second, months, True)
        roles = {e.value for e in planned if e.person == person.key and e.attribute == "role"}
        event(
            person, "role", rng.choice([r for r in sorted(ROLES) if r not in roles]), months, True
        )
        drinks = [e for e in planned if e.person == person.key and e.attribute == "drink"]
        current = max(drinks, key=lambda e: (e.month, e.key)).value
        event(person, "drink", next(d for d in sorted(DRINKS) if d != current), months, True)
    return events


def _turn(
    corpus: Corpus, scope: str, language: str, month: int, text: str, event: str | None
) -> Turn:
    return Turn(
        key=f"t{len(corpus.turns):03d}",
        scope=scope,
        session=f"{scope}#{YEAR}-{month:02d}",
        language=language,
        month=month,
        text=text,
        event=event,
    )


def gold_answer(person: Person, event: Event, text: str, name: str, written: str) -> dict[str, Any]:
    """What an ideal model answers to the extraction prompt for this turn: the
    known name reused, the source's spelling in ``subject_as_written`` (SPEC 4.2)."""
    return {
        "nodes": [{"name": name, "kind": "person", "aliases": [], "summary": ""}],
        "facts": [
            {
                "subject": name,
                "subject_as_written": written if written != name else None,
                "statement": text,
                "attribute": GOLD_KEYS[event.attribute],
                "value": value_text(
                    event.attribute, event.value, event.language, feminine=person.feminine
                ),
                "valid_from": f"{YEAR}-{event.month:02d}",
                "quote": text,
                "confidence": 0.9,
            }
        ],
        "edges": [],
    }


def owner_gold_answer(event: Event, text: str) -> dict[str, Any]:
    """What an ideal model answers for the owner's own words (SPEC 4.2): the
    owner's name as the subject (the source labels her part with it); the
    statement names her, by her name, or by a speaker word in Russian."""
    if event.language == "ru":
        speaker = _person("alice", ALICE, False, "ru")
        statement = sentence(speaker, event.attribute, event.value, "ru", change=event.change)
        statement = statement.replace("Алиса", RU_SPEAKER, 1)
    else:
        statement = sentence(
            OWNER_PERSON, event.attribute, event.value, event.language, change=event.change
        )
    return {
        "nodes": [{"name": OWNER_NAME, "kind": "person", "aliases": [], "summary": ""}],
        "facts": [
            {
                "subject": OWNER_NAME,
                "subject_as_written": None,
                "statement": statement,
                "attribute": GOLD_KEYS[event.attribute],
                "value": value_text(event.attribute, event.value, event.language, feminine=True),
                "valid_from": f"{YEAR}-{event.month:02d}",
                "quote": text,
                "confidence": 0.9,
            }
        ],
        "edges": [],
    }


def _questions(corpus: Corpus) -> list[Question]:
    out: list[Question] = []
    for person in corpus.people:
        language = ASK_LANGUAGE[person.scope]
        asked = QUESTIONS_TO_SELF if person.key == OWNER else QUESTIONS
        for attribute in (*ATTRIBUTES, TRAP, TEAM):
            told = [
                e
                for e in corpus.events.values()
                if e.person == person.key and e.attribute == attribute
            ]
            if not told:
                continue
            chain = sorted((e for e in told if not e.added), key=lambda e: (e.month, e.key))
            added = tuple(e.key for e in told if e.added)
            current, stale = chain[-1], chain[:-1]
            out.append(
                Question(
                    key=f"q{len(out):03d}",
                    scope=person.scope,
                    language=language,
                    query=asked[attribute][language].format(name=person.name(language)),
                    person=person.key,
                    attribute=attribute,
                    expected=(current.key, *added),  # an added value holds too
                    stale=tuple(e.key for e in stale),
                    same_language_change=(current.language == stale[-1].language)
                    if stale
                    else None,
                )
            )
    return out


def scopes_of_turns(corpus: Corpus) -> Mapping[str, str]:
    return {t.key: t.scope for t in corpus.turns}
