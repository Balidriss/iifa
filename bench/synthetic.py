"""The synthetic multilingual corpus of the benchmark.

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

Options make other corpora, each compared only with runs that used it:
``rephrase`` adds, after some event turns, a turn of the same
month restating the same value in another of the person's languages (not a
change); ``speaker`` adds the owner of ``user:alice``, Alice, who speaks in the
first person in her own turns and is asked about in the first person;
``name_traps`` adds three people whose names are near another's of the same
scope (Karima beside Karim, Ivana beside Ivan, Nadine beside Nadia);
``neighbours`` adds turns that describe two of the owner's projects side by
side (a web service and a command-line tool, each with a sentence of its own),
in English or French: lasting facts, no event, no question, each project's
fact judged on its own node.

``intent`` adds what was not meant as a fact: for every person and
each of their languages among English, French and Russian, a turn that names
one of their real values without stating it - a hypothesis, a question, a
guess, a possibility, an example - told just before the sincere turn that
states the same value (its **control**); and in ``user:alice``, test personas:
a frame over several turns ("for this exercise, I am a pilot named..." until
"end of the exercise") and a persona the task sets ("answer as the candidate..."), each
with the one episode an ideal model leaves. Their gold answers hold the
content as an ideal model reads it (labelled ``hypothesis``, ``pretend``...):
the service must write none of it. ``worksession`` turns the last month of
``user:alice`` into a work session on a slide deck: iterations whose facts
hold only for the task, with that month's lasting events told in between,
then the session's close, whose gold is one episode.

``observed`` adds, in ``user:alice``, a few things whose state changes without
anyone telling the memory (a shelf, a service, a queue): each is told once
something lasting about it, then observed by the owner two or three times in
later months, the turn dated that month (an observation is true at a
date), the state under the same generic key as the lasting fact.
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
# --traps: where each person works, a city other than where they live (a key
# lumping both together closes a home that is still true)
TRAP = "office"
# --traps: a team, then values ADDED in the last month to what already holds -
# a second role, a second team, a second favourite drink (a
# "replaces" that closes one of them closes a value still true)
TEAM = "team"
# the keys of the base vocabulary an ideal model gives them
GOLD_KEYS = {
    "city": "home",
    "role": "role",
    "drink": "preference.drink",
    TRAP: "workplace",
    TEAM: "team",
}
YEAR = 2026
# --speaker: the declared owner of user:alice, who says "I"
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

# --neighbours: two of the owner's projects described side by side in one
# turn, lasting facts, never a one-off description of a file (what
# a tool printed is not memory). English and French only: a Cyrillic
# statement naming a Latin project name is dropped as a fragment,
# a defect of its own
NEIGHBOUR_LANGUAGES = ("en", "fr")
SERVICES = ("atlas", "borealis", "cobalt", "delta")
TOOLS = ("ember", "fjord", "granite", "harbor")
NEIGHBOUR_INTRO = {
    "en": "I maintain two side projects, {a} and {b}.",
    "fr": "Je maintiens deux projets perso, {a} et {b}.",
}
# (sentence, value) per language: the web service's, then the tool's
NEIGHBOUR_SERVICE = {
    "en": (
        "{name} is a Go web service with {n} contributors.",
        "a Go web service with {n} contributors",
    ),
    "fr": (
        "{name} est un service web en Go avec {n} contributeurs.",
        "un service web en Go avec {n} contributeurs",
    ),
}
NEIGHBOUR_TOOL = {
    "en": (
        "{name} is a Rust command-line tool with {n} plugins.",
        "a Rust command-line tool with {n} plugins",
    ),
    "fr": (
        "{name} est un outil en ligne de commande en Rust avec {n} plugins.",
        "un outil en ligne de commande en Rust avec {n} plugins",
    ),
}
# --neighbours: the web service again, told something unrelated to its
# description under the same generic key, as a model may key it (a
# shared free-form key alone does not prove a replacement; both stay live)
NEIGHBOUR_ASIDE = {
    "en": ("{name} is deployed on the shared server.", "deployed on the shared server"),
    "fr": ("{name} est déployé sur le serveur partagé.", "déployé sur le serveur partagé"),
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
    the value adjudication with."""
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
    """One complete sentence naming its subject in the turn's script."""
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


# --------------------------------------------------------------------------- --intent

INTENT_LANGUAGES = ("en", "fr", "ru")
# what names a value without stating it: a modality, or the example frame
INTENT_KINDS = ("hypothesis", "question", "guess", "plan", "example")


def intent_sentence(person: Person, kind: str, city: str, language: str) -> str:
    """A turn naming ``city`` as the person's home without stating it."""
    n, f = person.name(language), person.feminine
    v = CITIES[city][language]
    moved = "переехала" if f else "переехал"
    sure = "уверена" if f else "уверен"
    return {
        "hypothesis": {
            "en": f"If {n} moved to {v}, the commute would be long.",
            "fr": f"Si {n} déménageait à {v}, le trajet serait long.",
            "ru": f"Если бы {n} {moved} в {RU_TO[city]}, дорога была бы долгой.",
        },
        "question": {
            "en": f"Does {n} live in {v} these days?",
            "fr": f"Est-ce que {n} habite à {v} en ce moment ?",
            "ru": f"{n} сейчас живёт в {RU_IN[city]}?",
        },
        "guess": {
            "en": f"{n} probably lives in {v}, I am not sure.",
            "fr": f"{n} habite sans doute à {v}, je n'en suis pas sûr.",
            "ru": f"{n}, наверное, живёт в {RU_IN[city]}, я не {sure}.",
        },
        "plan": {
            "en": f"{n} might move to {v} one day.",
            "fr": f"{n} pourrait déménager à {v} un jour.",
            "ru": f"{n}, может быть, когда-нибудь переедет в {RU_TO[city]}.",
        },
        "example": {
            "en": f"For example, say {n} lives in {v}: then the map shows it.",
            "fr": f"Par exemple, disons que {n} habite à {v} : la carte le montre.",
            "ru": f"Например, допустим, {n} живёт в {RU_IN[city]}: тогда карта это покажет.",
        },
    }[kind][language]


PERSONA = "Dana Morel"  # a role: never a person of the corpus
# (frame-opening turn, a turn inside the frame, the turn that ends it), per language
PERSONA_FRAME: dict[str, tuple[str, str, str]] = {
    "en": (
        f"For this exercise, I am a pilot named {PERSONA} and I live in Kazan.",
        "My phone number is 0612345678 and my e-mail is dana.morel@example.org.",
        "End of the exercise, thanks.",
    ),
    "fr": (
        f"Pour cet exercice, je suis une pilote nommée {PERSONA} et j'habite à Kazan.",
        "Mon numéro est le 0612345678 et mon e-mail dana.morel@example.org.",
        "Fin de l'exercice, merci.",
    ),
    "ru": (
        f"Для этого упражнения я пилот по имени {PERSONA}, и я живу в Казани.",
        "Мой телефон 0612345678, а почта dana.morel@example.org.",
        "Конец упражнения, спасибо.",
    ),
}
# the persona a task sets: one turn
PERSONA_TASK: dict[str, str] = {
    "en": f"Answer the recruiter's questions as the candidate {PERSONA}, born 1990-04-02,"
    " 12 Lilac Street.",
    "fr": f"Réponds aux questions du recruteur comme la candidate {PERSONA},"
    " née le 1990-04-02, 12 rue des Lilas.",
    "ru": f"Ответь на вопросы рекрутера как кандидатка {PERSONA}, родилась"
    " 1990-04-02, улица Сиреневая, 12.",
}
# what an ideal model leaves of each: the activity, never the persona's values
PERSONA_EPISODE: dict[str, tuple[str, str]] = {  # (frame, task)
    "en": (
        "{s} played a pilot in an exercise.",
        "{s} rehearsed a job interview as a fictitious candidate.",
    ),
    "fr": (
        "{s} a joué une pilote dans un exercice.",
        "{s} a répété un entretien d'embauche comme une candidate fictive.",
    ),
    "ru": (
        "{s} сыграл пилота в упражнении.",
        "{s} отрепетировал собеседование как вымышленная кандидатка.",
    ),
}

# --------------------------------------------------------------------------- --worksession

WORK_SESSION = f"{ALICE}#work"
DECK = "Harbor Deck"
# an evening iterating on one thing: every fact holds for the task only
WORK_TURNS: tuple[tuple[str, str], ...] = (  # (turn, the fact an ideal model reads)
    (
        f"Let's build a short slide deck called {DECK}.",
        f"{DECK} is a slide deck being built.",
    ),
    ("Make the cover title 32 points high.", f"{DECK}'s cover title is 32 points high."),
    ("Change the background to dark blue.", f"{DECK}'s background is dark blue."),
    ("The text is too dense, set the line spacing to 1.6.", f"{DECK}'s line spacing is 1.6."),
    ("Current slide count in my draft: 27.", f"{DECK}'s current slide count is 27."),
    ("Now make the cover title 48 points high.", f"{DECK}'s cover title is 48 points high."),
    (
        "Fix the bug where the chart overflows the slide.",
        f"{DECK} has a bug where the chart overflows the slide.",
    ),
    (
        "Please put the export in the slides archive.",
        "{s} wants the export put in the slides archive.",
    ),
)
WORK_EPISODE = "{s} worked on a short slide deck, " + DECK + ", with a timeline and charts."


# --------------------------------------------------------------------------- --observed

# (name, key, lasting sentence and value, observations: sentence and value),
# per language; every sentence names the thing
OBSERVED_LANGUAGES = ("en", "fr")
OBSERVED_THINGS: dict[str, dict[str, Any]] = {
    "Pantry": {
        "key": "content",
        "lasting": {
            "en": ("Pantry is where spare jars go.", "spare jars"),
            "fr": ("Pantry sert aux bocaux de réserve.", "bocaux de réserve"),
        },
        "observations": {
            "en": (
                ("Pantry is empty.", "empty"),
                ("Pantry holds 3 jars.", "3 jars"),
                ("Pantry holds the spices.", "the spices"),
            ),
            "fr": (
                ("Pantry est vide.", "vide"),
                ("Pantry contient 3 bocaux.", "3 bocaux"),
                ("Pantry contient les épices.", "les épices"),
            ),
        },
    },
    "Mailbridge": {
        "key": "state",
        "lasting": {
            "en": ("Mailbridge is the service that relays the team's mail.", "mail relay"),
            "fr": ("Mailbridge est le service qui relaie le courrier de l'équipe.", "relais"),
        },
        "observations": {
            "en": (("Mailbridge is down.", "down"), ("Mailbridge is up again.", "up")),
            "fr": (
                ("Mailbridge est en panne.", "en panne"),
                ("Mailbridge fonctionne de nouveau.", "fonctionne"),
            ),
        },
    },
    "Printq": {
        "key": "content",
        "lasting": {
            "en": ("Printq is the queue for poster prints.", "poster prints"),
            "fr": ("Printq est la file des impressions d'affiches.", "impressions d'affiches"),
        },
        "observations": {
            "en": (
                ("Printq holds 4 jobs.", "4 jobs"),
                ("Printq holds 1 job.", "1 job"),
                ("Printq is empty.", "empty"),
            ),
            "fr": (
                ("Printq contient 4 tâches.", "4 tâches"),
                ("Printq contient 1 tâche.", "1 tâche"),
                ("Printq est vide.", "vide"),
            ),
        },
    },
}
OBSERVED_SAID = {"en": "I checked: {sentence}", "fr": "J'ai vérifié : {sentence}"}
OBSERVED_QUESTION = {"en": "What is the state of {name}?", "fr": "Quel est l'état de {name} ?"}


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
    neighbours: bool = False  # --neighbours: two projects described side by side
    aside: str | None = None  # --neighbours: the project told something unrelated
    # --intent: what the turn is when not meant as a fact (a modality, a frame),
    # "task" for --worksession; and the event a sincere control turn states
    intent: str | None = None
    control: bool = False
    episode: bool = False  # an ideal model leaves one episode from this turn
    ends_session: bool = False  # the host closes the session after this turn
    # --observed: the thing this turn is about, its observation's index (None:
    # the lasting fact), and the day the turn says it happened
    observed: str | None = None
    observation: int | None = None
    occurred_at: str | None = None
    # --self: the case this turn is (``bench/self_bench.py``), the assistant's
    # reply, the external parts (kind, text, ref), and the source's text as the
    # extraction reads it after its first label (the oracle's key)
    self_case: str | None = None
    assistant: str = ""
    external: tuple[tuple[str, str, str | None], ...] = ()
    vouched: bool = False  # --self: the host vouched for every external part
    source_key: str | None = None


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
    neighbours: bool = False
    intent: bool = False
    worksession: bool = False
    observed: bool = False
    agent: bool = False  # --self (the agent perspective)
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
        return (
            turn.event is None
            and turn.restates is None
            and not turn.neighbours
            and turn.aside is None
            and turn.intent is None
            and turn.observed is None
            and turn.self_case is None
        )

    @property
    def speaker_subject(self) -> str:
        """How an ideal model names the person speaking: the owner, or "User"."""
        return OWNER_NAME if self.speaker else "User"

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
            "neighbours": self.neighbours,
            "turns": len(self.turns),
            "event_turns": sum(1 for t in self.turns if t.event),
            "chatter_turns": sum(1 for t in self.turns if self.is_chatter(t)),
            "rephrase_turns": sum(1 for t in self.turns if t.restates),
            "added_events": sum(1 for e in self.events.values() if e.added),
            "speaker_turns": sum(1 for t in self.turns if t.speaker),
            "neighbour_turns": sum(1 for t in self.turns if t.neighbours),
            "aside_turns": sum(1 for t in self.turns if t.aside),
            "intent": self.intent,
            "worksession": self.worksession,
            "intent_turns": sum(1 for t in self.turns if t.intent and t.intent != "task"),
            "control_turns": sum(1 for t in self.turns if t.control),
            "task_turns": sum(1 for t in self.turns if t.intent == "task"),
            "expected_episodes": sum(1 for t in self.turns if t.episode) + int(self.worksession),
            "observed": self.observed,
            "observed_turns": sum(1 for t in self.turns if t.observation is not None),
            "self": self.agent,
            "self_turns": sum(1 for t in self.turns if t.self_case is not None),
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
    neighbours: bool = False,
    intent: bool = False,
    worksession: bool = False,
    observed: bool = False,
) -> Corpus:
    """The corpus for ``seed``: ``months`` months of events (at least 2), about
    ``chatter`` chatter turns per event turn. ``traps`` adds, for every person,
    where they work (a city other than their first home), drawn from a random
    generator of its own: without traps the corpus is unchanged. ``rephrase``
    ``speaker``, ``name_traps``, ``neighbours``, ``intent`` and ``worksession``
    (see the module) draw from generators of their own too, or none."""
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
        neighbours=neighbours,
        intent=intent,
        worksession=worksession,
        observed=observed,
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
    side = random.Random(f"neighbours:{seed}")
    known: dict[str, str] = {}  # person -> the name the first turn spelt (the node's name)
    unmeant = _intent_plan(random.Random(f"intent:{seed}"), planned) if intent else {}
    personas = intent
    held: list[Event] = []  # --worksession: the last month of user:alice
    for event in planned:
        person = next(p for p in people if p.key == event.person)
        corpus.events[event.key] = event
        if worksession and event.scope == ALICE and event.month == months:
            held.append(event)
            continue
        if personas and event.scope == ALICE and event.month >= 2:
            _personas(corpus, event.month)
            personas = False
        for kind, language in unmeant.get(event.key, ()):
            _tell_unmeant(corpus, person, event, kind, language, known)
        _tell(corpus, person, event, known, control=event.key in unmeant)
        if rephrase and len(person.languages) > 1 and restate.random() < 0.5:
            other = restate.choice([lang for lang in person.languages if lang != event.language])
            _tell(corpus, person, dataclasses.replace(event, language=other), known, again=True)
        if neighbours and side.random() < 0.25:
            _neighbour_turn(corpus, side, event.scope, event.month)
        extra = int(chatter) + (1 if rng.random() < chatter - int(chatter) else 0)
        for _ in range(extra):
            language = rng.choice(person.languages)
            line = rng.choice(CHATTER[language])
            corpus.turns.append(_turn(corpus, event.scope, language, event.month, line, None))
    if worksession:
        _work_session(corpus, held, people, known, months)
    if observed:
        _observed_turns(corpus, random.Random(f"observed:{seed}"), months)
    corpus.questions = _questions(corpus)
    return corpus


def observed_months(months: int, count: int) -> list[int]:
    """--observed: the months a thing is observed in, after the first (its
    lasting fact's), spread to the last."""
    if count == 1:
        return [months]
    step = (months - 2) / (count - 1)
    return sorted({round(2 + i * step) for i in range(count)})


def _observed_turns(corpus: Corpus, rng: random.Random, months: int) -> None:
    """--observed: each thing's lasting fact in month 1, then its observations
    in later months, every turn dated the 15th of its month; the gold reads the
    lasting fact as lasting and each observation as observed."""
    planned: list[tuple[int, str, str, int | None]] = []
    for name in OBSERVED_THINGS:
        language = rng.choice(OBSERVED_LANGUAGES)
        planned.append((1, name, language, None))
        count = len(OBSERVED_THINGS[name]["observations"][language])
        for index, month in enumerate(observed_months(months, count)):
            planned.append((month, name, language, index))
    for month, name, language, index in sorted(planned, key=lambda p: (p[0], p[1])):
        thing = OBSERVED_THINGS[name]
        if index is None:
            sentence, value = thing["lasting"][language]
            text, duration = sentence, "lasting"
        else:
            sentence, value = thing["observations"][language][index]
            text, duration = OBSERVED_SAID[language].format(sentence=sentence), "observed"
        turn = _turn(corpus, ALICE, language, month, text, None)
        corpus.turns.append(
            dataclasses.replace(
                turn,
                observed=name,
                observation=index,
                occurred_at=f"{YEAR}-{month:02d}-15T10:00:00Z",
            )
        )
        fact = _fact(
            name, None, sentence, None, value, month, sentence.rstrip("."), duration=duration
        )
        corpus.gold[text] = {
            "nodes": [{"name": name, "kind": "object", "aliases": [], "summary": ""}],
            "facts": [{**fact, "attribute": thing["key"], "cardinality": "one"}],
            "edges": [],
            "frame": "sincere",
            "frame_holds": False,
            "frame_ends": False,
        }


def _tell(
    corpus: Corpus,
    person: Person,
    event: Event,
    known: dict[str, str],
    *,
    again: bool = False,
    control: bool = False,
    session: str | None = None,
) -> None:
    """The turn telling ``event`` in ``event.language`` and its gold answer; with
    ``again``, a restatement of an event already told (``--rephrase``); with
    ``control``, the sincere turn an ``--intent`` turn is paired with."""
    owner = person.key == OWNER
    if owner:
        text = first_person(event.attribute, event.value, event.language, change=event.change)
    elif event.added:
        text = added_sentence(person, event.attribute, event.value, event.language)
    else:
        text = sentence(person, event.attribute, event.value, event.language, change=event.change)
    turn = _turn(
        corpus,
        event.scope,
        event.language,
        event.month,
        text,
        None if again else event.key,
        session=session,
    )
    corpus.turns.append(
        dataclasses.replace(
            turn, restates=event.key if again else None, speaker=owner, control=control
        )
    )
    if owner:
        corpus.gold[text] = owner_gold_answer(event, text)
        return
    written = person.name(event.language)
    name = known.setdefault(person.key, written)
    corpus.gold[text] = gold_answer(person, event, text, name, written)


def _neighbour_turn(corpus: Corpus, rng: random.Random, scope: str, month: int) -> None:
    """--neighbours: a web service and a command-line tool described side by
    side, each in a sentence of its own, and the gold answer: one lasting fact
    per project; then a turn telling the service something
    unrelated (``_aside_turn``)."""
    language = rng.choice(NEIGHBOUR_LANGUAGES)
    count = sum(1 for t in corpus.turns if t.neighbours)
    service = f"{rng.choice(SERVICES).capitalize()}-{count + 1}"
    tool = f"{rng.choice(TOOLS).capitalize()}-{count + 1}"
    told = []
    for name, (said, value) in (
        (service, NEIGHBOUR_SERVICE[language]),
        (tool, NEIGHBOUR_TOOL[language]),
    ):
        n = rng.randint(2, 9)
        told.append((name, said.format(name=name, n=n), value.format(n=n)))
    rng.shuffle(told)
    intro = NEIGHBOUR_INTRO[language].format(a=told[0][0], b=told[1][0])
    text = " ".join([intro, *(sentence for _, sentence, _ in told)])
    turn = _turn(corpus, scope, language, month, text, None)
    corpus.turns.append(dataclasses.replace(turn, neighbours=True))
    corpus.gold[text] = {
        "nodes": [
            {"name": name, "kind": "project", "aliases": [], "summary": ""} for name, _, _ in told
        ],
        "facts": [
            {
                "subject": name,
                "subject_as_written": None,
                "statement": sentence,
                "attribute": "description",
                "value": value,
                "valid_from": f"{YEAR}-{month:02d}",
                "quote": sentence.rstrip("."),
                "confidence": 0.9,
                "modality": "fact",
                "duration": "lasting",
            }
            for name, sentence, value in told
        ],
        "edges": [],
        "frame": "sincere",
    }
    _aside_turn(corpus, scope, language, month, service)


def _aside_turn(corpus: Corpus, scope: str, language: str, month: int, name: str) -> None:
    """--neighbours: the web service told something unrelated to its description,
    keyed ``description`` too and said to hold one value - the slip of a real
    model on a generic key. Its gold: both facts true at once. No
    random draw: the rest of the corpus stays as it was."""
    said, value = NEIGHBOUR_ASIDE[language]
    text = said.format(name=name)
    turn = _turn(corpus, scope, language, month, text, None)
    corpus.turns.append(dataclasses.replace(turn, aside=name))
    corpus.gold[text] = {
        "nodes": [{"name": name, "kind": "project", "aliases": [], "summary": ""}],
        "facts": [
            {
                "subject": name,
                "subject_as_written": None,
                "statement": text,
                "attribute": "description",
                "value": value,
                "valid_from": f"{YEAR}-{month:02d}",
                "cardinality": "one",
                "quote": text.rstrip("."),
                "confidence": 0.9,
                "modality": "fact",
                "duration": "lasting",
            }
        ],
        "edges": [],
        "frame": "sincere",
    }


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
    corpus: Corpus,
    scope: str,
    language: str,
    month: int,
    text: str,
    event: str | None,
    *,
    session: str | None = None,
) -> Turn:
    return Turn(
        key=f"t{len(corpus.turns):03d}",
        scope=scope,
        session=session or f"{scope}#{YEAR}-{month:02d}",
        language=language,
        month=month,
        text=text,
        event=event,
    )


def _intent_plan(rng: random.Random, planned: list[Event]) -> dict[str, list[tuple[str, str]]]:
    """--intent: for every person and each of their languages among English,
    French and Russian, one turn not meant as a fact about one of their homes,
    told before the event that states that home (the control)."""
    out: dict[str, list[tuple[str, str]]] = {}
    for person in PEOPLE:
        homes = [e for e in planned if e.person == person.key and e.attribute == "city"]
        for language in (lang for lang in person.languages if lang in INTENT_LANGUAGES):
            event = rng.choice(homes)
            out.setdefault(event.key, []).append((rng.choice(INTENT_KINDS), language))
    return out


def _tell_unmeant(
    corpus: Corpus, person: Person, event: Event, kind: str, language: str, known: dict[str, str]
) -> None:
    """The --intent turn of ``kind`` naming ``event``'s home, and its gold: the
    content as an ideal model reads it - a hypothesis, a question..., or the
    example frame - of which the service must write nothing."""
    text = intent_sentence(person, kind, event.value, language)
    written = person.name(language)
    name = known.get(person.key, written)
    statement = sentence(person, "city", event.value, language, change=False)
    fact = _fact(
        name,
        written,
        statement,
        "city",
        value_text("city", event.value, language),
        event.month,
        text,
        modality="fact" if kind == "example" else kind,
    )
    corpus.gold[text] = {
        "nodes": [{"name": name, "kind": "person", "aliases": [], "summary": ""}],
        "facts": [fact],
        "edges": [],
        "frame": "example" if kind == "example" else "sincere",
    }
    turn = _turn(corpus, event.scope, language, event.month, text, None)
    corpus.turns.append(dataclasses.replace(turn, intent=kind))


def _personas(corpus: Corpus, month: int) -> None:
    """--intent, in user:alice: in each persona language, a frame over three
    turns (opened, kept, ended) and a persona the task sets, each leaving one
    episode in an ideal answer and nothing of the persona's values."""
    who = corpus.speaker_subject
    for language in INTENT_LANGUAGES:
        opened, inside, ended = PERSONA_FRAME[language]
        subject = RU_SPEAKER if language == "ru" else who
        frame_episode, task_episode = PERSONA_EPISODE[language]
        persona = [
            _fact(PERSONA, PERSONA, f"{PERSONA} lives in Kazan.", "city", "Kazan", month, opened)
        ]
        corpus.gold[opened] = {
            "nodes": [{"name": PERSONA, "kind": "person", "aliases": [], "summary": ""}],
            "facts": [*persona, _episode(who, frame_episode.format(s=subject), month, opened)],
            "edges": [],
            "frame": "pretend",
            "frame_holds": True,
        }
        corpus.gold[inside] = {
            "nodes": [],
            "facts": [
                _fact(
                    PERSONA,
                    PERSONA,
                    f"{PERSONA}'s e-mail is dana.morel@example.org.",
                    "email",
                    "dana.morel@example.org",
                    month,
                    inside,
                )
            ],
            "edges": [],
            "frame": "pretend",
        }
        corpus.gold[ended] = {
            "nodes": [],
            "facts": [],
            "edges": [],
            "frame": "sincere",
            "frame_ends": True,
        }
        task = PERSONA_TASK[language]
        corpus.gold[task] = {
            "nodes": [{"name": PERSONA, "kind": "person", "aliases": [], "summary": ""}],
            "facts": [
                _fact(
                    PERSONA,
                    PERSONA,
                    f"{PERSONA} was born on 1990-04-02.",
                    "birth_date",
                    "1990-04-02",
                    month,
                    task,
                ),
                _episode(who, task_episode.format(s=subject), month, task),
            ],
            "edges": [],
            "frame": "pretend",
        }
        for text, kind, episode in (
            (opened, "pretend", True),
            (inside, "pretend", False),
            (ended, "pretend", False),
            (task, "pretend", True),
        ):
            turn = _turn(corpus, ALICE, language, month, text, None)
            corpus.turns.append(dataclasses.replace(turn, intent=kind, episode=episode))


def _work_session(
    corpus: Corpus,
    held: list[Event],
    people: tuple[Person, ...],
    known: dict[str, str],
    month: int,
) -> None:
    """--worksession: the iterations on one slide deck, the held lasting events of the
    month told in between, in one session the host then closes."""
    who = corpus.speaker_subject
    lasting = list(held)
    every = max(1, len(WORK_TURNS) // max(1, len(lasting))) if lasting else len(WORK_TURNS)
    for index, (text, statement) in enumerate(WORK_TURNS):
        corpus.gold[text] = {
            "nodes": [],
            "facts": [
                _fact(
                    who if "{s}" in statement else DECK,
                    None,
                    statement.format(s=who),
                    None,
                    None,
                    month,
                    text,
                    duration="task",
                )
            ],
            "edges": [],
            "frame": "sincere",
        }
        turn = _turn(corpus, ALICE, "en", month, text, None, session=WORK_SESSION)
        corpus.turns.append(dataclasses.replace(turn, intent="task"))
        if lasting and index % every == every - 1:
            event = lasting.pop(0)
            person = next(p for p in people if p.key == event.person)
            _tell(corpus, person, event, known, session=WORK_SESSION)
    for event in lasting:
        person = next(p for p in people if p.key == event.person)
        _tell(corpus, person, event, known, session=WORK_SESSION)
    last = corpus.turns[-1]
    corpus.turns[-1] = dataclasses.replace(last, ends_session=True)
    corpus.gold[WORK_SESSION] = {"episode": WORK_EPISODE.format(s=who), "keep": []}


def _fact(
    subject: str,
    written: str | None,
    statement: str,
    attribute: str | None,
    value: str | None,
    month: int,
    quote: str,
    *,
    modality: str = "fact",
    duration: str = "lasting",
) -> dict[str, Any]:
    return {
        "subject": subject,
        "subject_as_written": written if written and written != subject else None,
        "statement": statement,
        "attribute": GOLD_KEYS.get(attribute or "", attribute),
        "value": value,
        "valid_from": f"{YEAR}-{month:02d}",
        "quote": quote,
        "confidence": 0.9,
        "modality": modality,
        "duration": duration,
    }


def _episode(subject: str, statement: str, month: int, quote: str) -> dict[str, Any]:
    return {**_fact(subject, None, statement, None, None, month, quote), "episode": True}


def gold_answer(person: Person, event: Event, text: str, name: str, written: str) -> dict[str, Any]:
    """What an ideal model answers to the extraction prompt for this turn: the
    known name reused, the source's spelling in ``subject_as_written``."""
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
                "modality": "fact",
                "duration": "lasting",
            }
        ],
        "edges": [],
        "frame": "sincere",
    }


def owner_gold_answer(event: Event, text: str) -> dict[str, Any]:
    """What an ideal model answers for the owner's own words: the
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
                "modality": "fact",
                "duration": "lasting",
            }
        ],
        "edges": [],
        "frame": "sincere",
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
