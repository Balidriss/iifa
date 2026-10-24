"""Settings: a TOML file, overridden by environment variables.

The database key is supplied by the operator (P5): ``IIFA_DB_KEY`` or
``IIFA_DB_KEY_FILE``. iifa never generates one and never stores one next to the
database. Tokens are kept as SHA-256 digests only; a plaintext token given in
the configuration is hashed at load time and dropped.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from iifa.language import is_code
from iifa.llm import RESERVED_REQUEST_KEYS
from iifa.scopes import ROOT_SCOPE, ScopeError, validate_scope

PERMISSIONS: frozenset[str] = frozenset({"read", "write", "forget", "dream", "review", "admin"})
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
BACKGROUND_MODES: tuple[str, ...] = ("auto", "paused", "on_demand")
MIN_ANSWER_TOKENS = 512  # the answer cap's base (SPEC 7.4b): a bound below it caps nothing
RISK_CLASSES: tuple[str, ...] = ("low", "medium", "high")  # SPEC 5.6 rule 5
REVIEW_ACTIONS: tuple[str, ...] = ("auto", "review")
MAX_ADJUDICATIONS_PER_UNIT = 20  # SPEC 5.6: merge candidates asked about per unit
MIN_PROMPT_TOKENS = 1000  # SPEC 5.6: below it no dream prompt fits its own rules


class ConfigError(ValueError):
    """The configuration is missing something iifa refuses to guess."""


@dataclass(frozen=True)
class TokenConfig:
    """A client token: a digest, the scopes it may use and its permissions."""

    name: str
    sha256: str
    scopes: tuple[str, ...]
    permissions: frozenset[str]

    @staticmethod
    def from_plaintext(
        token: str, *, name: str, scopes: tuple[str, ...], permissions: frozenset[str]
    ) -> TokenConfig:
        return TokenConfig(name, hash_token(token), scopes, permissions)


@dataclass(frozen=True)
class Settings:
    db_path: Path
    db_key: str = field(repr=False, default="")
    tokens: tuple[TokenConfig, ...] = ()
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    tokenizer: str = "conservative"
    default_budget_tokens: int = 400
    max_budget_tokens: int = 16000
    retain_sources: bool = True
    checkpoint_interval: float = 0.5
    recall_candidates: int = 200
    graph_neighbour_facts: int = 3
    mcp_http: bool = True
    mcp_expose_forget: bool = False
    mcp_default_scope: str | None = None
    background_mode: str = "auto"
    chat_concurrency: int = 1
    embeddings_concurrency: int = 1
    yield_timeout: float = 30.0
    max_attempts: int = 3
    retry_delay: float = 30.0
    worker_poll: float = 1.0
    chat_url: str | None = None
    chat_model: str | None = None
    chat_api_key: str = field(repr=False, default="")
    embeddings_url: str | None = None
    embeddings_model: str | None = None
    embeddings_api_key: str = field(repr=False, default="")
    model_timeout: float = 120.0
    chat_extra: Mapping[str, Any] = field(default_factory=dict)
    embeddings_extra: Mapping[str, Any] = field(default_factory=dict)
    chat_json_mode: bool = True
    answer_max_tokens: int = 4096
    recall_dense: bool = True
    query_timeout: float = 1.5
    # None: from the embedding model's profile (SPEC 6.1, iifa.calibration)
    resolve_threshold: float | None = None
    merge_threshold: float | None = None
    link_threshold: float | None = None
    key_threshold: float | None = None  # SPEC 5.4: attribute keys aligned by embeddings
    embed_batch: int = 32
    extraction_chunk_chars: int = 12000
    confirm_after_sources: int = 2
    edge_half_life_days: float = 30.0
    edge_prune_below: float = 0.1
    summary_nodes_per_session: int = 20
    language_overrides: tuple[tuple[str, str], ...] = ()
    # SPEC 5.4: role, team and preferences asked "replaces, or both true?" - on
    # by default (measured on qwen3.5:9b with the source's quotes)
    replacement_question: bool = True
    # SPEC 4.6: (scope, name, aliases) - who the memory of a scope belongs to
    owners: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    # SPEC 5.6: the dream's own model (rule 10), its probe tolerance (rule 9), the
    # merge candidates adjudicated per unit (0: none), the prompt budget of one
    # call (counted with ``tokenizer``) and the review policy per
    # scope: (scope, risk, auto | review), nearest wins
    dream_url: str | None = None
    dream_model: str | None = None
    dream_api_key: str = field(repr=False, default="")
    dream_extra: Mapping[str, Any] = field(default_factory=dict)
    dream_answer_max_tokens: int = 2048
    dream_probe_tolerance: float = 0.02
    dream_adjudications_per_unit: int = 3
    dream_prompt_max_tokens: int = 4000
    dream_review: tuple[tuple[str, str, str], ...] = ()
    plaintext_database_for_tests: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_background()
        self._validate_models()
        self._validate_extraction()
        self._validate_owners()
        self._validate_embeddings()
        self._validate_dream()
        if not self.db_key and not self.plaintext_database_for_tests:
            raise ConfigError(
                "no database key: set IIFA_DB_KEY or IIFA_DB_KEY_FILE "
                "(iifa never generates a key for you)"
            )
        if self.default_budget_tokens < 1 or self.default_budget_tokens > self.max_budget_tokens:
            raise ConfigError("default_budget_tokens must be within 1..max_budget_tokens")
        if not (0 < self.port < 65536):
            raise ConfigError("port must be within 1..65535")
        if self.checkpoint_interval < 0:
            raise ConfigError("checkpoint_interval must be 0 (off) or a number of seconds")
        if self.mcp_default_scope is not None:
            try:
                validate_scope(self.mcp_default_scope)
            except ScopeError as exc:
                raise ConfigError(f"mcp default_scope: {exc}") from exc
        for token in self.tokens:
            unknown = token.permissions - PERMISSIONS
            if unknown:
                raise ConfigError(f"token {token.name!r}: unknown permissions {sorted(unknown)}")
            if not token.scopes:
                raise ConfigError(f"token {token.name!r} is bound to no scope")
            for scope in token.scopes:
                if scope != ROOT_SCOPE:
                    try:
                        validate_scope(scope)
                    except ScopeError as exc:
                        raise ConfigError(f"token {token.name!r}: {exc}") from exc

    def _validate_background(self) -> None:
        if self.background_mode not in BACKGROUND_MODES:
            raise ConfigError(
                f"background mode {self.background_mode!r}: expected one of {BACKGROUND_MODES}"
            )
        for name in ("chat_concurrency", "embeddings_concurrency", "max_attempts"):
            if int(getattr(self, name)) < 1:
                raise ConfigError(f"{name} must be at least 1")
        for name in ("yield_timeout", "model_timeout", "worker_poll"):
            if float(getattr(self, name)) <= 0:
                raise ConfigError(f"{name} must be a positive number of seconds")
        if self.retry_delay < 0:
            raise ConfigError("retry_delay must be 0 or a number of seconds")

    def _validate_extraction(self) -> None:
        if self.extraction_chunk_chars < 500:
            raise ConfigError("extraction_chunk_chars must be at least 500")
        if self.answer_max_tokens < MIN_ANSWER_TOKENS:
            raise ConfigError(f"answer_max_tokens must be at least {MIN_ANSWER_TOKENS}")
        if self.confirm_after_sources < 1:
            raise ConfigError("confirm_after_sources must be at least 1")
        if self.edge_half_life_days <= 0:
            raise ConfigError("edge_half_life_days must be a positive number of days")
        if not (0 <= self.edge_prune_below < 1):
            raise ConfigError("edge_prune_below must be within 0..1")
        if self.summary_nodes_per_session < 0:
            raise ConfigError("summary_nodes_per_session must be 0 or more")
        for scope, code in self.language_overrides:
            if scope != ROOT_SCOPE:
                try:
                    validate_scope(scope)
                except ScopeError as exc:
                    raise ConfigError(f"language override: {exc}") from exc
            if not is_code(code):
                raise ConfigError(
                    f"language override for {scope!r}: {code!r} is not an ISO 639 code"
                )

    def _validate_owners(self) -> None:
        """SPEC 4.6: who the memory of a scope belongs to, by name."""
        for scope, name, aliases in self.owners:
            if scope != ROOT_SCOPE:
                try:
                    validate_scope(scope)
                except ScopeError as exc:
                    raise ConfigError(f"owner: {exc}") from exc
            if not name.strip() or any(not a.strip() for a in aliases):
                raise ConfigError(f"owner of {scope!r}: a name (and aliases) must not be empty")

    def _validate_embeddings(self) -> None:
        """SPEC 6.1: the thresholds are cosines in 0..1; a merge proposal must be
        stricter than a resolution candidate."""
        if self.query_timeout <= 0:
            raise ConfigError("embeddings query_timeout must be a positive number of seconds")
        for name in ("resolve_threshold", "merge_threshold", "link_threshold", "key_threshold"):
            value = getattr(self, name)
            if value is not None and not (0 < float(value) <= 1):
                raise ConfigError(f"embeddings {name} must be within 0..1")
        resolve, merge = self.resolve_threshold, self.merge_threshold
        if resolve is not None and merge is not None and merge < resolve:
            raise ConfigError("embeddings merge_threshold must be at least resolve_threshold")
        if self.embed_batch < 1:
            raise ConfigError("embeddings batch must be at least 1")

    def _validate_dream(self) -> None:
        """SPEC 5.6: the probe tolerance is a fraction; the review policy names a
        scope, a risk class and what happens to it."""
        if not (0 <= self.dream_probe_tolerance < 1):
            raise ConfigError("dream probe_tolerance must be within 0..1")
        if self.dream_prompt_max_tokens < MIN_PROMPT_TOKENS:
            raise ConfigError(f"dream prompt_max_tokens must be at least {MIN_PROMPT_TOKENS}")
        if not (0 <= self.dream_adjudications_per_unit <= MAX_ADJUDICATIONS_PER_UNIT):
            raise ConfigError(
                f"dream adjudications_per_unit must be within 0..{MAX_ADJUDICATIONS_PER_UNIT}"
            )
        if self.dream_answer_max_tokens < MIN_ANSWER_TOKENS:
            raise ConfigError(f"dream answer_max_tokens must be at least {MIN_ANSWER_TOKENS}")
        for scope, risk, action in self.dream_review:
            if scope != ROOT_SCOPE:
                try:
                    validate_scope(scope)
                except ScopeError as exc:
                    raise ConfigError(f"dream review: {exc}") from exc
            if risk not in RISK_CLASSES:
                raise ConfigError(f"dream review for {scope!r}: unknown risk class {risk!r}")
            if action not in REVIEW_ACTIONS:
                raise ConfigError(
                    f"dream review for {scope!r}: {risk} must be one of {REVIEW_ACTIONS}"
                )

    def _validate_models(self) -> None:
        for kind in ("chat", "embeddings", "dream"):
            url = getattr(self, f"{kind}_url")
            model = getattr(self, f"{kind}_model")
            if bool(url) != bool(model):
                raise ConfigError(
                    f"{kind} endpoint needs both {kind}_url and {kind}_model:"
                    " the model is explicit on every call"
                )
            if url and not str(url).startswith(("http://", "https://")):
                raise ConfigError(f"{kind}_url must start with http:// or https://")
            extra = getattr(self, f"{kind}_extra")
            if not isinstance(extra, Mapping):
                raise ConfigError(f"{kind}_extra must be a table of request parameters")
            clash = sorted(set(extra) & RESERVED_REQUEST_KEYS)
            if clash:
                raise ConfigError(
                    f"{kind}_extra may not replace what iifa sets itself: {', '.join(clash)}"
                )
            try:
                json.dumps(dict(extra))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{kind}_extra must hold JSON values only: {exc}") from exc


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- loading


def _token_from_mapping(raw: Mapping[str, Any], index: int) -> TokenConfig:
    name = str(raw.get("name") or f"token{index}")
    scopes_raw = raw.get("scopes", [ROOT_SCOPE])
    if isinstance(scopes_raw, str):
        scopes_raw = [scopes_raw]
    permissions_raw = raw.get("permissions", ["read", "write"])
    if isinstance(permissions_raw, str):
        permissions_raw = [permissions_raw]
    scopes = tuple(str(s) for s in scopes_raw)
    permissions = frozenset(str(p) for p in permissions_raw)
    if "token_sha256" in raw:
        return TokenConfig(name, str(raw["token_sha256"]).lower(), scopes, permissions)
    if "token" in raw:
        return TokenConfig.from_plaintext(
            str(raw["token"]), name=name, scopes=scopes, permissions=permissions
        )
    raise ConfigError(f"token {name!r} needs 'token' or 'token_sha256'")


def _read_key(env: Mapping[str, str], file_section: Mapping[str, Any]) -> str:
    if env.get("IIFA_DB_KEY"):
        return env["IIFA_DB_KEY"]
    key_file = env.get("IIFA_DB_KEY_FILE") or file_section.get("key_file")
    if key_file:
        path = Path(str(key_file)).expanduser()
        if not path.is_file():
            raise ConfigError(f"database key file not found: {path}")
        return path.read_text(encoding="utf-8").strip()
    return ""


def load_settings(
    env: Mapping[str, str] | None = None, config_file: str | os.PathLike[str] | None = None
) -> Settings:
    """Build ``Settings`` from a TOML file (``IIFA_CONFIG``/``config_file``) and the environment."""
    env = os.environ if env is None else env
    data: dict[str, Any] = {}
    path = config_file or env.get("IIFA_CONFIG")
    if path:
        with Path(path).open("rb") as handle:
            data = tomllib.load(handle)
    service = data.get("service", {})
    store = data.get("store", {})
    recall = data.get("recall", {})
    mcp = data.get("mcp", {})
    background = data.get("background", {})
    models = data.get("models", {})
    extraction = data.get("extraction", {})
    embeddings = data.get("embeddings", {})
    dream = data.get("dream", {})

    tokens: list[TokenConfig] = [
        _token_from_mapping(raw, i) for i, raw in enumerate(data.get("tokens", []))
    ]
    if env.get("IIFA_TOKENS"):
        tokens.extend(
            _token_from_mapping(raw, i) for i, raw in enumerate(json.loads(env["IIFA_TOKENS"]))
        )
    if env.get("IIFA_TOKEN"):
        tokens.append(
            TokenConfig.from_plaintext(
                env["IIFA_TOKEN"], name="env", scopes=(ROOT_SCOPE,), permissions=PERMISSIONS
            )
        )

    db_path = Path(env.get("IIFA_DB_PATH") or store.get("path") or "iifa.db").expanduser()
    return Settings(
        db_path=db_path,
        db_key=_read_key(env, store),
        tokens=tuple(tokens),
        host=str(env.get("IIFA_HOST") or service.get("host") or DEFAULT_HOST),
        port=int(env.get("IIFA_PORT") or service.get("port") or DEFAULT_PORT),
        tokenizer=str(env.get("IIFA_TOKENIZER") or recall.get("tokenizer") or "conservative"),
        default_budget_tokens=int(
            env.get("IIFA_DEFAULT_BUDGET_TOKENS") or recall.get("default_budget_tokens") or 400
        ),
        max_budget_tokens=int(recall.get("max_budget_tokens") or 16000),
        retain_sources=bool(store.get("retain_sources", True)),
        checkpoint_interval=float(store.get("checkpoint_interval", 0.5)),
        mcp_http=_flag(env.get("IIFA_MCP_HTTP"), mcp.get("http", True)),
        mcp_expose_forget=_flag(env.get("IIFA_MCP_EXPOSE_FORGET"), mcp.get("expose_forget", False)),
        mcp_default_scope=str(env.get("IIFA_MCP_DEFAULT_SCOPE") or mcp.get("default_scope") or "")
        or None,
        background_mode=str(env.get("IIFA_BACKGROUND_MODE") or background.get("mode") or "auto"),
        chat_concurrency=int(
            env.get("IIFA_CHAT_CONCURRENCY") or background.get("chat_concurrency") or 1
        ),
        embeddings_concurrency=int(
            env.get("IIFA_EMBEDDINGS_CONCURRENCY") or background.get("embeddings_concurrency") or 1
        ),
        yield_timeout=float(background.get("yield_timeout", 30.0)),
        max_attempts=int(background.get("max_attempts", 3)),
        retry_delay=float(background.get("retry_delay", 30.0)),
        chat_url=_optional(env.get("IIFA_CHAT_URL") or models.get("chat_url")),
        chat_model=_optional(env.get("IIFA_CHAT_MODEL") or models.get("chat_model")),
        chat_api_key=str(env.get("IIFA_CHAT_API_KEY") or models.get("chat_api_key") or ""),
        embeddings_url=_optional(env.get("IIFA_EMBEDDINGS_URL") or models.get("embeddings_url")),
        embeddings_model=_optional(
            env.get("IIFA_EMBEDDINGS_MODEL") or models.get("embeddings_model")
        ),
        embeddings_api_key=str(
            env.get("IIFA_EMBEDDINGS_API_KEY") or models.get("embeddings_api_key") or ""
        ),
        model_timeout=float(env.get("IIFA_MODEL_TIMEOUT") or models.get("timeout") or 120.0),
        chat_extra=_extra(env.get("IIFA_CHAT_EXTRA"), models.get("chat_extra", {}), "chat"),
        embeddings_extra=_extra(
            env.get("IIFA_EMBEDDINGS_EXTRA"), models.get("embeddings_extra", {}), "embeddings"
        ),
        chat_json_mode=_flag(env.get("IIFA_CHAT_JSON_MODE"), models.get("chat_json_mode", True)),
        answer_max_tokens=int(
            env.get("IIFA_ANSWER_MAX_TOKENS") or extraction.get("answer_max_tokens") or 4096
        ),
        recall_dense=_flag(env.get("IIFA_RECALL_DENSE"), embeddings.get("recall_dense", True)),
        query_timeout=float(
            env.get("IIFA_EMBEDDINGS_QUERY_TIMEOUT") or embeddings.get("query_timeout", 1.5)
        ),
        resolve_threshold=_optional_float(embeddings.get("resolve_threshold")),
        merge_threshold=_optional_float(embeddings.get("merge_threshold")),
        link_threshold=_optional_float(embeddings.get("link_threshold")),
        key_threshold=_optional_float(embeddings.get("key_threshold")),
        embed_batch=int(embeddings.get("batch", 32)),
        extraction_chunk_chars=int(extraction.get("chunk_chars", 12000)),
        confirm_after_sources=int(extraction.get("confirm_after_sources", 2)),
        edge_half_life_days=float(extraction.get("edge_half_life_days", 30.0)),
        edge_prune_below=float(extraction.get("edge_prune_below", 0.1)),
        summary_nodes_per_session=int(extraction.get("summary_nodes_per_session", 20)),
        language_overrides=_overrides(
            env.get("IIFA_LANGUAGE_OVERRIDES"), extraction.get("language_overrides", {})
        ),
        owners=_owners(env.get("IIFA_OWNERS"), extraction.get("owners", {})),
        replacement_question=_flag(
            env.get("IIFA_REPLACEMENT_QUESTION"), extraction.get("replacement_question", True)
        ),
        dream_url=_optional(env.get("IIFA_DREAM_URL") or dream.get("url")),
        dream_model=_optional(env.get("IIFA_DREAM_MODEL") or dream.get("model")),
        dream_api_key=str(env.get("IIFA_DREAM_API_KEY") or dream.get("api_key") or ""),
        dream_extra=_extra(env.get("IIFA_DREAM_EXTRA"), dream.get("extra", {}), "dream"),
        dream_answer_max_tokens=int(dream.get("answer_max_tokens", 2048)),
        dream_probe_tolerance=float(dream.get("probe_tolerance", 0.02)),
        dream_adjudications_per_unit=int(dream.get("adjudications_per_unit", 3)),
        dream_prompt_max_tokens=int(dream.get("prompt_max_tokens", 4000)),
        dream_review=_review(dream.get("review", {})),
    )


def _review(file_value: object) -> tuple[tuple[str, str, str], ...]:
    """``{"user:alice": {"medium": "review"}}`` from the file: what the dream does
    with each risk class in a scope (SPEC 5.6 rule 5)."""
    if not isinstance(file_value, Mapping):
        raise ConfigError("dream review must be a table of scope to {low, medium, high}")
    out: list[tuple[str, str, str]] = []
    for scope, entry in file_value.items():
        if not isinstance(entry, Mapping):
            raise ConfigError(f"dream review for {scope!r}: a table of risk class to action")
        out.extend((str(scope), str(risk), str(action)) for risk, action in entry.items())
    return tuple(out)


def _overrides(env_value: str | None, file_value: object) -> tuple[tuple[str, str], ...]:
    """``{"user:alice": "fr"}`` from the environment (JSON) or the file (a table):
    the language a scope forces on what is written into it (P12)."""
    raw: object = file_value
    if env_value is not None and env_value.strip():
        raw = json.loads(env_value)
    if not isinstance(raw, Mapping):
        raise ConfigError("language_overrides must be a mapping of scope to language code")
    return tuple((str(k), str(v).strip().lower()) for k, v in raw.items())


def _owners(
    env_value: str | None, file_value: object
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """``{"user:alice": {"name": "Alice", "aliases": ["Алиса"]}}`` (or just a
    name) from the environment (JSON) or the file (a table): who the memory of a
    scope belongs to (SPEC 4.6)."""
    raw: object = file_value
    if env_value is not None and env_value.strip():
        try:
            raw = json.loads(env_value)
        except ValueError as exc:
            raise ConfigError(f"owners: not JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError("owners must be a mapping of scope to a name or {name, aliases}")
    out: list[tuple[str, str, tuple[str, ...]]] = []
    for scope, entry in raw.items():
        name: object
        aliases: object
        if isinstance(entry, str):
            name, aliases = entry, []
        elif isinstance(entry, Mapping):
            name, aliases = entry.get("name"), entry.get("aliases", [])
        else:
            raise ConfigError(f"owner of {scope!r}: a name or {{name, aliases}}")
        if not isinstance(name, str) or not isinstance(aliases, list):
            raise ConfigError(f"owner of {scope!r}: a name and a list of aliases")
        out.append((str(scope), name.strip(), tuple(str(a).strip() for a in aliases)))
    return tuple(out)


def _extra(env_value: str | None, file_value: object, kind: str) -> dict[str, Any]:
    """Extra request parameters of one endpoint (SPEC 7.4b), from the environment
    (a JSON object) or the file (a table)."""
    raw: object = file_value
    if env_value is not None and env_value.strip():
        try:
            raw = json.loads(env_value)
        except ValueError as exc:
            raise ConfigError(f"{kind}_extra: not JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{kind}_extra must be a JSON object / TOML table")
    return {str(k): v for k, v in raw.items()}


def _optional_float(value: object) -> float | None:
    return None if value is None else float(str(value))


def _optional(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _flag(env_value: str | None, file_value: object) -> bool:
    """A boolean from the environment (``1/true/yes/on`` or ``0/false/no/off``), else the file."""
    if env_value is not None and env_value.strip() != "":
        lowered = env_value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"not a boolean: {env_value!r}")
    return bool(file_value)
