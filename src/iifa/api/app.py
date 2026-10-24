"""The FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from iifa import __version__
from iifa.api.auth import TokenRegistry
from iifa.api.dream import router as dream_router
from iifa.api.routes import router
from iifa.config import Settings
from iifa.dream.adjudicate import NameNeighbours, NamesClose, name_neighbours, names_close
from iifa.dream.run import DREAM_JOB, DreamConfig, Dreamer, dream_job
from iifa.ingest.extract import PipelineConfig
from iifa.ingest.jobs import HANDLERS
from iifa.ingest.queue import Queue
from iifa.ingest.worker import Gate, Worker
from iifa.llm import CHAT, EMBEDDINGS, Endpoint, ModelClient, QueryEmbedder
from iifa.mcp.http import mcp_lifespan, mount_mcp
from iifa.recall.dense import Dense
from iifa.recall.engine import RecallEngine
from iifa.recall.probes import run_probe_set
from iifa.store.repo import NotFound, Store, utc_now
from iifa.tokens import get_tokenizer


def build_models(settings: Settings, gate: Gate) -> ModelClient:
    """The model client toward the endpoints the operator configured (P4): none,
    chat only (tier 2), or chat and embeddings (tier 3)."""
    chat = (
        Endpoint(
            CHAT,
            settings.chat_url,
            settings.chat_model,
            settings.chat_api_key,
            settings.model_timeout,
            settings.chat_extra,
        )
        if settings.chat_url and settings.chat_model
        else None
    )
    embeddings = (
        Endpoint(
            EMBEDDINGS,
            settings.embeddings_url,
            settings.embeddings_model,
            settings.embeddings_api_key,
            settings.model_timeout,
            settings.embeddings_extra,
        )
        if settings.embeddings_url and settings.embeddings_model
        else None
    )
    return ModelClient(chat, embeddings, gate=gate)


def build_dream_models(settings: Settings, gate: Gate) -> ModelClient | None:
    """The dream's own model (SPEC 5.6 rule 10), through the chat gate: None when
    the operator configured none - the extraction model never stands in."""
    if not (settings.dream_url and settings.dream_model):
        return None
    endpoint = Endpoint(
        CHAT,
        settings.dream_url,
        settings.dream_model,
        settings.dream_api_key,
        settings.model_timeout,
        settings.dream_extra,
    )
    return ModelClient(endpoint, None, gate=gate)


def build_pipeline(settings: Settings) -> PipelineConfig:
    return PipelineConfig(
        chunk_chars=settings.extraction_chunk_chars,
        answer_max_tokens=settings.answer_max_tokens,
        json_mode=settings.chat_json_mode,
        resolve_threshold=settings.resolve_threshold,
        merge_threshold=settings.merge_threshold,
        link_threshold=settings.link_threshold,
        key_threshold=settings.key_threshold,
        embed_batch=settings.embed_batch,
        confirm_after_sources=settings.confirm_after_sources,
        edge_half_life_days=settings.edge_half_life_days,
        edge_prune_below=settings.edge_prune_below,
        summary_nodes_per_session=settings.summary_nodes_per_session,
        language_overrides=settings.language_overrides,
        owners=settings.owners,
        replacement_question=settings.replacement_question,
    )


def _name_neighbours(
    store: Store, model: str | None, pipeline: PipelineConfig
) -> NameNeighbours | None:
    """SPEC 5.6: name vectors give merge candidates only at tier 3 with a
    calibrated embedding model (SPEC 6.1); otherwise proposals alone do."""
    thresholds = pipeline.thresholds(model) if model is not None else None
    if model is None or thresholds is None:
        return None
    return name_neighbours(store, model, thresholds.resolve)


def _names_close(store: Store, model: str | None, pipeline: PipelineConfig) -> NamesClose | None:
    """SPEC 5.6: a name cosine lets a ``same`` stand only at tier 3 with a
    calibrated embedding model, at its ``merge`` threshold."""
    thresholds = pipeline.thresholds(model) if model is not None else None
    if model is None or thresholds is None:
        return None
    return names_close(store, model, thresholds.merge)


def tier_of(models: ModelClient) -> int:
    """SPEC 6: 0 with no model, 2 with a chat endpoint, 3 with chat and embeddings."""
    if models.chat_endpoint is None:
        return 0
    return 3 if models.embeddings_endpoint is not None else 2


def create_app(
    settings: Settings, *, store: Store | None = None, clock: Callable[[], datetime] = utc_now
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.worker.start()
        try:
            async with mcp_lifespan(application):
                yield
        finally:
            application.state.worker.stop()
            application.state.models.close()
            if application.state.dream_models is not None:
                application.state.dream_models.close()
            application.state.store.close()

    app = FastAPI(
        lifespan=lifespan,
        title="iifa",
        version=__version__,
        description="Rich long-term memory for your agent, delivered under a token budget you set.",
        docs_url=None,
        redoc_url=None,
    )
    if store is None:
        store = Store(
            settings.db_path,
            settings.db_key,
            clock=clock,
            retain_sources=settings.retain_sources,
            plaintext_for_tests=settings.plaintext_database_for_tests,
            checkpoint_interval=settings.checkpoint_interval,
        )
    app.state.settings = settings
    app.state.store = store
    app.state.registry = TokenRegistry(settings.tokens)
    gate = Gate({CHAT: settings.chat_concurrency, EMBEDDINGS: settings.embeddings_concurrency})
    app.state.models = build_models(settings, gate)
    app.state.engine = RecallEngine(
        store,
        get_tokenizer(settings.tokenizer),
        max_candidates=settings.recall_candidates,
        neighbour_facts=settings.graph_neighbour_facts,
        tier=tier_of(app.state.models),
        dense=(
            Dense(
                store,
                QueryEmbedder(app.state.models),
                enabled=settings.recall_dense,
                timeout=settings.query_timeout,
            )
            if app.state.models.embeddings_endpoint is not None
            else None
        ),
    )
    app.state.queue = Queue(store)
    pipeline = build_pipeline(settings)
    app.state.dream_models = build_dream_models(settings, gate)
    engine = app.state.engine
    app.state.dreamer = Dreamer(
        store,
        client=app.state.dream_models,
        config=DreamConfig(
            answer_max_tokens=settings.dream_answer_max_tokens,
            json_mode=settings.chat_json_mode,
            probe_tolerance=settings.dream_probe_tolerance,
            adjudications_per_unit=settings.dream_adjudications_per_unit,
            prompt_max_tokens=settings.dream_prompt_max_tokens,
            tokenizer=get_tokenizer(settings.tokenizer),
            edge_half_life_days=settings.edge_half_life_days,
            edge_prune_below=settings.edge_prune_below,
            review=settings.dream_review,
            forced_language=pipeline.forced_language,
        ),
        probe_scorer=lambda probe_set: run_probe_set(engine, store, probe_set),
        has_embeddings=app.state.models.embeddings_endpoint is not None,
        name_neighbours=_name_neighbours(
            store,
            app.state.models.embeddings_endpoint.model
            if app.state.models.embeddings_endpoint is not None
            else None,
            pipeline,
        ),
        names_close=_names_close(
            store,
            app.state.models.embeddings_endpoint.model
            if app.state.models.embeddings_endpoint is not None
            else None,
            pipeline,
        ),
    )
    app.state.worker = Worker(
        store,
        app.state.queue,
        mode=settings.background_mode,
        gate=gate,
        models=app.state.models,
        handlers={**HANDLERS, DREAM_JOB: dream_job},
        yield_timeout=settings.yield_timeout,
        max_attempts=settings.max_attempts,
        retry_delay=settings.retry_delay,
        idle_poll=settings.worker_poll,
        pipeline=pipeline,
        dream=app.state.dreamer,
    )
    app.state.dreamer.request_embeddings = app.state.worker.request_embeddings
    app.include_router(router)
    app.include_router(dream_router)

    @app.exception_handler(NotFound)
    async def _not_found(request: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse({"detail": "no such record"}, status_code=404)

    if settings.mcp_http:
        mount_mcp(app, settings)
    return app
