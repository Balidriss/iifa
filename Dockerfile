# syntax=docker/dockerfile:1
# iifa: the memory service (HTTP API, MCP at /mcp). docs/docker.md explains the
# volumes, the key and how to publish the port safely.
#
#   docker build -t iifa .
#   docker build -t iifa --build-arg IIFA_EXTRAS=vectors,exact-tokens .
ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM ${PYTHON_IMAGE}
ARG IIFA_EXTRAS=""
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    wheel="$(ls /wheels/iifa-*.whl)" \
    && python -m pip install --no-cache-dir "${wheel}${IIFA_EXTRAS:+[$IIFA_EXTRAS]}" \
    && groupadd --system --gid 10001 iifa \
    && useradd --system --uid 10001 --gid iifa --no-create-home \
       --home-dir /nonexistent --shell /usr/sbin/nologin iifa \
    && mkdir -p /data \
    && chown iifa:iifa /data \
    && chmod 0700 /data

# The database lives on its own volume. The key is never in the image and never
# on that volume: mount it read-only as a file (a Docker secret lands there).
ENV IIFA_DB_PATH=/data/iifa.db \
    IIFA_DB_KEY_FILE=/run/secrets/iifa_db_key \
    IIFA_HOST=127.0.0.1 \
    IIFA_PORT=8765
VOLUME ["/data"]

USER 10001:10001
EXPOSE 8765
# A connection check from inside the container (/v1/health needs a token).
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, socket; socket.create_connection(('127.0.0.1', int(os.environ['IIFA_PORT'])), 2).close()"]
ENTRYPOINT ["iifa"]
