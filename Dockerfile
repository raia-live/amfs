FROM python:3.12-slim AS base

LABEL org.opencontainers.image.source="https://github.com/raia-live/amfs"
LABEL org.opencontainers.image.description="AMFS — Agent Memory File System HTTP Server"

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

WORKDIR /build
COPY packages/core packages/core
COPY packages/adapters/filesystem packages/adapters/filesystem
COPY packages/adapters/postgres packages/adapters/postgres
COPY packages/adapters/s3 packages/adapters/s3
COPY packages/sdk-python packages/sdk-python
COPY packages/http-server packages/http-server
COPY packages/mcp-server packages/mcp-server
COPY packages/cortex packages/cortex

RUN pip install --no-cache-dir \
    ./packages/core \
    ./packages/adapters/filesystem \
    ./packages/adapters/postgres \
    ./packages/adapters/s3 \
    ./packages/sdk-python \
    ./packages/http-server \
    ./packages/mcp-server \
    ./packages/cortex

FROM base AS runtime

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN useradd --create-home --shell /bin/bash amfs
USER amfs
WORKDIR /home/amfs

ENV AMFS_DATA_DIR=/data/.amfs
ENV AMFS_AGENT_ID=amfs-server

VOLUME ["/data"]

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

ENTRYPOINT ["amfs-http"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
