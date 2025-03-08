FROM python:3.10-slim as builder

RUN pip install poetry==1.4.0

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN apt-get update && \
    apt-get install -y python3 gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./pyproject.toml ./poetry.lock ./
RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-root

FROM python:3.10-slim as runtime

ENV VIRTUAL_ENV=app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY src /app/src/
# This is optional.
# You can include model weights and encoders inside the container 
# by running the training script during the build process.
COPY cache/test_run /app/data

COPY main_inference.py main_train.py /app/

ENTRYPOINT ["python", "app/main_inference.py"]