# syntax=docker/dockerfile:1

FROM python:3.12.4-slim-bookworm AS python
WORKDIR /app
RUN python3.12 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
# TODO: Move this out later into into somewhere that local users can use too
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY ./tmp/deps/requirements.txt ./deps/requirements.txt
RUN python3.12 -m pip install --no-cache-dir -r /app/deps/requirements.txt
# TODO: Parameterise
COPY ./tmp/deps/dist/mancala_env-0.0.1-py3-none-any.whl ./deps/mancala_env-0.0.1-py3-none-any.whl
RUN pip install ./deps/mancala_env-0.0.1-py3-none-any.whl
RUN rm -rf ./deps
COPY ./tmp/run .
ENV PATH="/app/venv/bin:$PATH"
CMD [ "python3.12", "./pkg/mancala_agent_pkg/inference_api/server.py" ]
EXPOSE ${PORT}
