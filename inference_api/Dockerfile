# syntax=docker/dockerfile:1

FROM python:3.12.4-slim-bookworm AS python
WORKDIR /app
RUN python3.12 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
COPY . .
RUN python3.12 -m pip install --no-cache-dir -r /app/inference_api/inference_requirements.txt
RUN pip install -e .
RUN rm -rf ./inference_api/inference_requirements.txt
RUN rm -rf ./inference_api/Dockerfile

FROM python:3.12.4-slim-bookworm
WORKDIR /app
COPY --from=python /app/venv/ ./venv/
COPY --from=python /app/inference_api/ ./inference_api/
COPY --from=python /app/model/infer.py ./model/infer.py
COPY --from=python /app/saved_models/prod/ ./saved_models/prod/
ENV PATH="/app/venv/bin:$PATH"
CMD [ "python3.12", "./inference_api/server.py" ]
EXPOSE ${PORT}
