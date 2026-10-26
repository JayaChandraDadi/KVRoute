FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY kvroute ./kvroute
RUN pip install --no-cache-dir ".[hf]"

EXPOSE 8000
ENTRYPOINT ["kvroute"]
