FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY main.py ./
COPY fixtures ./fixtures
COPY demo/__init__.py ./demo/__init__.py
COPY demo/api ./demo/api

RUN pip install --no-cache-dir -e ".[demo]"

EXPOSE 8000

CMD ["uvicorn", "demo.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
