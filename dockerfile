FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install -r pyproject.toml

COPY . .

CMD ["python", "agent.py", "--mode", "infer"]