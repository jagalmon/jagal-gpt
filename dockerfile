FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y build-essential g++ \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get remove -y build-essential g++ \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["python", "agent.py", "--mode", "infer"]