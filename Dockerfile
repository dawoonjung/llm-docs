# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# uv 설치
RUN pip install --no-cache-dir uv

# 프로젝트 파일 복사
COPY pyproject.toml .
COPY main.py .
COPY README.md .

# 의존성 설치 (uv 사용)
RUN uv pip compile pyproject.toml > requirements.txt
RUN uv pip install --system -r requirements.txt

# .env 파일 복사 (있을 경우)
COPY .env .

# 실행 명령
CMD ["python", "main.py"] 