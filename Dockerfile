# NIFTY AI Agent — container image for 24/7 deployment.
#
# Timezone is pinned to Asia/Kolkata because the scheduler's fixed-time jobs
# (schedule.every().day.at("08:00")) fire on the container's LOCAL wall clock.
# Without this a job set for 08:00 would run at 08:00 UTC = 13:30 IST.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata

# tzdata makes the TZ above effective; the rest are pulled in by playwright's
# --with-deps step below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps + the Postgres driver (psycopg is only needed in deployment,
# so it is added here rather than in requirements.txt which stays sqlite-friendly).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "psycopg[binary]>=3.1" \
    && playwright install --with-deps chromium

COPY . .

# FileHandler in configure_logging() does not create parent dirs — ensure it exists.
RUN mkdir -p nifty_ai_agent/logs

CMD ["python", "main.py"]
