# Deploying the agent for others (24/7)

This runs the agent as a container alongside a Postgres database, so it serves
Telegram subscribers around the clock without your laptop. Everything is driven
by `docker-compose.yml` + your `.env`.

> **Before you go live for others, read the compliance note at the bottom.**

## What you need
- A host that's always on: any small cloud VM (a ₹400–800/month / \$5–10 VPS is plenty)
  or a Raspberry Pi. 1 vCPU / 1 GB RAM works.
- **Docker** + the **Docker Compose** plugin installed on that host.
- Your API keys (Pushover, and — for the multi-user bot — a **Telegram bot token**;
  Upstox/Anthropic optional).

## One-time setup
1. Copy the project to the host (e.g. `git clone <your repo>`), then:
   ```bash
   cp .env.example .env
   ```
2. Fill in `.env`:
   - `PUSHOVER_USER_KEY`, `PUSHOVER_API_TOKEN` — your own alerts.
   - `TELEGRAM_BOT_TOKEN` — from **@BotFather** (`/newbot`). This is what makes it
     multi-user: others `/start` your bot to subscribe.
   - `UPSTOX_ACCESS_TOKEN` — optional but recommended for live premiums.
   - `POSTGRES_PASSWORD` — pick any strong value (compose passes it to both services).
   - Leave `DATABASE_URL` as-is; compose overrides it to point at the Postgres service.

## Run it
```bash
docker compose up -d --build     # build + start in the background
docker compose logs -f agent     # watch it boot (look for "Telegram bot polling started")
```
That's it. Share your bot link (`https://t.me/<your_bot_username>`); anyone who sends
**/start** begins receiving signals.

### Day-to-day
```bash
docker compose logs -f agent     # live logs
docker compose restart agent     # restart just the agent
docker compose down              # stop everything (Postgres data survives in the volume)
docker compose up -d --build     # after pulling new code, rebuild + restart
```

## Why these choices
- **Timezone.** The container is pinned to `Asia/Kolkata` (`TZ` + tzdata in the
  Dockerfile). The scheduler's fixed-time jobs (`08:00`, `16:00`, …) fire on the
  container's local clock, so without this they'd run on UTC — 5½ hours off.
- **Postgres, not SQLite.** The command-polling thread and the signal scheduler both
  touch the DB; Postgres handles concurrent access cleanly where SQLite would lock.
  The schema is created automatically on first boot — no migrations to run.
- **Secrets.** `.env` is **git-ignored and docker-ignored**; it's mounted at runtime,
  never baked into the image.
- **Restart policy.** Both services use `restart: unless-stopped`, so they come back
  after a crash or a host reboot.

## Upstox token upkeep
If you use a daily OAuth token it expires nightly and the agent falls back to
estimated premiums until refreshed. For an unattended deployment use the **~1-year
Analytics Access Token** instead (see the main README's Upstox section) so there's
nothing to refresh. The built-in token-health check alerts you if it ever goes stale.

## Auto-deploy on git push (GitHub Actions)
`.github/workflows/deploy.yml` runs the test suite on every push to `main`, then —
only if it passes — SSHes to your VPS and redeploys. One-time setup:

1. **On the VPS**, clone the repo once and create `.env` (as above). The workflow
   only ever *updates* an existing checkout.
2. **Generate a dedicated deploy key** (on your laptop):
   ```bash
   ssh-keygen -t ed25519 -f deploy_key -N ""
   ```
   Append `deploy_key.pub` to the VPS's `~/.ssh/authorized_keys`.
3. **Add repository secrets** on GitHub (Settings → Secrets and variables → Actions):
   - `VPS_HOST` — server IP/hostname
   - `VPS_USER` — SSH user
   - `VPS_SSH_KEY` — the **private** `deploy_key` contents
   - `VPS_APP_DIR` — path to the repo on the server (e.g. `/home/deploy/market_analysyis`)
   - (non-standard SSH port? add `VPS_PORT` and wire it in the workflow's `with:`)

Now `git push` → tests run → on green, the VPS pulls and rebuilds automatically.
Your `.env` on the server is never touched (it's untracked, so `git reset` leaves
it alone). Pull requests run the tests but do **not** deploy.

## Optional: the dashboard
The image runs the agent (`python main.py`). To also expose the Streamlit dashboard,
add a second service that runs `streamlit run dashboard/app.py --server.port 8501`
against the same database, and publish port 8501 — but note the dashboard has **no
authentication**, so put it behind a reverse proxy / auth before exposing it publicly.

## ⚠️ Compliance (India / SEBI)
Distributing buy/sell recommendations to others — **especially for a fee** — falls
under SEBI's Research Analyst / Investment Adviser regulations. Keep the bot **free
and framed as educational/informational** (the disclaimers in each message already do
this), and **consult a SEBI-registration professional before charging or marketing it
publicly**. This document is deployment guidance, not legal advice.
