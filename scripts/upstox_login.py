"""One-time-per-day helper: obtain a fresh Upstox access token and write it to .env.

Upstox's v2 API tokens expire nightly (~3:30 AM IST) — there is no long-lived
refresh token, so this needs to run once each morning before the agent needs
live option chain data.

Usage:
    python scripts/upstox_login.py
"""

import sys
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nifty_ai_agent.config import get_settings  # noqa: E402

_AUTH_URL = "https://api.upstox.com/v2/login/authorization/dialog"
_TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
_DEFAULT_REDIRECT_URI = "https://www.google.com/upstox-callback"
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def extract_code(user_input: str) -> str:
    """Accept either a bare auth code or the full redirected URL and return the code."""
    user_input = user_input.strip()
    if user_input.startswith("http"):
        query = parse_qs(urlparse(user_input).query)
        code = query.get("code", [""])[0]
        if not code:
            raise ValueError("No 'code' parameter found in that URL")
        return code
    return user_input


def update_env_token(token: str, env_path: Path = _ENV_PATH) -> None:
    """Rewrite UPSTOX_ACCESS_TOKEN=... in .env, appending it if not already present."""
    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("UPSTOX_ACCESS_TOKEN="):
            lines[i] = f"UPSTOX_ACCESS_TOKEN={token}\n"
            break
    else:
        lines.append(f"UPSTOX_ACCESS_TOKEN={token}\n")
    env_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    settings = get_settings()
    if not settings.upstox_api_key or not settings.upstox_api_secret:
        print("UPSTOX_API_KEY / UPSTOX_API_SECRET are not set in .env — add them first.")
        sys.exit(1)

    redirect_uri = settings.upstox_redirect_uri or _DEFAULT_REDIRECT_URI
    auth_url = f"{_AUTH_URL}?" + urlencode({
        "response_type": "code",
        "client_id": settings.upstox_api_key,
        "redirect_uri": redirect_uri,
    })

    print("Opening the Upstox login page in your browser...")
    print(auth_url)
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    print()
    print("Log in, then paste the FULL URL you land on afterwards")
    print("(or just the 'code' value from it) below:")
    code = extract_code(input("> "))

    resp = requests.post(
        _TOKEN_URL,
        data={
            "code": code,
            "client_id": settings.upstox_api_key,
            "client_secret": settings.upstox_api_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        headers={"Accept": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()
    token = payload.get("access_token")
    if not token:
        print(f"No access_token in response: {payload}")
        sys.exit(1)

    update_env_token(token)
    print(f"\nUPSTOX_ACCESS_TOKEN updated in {_ENV_PATH} — valid until ~3:30 AM IST tonight.")


if __name__ == "__main__":
    main()
