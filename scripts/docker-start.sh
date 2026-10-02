#!/usr/bin/env bash
# Start DocBlendAI in Docker on Linux / macOS and open it in the browser.
#
#   scripts/docker-start.sh             start (build only if the image is missing)
#   scripts/docker-start.sh --build     build the image from this checkout
#   scripts/docker-start.sh --pull      use the pre-built image (DOCBLENDAI_IMAGE) instead of building
#   options: --no-browser, --port 8001
#
# 1. checks Docker is running; 2. creates .env from .env.example if missing, asking for the Gemini
# API key (hidden, never printed); 3. docker compose up -d; 4. waits for /health and opens the app.
set -euo pipefail
cd "$(dirname "$0")/.."   # the repository root

MODE="" BROWSER=1 PORT=8000 TIMEOUT_MIN=20
while [ $# -gt 0 ]; do
  case "$1" in
    --build) MODE=build ;;
    --pull) MODE=pull ;;
    --no-browser) BROWSER=0 ;;
    --port) PORT="$2"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
fail() { echo "ERROR: $*" >&2; exit 1; }

# 1. Docker installed and running?
command -v docker >/dev/null 2>&1 || fail "Docker is not installed: https://docs.docker.com/get-docker/"
docker info >/dev/null 2>&1 || fail "Docker is not running (start Docker Desktop, or: sudo systemctl start docker)."
docker compose version >/dev/null 2>&1 || fail "'docker compose' is missing: install the Docker Compose plugin."

# 2. .env with the Gemini key (never printed)
if [ ! -f .env ]; then
  [ -f .env.example ] || fail ".env.example is missing: run this from a DocBlendAI checkout."
  cp .env.example .env
  echo "Created .env from .env.example."
  printf "Gemini API key (https://aistudio.google.com/apikey; Enter to skip): "
  IFS= read -rs KEY || KEY=""
  echo
  if [ -n "$KEY" ]; then
    # Replace the GEMINI_API_KEY= line without putting the key on a command line.
    tmp="$(mktemp)"
    while IFS= read -r line || [ -n "$line" ]; do
      case "$line" in GEMINI_API_KEY=*) printf 'GEMINI_API_KEY=%s\n' "$KEY" ;; *) printf '%s\n' "$line" ;; esac
    done < .env > "$tmp"
    mv "$tmp" .env
    chmod 600 .env
    echo "Saved the key in .env (not shown)."
  else
    echo "No key: answers and LLM refinement stay off until you set GEMINI_API_KEY in .env."
  fi
  unset KEY
fi

# Linux: the container runs as uid 1000 and writes to ./data; make sure that works.
mkdir -p data
if [ "$(uname -s)" = "Linux" ] && [ "$(id -u)" != "1000" ]; then
  echo "Note: you are uid $(id -u); if uploads fail with 'Permission denied', run: sudo chown -R 1000:1000 data"
fi

# 3. Start
export DOCBLENDAI_PORT="$PORT"
case "$MODE" in
  pull) docker compose pull && docker compose up -d --no-build ;;
  build) echo "Building the image (first time: 10-20 minutes)..."; docker compose up -d --build ;;
  *) docker compose up -d ;;
esac || fail "docker compose up failed. If port $PORT is busy, run again with --port 8001."

# 4. Wait for /health
URL="http://localhost:$PORT"
echo "Waiting for $URL/health (up to $TIMEOUT_MIN min)..."
deadline=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
until curl -fsS "$URL/health" >/dev/null 2>&1; do
  [ "$(date +%s)" -lt "$deadline" ] || fail "DocBlendAI did not answer in time. Check: docker compose logs --tail 100"
  sleep 3
done
echo "DocBlendAI is running: $URL  (Experience Center: $URL/studio)"
echo "Stop it with: docker compose down"
if [ "$BROWSER" = 1 ]; then
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then open "$URL"
  fi
fi
