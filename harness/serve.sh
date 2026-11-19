#!/bin/zsh
# serve.sh — start a llama.cpp server for one evaluation model under an OS-level
# network-deny sandbox, bound to 127.0.0.1. Nothing leaves the machine.
#
#   harness/serve.sh <model.gguf> <port> [embed|chat] [ctx]
#   harness/serve.sh stop <port>
#
# Sandbox profile: SANDBOX_PROFILE (default harness/llm-net-deny.sb),
# which denies all network except loopback bind/inbound/outbound.
set -euo pipefail
PROFILE="${SANDBOX_PROFILE:-$(dirname "$0")/llm-net-deny.sb}"
RUN="${EVAL_RUN_DIR:-$(dirname "$0")/../private/run}"
mkdir -p "$RUN"

if [[ "${1:-}" == "stop" ]]; then
  pidf="$RUN/server-$2.pid"; [[ -f "$pidf" ]] && kill "$(cat "$pidf")" 2>/dev/null || true; rm -f "$pidf"; exit 0
fi

model="$1"; port="$2"; mode="${3:-chat}"; ctx="${4:-8192}"
[[ -f "$PROFILE" ]] || { echo "sandbox profile missing: $PROFILE" >&2; exit 1; }
extra=(--jinja)
[[ "$mode" == "embed" ]] && extra=(--embedding --pooling mean -ub 2048 -b 2048)
nohup env -i HOME="$HOME" PATH="/opt/homebrew/bin:/usr/bin:/bin" LLAMA_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /usr/bin/sandbox-exec -f "$PROFILE" "$(command -v llama-server)" \
  -m "$model" --host 127.0.0.1 --port "$port" -c "$ctx" -ngl 999 --parallel 1 --offline \
  "${extra[@]}" >"$RUN/server-$port.log" 2>&1 &
echo $! > "$RUN/server-$port.pid"
for _ in $(seq 1 300); do
  kill -0 "$(cat "$RUN/server-$port.pid")" 2>/dev/null || { echo "server exited; see $RUN/server-$port.log" >&2; exit 1; }
  curl -s --noproxy '*' -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" | grep -q 200 && break
  sleep 1
done
# Verify the listener is loopback-only.
if lsof -nP -iTCP:"$port" -sTCP:LISTEN | awk 'NR>1{print $9}' | grep -qv '^127\.0\.0\.1:'; then
  echo "listener not loopback-only; stopping" >&2; "$0" stop "$port"; exit 1
fi
echo "ready: $(basename "$model") on 127.0.0.1:$port ($mode, sandboxed)"
