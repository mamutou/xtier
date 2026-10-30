#!/usr/bin/env bash
# Launch the server inside a transient user scope with a hard memory cap.
#
# Why: on a GB10 the CPU expert engine and the GPU share one 121 GB pool, and the checkpoint is
# ~156 GB. When the cgroup allowance is exceeded the kernel would otherwise spend all its time in
# reclaim and the machine stops answering ssh (a wedge that needs a manual power cycle). With
# MemoryMax the cgroup OOM-kills this scope instead, and the box stays reachable.
#
#   MemoryHigh  soft limit: reclaim first, do not kill. Keep this ~10 GB under MemoryMax.
#   MemoryMax   hard limit: the scope is killed when it gets here.
#   MemorySwapMax=0  no swap on this box; swapping would only make the thrash longer.
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
LOG=${LOG:-$HOME/logs/xtier-spark.log}
mkdir -p "$(dirname "$LOG")"

# Competing IO (a checkpoint download saturating the NVMe) makes the load phase several times slower
# and eats the page cache the cold-expert pool needs. Set STOP_PATTERN to whatever you want killed.
if [ -n "${STOP_PATTERN:-}" ]; then
  pkill -f "$STOP_PATTERN" 2>/dev/null || true
  sleep 2
fi
pkill -f '[s]glang.launch_server' 2>/dev/null && sleep 3 || true

setsid nohup systemd-run --user --scope --collect --unit="xtier-spark-$(date +%s)" \
  -p MemoryMax="${MEMCAP:-118G}" -p MemoryHigh="${MEMHIGH:-108G}" -p MemorySwapMax=0 \
  bash "$HERE/run_server.sh" "$@" > "$LOG" 2>&1 < /dev/null &

sleep 20
echo "--- server processes: $(pgrep -c -f '[s]glang.launch_server' || echo 0)"
echo "--- log: $LOG"
tail -20 "$LOG"
