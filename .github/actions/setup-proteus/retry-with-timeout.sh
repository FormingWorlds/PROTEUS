# Usage: retry_with_timeout <description> <timeout-secs> <kill-after-secs> <pre-attempt-hook-or-empty> <command...>
# Source this, then call retry_with_timeout to run a command under a
# bounded timeout, retrying once with backoff.

# GNU coreutils' `timeout` ships on Linux but not macOS; resolve gtimeout
# from Homebrew instead, installing coreutils only if neither is already
# on PATH. That install is bounded by a background watchdog that kills
# the whole process group, since brew forks children of its own.
if command -v timeout >/dev/null 2>&1; then
  _RWT_TIMEOUT_BIN="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  _RWT_TIMEOUT_BIN="gtimeout"
else
  ( set -m
    brew install coreutils >/dev/null 2>&1 &
    _rwt_brew_pid=$!
    ( sleep 90; kill -9 -- "-$_rwt_brew_pid" ) >/dev/null 2>&1 &
    _rwt_watch_pid=$!
    wait "$_rwt_brew_pid" 2>/dev/null
    kill -9 -- "-$_rwt_watch_pid" >/dev/null 2>&1
    wait "$_rwt_watch_pid" 2>/dev/null
  ) || true
  if command -v gtimeout >/dev/null 2>&1; then
    _RWT_TIMEOUT_BIN="gtimeout"
  elif command -v timeout >/dev/null 2>&1; then
    _RWT_TIMEOUT_BIN="timeout"
  else
    echo "retry-with-timeout.sh: neither timeout nor gtimeout is on PATH after installing coreutils; cannot bound package-manager calls, stopping here" >&2
    exit 1
  fi
fi

retry_with_timeout() {
  local desc="$1" timeout_secs="$2" kill_after="$3" pre_hook="$4"
  shift 4
  local delays=(45)
  local attempts=$(( ${#delays[@]} + 1 ))
  local n=1 rc
  local use_sudo=""
  if [ "$1" = "sudo" ]; then
    use_sudo=1
    shift
  fi

  while true; do
    if [ -n "$pre_hook" ]; then
      "$pre_hook" || true
    fi
    if [ -n "$use_sudo" ]; then
      if sudo "$_RWT_TIMEOUT_BIN" --kill-after="$kill_after" "$timeout_secs" "$@"; then
        rc=0
      else
        rc=$?
      fi
    else
      if "$_RWT_TIMEOUT_BIN" --kill-after="$kill_after" "$timeout_secs" "$@"; then
        rc=0
      else
        rc=$?
      fi
    fi
    if [ "$rc" -eq 0 ]; then
      echo "$desc succeeded on attempt $n"
      return 0
    fi
    if [ "$n" -ge "$attempts" ]; then
      echo "$desc failed after $attempts attempts (exit $rc)" >&2
      return "$rc"
    fi
    local delay="${delays[$((n - 1))]}"
    echo "$desc attempt $n failed or timed out (exit $rc); retrying in ${delay}s" >&2
    sleep "$delay"
    n=$((n + 1))
  done
}
