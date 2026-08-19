# Shared retry-with-timeout helper for the package-manager install steps in
# this action. Source it, then call retry_with_timeout to run a command
# under a bounded timeout, retrying once with backoff if it fails or times
# out.
#
# Usage: retry_with_timeout <description> <timeout-secs> <kill-after-secs> <pre-attempt-hook-or-empty> <command...>
# The pre-attempt hook, if non-empty, is called before every attempt and is
# not itself subject to the timeout; give it its own bound if it needs one.
# A non-zero return from the hook does not abort the caller; a hook that
# calls `exit` directly bypasses that and ends the whole script.
# A command beginning with `sudo` runs as `sudo <timeout-bin> ... <rest>`
# rather than `<timeout-bin> ... sudo <rest>`, so a kill-after signal reaches
# the privileged process directly instead of only reaching the sudo wrapper
# around it, which sudo would not relay.

# GNU coreutils' `timeout` ships on Linux runners but not on macOS; Homebrew
# installs the same binary as `gtimeout` if asked. Resolve which name is on
# PATH once per sourcing shell, installing coreutils only when neither exists.
# Neither timeout binary is available yet at this point, so the install
# itself is bounded by hand with a background watchdog. `set -m` gives the
# install its own process group so the watchdog can kill the whole group
# (`-$_rwt_brew_pid`, not just that one PID): brew forks child processes of
# its own, and killing only the top PID would leave those running.
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
