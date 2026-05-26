#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

TAL_ROOT="${TAL_ROOT:-${SCRIPT_DIR}/TAL2}"
OPENPI_DIR="${OPENPI_DIR:-${SCRIPT_DIR}/openpi}"
ROBOT_WS="${ROBOT_WS:-/root/gpufree-data/code/robot_ws}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/root/isaacsim/python.sh}"
OPENPI_PYTHON="${OPENPI_PYTHON:-${OPENPI_DIR}/.venv/bin/python}"

OPENPI_HOST="${OPENPI_HOST:-127.0.0.1}"
OPENPI_PORT="${OPENPI_PORT:-8000}"
OPENPI_POLICY_CONFIG="${OPENPI_POLICY_CONFIG:-pi05_pro630_lora}"
OPENPI_POLICY_DIR="${OPENPI_POLICY_DIR:-/root/gpufree-data/code/openpi/checkpoints/pi05_pro630_lora/pAndp/14999}"

PROMPT="${PROMPT:-pick up the block}"
MAX_STEPS="${MAX_STEPS:--1}"
QWEN_MODEL="${QWEN_MODEL:-qwen3-max}"
QWEN_API_KEY_ENV="${QWEN_API_KEY_ENV:-DASHSCOPE_API_KEY}"
NAV_GOAL_TIMEOUT_SEC="${NAV_GOAL_TIMEOUT_SEC:-120}"
NAV_WARMUP_SEC="${NAV_WARMUP_SEC:-4}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/tal_nav_openpi_${TIMESTAMP}}"

HEADLESS=0
KEEP_SERVICES=0
DRY_RUN=0
RESTART_NAV2=0
OPENPI_STARTED_PID=""
NAV2_STARTED_PID=""
ISAAC_EXTRA_ARGS=()

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [options] [-- extra args for sim_inference_tal_controller2.py]

Options:
  --prompt TEXT                 Task prompt. Default: "${PROMPT}"
  --max-steps N                 Main TAL/OpenPI loop steps. -1 means unlimited. Default: ${MAX_STEPS}
  --headless                    Run Isaac Sim headless.
  --keep-services               Keep OpenPI/Nav2 processes started by this script after Isaac exits.
  --restart-nav2                Stop an existing Nav2 launch before starting a fresh one.
  --server-port PORT            OpenPI policy server port. Default: ${OPENPI_PORT}
  --qwen-model NAME             DashScope model for TAL. Default: ${QWEN_MODEL}
  --nav-goal-timeout-sec SEC    Timeout for one Nav2 goal. Default: ${NAV_GOAL_TIMEOUT_SEC}
  --nav-warmup-sec SEC          Isaac bridge warmup before first plan. Default: ${NAV_WARMUP_SEC}
  --log-dir DIR                 Directory for logs. Default: ${LOG_DIR}
  --dry-run                     Print commands without running them.
  -h, --help                    Show this help.

Environment overrides:
  DASHSCOPE_API_KEY             Required unless QWEN_API_KEY_ENV points to another populated variable.
  TAL_ROOT                      Default: ${TAL_ROOT}
  OPENPI_DIR                    Default: ${OPENPI_DIR}
  OPENPI_POLICY_DIR             Default: ${OPENPI_POLICY_DIR}
  ROBOT_WS                      Default: ${ROBOT_WS}
  ISAAC_PYTHON                  Default: ${ISAAC_PYTHON}
  TAL_NAV_KINEMATIC_BASE        Defaulted to 1 by this script.
USAGE
}

die() {
  echo "[tal-nav-openpi] ERROR: $*" >&2
  exit 1
}

log() {
  echo "[tal-nav-openpi] $*"
}

quote_cmd() {
  printf "%q " "$@"
  printf "\n"
}

while (($#)); do
  case "$1" in
    --prompt)
      PROMPT="${2:?missing value for --prompt}"
      shift 2
      ;;
    --prompt=*)
      PROMPT="${1#*=}"
      shift
      ;;
    --max-steps)
      MAX_STEPS="${2:?missing value for --max-steps}"
      shift 2
      ;;
    --max-steps=*)
      MAX_STEPS="${1#*=}"
      shift
      ;;
    --headless)
      HEADLESS=1
      shift
      ;;
    --keep-services)
      KEEP_SERVICES=1
      shift
      ;;
    --restart-nav2)
      RESTART_NAV2=1
      shift
      ;;
    --server-port)
      OPENPI_PORT="${2:?missing value for --server-port}"
      shift 2
      ;;
    --server-port=*)
      OPENPI_PORT="${1#*=}"
      shift
      ;;
    --qwen-model)
      QWEN_MODEL="${2:?missing value for --qwen-model}"
      shift 2
      ;;
    --qwen-model=*)
      QWEN_MODEL="${1#*=}"
      shift
      ;;
    --nav-goal-timeout-sec)
      NAV_GOAL_TIMEOUT_SEC="${2:?missing value for --nav-goal-timeout-sec}"
      shift 2
      ;;
    --nav-goal-timeout-sec=*)
      NAV_GOAL_TIMEOUT_SEC="${1#*=}"
      shift
      ;;
    --nav-warmup-sec)
      NAV_WARMUP_SEC="${2:?missing value for --nav-warmup-sec}"
      shift 2
      ;;
    --nav-warmup-sec=*)
      NAV_WARMUP_SEC="${1#*=}"
      shift
      ;;
    --log-dir)
      LOG_DIR="${2:?missing value for --log-dir}"
      shift 2
      ;;
    --log-dir=*)
      LOG_DIR="${1#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      ISAAC_EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      ISAAC_EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

API_KEY_VALUE="${!QWEN_API_KEY_ENV:-}"
[[ -n "${API_KEY_VALUE}" ]] || die "${QWEN_API_KEY_ENV} is not set. Run: export ${QWEN_API_KEY_ENV}=..."

[[ -x "${ISAAC_PYTHON}" ]] || die "Isaac Python is not executable: ${ISAAC_PYTHON}"
[[ -x "${OPENPI_PYTHON}" ]] || die "OpenPI Python is not executable: ${OPENPI_PYTHON}"
[[ -f "${OPENPI_DIR}/scripts/serve_policy.py" ]] || die "OpenPI serve_policy.py not found under: ${OPENPI_DIR}"
[[ -d "${OPENPI_POLICY_DIR}" ]] || die "OpenPI checkpoint dir not found: ${OPENPI_POLICY_DIR}"
[[ -d "${TAL_ROOT}/src" ]] || die "TAL root is invalid: ${TAL_ROOT}"
[[ -x "${ROBOT_WS}/run_nav2_clean.sh" ]] || die "Nav2 launcher is not executable: ${ROBOT_WS}/run_nav2_clean.sh"

mkdir -p "${LOG_DIR}"

export TAL_NAV_KINEMATIC_BASE="${TAL_NAV_KINEMATIC_BASE:-1}"
export TAL_NAV_MAP_YAML="${TAL_NAV_MAP_YAML:-${ROBOT_WS}/src/robot_navigation/maps/expff_map.yaml}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

is_port_open() {
  python3 - "$OPENPI_HOST" "$OPENPI_PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.5)
try:
    sock.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

wait_for_port() {
  local timeout_sec="$1"
  local start
  start="$(date +%s)"
  while true; do
    if is_port_open; then
      return 0
    fi
    if (( "$(date +%s)" - start >= timeout_sec )); then
      return 1
    fi
    sleep 1
  done
}

nav2_running() {
  pgrep -f "ros2 launch robot_navigation nav2_launch.py" >/dev/null 2>&1 \
    || pgrep -f "robot_navigation.*nav2_launch.py" >/dev/null 2>&1
}

wait_for_nav2_launch_process() {
  local timeout_sec="$1"
  local start
  start="$(date +%s)"
  while true; do
    if nav2_running; then
      return 0
    fi
    if (( "$(date +%s)" - start >= timeout_sec )); then
      return 1
    fi
    sleep 1
  done
}

stop_existing_nav2() {
  local pids
  mapfile -t pids < <(pgrep -f "ros2 launch robot_navigation nav2_launch.py|robot_navigation.*nav2_launch.py|nav2_controller/controller_server|nav2_planner/planner_server|nav2_bt_navigator/bt_navigator|nav2_behaviors/behavior_server|nav2_lifecycle_manager/lifecycle_manager|nav2_map_server/map_server" || true)
  ((${#pids[@]})) || return 0
  log "Stopping existing Nav2 processes: ${pids[*]}"
  kill -TERM "${pids[@]}" >/dev/null 2>&1 || true
  sleep 2
  mapfile -t pids < <(pgrep -f "ros2 launch robot_navigation nav2_launch.py|robot_navigation.*nav2_launch.py|nav2_controller/controller_server|nav2_planner/planner_server|nav2_bt_navigator/bt_navigator|nav2_behaviors/behavior_server|nav2_lifecycle_manager/lifecycle_manager|nav2_map_server/map_server" || true)
  ((${#pids[@]})) || return 0
  kill -KILL "${pids[@]}" >/dev/null 2>&1 || true
}

wait_for_nav2_node() {
  local timeout_sec="$1"
  timeout "${timeout_sec}" bash -lc "
    set -e
    set +u
    source /opt/ros/humble/setup.bash
    source '${ROBOT_WS}/install/local_setup.bash'
    set -u
    until /opt/ros/humble/bin/ros2 node list 2>/dev/null | grep -q '/bt_navigator'; do
      sleep 1
    done
  " >/dev/null 2>&1
}

start_openpi_if_needed() {
  if is_port_open; then
    log "OpenPI policy server already listens on ${OPENPI_HOST}:${OPENPI_PORT}; reusing it."
    return 0
  fi

  local log_file="${LOG_DIR}/openpi_server.log"
  local cmd=(
    "${OPENPI_PYTHON}"
    scripts/serve_policy.py
    --port "${OPENPI_PORT}"
    policy:checkpoint
    "--policy.config=${OPENPI_POLICY_CONFIG}"
    "--policy.dir=${OPENPI_POLICY_DIR}"
  )

  log "Starting OpenPI policy server. Log: ${log_file}"
  if ((DRY_RUN)); then
    (cd "${OPENPI_DIR}" && quote_cmd "${cmd[@]}")
    return 0
  fi

  setsid bash -lc "cd '${OPENPI_DIR}' && exec \"\$@\"" bash "${cmd[@]}" >"${log_file}" 2>&1 &
  OPENPI_STARTED_PID="$!"

  if ! wait_for_port 240; then
    tail -80 "${log_file}" >&2 || true
    die "OpenPI policy server did not become ready on ${OPENPI_HOST}:${OPENPI_PORT}"
  fi
  log "OpenPI is ready on ${OPENPI_HOST}:${OPENPI_PORT}."
}

start_nav2_if_needed() {
  if ((RESTART_NAV2)); then
    if ((DRY_RUN)); then
      log "Would stop existing Nav2 processes because --restart-nav2 was set."
    else
      stop_existing_nav2
    fi
  fi

  if ! ((RESTART_NAV2 && DRY_RUN)) && nav2_running; then
    log "Nav2 launch process already exists; reusing it."
    log "Isaac will publish /clock, /odom, and odom -> base_link after it starts."
    return 0
  fi

  local log_file="${LOG_DIR}/nav2.log"
  log "Starting Nav2. Log: ${log_file}"
  if ((DRY_RUN)); then
    quote_cmd "${ROBOT_WS}/run_nav2_clean.sh"
    return 0
  fi

  setsid bash -lc "cd '${ROBOT_WS}' && exec '${ROBOT_WS}/run_nav2_clean.sh'" >"${log_file}" 2>&1 &
  NAV2_STARTED_PID="$!"

  if ! wait_for_nav2_launch_process 15; then
    tail -120 "${log_file}" >&2 || true
    die "Nav2 launch process did not start within timeout."
  fi
  log "Nav2 launch is running. Isaac startup will connect the TF tree."
}

kill_process_group() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0
  kill -0 "${pid}" >/dev/null 2>&1 || return 0
  kill -TERM "-${pid}" >/dev/null 2>&1 || kill -TERM "${pid}" >/dev/null 2>&1 || true
  sleep 2
  kill -0 "${pid}" >/dev/null 2>&1 || return 0
  kill -KILL "-${pid}" >/dev/null 2>&1 || kill -KILL "${pid}" >/dev/null 2>&1 || true
}

cleanup() {
  local status=$?
  if ((KEEP_SERVICES)); then
    log "Keeping services alive because --keep-services was set."
    return "${status}"
  fi
  if [[ -n "${NAV2_STARTED_PID}" ]]; then
    log "Stopping Nav2 started by this script."
    kill_process_group "${NAV2_STARTED_PID}"
  fi
  if [[ -n "${OPENPI_STARTED_PID}" ]]; then
    log "Stopping OpenPI started by this script."
    kill_process_group "${OPENPI_STARTED_PID}"
  fi
  return "${status}"
}
trap cleanup EXIT

start_openpi_if_needed
start_nav2_if_needed

ISAAC_CMD=(
  "${ISAAC_PYTHON}"
  "${SCRIPT_DIR}/sim_inference_tal_controller2.py"
  --prompt "${PROMPT}"
  --server-host "${OPENPI_HOST}"
  --server-port "${OPENPI_PORT}"
  --tal-root "${TAL_ROOT}"
  --qwen-model "${QWEN_MODEL}"
  --qwen-api-key-env "${QWEN_API_KEY_ENV}"
  --max-steps "${MAX_STEPS}"
  --nav-goal-timeout-sec "${NAV_GOAL_TIMEOUT_SEC}"
  --nav-warmup-sec "${NAV_WARMUP_SEC}"
)

if ((HEADLESS)); then
  ISAAC_CMD+=(--headless)
fi
ISAAC_CMD+=("${ISAAC_EXTRA_ARGS[@]}")

log "Logs: ${LOG_DIR}"
log "Launching TAL -> Nav2/OpenPI Isaac controller:"
quote_cmd "${ISAAC_CMD[@]}"

if ((DRY_RUN)); then
  exit 0
fi

set +e
"${ISAAC_CMD[@]}" 2>&1 | tee "${LOG_DIR}/isaac_main.log"
ISAAC_STATUS="${PIPESTATUS[0]}"
set -e

if [[ "${ISAAC_STATUS}" -eq 0 ]]; then
  log "Isaac TAL/Nav/OpenPI flow exited successfully."
else
  log "Isaac TAL/Nav/OpenPI flow exited with status ${ISAAC_STATUS}."
fi

exit "${ISAAC_STATUS}"
