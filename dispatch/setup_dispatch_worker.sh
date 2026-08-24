#!/bin/bash
# Set up a pipeline dispatch worker on any host. Run it ON that host, as the pipeline user.
# Every variable below can be overridden from the environment.
#
# The worker never opens the main host's scheduler sqlite: every scheduler write runs THERE via
# cli/scheduler_rpc over ssh (WAL's index is coherent only within one host, so opening it over NFS
# hands the same task to two hosts). There is therefore no /var/db export and no mount of it.

set -euo pipefail

MAIN_SSH="${MAIN_SSH:-pipeline-stable@proton.snu.ac.kr}"   # the host running the scheduler + queue daemons
MAIN_SSH_PORT="${MAIN_SSH_PORT:-45204}"                    # its sshd binds the public interface only
MAIN_SCHEDULER_RPC="${MAIN_SCHEDULER_RPC:-/home/pipeline-stable/pipeline/pipeline/cli/scheduler_rpc}"
LYMAN_IP="${LYMAN_IP:-20.20.20.11}"                  # NFS server for /lyman/data1 and /lyman/data2
PIPELINE_ROOT="${PIPELINE_ROOT:-$HOME/pipeline}"
PYTHON_BIN_DIR="${PYTHON_BIN_DIR:-$(dirname "$(command -v python3)")}"
WORKER_USER="${WORKER_USER:-$(id -un)}"
# Existing .env wins over these defaults, so re-running the script never downgrades a tuned host.
env_get() { grep -E "^$1=" "$PIPELINE_ROOT/.env" 2>/dev/null | tail -1 | cut -d= -f2-; }
SERVER_NAME="${SERVER_NAME:-$(env_get DISPATCH_SERVER_NAME)}"; SERVER_NAME="${SERVER_NAME:-$(hostname -s)}"
MAX_WORKERS="${MAX_WORKERS:-$(env_get DISPATCH_MAX_WORKERS)}"; MAX_WORKERS="${MAX_WORKERS:-6}"
# Narrow while a worker is new; no stage needs a GPU, so "any" is safe when you want it.
CONFIG_TYPES="${CONFIG_TYPES:-$(env_get DISPATCH_CONFIG_TYPES)}"; CONFIG_TYPES="${CONFIG_TYPES:-science}"
INPUT_TYPE="${INPUT_TYPE:-$(env_get DISPATCH_INPUT_TYPE)}"  # relabels each claimed row, e.g. single-reduction-balmer

NFSOPT="rw,noatime,nodiratime,vers=4.1,rsize=1048576,wsize=1048576,hard,proto=tcp,timeo=600,_netdev 0 0"
UNIT_TEMPLATE="$PIPELINE_ROOT/systemd/pipeline-dispatch-worker.service"

echo "=== worker: $SERVER_NAME  user: $WORKER_USER  root: $PIPELINE_ROOT  python: $PYTHON_BIN_DIR ==="

echo "=== 1. Data mounts (read-write: PathHandler creates directories on attribute read) ==="
for dst in /lyman/data1 /lyman/data2; do
  sudo mkdir -p "$dst"
  grep -qE "^[^#]*[[:space:]]${dst}[[:space:]]" /etc/fstab 2>/dev/null \
    || echo "${LYMAN_IP}:${dst} ${dst} nfs4 ${NFSOPT}" | sudo tee -a /etc/fstab >/dev/null
  mountpoint -q "$dst" || sudo mount "$dst"
done

echo "=== 2. Host directories ==="
sudo install -d -m 0775 -o root -g pipeline /var/log/pipeline
sudo install -d -m 0775 -o root -g pipeline /var/lock/py7dt
sudo install -d -m 1777 /tmp/pipeline

echo "=== 3. ssh to the main host (key-based, no password) ==="
test -f ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
# The daemon connects with BatchMode=yes, which never prompts, so the host key must already be
# known. Accept it by connecting once by hand 
MAIN_HOSTNAME="${MAIN_SSH#*@}"
if ! ssh-keygen -F "[${MAIN_HOSTNAME}]:${MAIN_SSH_PORT}" >/dev/null 2>&1; then
  echo "  ${MAIN_HOSTNAME} host key is unknown. Connect once by hand and accept it after checking"
  echo "  the fingerprint against 'ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub' on that host:"
  echo "    ssh -p ${MAIN_SSH_PORT} ${MAIN_SSH}"
  exit 1
fi
if ! ssh -p "$MAIN_SSH_PORT" -o BatchMode=yes -o ConnectTimeout=10 "$MAIN_SSH" true 2>/dev/null; then
  echo "  Authorize this key on ${MAIN_SSH}, then re-run:"
  echo "    $(cat ~/.ssh/id_ed25519.pub)"
  exit 1
fi
echo "  OK: ssh $MAIN_SSH"

echo "=== 4. Pipeline env ==="
for kv in "PROTON_SSH=${MAIN_SSH}" "PROTON_SSH_PORT=${MAIN_SSH_PORT}" \
          "PROTON_SCHEDULER_RPC=${MAIN_SCHEDULER_RPC}" "DISPATCH_SERVER_NAME=${SERVER_NAME}" \
          "DISPATCH_CONFIG_TYPES=${CONFIG_TYPES}" "DISPATCH_INPUT_TYPE=${INPUT_TYPE}" \
          "DISPATCH_MAX_WORKERS=${MAX_WORKERS}" "DISPATCH_POLL_INTERVAL=5"; do
  # upsert, so re-running with a new value actually changes it
  if grep -q "^${kv%%=*}=" "$PIPELINE_ROOT/.env" 2>/dev/null; then
    sed -i "s|^${kv%%=*}=.*|${kv}|" "$PIPELINE_ROOT/.env"
  else
    echo "$kv" >> "$PIPELINE_ROOT/.env"
  fi
done
grep -q "^DB_BACKEND=" "$PIPELINE_ROOT/.env" 2>/dev/null \
  || echo "  Postgres runs on the main host: add DB_BACKEND=remote and REMOTE_DBHOST to .env."

echo "=== 5. Interactive shell: cli on PATH ==="
CLI_LINE="export PATH=\"${PIPELINE_ROOT}/pipeline/cli:\$PATH\""
grep -qF "${PIPELINE_ROOT}/pipeline/cli" ~/.bashrc 2>/dev/null || echo "$CLI_LINE" >> ~/.bashrc
echo "  ${PIPELINE_ROOT}/pipeline/cli on PATH in ~/.bashrc (daemons use absolute paths and do not need it)"

echo "=== 6. External binaries ==="
for bin in source-extractor swarp scamp; do
  command -v "$bin" >/dev/null || echo "  MISSING: $bin (or set SEXTRACTOR_COMMAND / SWARP_COMMAND in .env)"
done

echo "=== 7. systemd unit, rendered from the template for this host ==="
sed -e "s|@USER@|${WORKER_USER}|g" \
    -e "s|@PIPELINE_ROOT@|${PIPELINE_ROOT}|g" \
    -e "s|@PYTHON_BIN_DIR@|${PYTHON_BIN_DIR}|g" \
    "$UNIT_TEMPLATE" | sudo tee /etc/systemd/system/pipeline-dispatch-worker.service >/dev/null
sudo systemctl daemon-reload
echo "  installed /etc/systemd/system/pipeline-dispatch-worker.service"

echo "=== Done. Verify, then start ==="
echo "    $PIPELINE_ROOT/pipeline/cli/dispatch_worker_daemon --check"
echo "    sudo systemctl enable --now pipeline-dispatch-worker"
echo "    journalctl -u pipeline-dispatch-worker -f"
