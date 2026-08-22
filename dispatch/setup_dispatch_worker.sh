#!/bin/bash
# Set up a pipeline dispatch worker on a new host.
# Run on the worker host as a user with sudo. Override any variable from the environment.

set -euo pipefail

ORIGIN_IP="${ORIGIN_IP:-10.1.1.51}"          # host running the scheduler + trigger + queue daemons
LYMAN_IP="${LYMAN_IP:-10.1.1.50}"            # NFS server for /lyman/data1 and /lyman/data2
BALMER_IP="${BALMER_IP:-10.1.1.52}"          # NFS server for /balmer/data1 (the *_DIR_2 roots)
MOUNT="${MOUNT:-/mnt/origin/var/db}"
ORIGIN_DB="${MOUNT}/scheduler.db"
PIPELINE_ROOT="${PIPELINE_ROOT:-/opt/pipeline}"
CONDA_PYTHON="${CONDA_PYTHON:-/opt/conda/envs/pipeline/bin/python3}"
SERVER_NAME="${SERVER_NAME:-$(hostname -s)}"

NFSOPT="rw,noatime,nodiratime,vers=4.1,rsize=1048576,wsize=1048576,hard,proto=tcp,timeo=600,acregmin=1,acregmax=1,acdirmin=1,acdirmax=1,_netdev 0 0"

echo "=== 1. Data mounts (read-write: PathHandler creates directories on attribute read) ==="
for spec in "${LYMAN_IP}:/lyman/data1 /lyman/data1" \
            "${LYMAN_IP}:/lyman/data2 /lyman/data2" \
            "${BALMER_IP}:/balmer/data1 /balmer/data1"; do
  src="${spec%% *}"; dst="${spec##* }"
  sudo mkdir -p "$dst"
  grep -qE "^[^#]*[[:space:]]${dst}[[:space:]]" /etc/fstab 2>/dev/null \
    || echo "${src} ${dst} nfs4 ${NFSOPT}" | sudo tee -a /etc/fstab >/dev/null
  mountpoint -q "$dst" || sudo mount "$dst"
done

echo "=== 2. Origin scheduler DB mount ==="
sudo mkdir -p "$MOUNT"
grep -qE "^[^#]*[[:space:]]${MOUNT}[[:space:]]" /etc/fstab 2>/dev/null \
  || echo "${ORIGIN_IP}:/var/db ${MOUNT} nfs4 ${NFSOPT}" | sudo tee -a /etc/fstab >/dev/null
mountpoint -q "$MOUNT" || sudo mount "$MOUNT"
test -f "$ORIGIN_DB" || { echo "Missing $ORIGIN_DB - export /var/db from ${ORIGIN_IP} first"; exit 1; }
echo "OK: $ORIGIN_DB"

echo "=== 3. Host directories ==="
sudo install -d -m 0775 -o root -g pipeline /var/log/pipeline
sudo install -d -m 0775 -o root -g pipeline /var/lock/py7dt
sudo install -d -m 1777 /tmp/pipeline

echo "=== 4. Pipeline env ==="
grep -q ORIGIN_SCHEDULER_DB_PATH "$PIPELINE_ROOT/.env" 2>/dev/null \
  || echo "ORIGIN_SCHEDULER_DB_PATH=${ORIGIN_DB}" >> "$PIPELINE_ROOT/.env"
grep -q DISPATCH_SERVER_NAME "$PIPELINE_ROOT/.env" 2>/dev/null \
  || echo "DISPATCH_SERVER_NAME=${SERVER_NAME}" >> "$PIPELINE_ROOT/.env"

echo "=== 5. External binaries ==="
for bin in source-extractor swarp scamp missfits; do
  command -v "$bin" >/dev/null || echo "  MISSING: $bin (set it in ref/deployment.yml commands:, or install it)"
done

echo "=== 6. systemd unit ==="
echo "  Edit systemd/pipeline-dispatch-worker.service for this host's paths and user, then:"
echo "    sudo cp $PIPELINE_ROOT/systemd/pipeline-dispatch-worker.service /etc/systemd/system/"
echo "    sudo systemctl daemon-reload && sudo systemctl enable --now pipeline-dispatch-worker"

echo "=== Done ==="
echo "  systemctl status pipeline-dispatch-worker"
echo "  journalctl -u pipeline-dispatch-worker -f"
echo "  Smoke test in the foreground (do NOT use --once, see the guide):"
echo "    DISPATCH_MAX_WORKERS=1 $CONDA_PYTHON $PIPELINE_ROOT/pipeline/cli/dispatch_worker_daemon"
