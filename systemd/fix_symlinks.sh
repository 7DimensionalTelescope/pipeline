#!/bin/bash
# Fix broken symlinks in /etc/systemd/system/

SYSTEMD_DIR="/home/pipeline-stable/pipeline/systemd"
TARGET_DIR="/etc/systemd/system"

# Remove broken symlinks
sudo rm -f "$TARGET_DIR/pipeline-clear-schedules.service"
sudo rm -f "$TARGET_DIR/pipeline-clear-schedules.timer"
sudo rm -f "$TARGET_DIR/pipeline-queue.service"
sudo rm -f "$TARGET_DIR/pipeline-timer-logrotate"
sudo rm -f "$TARGET_DIR/pipeline-trigger.service"

# Create correct symlinks with absolute paths
sudo ln -s "$SYSTEMD_DIR/pipeline-clear-schedules.service" "$TARGET_DIR/pipeline-clear-schedules.service"
sudo ln -s "$SYSTEMD_DIR/pipeline-clear-schedules.timer" "$TARGET_DIR/pipeline-clear-schedules.timer"
sudo ln -s "$SYSTEMD_DIR/pipeline-queue.service" "$TARGET_DIR/pipeline-queue.service"
sudo ln -s "$SYSTEMD_DIR/pipeline-timer-logrotate" "$TARGET_DIR/pipeline-timer-logrotate"
sudo ln -s "$SYSTEMD_DIR/pipeline-trigger.service" "$TARGET_DIR/pipeline-trigger.service"

# Every cli/ shebang hardcodes this interpreter path on EVERY host; worker hosts satisfy it with the
# same symlink (dispatch/setup_dispatch_worker.sh). Repoint it whenever the conda install moves.
CONDA_ENV="${CONDA_ENV:-/home/pipeline-stable/miniconda3/envs/pipeline}"
SHEBANG_ENV="/home/pipeline-stable/.conda/envs/pipeline"
if [ ! -e "$SHEBANG_ENV" ]; then
  mkdir -p "$(dirname "$SHEBANG_ENV")"
  ln -sfn "$CONDA_ENV" "$SHEBANG_ENV"
  echo "Linked $SHEBANG_ENV -> $CONDA_ENV"
fi

# /etc/tmpfiles.d is the directory systemd-tmpfiles actually reads (unlike pipeline-timer-logrotate,
# which is symlinked into one logrotate never reads). tmpfiles-setup is After=local-fs.target, so this
# symlink into /home resolves at boot; --create applies it now instead of waiting for one.
sudo rm -f /etc/tmpfiles.d/pipeline.conf
sudo ln -s "$SYSTEMD_DIR/pipeline-tmpfiles.conf" /etc/tmpfiles.d/pipeline.conf
sudo systemd-tmpfiles --create /etc/tmpfiles.d/pipeline.conf

# Reload systemd daemon
sudo systemctl daemon-reload

echo "Symlinks fixed. Verifying..."
ls -la "$TARGET_DIR/pipeline*"









