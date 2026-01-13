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

# Reload systemd daemon
sudo systemctl daemon-reload

echo "Symlinks fixed. Verifying..."
ls -la "$TARGET_DIR/pipeline*"









