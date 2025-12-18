#!/bin/bash
# Script to check queue service status and logs

# Load environment variables
if [ -f "/home/pipeline/pipeline/.env" ]; then
    set -a
    source /home/pipeline/pipeline/.env
    set +a
fi

echo "=== Queue Service Status ==="
systemctl status queue.service --no-pager -l

echo -e "\n=== Recent Journal Logs (last 50 lines) ==="
journalctl -u queue.service -n 50 --no-pager

echo -e "\n=== Socket Status ==="
if [ -S "/run/queue/queue.sock" ]; then
    echo "✓ Socket exists: /run/queue/queue.sock"
    ls -l /run/queue/queue.sock
else
    echo "✗ Socket not found: /run/queue/queue.sock"
fi

echo -e "\n=== Recent Log Files ==="
if [ -d "/var/log/pipeline" ]; then
    echo "Recent log files:"
    ls -lht /var/log/pipeline/*.log 2>/dev/null | head -5
    echo -e "\nLatest log file content (last 20 lines):"
    LATEST_LOG=$(ls -t /var/log/pipeline/*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        tail -20 "$LATEST_LOG"
    else
        echo "No log files found"
    fi
else
    echo "Log directory /var/log/pipeline not found"
fi

echo -e "\n=== Database Status ==="
if [ -n "$SCHEDULER_DB_PATH" ] && [ -f "$SCHEDULER_DB_PATH" ]; then
    echo "✓ Database exists: $SCHEDULER_DB_PATH"
    echo "Database size: $(du -h "$SCHEDULER_DB_PATH" | cut -f1)"
    echo -e "\nJob counts:"
    sqlite3 "$SCHEDULER_DB_PATH" "SELECT status, COUNT(*) FROM scheduler GROUP BY status;" 2>/dev/null || echo "Could not query database"
else
    echo "✗ Database not found: ${SCHEDULER_DB_PATH:-SCHEDULER_DB_PATH not set}"
fi

echo -e "\n=== Active Processes ==="
pgrep -af queue_daemon || echo "No queue_daemon processes found"


