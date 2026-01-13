#!/bin/bash
# Wrapper script to run pipeline trigger in tmux session named "monitor"
# This runs the main monitoring service that watches for new images

# Kill existing session if it exists (clean start)
if tmux has-session -t monitor 2>/dev/null; then
    tmux kill-session -t monitor 2>/dev/null || true
    sleep 1
fi

# Create new session and run the trigger script
tmux new-session -d -s monitor "source ~/.bashrc && /home/pipeline-stable/.conda/envs/pipeline/bin/python /home/pipeline-stable/pipeline/pipeline/cli/run_trigger"

# Verify session was created successfully
sleep 2
if ! tmux has-session -t monitor 2>/dev/null; then
    echo "ERROR: Failed to create tmux session 'monitor'" >&2
    exit 1
fi

# Keep monitoring the session - if it dies, exit (systemd will restart)
while tmux has-session -t monitor 2>/dev/null; do
    sleep 10
done

# Session died, exit with error so systemd knows to restart
exit 1

