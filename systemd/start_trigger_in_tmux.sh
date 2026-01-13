#!/bin/bash
# Wrapper script to run pipeline trigger in tmux session named "monitor"
# This runs the main monitoring service that watches for new images

# Create session if it doesn't exist, then run the trigger script
if ! tmux has-session -t monitor 2>/dev/null; then
    tmux new-session -d -s monitor "cd /tmp/pipeline && source ~/.bashrc && /home/pipeline/.conda/envs/pipeline/bin/python /home/pipeline/pipeline/pipeline/bin/run_trigger"
else
    # Session exists - restart the trigger script in it
    tmux send-keys -t monitor C-c 2>/dev/null || true
    sleep 1
    tmux send-keys -t monitor "cd /tmp/pipeline && source ~/.bashrc && /home/pipeline/.conda/envs/pipeline/bin/python /home/pipeline/pipeline/pipeline/bin/run_trigger" Enter
fi

# Keep monitoring the session
while tmux has-session -t monitor 2>/dev/null; do
    sleep 10
done

