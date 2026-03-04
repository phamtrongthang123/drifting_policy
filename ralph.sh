#!/bin/bash
LOGDIR="ralph_logs"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/ralph_$(date +%Y%m%d_%H%M%S).log"

count=0
failures=0
MAX_CONSECUTIVE_FAILURES=3

echo "Logging to $LOGFILE"

while true; do
    count=$((count + 1))
    start_ts=$(date +%s)
    echo "=== Loop #${count} | $(date) ===" | tee -a "$LOGFILE"

    cat prompt.md | claude --model sonnet --dangerously-skip-permissions -p 2>&1 | tee -a "$LOGFILE"
    exit_code=${PIPESTATUS[1]}

    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    echo "--- Loop #${count} finished | exit=$exit_code | ${elapsed}s elapsed ---" | tee -a "$LOGFILE"

    if [ "$exit_code" -ne 0 ]; then
        failures=$((failures + 1))
        echo "WARNING: failure $failures/$MAX_CONSECUTIVE_FAILURES" | tee -a "$LOGFILE"
        if [ "$failures" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
            echo "ERROR: $MAX_CONSECUTIVE_FAILURES consecutive failures, stopping." | tee -a "$LOGFILE"
            exit 1
        fi
    else
        failures=0
    fi
done