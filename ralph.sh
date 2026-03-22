#!/bin/bash
LOGDIR="ralph_logs"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/ralph_$(date +%Y%m%d_%H%M%S).log"

count=0
failures=0
MAX_CONSECUTIVE_FAILURES=3
REFLECT_EVERY=2

echo "Logging to $LOGFILE"

while true; do
    count=$((count + 1))
    start_ts=$(date +%s)

    # Every REFLECT_EVERY loops, run reflect instead of prompt
    if [ $((count % REFLECT_EVERY)) -eq 0 ]; then
        echo "=== Loop #${count} [REFLECT] | $(date) ===" | tee -a "$LOGFILE"
        cat reflect.md | claude --model opus --effort max --dangerously-skip-permissions -p 2>&1 | tee -a "$LOGFILE"
    else
        echo "=== Loop #${count} | $(date) ===" | tee -a "$LOGFILE"
        cat prompt.md | claude --model opus --effort max --dangerously-skip-permissions -p 2>&1 | tee -a "$LOGFILE"
    fi
    exit_code=${PIPESTATUS[1]}

    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    echo "--- Loop #${count} finished | exit=$exit_code | ${elapsed}s elapsed ---" | tee -a "$LOGFILE"

    # Detect rate-limit / quota hit in the last chunk of the log
    if tail -20 "$LOGFILE" | grep -qiE "you've hit your limit|hit your limit|rate.limit|resets [0-9]+(am|pm)|account.*limit|usage.*limit|quota.*exceeded"; then
        WAIT_HOURS=3
        echo "*** RATE LIMIT DETECTED — sleeping ${WAIT_HOURS} hours until $(date -d "+${WAIT_HOURS} hours") ***" | tee -a "$LOGFILE"
        sleep $((WAIT_HOURS * 3600))
        failures=0  # reset after waiting
        continue
    fi

    if [ "$count" -ge 6 ]; then
        echo "Reached $count loops, stopping." | tee -a "$LOGFILE"
        exit 0
    fi

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