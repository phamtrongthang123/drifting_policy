#!/bin/bash
while true; do
    cat prompt.md | claude --model sonnet --dangerously-skip-permissions -p 
done