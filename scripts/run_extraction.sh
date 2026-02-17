#!/bin/bash
# Automated census extraction loop using Claude Code CLI.
#
# Parallel-safe: uses .lock files so multiple workers don't collide.
# Each iteration spawns a fresh Claude conversation that reads one PDF,
# extracts census data, writes JSON, and validates.
#
# Usage:
#   ./opus46/run_extraction.sh                  # Run until all done
#   ./opus46/run_extraction.sh --limit 10       # Process at most 10 PDFs
#   ./opus46/run_extraction.sh --workers 3      # Launch 3 parallel workers
#   ./opus46/run_extraction.sh --dry-run        # Show what would be processed

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OPUS46_DIR="$PROJECT_DIR/opus46"
TOTAL=188
LIMIT=0       # 0 = unlimited
WORKERS=1
DRY_RUN=false

# ── Parse args ─────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)     LIMIT="$2"; shift 2 ;;
        --workers)   WORKERS="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--limit N] [--workers N] [--dry-run]"
            echo "  --limit N     Process at most N PDFs per worker (default: unlimited)"
            echo "  --workers N   Launch N parallel workers (default: 1)"
            echo "  --dry-run     Show what would be processed without running"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$PROJECT_DIR"

# ── Worker function ────────────────────────────────────────────────────────

run_worker() {
    local worker_id=$1
    local log_file="$OPUS46_DIR/worker_${worker_id}.log"
    local processed=0

    log() {
        local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [W$worker_id] $*"
        echo "$msg"
        echo "$msg" >> "$log_file"
    }

    log "Worker $worker_id started"

    while true; do
        # Check limit
        if [ "$LIMIT" -gt 0 ] && [ "$processed" -ge "$LIMIT" ]; then
            log "Reached limit of $LIMIT. Stopping."
            break
        fi

        # Atomically claim next PDF using census_extractor.py (lock-aware)
        next_output=$(python3 opus46/census_extractor.py next 2>/dev/null)
        next_id=$(echo "$next_output" | grep -oE 'pair_[0-9]+' | head -1)

        if [ -z "$next_id" ]; then
            log "No more PDFs to process. Done."
            break
        fi

        local lock_file="$OPUS46_DIR/${next_id}.lock"
        local json_file="$OPUS46_DIR/${next_id}.json"
        local pdf_file="$PROJECT_DIR/2page_pdfs/${next_id}.pdf"

        # Atomic lock: create lock file (skip if another worker got it)
        if ! (set -o noclobber; echo "$$" > "$lock_file") 2>/dev/null; then
            # Another worker claimed this one — try again immediately
            sleep 0.5
            continue
        fi

        local done_count
        done_count=$(find "$OPUS46_DIR" -maxdepth 1 -name 'pair_*.json' | wc -l | tr -d ' ')
        log "[$((done_count + 1))/$TOTAL] Processing $next_id..."

        if $DRY_RUN; then
            echo "  DRY RUN [W$worker_id]: would process $next_id"
            rm -f "$lock_file"
            processed=$((processed + 1))
            break  # dry run only shows one per worker
        fi

        # Build a prompt with the SPECIFIC PDF path (no race condition)
        local prompt="Extract census data from this specific PDF:
PDF file: $pdf_file
Output to: $json_file

Steps:
1. Read the PDF file (use the Read tool on the path above)
2. Extract ALL municipality rows from the census table into JSON following the schema in CLAUDE.md
3. Write the JSON to the output path above
4. Run: python3 opus46/census_extractor.py validate $json_file
5. Run: python3 opus46/census_extractor.py compare $json_file
6. If validate shows errors, fix the JSON and re-run validation
7. If you discover anything noteworthy, append a line to opus46/notes.md"

        # Run Claude in print mode, capture output to check for rate limits
        local claude_output
        claude_output=$(claude -p "$prompt" \
            --max-turns 25 \
            --allowedTools "Read Write Edit Bash Glob Grep" \
            2>&1) || true
        echo "$claude_output" >> "$log_file"

        # Check for rate limit / usage errors — stop worker immediately
        if echo "$claude_output" | grep -qiE "out of (extra )?usage|rate limit|quota exceeded|resets [0-9]"; then
            log "RATE LIMITED. Stopping worker $worker_id."
            rm -f "$lock_file"
            break
        fi

        # Check result and clean up lock
        if [ -f "$json_file" ]; then
            log "SUCCESS: $next_id extracted."
        else
            log "FAILED: No JSON created for $next_id. Will retry next run."
        fi

        rm -f "$lock_file"
        processed=$((processed + 1))

        sleep 2
    done

    log "Worker $worker_id finished ($processed processed)"
}

# ── Main ───────────────────────────────────────────────────────────────────

echo "=== Census Extraction ==="
echo "  Workers: $WORKERS"
echo "  Limit:   ${LIMIT:-unlimited} per worker"
echo ""

python3 opus46/census_extractor.py status

if [ "$WORKERS" -eq 1 ]; then
    # Single worker — run in foreground
    run_worker 1
else
    # Multiple workers — run in parallel
    pids=()
    for i in $(seq 1 "$WORKERS"); do
        run_worker "$i" &
        pids+=($!)
        sleep 2  # stagger starts slightly
    done

    echo "Launched $WORKERS workers: PIDs ${pids[*]}"
    echo "Logs: opus46/worker_*.log"
    echo ""
    echo "Monitor progress:  python3 opus46/census_extractor.py status"
    echo "Follow a worker:   tail -f opus46/worker_1.log"
    echo "Stop all:          kill ${pids[*]}"

    # Wait for all workers
    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done
fi

echo ""
echo "=== Final Status ==="
python3 opus46/census_extractor.py status
