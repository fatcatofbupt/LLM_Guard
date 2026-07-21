#!/bin/bash
# Start the safety-guard chat service.
#
# Usage:
#   ./start_server.sh              # foreground
#   ./start_server.sh --daemon     # background (nohup)
#   ./start_server.sh --stop       # stop background process
#   ./start_server.sh --status     # check if running
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PIDFILE=".safety_service.pid"
LOGFILE="logs/safety_service.log"
PYTHON="${PYTHON:-python3}"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
print_header() {
    echo "=========================================="
    echo "  Safety-Guard Chat Service"
    echo "=========================================="
}

check_deps() {
    echo "[check] Verifying Python dependencies ..."
    local missing=()
    for pkg in fastapi uvicorn openai loguru transformers torch; do
        if ! $PYTHON -c "import $pkg" 2>/dev/null; then
            missing+=("$pkg")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "[error] Missing packages: ${missing[*]}"
        echo "[hint] Run: ./install_deps.sh"
        exit 1
    fi
    echo "[check] All dependencies OK."
}

check_gpu() {
    echo "[check] Verifying GPU 3 availability ..."
    if ! $PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; assert torch.cuda.device_count() > 3, 'GPU 3 not found'; print(f'GPU 3: {torch.cuda.get_device_name(3)}')" 2>/dev/null; then
        echo "[warn] GPU 3 check failed. The service will fall back to available device."
    fi
}

check_model() {
    if [ ! -d "finetune_qwen3guard/output/lora_v5_1/merged_model" ]; then
        echo "[error] Fine-tuned model not found at:"
        echo "  finetune_qwen3guard/output/lora_v5_1/merged_model"
        echo "[hint] Check the model path or run merge_adapter.py first."
        exit 1
    fi
    echo "[check] Model found."
}

start_foreground() {
    print_header
    #check_deps
    #check_gpu
    #check_model

    echo ""
    echo "[start] Launching service on 0.0.0.0:32469 (GPU 3) ..."
    echo "[start] Press Ctrl+C to stop."
    echo ""

    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES=3
    exec $PYTHON scripts/services/safety_service.py
}

start_daemon() {
    print_header
    # check_deps
    # check_gpu
    # check_model

    # Kill any running instance first
    if [ -f "$PIDFILE" ]; then
        local old_pid
        old_pid=$(cat "$PIDFILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "[stop] Stopping existing service (PID: $old_pid) ..."
            kill "$old_pid" 2>/dev/null || true
            sleep 2
            if kill -0 "$old_pid" 2>/dev/null; then
                echo "[stop] Force killing ..."
                kill -9 "$old_pid" 2>/dev/null || true
            fi
            echo "[ok] Old instance stopped."
        fi
        rm -f "$PIDFILE"
    fi

    # Also kill any orphaned safety_service.py processes
    local orphaned
    orphaned=$(pgrep -f "python.*safety_service.py" || true)
    if [ -n "$orphaned" ]; then
        echo "[stop] Killing orphaned processes: $orphaned"
        kill -9 $orphaned 2>/dev/null || true
        sleep 1
    fi

    mkdir -p logs

    echo ""
    echo "[start] Launching service in background ..."
    echo "[start] Log file: $LOGFILE"
    echo "[start] PID file: $PIDFILE"
    echo ""

    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES=3

    CUDA_VISIBLE_DEVICES=3 nohup $PYTHON scripts/services/safety_service.py > "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"

    sleep 2
    if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "[ok] Service started successfully (PID: $(cat "$PIDFILE"))"
        echo "[ok] Health check: curl http://localhost:32469/health"
    else
        echo "[error] Service failed to start. Check log: $LOGFILE"
        rm -f "$PIDFILE"
        exit 1
    fi
}

stop_daemon() {
    if [ ! -f "$PIDFILE" ]; then
        echo "[info] No PID file found. Service not running?"
        exit 0
    fi

    local pid
    pid=$(cat "$PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
        echo "[stop] Stopping service (PID: $pid) ..."
        kill "$pid"
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
            echo "[stop] Force killing ..."
            kill -9 "$pid" 2>/dev/null || true
        fi
        echo "[ok] Service stopped."
    else
        echo "[info] Service not running (stale PID file)."
    fi
    rm -f "$PIDFILE"
}

check_status() {
    if [ -f "$PIDFILE" ]; then
        local pid
        pid=$(cat "$PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[status] Running (PID: $pid)"
            echo "[status] Log: $LOGFILE"
            echo "[status] Try: curl -s http://localhost:32469/health | python3 -m json.tool"
        else
            echo "[status] Not running (stale PID file)."
            rm -f "$PIDFILE"
        fi
    else
        echo "[status] Not running."
    fi
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
case "${1:-}" in
    --stop|-s)
        stop_daemon
        ;;
    --status|-S)
        check_status
        ;;
    --help|-h)
        echo "Usage: $0 [--stop|--status|--help]"
        echo ""
        echo "Options:"
        echo "  (none)       Start in background (kill old instance first)"
        echo "  --stop       Stop background service"
        echo "  --status     Check service status"
        echo "  --help       Show this help"
        ;;
    *)
        start_daemon
        ;;
esac
