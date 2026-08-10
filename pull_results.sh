#!/usr/bin/env bash
# Pull results from GPU server to local machine.
# Usage: bash pull_results.sh user@server [remote_dir]
set -euo pipefail

SERVER="${1:?Usage: bash pull_results.sh user@server [remote_dir]}"
REMOTE_DIR="${2:-ovseg-probes}"

echo "Pulling results from ${SERVER}:${REMOTE_DIR}/ ..."

rsync -avz --progress \
    "${SERVER}:${REMOTE_DIR}/results/" \
    ./results/

rsync -avz --progress \
    "${SERVER}:${REMOTE_DIR}/predictions/" \
    ./predictions/

echo ""
echo "Done. Results in ./results/, predictions in ./predictions/"
