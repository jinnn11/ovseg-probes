#!/usr/bin/env bash
# Pull results from GPU server to local machine.
# Usage:
#   bash pull_results.sh user@server                          # default SSH port
#   bash pull_results.sh user@server 22 /workspace/ovseg-probes  # custom port + path
#   bash pull_results.sh root@1.2.3.4 12345                   # Vast.ai style
set -euo pipefail

SERVER="${1:?Usage: bash pull_results.sh user@server [ssh_port] [remote_dir]}"
SSH_PORT="${2:-22}"
REMOTE_DIR="${3:-ovseg-probes}"

echo "Pulling results from ${SERVER}:${REMOTE_DIR}/ (port ${SSH_PORT}) ..."

rsync -avz --progress -e "ssh -p ${SSH_PORT}" \
    "${SERVER}:${REMOTE_DIR}/results/" \
    ./results/

rsync -avz --progress -e "ssh -p ${SSH_PORT}" \
    "${SERVER}:${REMOTE_DIR}/predictions/" \
    ./predictions/

echo ""
echo "Done. Results in ./results/, predictions in ./predictions/"
