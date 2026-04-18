#!/usr/bin/env bash

set -euo pipefail

apt update && apt install -y git openssh-client
apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

SSH_DIR="$HOME/.ssh"
KEY_PATH="$SSH_DIR/id_ed25519_github"
CONFIG_PATH="$SSH_DIR/config"
DEFAULT_REPO_URL="git@github.com:vodkar/llm4codesec-framework.git"

mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

if [[ -f "$KEY_PATH" ]]; then
    read -r -p "SSH key already exists at $KEY_PATH. Overwrite? [y/N]: " overwrite_key
    if [[ ! "$overwrite_key" =~ ^[Yy]$ ]]; then
        echo "Keeping existing key file."
    else
        echo "Paste your GitHub private key below. Press Ctrl-D when finished:"
        cat > "$KEY_PATH"
        chmod 600 "$KEY_PATH"
    fi
else
    echo "Paste your GitHub private key below. Press Ctrl-D when finished:"
    cat > "$KEY_PATH"
    chmod 600 "$KEY_PATH"
fi

touch "$CONFIG_PATH"
chmod 600 "$CONFIG_PATH"

if ! grep -Eq '^Host github\.com$' "$CONFIG_PATH"; then
    cat >> "$CONFIG_PATH" <<EOF

Host github.com
    HostName github.com
    User git
    IdentityFile $KEY_PATH
    IdentitiesOnly yes
EOF
fi

touch "$SSH_DIR/known_hosts"
chmod 600 "$SSH_DIR/known_hosts"
if ! ssh-keygen -F github.com -f "$SSH_DIR/known_hosts" >/dev/null 2>&1; then
    ssh-keyscan -H github.com >> "$SSH_DIR/known_hosts"
fi

echo "Testing GitHub SSH connectivity (a success message is not always returned by GitHub):"
ssh -T git@github.com || true

repo_url="${1:-$DEFAULT_REPO_URL}"

if [[ -z "$repo_url" ]]; then
    echo "Unable to determine repository URL. Pass it as the first argument."
    echo "Example: $0 $DEFAULT_REPO_URL"
    exit 1
fi

if [[ "$repo_url" == https://github.com/* || "$repo_url" == http://github.com/* ]]; then
    repo_url="${repo_url#https://github.com/}"
    repo_url="${repo_url#http://github.com/}"
    repo_url="${repo_url%/}"
    repo_url="${repo_url%.git}"
    repo_url="git@github.com:${repo_url}.git"
    echo "Converted GitHub URL to SSH: $repo_url"
fi

clone_dir="${2:-$(basename -s .git "$repo_url")}"

if [[ -d "$clone_dir/.git" ]]; then
    echo "Repository already cloned in $clone_dir. Reusing existing directory."
else
    git clone "$repo_url" "$clone_dir"
fi

cd "$clone_dir"

cat > .env <<'EOF'
# Environment variables for LLM Code Security Benchmark Framework
PYTHONPATH=src/

LOG_LEVEL=INFO
LOG_FORMAT='%(asctime)s - %(name)s - %(levelname)s - %(message)s'

HF_TOKEN=''

PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

MAX_NUM_SEQS=32

# llama.cpp: offload all layers to GPU (-1 = all)
LLAMA_CPP_N_GPU_LAYERS=-1
# llama.cpp: use the HF cache as model download dir (avoids re-downloading)
LLAMA_CPP_MODEL_DIR=/app/.cache/huggingface

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
EOF

git submodule update --init --recursive --remote

if [[ -x "./build-docker.sh" ]]; then
    ./build-docker.sh
elif [[ -x "./build_docker.sh" ]]; then
    ./build_docker.sh
else
    echo "Build script not found: expected ./build-docker.sh or ./build_docker.sh"
    exit 1
fi



./build_docker.sh
./scripts/run_build_datasets.sh
./scripts/run_context_assembler_compare_rankings_rtx6000pro.sh

echo "Completed: build + datasets + context_assembler_compare_rankings_rtx6000pro."