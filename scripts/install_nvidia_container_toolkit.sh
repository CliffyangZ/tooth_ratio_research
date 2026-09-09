#!/usr/bin/env bash
set -euo pipefail

CVAT_DIR="/home/cliffyang/Program/Medical-CV/cvat"
TOOLKIT_VERSION="1.20.0-1"

sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes \
      -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -sSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

sudo apt-get update
sudo apt-get install -y \
  "nvidia-container-toolkit=${TOOLKIT_VERSION}" \
  "nvidia-container-toolkit-base=${TOOLKIT_VERSION}" \
  "libnvidia-container-tools=${TOOLKIT_VERSION}" \
  "libnvidia-container1=${TOOLKIT_VERSION}"

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

cd "$CVAT_DIR"
CVAT_HOST=localhost docker compose \
  -f docker-compose.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d
