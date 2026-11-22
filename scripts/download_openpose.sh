#!/usr/bin/env bash
# Baixa o modelo OpenPose COCO (CMU) para uso com --pose-backend openpose (~200 MB).
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)/models/openpose"
mkdir -p "$DIR"
curl -L -o "$DIR/pose_deploy_linevec.prototxt" \
  https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt
echo "Baixe pose_iter_440000.caffemodel (link em models/getModels.sh do repositório OpenPose) para $DIR"
