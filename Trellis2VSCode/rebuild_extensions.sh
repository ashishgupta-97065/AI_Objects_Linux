#!/bin/bash
set -e

source /home/ag/miniconda3/etc/profile.d/conda.sh
conda activate trellis2vscode

WORKDIR="/home/ag/AI_Objects_Linux/Trellis2VSCode"

echo "=== Rebuilding all CUDA extensions against torch $(python -c 'import torch; print(torch.__version__)') ==="

echo ""
echo "--- [1/5] flash-attn ---"
pip install flash-attn==2.7.3 --no-build-isolation
echo "--- flash-attn DONE ---"

echo ""
echo "--- [2/5] nvdiffrast ---"
mkdir -p /tmp/extensions
if [ ! -d /tmp/extensions/nvdiffrast ]; then
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
fi
pip install /tmp/extensions/nvdiffrast --no-build-isolation --force-reinstall
echo "--- nvdiffrast DONE ---"

echo ""
echo "--- [3/5] nvdiffrec ---"
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
pip install /tmp/extensions/nvdiffrec --no-build-isolation --force-reinstall
echo "--- nvdiffrec DONE ---"

echo ""
echo "--- [4/5] cumesh ---"
# --no-deps: prevent pip from resolving/upgrading torch via cumesh's torch>=2.4.0 dep
pip install /tmp/extensions/CuMesh --no-build-isolation --force-reinstall --no-deps
echo "--- cumesh DONE ---"

echo ""
echo "--- [5/5] o_voxel + flex_gemm ---"
# --no-deps on all three: o_voxel pulls cumesh/flex_gemm from git which then upgrade torch
cp -r $WORKDIR/o-voxel /tmp/extensions/o-voxel-rebuild
pip install /tmp/extensions/o-voxel-rebuild --no-build-isolation --force-reinstall --no-deps
pip install /tmp/extensions/FlexGEMM --no-build-isolation --force-reinstall --no-deps
echo "--- o_voxel + flex_gemm DONE ---"

echo ""
echo "--- Pinning torch back to 2.6.0 (in case any dep upgraded it) ---"
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps
echo "--- torch pin DONE ---"

echo ""
echo "=== All extensions rebuilt ==="
