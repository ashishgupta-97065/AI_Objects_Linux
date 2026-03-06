#!/bin/bash
set -e
LOGFILE="/home/ag/AI_Objects_Linux/Trellis2VSCode/install_missing_log.txt"
exec > >(tee -a "$LOGFILE") 2>&1

source /home/ag/miniconda3/etc/profile.d/conda.sh
conda activate trellis2vscode

echo "=== Starting missing installs: $(date) ==="

echo ""
echo "--- [1/2] Installing nvdiffrec ---"
mkdir -p /tmp/extensions
if [ ! -d /tmp/extensions/nvdiffrec ]; then
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
else
    echo "nvdiffrec already cloned, using existing."
fi
# Add CUDA stubs to linker path so -lcuda resolves during build
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
pip install /tmp/extensions/nvdiffrec --no-build-isolation
echo "--- nvdiffrec DONE ---"

echo ""
echo "--- [2/2] Installing flash-attn ---"
pip install flash-attn==2.7.3
echo "--- flash-attn DONE ---"

echo ""
echo "=== All installs complete: $(date) ==="
