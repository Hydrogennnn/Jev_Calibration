
#!/bin/bash

set -Eeuo pipefail

# ==========================================
# CoIN Dataset Downloader
# ==========================================

ROOT="/data1/wangjiedong/Valen/data/coin/images"
mkdir -p "$ROOT"

echo "Dataset root: $ROOT"

# ------------------------------------------
# Check dependencies
# ------------------------------------------

for cmd in curl unzip; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Please install $cmd first."
        exit 1
    fi
done

if ! command -v hf &>/dev/null; then
    echo "Installing Hugging Face CLI..."
    pip install -U huggingface_hub
fi

if ! command -v gdown &>/dev/null; then
    echo "Installing gdown..."
    pip install -U gdown
fi

# ------------------------------------------
# Download and extract ZIP
# ------------------------------------------

download_zip() {
    local name="$1"
    local url="$2"
    local dir="$3"

    local archive="$ROOT/archives/${name}.zip"

    mkdir -p "$ROOT/archives" "$dir"

    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "[SKIP] $name already extracted."
        return 0
    fi

    echo "[DOWNLOAD] $name"

    if [ ! -f "$archive" ]; then
        curl -fL --retry 5 --retry-delay 3 -k -L \
            -C - "$url" -o "$archive"
    fi

    echo "[EXTRACT] $name"
    unzip -q -o "$archive" -d "$dir"

    echo "[DONE] $name"
}

# ------------------------------------------
# Download CoIN itself
# ------------------------------------------

# echo "========== CoIN =========="

# if [ ! -d "$ROOT/CoIN" ]; then
#     hf download Zacks-Chen/CoIN \
#         --repo-type dataset \
#         --local-dir "$ROOT/CoIN"
# else
#     echo "[SKIP] CoIN repository already exists."
# fi

# ------------------------------------------
# COCO
# ------------------------------------------

echo "========== COCO =========="

download_zip "coco_train2014" \
    "https://images.cocodataset.org/zips/train2014.zip" \
    "$ROOT/COCO"

download_zip "coco_val2014" \
    "https://images.cocodataset.org/zips/val2014.zip" \
    "$ROOT/COCO"

download_zip "coco_test2015" \
    "https://images.cocodataset.org/zips/test2015.zip" \
    "$ROOT/COCO"

# ------------------------------------------
# RefCOCO / RefCOCO+ / RefCOCOg
# ------------------------------------------

# echo "========== RefCOCO =========="

# download_zip "refcoco" \
#     "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip" \
#     "$ROOT/RefCOCO"

# download_zip "refcoco_plus" \
#     "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip" \
#     "$ROOT/RefCOCO+"

# download_zip "refcocog" \
#     "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip" \
#     "$ROOT/RefCOCOg"

# ------------------------------------------
# GQA
# ------------------------------------------

echo "========== GQA =========="

download_zip "gqa_images" \
    "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" \
    "$ROOT/GQA"

# ------------------------------------------
# TextVQA
# ------------------------------------------

echo "========== TextVQA =========="

download_zip "textvqa_train" \
    "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" \
    "$ROOT/TextVQA"

download_zip "textvqa_test" \
    "https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip" \
    "$ROOT/TextVQA"

# ------------------------------------------
# VizWiz
# ------------------------------------------

echo "========== VizWiz =========="

download_zip "vizwiz_train" \
    "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip" \
    "$ROOT/VizWiz"

download_zip "vizwiz_val" \
    "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip" \
    "$ROOT/VizWiz"

download_zip "vizwiz_test" \
    "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip" \
    "$ROOT/VizWiz"

exit

# ------------------------------------------
# Google Drive datasets
# ------------------------------------------

echo "========== Google Drive datasets =========="

mkdir -p "$ROOT/OCR-VQA" "$ROOT/ScienceQA"

# OCR-VQA
echo "========== OCR-VQA =========="
gdown --folder \
    "https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_" \
    -O "$ROOT/OCR-VQA" || echo "[WARNING] OCR-VQA download failed."

# ScienceQA
echo "========== ScienceQA =========="
# gdown --folder \   
#     "https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev" \
#     -O "$ROOT/ScienceQA" || echo "[WARNING] ScienceQA download failed."

gdown --folder \
    "https://drive.google.com/drive/folders/16kuhXdM-MOhYcFIyRj91WvnDnjnF-xHw" \
    -O "$ROOT/ScienceQA" || echo "[WARNING] ScienceQA download failed."

# ------------------------------------------
# ImageNet
# ------------------------------------------

echo "========== ImageNet =========="

mkdir -p "$ROOT/ImageNet"

echo "[MANUAL] ImageNet requires registration and authorization."
echo "Download ImageNet from:"
echo "https://image-net.org/challenges/LSVRC/"
echo "Place the downloaded files under: $ROOT/ImageNet"

# ------------------------------------------
# Finish
# ------------------------------------------

echo ""
echo "=========================================="
echo "CoIN dataset download process completed."
echo "Dataset root: $ROOT"
echo "=========================================="
