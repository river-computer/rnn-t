#!/bin/bash
# Download EMG data and alignments from S3
#
# Usage:
#   chmod +x scripts/download_data.sh
#   ./scripts/download_data.sh [--local-only]
#
# Options:
#   --local-only: Don't download, just verify local data

set -e

DATA_DIR="${DATA_DIR:-./data}"
S3_BUCKET="${S3_BUCKET:-river-emg-speech}"

echo "========================================"
echo "EMG Data Download"
echo "========================================"
echo "Data directory: $DATA_DIR"
echo "S3 bucket: $S3_BUCKET"
echo ""

# Parse arguments
LOCAL_ONLY=false
if [ "$1" == "--local-only" ]; then
    LOCAL_ONLY=true
fi

# Create data directory
mkdir -p $DATA_DIR

# Function to download and extract
download_and_extract() {
    local s3_path=$1
    local local_path=$2
    local extract_dir=$3

    if [ -f "$local_path" ]; then
        echo "✓ Already exists: $local_path"
    elif [ "$LOCAL_ONLY" = true ]; then
        echo "✗ Not found (local-only mode): $local_path"
        return 1
    else
        echo "Downloading $s3_path..."
        aws s3 cp "s3://$S3_BUCKET/$s3_path" "$local_path"
        echo "✓ Downloaded: $local_path"
    fi

    # Extract if tarball
    if [[ "$local_path" == *.tar.gz ]]; then
        if [ -d "$extract_dir" ] && [ "$(ls -A $extract_dir)" ]; then
            echo "✓ Already extracted: $extract_dir"
        else
            echo "Extracting to $extract_dir..."
            mkdir -p "$extract_dir"
            tar -xzf "$local_path" -C "$extract_dir" --strip-components=1 || \
            tar -xzf "$local_path" -C "$DATA_DIR"
            echo "✓ Extracted"
        fi
    fi
}

# Download EMG data
echo ""
echo "=== EMG Dataset ==="
download_and_extract \
    "data/emg_data.tar.gz" \
    "$DATA_DIR/emg_data.tar.gz" \
    "$DATA_DIR/emg_data"

# Download alignments
echo ""
echo "=== Phoneme Alignments ==="
download_and_extract \
    "data/text_alignments.tar.gz" \
    "$DATA_DIR/text_alignments.tar.gz" \
    "$DATA_DIR/alignments"

# Verify data structure
echo ""
echo "=== Verifying Data Structure ==="

check_dir() {
    local dir=$1
    local desc=$2
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f | wc -l)
        echo "✓ $desc: $count files"
    else
        echo "✗ $desc: not found"
    fi
}

check_dir "$DATA_DIR/emg_data/voiced_parallel_data" "Voiced parallel data"
check_dir "$DATA_DIR/emg_data/silent_parallel_data" "Silent parallel data"
check_dir "$DATA_DIR/emg_data/nonparallel_data" "Non-parallel data"
check_dir "$DATA_DIR/alignments" "Alignments"

# Summary
echo ""
echo "========================================"
echo "Data Setup Complete"
echo "========================================"

# Count samples
if [ -d "$DATA_DIR/emg_data/voiced_parallel_data" ]; then
    emg_count=$(find "$DATA_DIR/emg_data" -name "*_emg.npy" | wc -l)
    echo "Total EMG samples: $emg_count"
fi

if [ -d "$DATA_DIR/alignments" ]; then
    align_count=$(find "$DATA_DIR/alignments" -name "*.TextGrid" | wc -l)
    echo "Total alignments: $align_count"
fi

echo ""
echo "Data is ready for training!"
echo "Run: python scripts/test_pipeline.py --config config/default.yaml"
