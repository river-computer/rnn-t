#!/bin/bash
# EC2 Setup Script for EMG RNN-T Training
#
# This script sets up an EC2 instance (g5.xlarge recommended) for training.
# Run this after launching a fresh Deep Learning AMI instance.
#
# Usage:
#   chmod +x scripts/setup_ec2.sh
#   ./scripts/setup_ec2.sh
#
# Prerequisites:
#   - AWS credentials configured (for S3 access)
#   - EC2 instance with GPU (g5.xlarge, A10G 24GB recommended)
#   - Ubuntu or Amazon Linux 2 AMI with CUDA pre-installed

set -e  # Exit on error

echo "========================================"
echo "EMG RNN-T EC2 Setup"
echo "========================================"

# 1. System updates
echo ""
echo "=== Step 1: System Updates ==="
sudo apt-get update -y || sudo yum update -y
echo "✓ System updated"

# 2. Check CUDA
echo ""
echo "=== Step 2: Checking CUDA ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✓ NVIDIA driver available"
else
    echo "✗ NVIDIA driver not found. Please use a Deep Learning AMI."
    exit 1
fi

# 3. Python environment
echo ""
echo "=== Step 3: Setting up Python environment ==="

# Create virtual environment
python3 -m venv ~/emg_venv
source ~/emg_venv/bin/activate

# Upgrade pip
pip install --upgrade pip

echo "✓ Virtual environment created at ~/emg_venv"

# 4. Install PyTorch with CUDA
echo ""
echo "=== Step 4: Installing PyTorch ==="
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo "✓ PyTorch installed"

# 5. Install k2 for pruned RNN-T loss
echo ""
echo "=== Step 5: Installing k2 (pruned RNN-T loss) ==="
# k2 installation can be tricky - try pip first
pip install k2 || {
    echo "pip install failed, trying from source..."
    pip install cmake
    git clone https://github.com/k2-fsa/k2.git /tmp/k2
    cd /tmp/k2
    python setup.py install
    cd -
}

# Verify k2
python -c "import k2; print(f'k2 {k2.__version__}')" || echo "⚠ k2 not available (will use torchaudio fallback)"
echo "✓ k2 installation attempted"

# 6. Install project dependencies
echo ""
echo "=== Step 6: Installing project dependencies ==="
pip install numpy scipy librosa praatio tslearn editdistance wandb boto3 pyyaml

echo "✓ Dependencies installed"

# 7. Clone/update project
echo ""
echo "=== Step 7: Setting up project ==="
PROJECT_DIR=~/emg-rnnt

if [ -d "$PROJECT_DIR" ]; then
    echo "Project directory exists, pulling latest..."
    cd $PROJECT_DIR
    git pull || true
else
    echo "Project directory not found."
    echo "Please copy project files to $PROJECT_DIR"
    mkdir -p $PROJECT_DIR
fi

echo "✓ Project directory: $PROJECT_DIR"

# 8. Download data from S3
echo ""
echo "=== Step 8: Downloading data from S3 ==="
DATA_DIR=$PROJECT_DIR/data
mkdir -p $DATA_DIR

# Check AWS credentials
if aws sts get-caller-identity &> /dev/null; then
    echo "AWS credentials configured"

    # Download EMG data
    if [ ! -f "$DATA_DIR/emg_data.tar.gz" ]; then
        echo "Downloading EMG data (3.7GB)..."
        aws s3 cp s3://river-emg-speech/data/emg_data.tar.gz $DATA_DIR/
        echo "Extracting..."
        tar -xzf $DATA_DIR/emg_data.tar.gz -C $DATA_DIR/
        echo "✓ EMG data downloaded and extracted"
    else
        echo "✓ EMG data already exists"
    fi

    # Download alignments
    if [ ! -f "$DATA_DIR/text_alignments.tar.gz" ]; then
        echo "Downloading alignments..."
        aws s3 cp s3://river-emg-speech/data/text_alignments.tar.gz $DATA_DIR/
        tar -xzf $DATA_DIR/text_alignments.tar.gz -C $DATA_DIR/
        echo "✓ Alignments downloaded and extracted"
    else
        echo "✓ Alignments already exist"
    fi
else
    echo "⚠ AWS credentials not configured. Please run: aws configure"
fi

# 9. Configure wandb (optional)
echo ""
echo "=== Step 9: Configure wandb (optional) ==="
echo "To enable experiment tracking, run: wandb login"

# 10. Verify setup
echo ""
echo "=== Step 10: Verifying setup ==="
cd $PROJECT_DIR

# Activate environment
source ~/emg_venv/bin/activate

# Run test pipeline
if [ -f "scripts/test_pipeline.py" ]; then
    python scripts/test_pipeline.py --config config/default.yaml || {
        echo "⚠ Some tests failed. Check output above."
    }
else
    echo "Test script not found - skipping verification"
fi

# Summary
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  source ~/emg_venv/bin/activate"
echo ""
echo "To start training:"
echo "  cd $PROJECT_DIR"
echo "  python scripts/train_all.py --config config/default.yaml"
echo ""
echo "For individual stages:"
echo "  python -m src.training.ctc_train --config config/default.yaml"
echo "  python -m src.training.rnnt_train --config config/default.yaml"
echo "  python -m src.training.silent_adapt --config config/default.yaml"
echo ""
echo "Estimated training time: 50-80 hours total"
echo "Estimated cost: \$50-80 on-demand, \$15-25 spot"
