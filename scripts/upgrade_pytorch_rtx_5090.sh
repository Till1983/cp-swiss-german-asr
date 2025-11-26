#!/bin/bash
# upgrade_pytorch_rtx5090.sh
# Quick PyTorch upgrade for RTX 5090 compatibility

set -e  # Exit on any error

echo "============================================"
echo "🚀 Upgrading PyTorch for RTX 5090"
echo "============================================"
echo ""

# 1. Backup current packages
echo "📦 Backing up current environment..."
cd /workspace/cp-swiss-german-asr
pip freeze > backup_requirements_$(date +%Y%m%d_%H%M%S).txt
echo "✅ Backup saved"
echo ""

# 2. Uninstall old PyTorch
echo "🗑️  Uninstalling PyTorch 2.6.0..."
pip uninstall -y torch torchvision torchaudio
echo "✅ Old PyTorch removed"
echo ""

# 3. Install PyTorch 2.8.0 with CUDA 12.8
echo "⬇️  Installing PyTorch 2.8.0+cu128..."
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
echo "✅ PyTorch 2.8.0 installed"
echo ""

# 4. Upgrade key dependencies
echo "📦 Upgrading transformers ecosystem..."
pip install --upgrade transformers==4.46.0 accelerate==1.2.1
echo "✅ Dependencies upgraded"
echo ""

# 5. Fix NumPy if needed
echo "🔧 Ensuring NumPy compatibility..."
pip install numpy==1.26.4
echo "✅ NumPy fixed"
echo ""

# 6. Verify installation
echo "============================================"
echo "✅ VERIFICATION"
echo "============================================"
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test actual computation
    print("\n🧪 Testing GPU computation...")
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)
    print("✅ GPU computation works!")
else:
    print("❌ CUDA not available!")
EOF

echo ""
echo "============================================"
echo "🎉 UPGRADE COMPLETE!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  cd /workspace/cp-swiss-german-asr"
echo "  ./scripts/adapt_on_cloud.sh"
echo ""
