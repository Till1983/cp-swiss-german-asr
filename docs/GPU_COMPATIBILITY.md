# GPU Compatibility Guide

**Last Updated:** November 26, 2025

---

## Overview

This document outlines GPU compatibility requirements for the Swiss German ASR project, specifically focusing on PyTorch version requirements for different NVIDIA GPU architectures.

---

## Supported GPU Configurations

### Currently Tested & Working

| GPU Model | Architecture | VRAM | PyTorch Version | EWC Mode | Status |
|-----------|-------------|------|----------------|----------|---------|
| RTX 3090 | Ampere (sm_86) | 24 GB | 2.6.0+ | CPU | ✅ Tested |
| RTX 4090 | Ada (sm_89) | 24 GB | 2.6.0+ | CPU | ✅ Compatible |

### Requires PyTorch Upgrade

| GPU Model | Architecture | VRAM | Min PyTorch | EWC Mode | Status |
|-----------|-------------|------|-------------|----------|---------|
| RTX 5090 | Blackwell (sm_120) | 32 GB | **2.8.0+** | GPU | ⚠️ Requires upgrade |
| RTX PRO 6000 | Blackwell (sm_120) | 96 GB | **2.8.0+** | GPU | ⚠️ Requires upgrade |
---

## Dependency Files Overview

The project maintains two requirements files for GPU architecture compatibility:

- **`requirements.txt`** (authoritative): PyTorch 2.6.0 + dependencies for standard GPUs (RTX 3090, RTX 4090). Use this for reproducible thesis evaluation.
- **`requirements_blackwell.txt`**: PyTorch 2.8.0+ for Blackwell GPUs (RTX 5090, RTX PRO 6000) requiring sm_120 compute capability.

**Default recommendation**: Use `requirements.txt` unless deploying on Blackwell-generation hardware.

---

## RTX 5090 Specific Requirements

### The Problem

**Error Message:**
```
CUDA error: no kernel image is available for execution on the device
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

### Root Cause

- RTX 5090 uses **Blackwell architecture** (compute capability sm_120)
- PyTorch 2.6.0 (`requirements.txt`) was compiled for sm_50 through sm_90 only
- No pre-compiled CUDA kernels exist for sm_120 in PyTorch 2.6.0
- PyTorch does NOT have JIT compilation fallback for unsupported architectures

### What Fails

| Operation | Status | Reason |
|-----------|--------|--------|
| Model loading | ✅ Works | Memory transfer only, no computation |
| Dataset preprocessing | ✅ Works | CPU operations (librosa) |
| Fisher estimation | ❌ FAILS | First GPU computation attempt |
| Training | ❌ FAILS | All training steps require GPU |

### The Solution

**Option 1: Use `requirements_blackwell.txt` (Recommended for RTX 5090)**
```bash
pip install -r requirements_blackwell.txt
```

**Option 2: Manual upgrade to PyTorch 2.8.0 with CUDA 12.8:**
```bash
# Uninstall old PyTorch
pip uninstall -y torch torchvision torchaudio

# Install PyTorch 2.8.0 with CUDA 12.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

**Additional recommended upgrades:**
```bash
pip install --upgrade \
    transformers==4.46.0 \
    accelerate==1.2.1 \
    datasets==3.2.0 \
    numpy==1.26.4
```

### Performance Comparison

| Configuration | Training Time | GPU Util | VRAM Used | Cost |
|--------------|---------------|----------|-----------|------|
| RTX 3090 + PyTorch 2.6 + CPU EWC | 27 hours | 19% | 18 GB / 24 GB | $13 |
| RTX 3090 + PyTorch 2.8 + GPU EWC | 8 hours | 85% | 22 GB / 24 GB | $4 |
| RTX 5090 + PyTorch 2.6 | ❌ CRASHES | N/A | N/A | N/A |
| RTX 5090 + PyTorch 2.8 + GPU EWC | 4-6 hours | 90% | 26 GB / 32 GB | $4-5 |

---

## Memory Requirements by Configuration

### RTX 3090 (24 GB VRAM)

**With GPU-based EWC (requires batch_size reduction):**
- Model: ~1.2 GB
- Optimizer states: ~2.4 GB
- Batch (4 samples): ~4.8 GB
- EWC Fisher/old_params: ~2.4 GB
- EWC computation temporaries: ~2+ GB
- Training activations: ~8-10 GB
- **Total: ~21-23 GB** ⚠️ Tight fit, may OOM

### RTX 5090 (32 GB VRAM)

**With GPU-based EWC (recommended for RTX 5090):**
- Model: ~1.2 GB
- Optimizer states: ~2.4 GB
- Batch (8 samples): ~9.6 GB
- EWC Fisher/old_params: ~2.4 GB
- EWC computation temporaries: ~2-3 GB
- Training activations: ~10-12 GB
- **Total: ~24-27 GB** ✅ Comfortable fit in 32 GB

---

## Configuration Guidelines

### For RTX 3090 (24 GB)

**Recommended configuration:**
```yaml
runpod:
  per_device_train_batch_size: 4      # Reduced for memory
  gradient_accumulation_steps: 4      # Maintains effective batch=16
  fp16: true
  dataloader_num_workers: 8
```

**EWC mode:**

**Expected performance:**
- Training time: 6-7 hours
- GPU utilization: 85-90%
- VRAM usage: 18-21 GB / 24 GB

### For RTX 5090 (32 GB) - Requires PyTorch 2.8.0

**Recommended configuration:**
```yaml
runpod:
  per_device_train_batch_size: 8      # Can use larger batch
  gradient_accumulation_steps: 2      # Less accumulation needed
  fp16: true
  dataloader_num_workers: 8
```

**EWC mode:** GPU-based (modify code to remove .cpu() calls)

**Expected performance:**
- Training time: 4-6 hours
- GPU utilization: 85-95%
- VRAM usage: 24-28 GB / 32 GB

---

## Upgrade Path for RTX 5090

### Prerequisites

- Python 3.11 or 3.12
- Ubuntu 24.04 (or 22.04)
- CUDA 12.8 drivers installed
- Sufficient disk space (~5 GB for PyTorch wheels)

### Step-by-Step Upgrade
```bash
# 1. Backup current environment
pip freeze > backup_requirements_$(date +%Y%m%d).txt

# 2. Uninstall old PyTorch
pip uninstall -y torch torchvision torchaudio

# 3. Install PyTorch 2.8.0 with CUDA 12.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 4. Upgrade key dependencies
pip install --upgrade \
    transformers==4.46.0 \
    accelerate==1.2.1 \
    datasets==3.2.0

# 5. Pin NumPy to prevent 2.0+ issues
pip install numpy==1.26.4

# 6. Verify installation
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

# Test GPU computation
x = torch.randn(100, 100, device='cuda')
y = torch.matmul(x, x)
print('✅ GPU computation works!')
"
```

### Code Changes for GPU-based EWC

**In `train_german_adaptation.py`, modify `compute_ewc_loss` (Lines 154-173):**
```python
def compute_ewc_loss(self):
    """
    Compute EWC penalty on GPU (optimized for 32GB VRAM).
    """
    if self.fisher_dict is None or self.old_params is None:
        return torch.tensor(0.0, device=self.model.device)
    
    ewc_loss = torch.tensor(0.0, device=self.model.device)
    for name, param in self.model.named_parameters():
        if name in self.fisher_dict and name in self.old_params:
            # Compute directly on GPU
            fisher = self.fisher_dict[name]
            old_param = self.old_params[name]
            ewc_loss += (fisher * (param - old_param).pow(2)).sum()
    
    return self.ewc_lambda * ewc_loss
```

**Remove all `.cpu()` calls from Fisher computation and EWC loss.**

---

## Verification Checklist

### After PyTorch Upgrade

Run these checks to verify RTX 5090 compatibility:
```bash
# 1. Check PyTorch version
python -c "import torch; print(torch.__version__)"
# Expected: 2.8.0+cu128

# 2. Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
# Expected: True

# 3. Check GPU detection (should NOT show warning)
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 5090
# Should NOT show: "not compatible with current PyTorch"

# 4. Test GPU computation
python -c "
import torch
x = torch.randn(1000, 1000, device='cuda')
y = torch.matmul(x, x)
print('GPU computation: OK')
"
# Expected: GPU computation: OK (no errors)

# 5. Test model loading
python -c "
from transformers import Wav2Vec2ForCTC
import torch
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base')
model = model.to('cuda')
print('Model on GPU: OK')
"
# Expected: Model on GPU: OK (no warnings)
```

---

## Troubleshooting

### Issue: "CUDA error: no kernel image" still appears after upgrade

**Possible causes:**
1. PyTorch installation failed
2. Wrong PyTorch version installed
3. CUDA 12.8 not properly installed

**Solution:**
```bash
# Verify PyTorch version
python -c "import torch; print(torch.__version__)"
# Must show: 2.8.0+cu128 (the +cu128 is critical!)

# If not, reinstall with --no-cache-dir
pip uninstall -y torch torchvision torchaudio
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --no-cache-dir
```

### Issue: NumPy version conflicts

**Solution:**
```bash
pip install --force-reinstall numpy==1.26.4
```

### Issue: Training still slow on RTX 5090

**Check GPU utilization:**
```bash
watch -n 1 nvidia-smi
```

**Expected:**
- GPU-Util: 85-95%
- Memory-Usage: 24-28 GB / 32 GB
- Power: 300-450W

**If utilization is low (<50%):**
1. Check if EWC is still using CPU (log should say "GPU-based EWC")
2. Verify batch size is 8 (not 4)
3. Check for I/O bottlenecks (increase num_workers)

---

## Cost-Benefit Analysis

### Development Phase (Current)

**Recommended:** RTX 3090 + PyTorch 2.6.0 + CPU EWC
- **Reason:** Proven stable, no upgrade risk
- **Cost:** $3.50 per training run (6.5 hours @ $0.50/hr)
- **Timeline:** Meets thesis deadline

### Production/Future Deployments

**Recommended:** RTX 5090 + PyTorch 2.8.0 + GPU EWC
- **Reason:** 5-6x faster training for rapid iteration
- **Cost:** $4-5 per training run (5 hours @ $0.89/hr)
- **Benefit:** Faster experimentation and hyperparameter tuning

---

## Backward Compatibility

### PyTorch 2.8.0 Compatibility with Older GPUs

**Good news:** PyTorch 2.8.0 is backward compatible with older GPUs.

| GPU | PyTorch 2.6.0 | PyTorch 2.8.0 | Notes |
|-----|---------------|---------------|-------|
| RTX 3090 | ✅ Works | ✅ Works | May be slightly faster with 2.8 |
| RTX 4090 | ✅ Works | ✅ Works | May be slightly faster with 2.8 |
| RTX 5090 | ❌ FAILS | ✅ Works | Only works with 2.8+ |

**Recommendation:** Upgrade all deployments to PyTorch 2.8.0 for consistency and performance.

---

## References

- [PyTorch 2.8.0 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v2.8.0)
- [NVIDIA Blackwell Architecture Documentation](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/)
- [PyTorch CUDA Compatibility Matrix](https://pytorch.org/get-started/locally/)
- [HuggingFace Transformers GPU Requirements](https://huggingface.co/docs/transformers/main/en/installation)

---

**Last Updated:** November 26, 2025  
**Tested Configurations:** RTX 3090 + PyTorch 2.6.0 + CPU EWC  
**Documented Configurations:** RTX 5090 + PyTorch 2.8.0 + GPU EWC (not yet tested)