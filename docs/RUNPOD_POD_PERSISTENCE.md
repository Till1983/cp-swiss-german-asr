# RunPod Pod Persistence and Package Management

## Understanding RunPod Storage Architecture

### Simple Explanation
Your RunPod setup is like a temporary laptop connected to an external hard drive:
- **The Pod (Container)** = Temporary laptop that gets wiped when you terminate it
- **Network Volume** = External hard drive that persists your data forever
- **Packages** = Software installed on the temporary laptop (disappears with pod)

### Technical Explanation
RunPod uses a two-tier storage system:
1. **Ephemeral Container Storage**: The pod's filesystem where packages, system files, and temporary data live. This is destroyed on termination.
2. **Persistent Network Volume**: Mounted at `/workspace`, this storage persists independently of pod lifecycle and survives termination.

## What Persists and What Doesn't

### ✅ PERSISTS (Network Volume - `/workspace`)
- Your code: `/workspace/cp-swiss-german-asr/`
- Your datasets: `/workspace/data/`
- Your models: `/workspace/models/`
- Your results: `/workspace/results/`
- Any files you explicitly put in `/workspace`

### ❌ DOES NOT PERSIST (Container)
- **Python packages** (installed via `pip install`)
- **System packages** (installed via `apt install`)
- **Environment modifications**
- **Files outside `/workspace`**
- The Docker image base system

## Why You Must Reinstall Packages Every Time

### The Reality
**Every new pod = completely fresh container.** 

When you:
1. Terminate a pod → Container destroyed (all packages gone)
2. Create a new pod → Fresh container from base Docker image
3. Attach same network volume → Your data reappears
4. Run a script → ❌ `ModuleNotFoundError` (packages not installed)
5. Run `pip install -r requirements.txt` → ✅ Packages installed
6. Run script again → ✅ Works

### Your Installation Log Analysis
Looking at your terminal output:
```python
Attempting uninstall: torch
  Found existing installation: torch 2.8.0+cu128
  Uninstalling torch-2.8.0+cu128:
    Successfully uninstalled torch-2.8.0+cu128
```

This shows:
- ✅ PyTorch 2.8.0 was in the base RunPod Docker image
- ✅ Your `requirements.txt` downgraded it to 2.6.0
- ✅ Packages install to **container filesystem** (not `/workspace`)
- ✅ Next pod will have 2.8.0 again (from base image)

## Dependency Hell: Why You Worried For Nothing

### The Good News: Dependency Hell is IMPOSSIBLE

**Each pod is isolated and disposable:**
```
Pod 1 (Today):
├── Try PyTorch 2.8.0
├── Install packages from requirements_blackwell.txt
└── Training fails? No problem!

Pod 2 (Tomorrow):
├── Completely fresh container
├── Install packages from requirements.txt (PyTorch 2.6.0)
└── Zero conflicts, clean slate
```

### Why This Works
- **Terminating = Formatting the computer**: All package conflicts disappear
- **Your data is safe**: Network volume is unaffected
- **Complete reproducibility**: Same base image + requirements file = identical environment

## Community Cloud vs Secure Cloud

### Community Cloud
- **Peer-to-peer GPU providers** (vetted third parties)
- **20-30% cheaper** than Secure Cloud
- **Can be stopped**: Pod pauses, GPU reserved, data in `/workspace` persists
- **Can be terminated**: Pod destroyed, only network volume data persists
- **Use case**: Cost-sensitive workloads, experimentation

### Secure Cloud (Your Setup)
- **T3/T4 data centres** with enterprise-grade reliability
- **Cannot be stopped** - only terminated
- **Single-tenant**: Your GPU is dedicated, not shared
- **Better for**: Thesis work, reproducibility, sensitive data
- **When terminated**: Container destroyed, `/workspace` persists

### Key Difference for Package Management
**Both work the same way**: Packages are always in the ephemeral container, not persistent storage.

## Standard Workflow (What You're Already Doing)

### Every New Pod Session
```bash
# 1. Create new Secure Cloud pod with network volume attached
# 2. SSH into pod
ssh root@your-pod-id.runpod.io

# 3. Navigate to project
cd /workspace/cp-swiss-german-asr

# 4. ALWAYS install packages first
pip install -r requirements.txt --break-system-packages

# 5. Now you can run scripts
python scripts/train_german_adaptation.py
```

### Why `--break-system-packages`?
RunPod's Python is system-managed. The flag allows pip installations without virtual environments. This is safe because:
- The container is disposable
- You're not modifying a long-lived system
- Next pod will be fresh anyway

## Best Practices for Package Installation

### Option 1: Standard Installation (Current Approach)
```bash
pip install -r requirements.txt --break-system-packages
```
- **Time**: ~5 minutes per pod
- **Reliability**: ✅ High (well-tested)
- **Reproducibility**: ✅ Excellent (versioned requirements)
- **Thesis-acceptable**: ✅ Yes (standard practice)

### Option 2: No-Cache Installation (Extra Security)
```bash
pip install -r requirements.txt --no-cache-dir --break-system-packages
```
- **Benefit**: Forces fresh downloads, avoids corrupted cache
- **Downside**: Slightly slower (~1-2 minutes extra)
- **When to use**: If you suspect caching issues

### Option 3: Custom Docker Image (Advanced)
```bash
# Build locally or on RunPod
docker build -t your-username/project:pytorch26 -f Dockerfile.ampere .
docker push your-username/project:pytorch26

# Deploy on RunPod using custom image
```
- **Time**: 1-2 hours initial setup
- **Benefit**: Zero package installation needed per pod
- **Downside**: Requires Docker expertise
- **Worth it**: Only if creating/destroying pods constantly (>10x/day)

## Experimenting with PyTorch 2.8.0 (Blackwell GPUs)

### Your Worry: "Will this create dependency conflicts?"
**Answer: NO. Impossible to create conflicts.**

### Safe Experimentation Workflow
```bash
# Pod 1: Test PyTorch 2.8.0 on RTX 5090
pip install -r requirements_blackwell.txt --break-system-packages
python scripts/train_german_adaptation.py

# Training fails due to PyTorch incompatibility?
# → Terminate pod (takes 10 seconds)

# Pod 2: Back to PyTorch 2.6.0 on RTX 3090
pip install -r requirements.txt --break-system-packages
python scripts/train_german_adaptation.py
# → Works perfectly, zero conflicts
```

### No-Cache Installation for Blackwell Experiment
```bash
# Extra cautious approach
pip install -r requirements_blackwell.txt --no-cache-dir --break-system-packages
```

**Why no-cache?**
- Ensures fresh package downloads
- Eliminates risk of version conflicts from pip's cache
- Adds ~1-2 minutes but guarantees clean environment
- **Recommended for first Blackwell test**

## Cost Implications

### Secure Cloud Costs
- **Running pod**: ~$0.47/hr (RTX 3090)
- **Network volume**: ~$0.01/hr (always charged, even when no pod)
- **Package installation time**: ~5 minutes = ~$0.04 per pod creation

### Optimisation Strategies
1. **Keep pod running during active work** (~8 hours = $3.76)
2. **Terminate between sessions** (data persists, no pod cost)
3. **Package installation is negligible cost** (don't optimise this)

## Thesis Documentation

### How to Document This in Your Thesis

#### Computational Environment Section
```markdown
## Computational Environment

All training was conducted on RunPod Secure Cloud infrastructure with persistent 
network volumes for data storage. Each pod deployment used ephemeral container 
storage with packages installed from versioned requirements files at pod creation.

**GPU Configuration - PyTorch 2.6.0 (Ampere Architecture)**
- Base Image: `runpod/pytorch:2.6.0-py3.11-cuda12.1.1-cudnn-devel-ubuntu22.04`
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- Dependencies: Installed via `requirements.txt` (see Appendix A)
- Setup: `pip install -r requirements.txt --break-system-packages`

**Storage Architecture**
- Persistent Storage: 100GB network volume mounted at `/workspace`
- Ephemeral Storage: 50GB container disk for system packages
- Data Persistence: Datasets, models, and results stored on network volume
- Package Management: Installed per pod session from version-locked requirements

**Reproducibility**
The computational environment is fully reproducible:
1. Base Docker image is versioned and publicly available
2. All Python packages are version-pinned in `requirements.txt`
3. Pod configuration is documented in project repository
4. Same setup can be replicated by anyone with RunPod account
```

### Why This Is Academically Sound
- ✅ **Transparent**: Clearly states ephemeral vs persistent storage
- ✅ **Reproducible**: Anyone can recreate exact environment
- ✅ **Standard practice**: Many ML papers use similar cloud GPU setups
- ✅ **Professional**: Shows understanding of infrastructure

## Common Questions

### Q: "Is it normal to reinstall packages every pod?"
**A: YES.** This is standard practice for cloud GPU platforms. Packages are ephemeral by design.

### Q: "Will I lose my training progress if pod terminates?"
**A: NO.** Model checkpoints are saved to `/workspace/models/` (persistent). You can resume from checkpoint in new pod.

### Q: "Can I pre-install packages to avoid reinstallation?"
**A: YES, with custom Docker image.** But for thesis work with limited pod recreations, not worth the complexity.

### Q: "Should I use virtual environments?"
**A: NO.** Unnecessary in disposable containers. Use `--break-system-packages` instead.

### Q: "What if I accidentally modify system files?"
**A: No problem.** Next pod will be completely fresh. Container isolation protects you.

## Troubleshooting

### Issue: ModuleNotFoundError after pod restart
**Cause**: Packages not installed in new pod
**Solution**: Run `pip install -r requirements.txt --break-system-packages`

### Issue: Different PyTorch version than expected
**Cause**: Base image has pre-installed PyTorch
**Solution**: Your requirements.txt will override it during installation

### Issue: Package installation fails
**Cause**: Network issues or disk space
**Solution**: 
```bash
# Check disk space
df -h

# Try no-cache installation
pip install -r requirements.txt --no-cache-dir --break-system-packages
```

### Issue: Want to verify package versions
**Solution**:
```bash
pip list | grep torch
pip list | grep transformers
```

## Key Takeaways

1. **Pods are disposable** - This is a feature, not a bug
2. **Network volume is your safety net** - Data always persists
3. **Package reinstallation is normal** - Budget 5 minutes per pod
4. **Dependency hell is impossible** - Each pod is isolated
5. **No-cache installation adds safety** - Use for critical experiments
6. **Current workflow is correct** - You're doing it right
7. **Thesis documentation is straightforward** - Standard practice in ML research

## References

- RunPod Documentation: [Storage Options](https://docs.runpod.io/pods/storage/types)
- RunPod Documentation: [Managing Pods](https://docs.runpod.io/pods/manage-pods)
- Docker Documentation: [Container Lifecycle](https://docs.docker.com/get-started/overview/)

---

**Last Updated**: 2025-11-27 (Week 4 - Infrastructure Understanding Phase)
**Author**: Till Ermold
**Context**: Understanding RunPod persistence for PyTorch 2.8.0 experiment planning
EOF
cat /tmp/runpod_persistence_guide.md
Output

# RunPod Pod Persistence and Package Management

## Understanding RunPod Storage Architecture

### Simple Explanation
Your RunPod setup is like a temporary laptop connected to an external hard drive:
- **The Pod (Container)** = Temporary laptop that gets wiped when you terminate it
- **Network Volume** = External hard drive that persists your data forever
- **Packages** = Software installed on the temporary laptop (disappears with pod)

### Technical Explanation
RunPod uses a two-tier storage system:
1. **Ephemeral Container Storage**: The pod's filesystem where packages, system files, and temporary data live. This is destroyed on termination.
2. **Persistent Network Volume**: Mounted at `/workspace`, this storage persists independently of pod lifecycle and survives termination.

## What Persists and What Doesn't

### ✅ PERSISTS (Network Volume - `/workspace`)
- Your code: `/workspace/cp-swiss-german-asr/`
- Your datasets: `/workspace/data/`
- Your models: `/workspace/models/`
- Your results: `/workspace/results/`
- Any files you explicitly put in `/workspace`

### ❌ DOES NOT PERSIST (Container)
- **Python packages** (installed via `pip install`)
- **System packages** (installed via `apt install`)
- **Environment modifications**
- **Files outside `/workspace`**
- The Docker image base system

## Why You Must Reinstall Packages Every Time

### The Reality
**Every new pod = completely fresh container.** 

When you:
1. Terminate a pod → Container destroyed (all packages gone)
2. Create a new pod → Fresh container from base Docker image
3. Attach same network volume → Your data reappears
4. Run a script → ❌ `ModuleNotFoundError` (packages not installed)
5. Run `pip install -r requirements.txt` → ✅ Packages installed
6. Run script again → ✅ Works

### Your Installation Log Analysis
Looking at your terminal output:
```python
Attempting uninstall: torch
  Found existing installation: torch 2.8.0+cu128
  Uninstalling torch-2.8.0+cu128:
    Successfully uninstalled torch-2.8.0+cu128
```

This shows:
- ✅ PyTorch 2.8.0 was in the base RunPod Docker image
- ✅ Your `requirements.txt` downgraded it to 2.6.0
- ✅ Packages install to **container filesystem** (not `/workspace`)
- ✅ Next pod will have 2.8.0 again (from base image)

## Dependency Hell: Why You Worried For Nothing

### The Good News: Dependency Hell is IMPOSSIBLE

**Each pod is isolated and disposable:**
```
Pod 1 (Today):
├── Try PyTorch 2.8.0
├── Install packages from requirements_blackwell.txt
└── Training fails? No problem!

Pod 2 (Tomorrow):
├── Completely fresh container
├── Install packages from requirements.txt (PyTorch 2.6.0)
└── Zero conflicts, clean slate
```

### Why This Works
- **Terminating = Formatting the computer**: All package conflicts disappear
- **Your data is safe**: Network volume is unaffected
- **Complete reproducibility**: Same base image + requirements file = identical environment

## Community Cloud vs Secure Cloud

### Community Cloud
- **Peer-to-peer GPU providers** (vetted third parties)
- **20-30% cheaper** than Secure Cloud
- **Can be stopped**: Pod pauses, GPU reserved, data in `/workspace` persists
- **Can be terminated**: Pod destroyed, only network volume data persists
- **Use case**: Cost-sensitive workloads, experimentation

### Secure Cloud (Your Setup)
- **T3/T4 data centres** with enterprise-grade reliability
- **Cannot be stopped** - only terminated
- **Single-tenant**: Your GPU is dedicated, not shared
- **Better for**: Thesis work, reproducibility, sensitive data
- **When terminated**: Container destroyed, `/workspace` persists

### Key Difference for Package Management
**Both work the same way**: Packages are always in the ephemeral container, not persistent storage.

## Standard Workflow (What You're Already Doing)

### Every New Pod Session
```bash
# 1. Create new Secure Cloud pod with network volume attached
# 2. SSH into pod
ssh root@your-pod-id.runpod.io

# 3. Navigate to project
cd /workspace/cp-swiss-german-asr

# 4. ALWAYS install packages first
pip install -r requirements.txt --break-system-packages

# 5. Now you can run scripts
python scripts/train_german_adaptation.py
```

### Why `--break-system-packages`?
RunPod's Python is system-managed. The flag allows pip installations without virtual environments. This is safe because:
- The container is disposable
- You're not modifying a long-lived system
- Next pod will be fresh anyway

## Best Practices for Package Installation

### Option 1: Standard Installation (Current Approach)
```bash
pip install -r requirements.txt --break-system-packages
```
- **Time**: ~5 minutes per pod
- **Reliability**: ✅ High (well-tested)
- **Reproducibility**: ✅ Excellent (versioned requirements)
- **Thesis-acceptable**: ✅ Yes (standard practice)

### Option 2: No-Cache Installation (Extra Security)
```bash
pip install -r requirements.txt --no-cache-dir --break-system-packages
```
- **Benefit**: Forces fresh downloads, avoids corrupted cache
- **Downside**: Slightly slower (~1-2 minutes extra)
- **When to use**: If you suspect caching issues

### Option 3: Custom Docker Image (Advanced)
```bash
# Build locally or on RunPod
docker build -t your-username/project:pytorch26 -f Dockerfile.ampere .
docker push your-username/project:pytorch26

# Deploy on RunPod using custom image
```
- **Time**: 1-2 hours initial setup
- **Benefit**: Zero package installation needed per pod
- **Downside**: Requires Docker expertise
- **Worth it**: Only if creating/destroying pods constantly (>10x/day)

## Experimenting with PyTorch 2.8.0 (Blackwell GPUs)

### Your Worry: "Will this create dependency conflicts?"
**Answer: NO. Impossible to create conflicts.**

### Safe Experimentation Workflow
```bash
# Pod 1: Test PyTorch 2.8.0 on RTX 5090
pip install -r requirements_blackwell.txt --break-system-packages
python scripts/train_german_adaptation.py

# Training fails due to PyTorch incompatibility?
# → Terminate pod (takes 10 seconds)

# Pod 2: Back to PyTorch 2.6.0 on RTX 3090
pip install -r requirements.txt --break-system-packages
python scripts/train_german_adaptation.py
# → Works perfectly, zero conflicts
```

### No-Cache Installation for Blackwell Experiment
```bash
# Extra cautious approach
pip install -r requirements_blackwell.txt --no-cache-dir --break-system-packages
```

**Why no-cache?**
- Ensures fresh package downloads
- Eliminates risk of version conflicts from pip's cache
- Adds ~1-2 minutes but guarantees clean environment
- **Recommended for first Blackwell test**

## Cost Implications

### Secure Cloud Costs
- **Running pod**: ~$0.47/hr (RTX 3090)
- **Network volume**: ~$0.01/hr (always charged, even when no pod)
- **Package installation time**: ~5 minutes = ~$0.04 per pod creation

### Optimisation Strategies
1. **Keep pod running during active work** (~8 hours = $3.76)
2. **Terminate between sessions** (data persists, no pod cost)
3. **Package installation is negligible cost** (don't optimise this)

## Thesis Documentation

### How to Document This in Your Thesis

#### Computational Environment Section
```markdown
## Computational Environment

All training was conducted on RunPod Secure Cloud infrastructure with persistent 
network volumes for data storage. Each pod deployment used ephemeral container 
storage with packages installed from versioned requirements files at pod creation.

**GPU Configuration - PyTorch 2.6.0 (Ampere Architecture)**
- Base Image: `runpod/pytorch:2.6.0-py3.11-cuda12.1.1-cudnn-devel-ubuntu22.04`
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- Dependencies: Installed via `requirements.txt` (see Appendix A)
- Setup: `pip install -r requirements.txt --break-system-packages`

**Storage Architecture**
- Persistent Storage: 100GB network volume mounted at `/workspace`
- Ephemeral Storage: 50GB container disk for system packages
- Data Persistence: Datasets, models, and results stored on network volume
- Package Management: Installed per pod session from version-locked requirements

**Reproducibility**
The computational environment is fully reproducible:
1. Base Docker image is versioned and publicly available
2. All Python packages are version-pinned in `requirements.txt`
3. Pod configuration is documented in project repository
4. Same setup can be replicated by anyone with RunPod account
```

### Why This Is Academically Sound
- ✅ **Transparent**: Clearly states ephemeral vs persistent storage
- ✅ **Reproducible**: Anyone can recreate exact environment
- ✅ **Standard practice**: Many ML papers use similar cloud GPU setups
- ✅ **Professional**: Shows understanding of infrastructure

## Common Questions

### Q: "Is it normal to reinstall packages every pod?"
**A: YES.** This is standard practice for cloud GPU platforms. Packages are ephemeral by design.

### Q: "Will I lose my training progress if pod terminates?"
**A: NO.** Model checkpoints are saved to `/workspace/models/` (persistent). You can resume from checkpoint in new pod.

### Q: "Can I pre-install packages to avoid reinstallation?"
**A: YES, with custom Docker image.** But for thesis work with limited pod recreations, not worth the complexity.

### Q: "Should I use virtual environments?"
**A: NO.** Unnecessary in disposable containers. Use `--break-system-packages` instead.

### Q: "What if I accidentally modify system files?"
**A: No problem.** Next pod will be completely fresh. Container isolation protects you.

## Troubleshooting

### Issue: ModuleNotFoundError after pod restart
**Cause**: Packages not installed in new pod
**Solution**: Run `pip install -r requirements.txt --break-system-packages`

### Issue: Different PyTorch version than expected
**Cause**: Base image has pre-installed PyTorch
**Solution**: Your requirements.txt will override it during installation

### Issue: Package installation fails
**Cause**: Network issues or disk space
**Solution**: 
```bash
# Check disk space
df -h

# Try no-cache installation
pip install -r requirements.txt --no-cache-dir
# or
pip install --no-cache-dir -r requirements.txt
# both are functionally equivalent
```

### Issue: Want to verify package versions
**Solution**:
```bash
pip list | grep torch
pip list | grep transformers
```

## Key Takeaways

1. **Pods are disposable** - This is a feature, not a bug
2. **Network volume is your safety net** - Data always persists
3. **Package reinstallation is normal** - Budget 5 minutes per pod
4. **Dependency hell is impossible** - Each pod is isolated
5. **No-cache installation adds safety** - Use for critical experiments
6. **Current workflow is correct** - You're doing it right
7. **Thesis documentation is straightforward** - Standard practice in ML research

## References

- RunPod Documentation: [Storage Options](https://docs.runpod.io/pods/storage/types)
- RunPod Documentation: [Managing Pods](https://docs.runpod.io/pods/manage-pods)
- Docker Documentation: [Container Lifecycle](https://docs.docker.com/get-started/overview/)

---

**Last Updated**: 2025-11-27 (Week 4 - Infrastructure Understanding Phase)
**Author**: Till Ermold
**Context**: Understanding RunPod persistence for PyTorch 2.8.0 experiment planning