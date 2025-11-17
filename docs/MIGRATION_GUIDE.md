# Configuration Migration Guide

## Summary of Changes

The configuration system has been improved to support multiple environments explicitly while maintaining backwards compatibility.

### What Changed?

**Before:**
- Environment auto-detected based on filesystem paths
- Implicit logic: "if /workspace exists and /app doesn't, assume RunPod"
- Brittle and hard to debug

**After:**
- Explicit `ENVIRONMENT` variable (recommended)
- Auto-detection still works (deprecated, with warnings)
- Clear error messages and validation
- Better logging for reproducibility

### Backwards Compatibility

**✅ Your existing code will work without any changes!**

The auto-detection logic is still present for backwards compatibility. However, you'll see deprecation warnings encouraging you to set `ENVIRONMENT` explicitly.

## Migration Steps

### Option A: Quick Migration (5 minutes)

1. **Update your local `.env`:**
```bash
   # Add this line at the top
   echo "ENVIRONMENT=local" >> .env
```

2. **Update `scripts/train_on_cloud.sh`:**
   - Already done in the provided file
   - Sets `export ENVIRONMENT=runpod` before running Python scripts

3. **Test:**
```bash
   # On your laptop
   python -c "from src import config; print(config.ENVIRONMENT)"
   # Should print: local
```

### Option B: Full Migration (15 minutes)

1. **Create clean environment files:**
```bash
   # Backup current .env
   cp .env .env.backup
   
   # Copy appropriate template
   cp .env.example.local .env
   
   # Edit .env with your values
   nano .env
```

2. **Set up RunPod environment:**
```bash
   # SSH into RunPod
   ssh root@your-pod-id.runpod.io
   
   # Navigate to project
   cd /workspace/cp-swiss-german-asr
   
   # Copy template
   cp .env.example.runpod .env
   
   # Edit with your HF token
   nano .env
```

3. **Test on both environments:**
```bash
   # Local
   python -m src.config  # Should show local environment
   
   # RunPod (via SSH)
   ssh root@your-pod-id.runpod.io "cd /workspace/cp-swiss-german-asr && python -m src.config"
   # Should show runpod environment
```

## Verification

### Quick Test Script

Create `scripts/test_config.py`:
```python
#!/usr/bin/env python3
"""Test configuration is working correctly."""

from src import config

print("\n" + "=" * 70)
print("CONFIGURATION TEST")
print("=" * 70)
print(f"Environment: {config.ENVIRONMENT}")
print(f"Auto-detected: {config.IS_AUTO_DETECTED}")
print(f"Project Root: {config.PROJECT_ROOT}")
print(f"Data Dir: {config.DATA_DIR}")
print(f"Models Dir: {config.MODELS_DIR}")
print(f"Results Dir: {config.RESULTS_DIR}")
print("=" * 70)

# Check if paths exist
print("\nPath Validation:")
for name, path in [
    ("PROJECT_ROOT", config.PROJECT_ROOT),
    ("DATA_DIR", config.DATA_DIR),
]:
    exists = "✅" if path.exists() else "❌"
    print(f"  {exists} {name}: {path}")

print("\n✅ Configuration test complete!\n")
```

Run it:
```bash
# Local
python scripts/test_config.py

# RunPod
ssh root@your-pod-id.runpod.io "cd /workspace/cp-swiss-german-asr && python scripts/test_config.py"
```

## Troubleshooting

### Issue: Deprecation warnings

**Symptom:**
```
⚠️  DEPRECATION WARNING: Auto-detecting RunPod environment.
```

**Solution:**
Add `ENVIRONMENT=runpod` (or `local`) to your environment:
```bash
export ENVIRONMENT=runpod  # In scripts or .env
```

### Issue: Wrong paths being used

**Symptom:**
```
FileNotFoundError: /app/data not found
```

**Solution:**
1. Check your `ENVIRONMENT` setting:
```python
   from src import config
   print(config.ENVIRONMENT)
```

2. Verify your `.env` file has the right `ENVIRONMENT` value

3. Check that paths are appropriate for your environment:
```python
   from src import config
   config.log_configuration()  # Shows all paths
```

### Issue: Need to override specific paths

**Solution:**
Set explicit environment variables in `.env`:
```dotenv
ENVIRONMENT=local
DATA_DIR=/custom/path/to/data
```

## Rollback

If you need to rollback to the old configuration:
```bash
# Restore old config.py from git
git checkout HEAD -- src/config.py

# Restore old .env
cp .env.backup .env
```

## Questions?

- Check the inline documentation in `src/config.py`
- Run `python -m src.config` to see current configuration
- Check logs for configuration details at startup