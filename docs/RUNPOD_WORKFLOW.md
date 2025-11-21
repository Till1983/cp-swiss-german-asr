# RunPod Training Workflow

## One-Time Setup
1. Create RunPod account
2. Create 100GB network volume
3. Deploy RTX 3090 Secure Cloud pod
4. Add SSH key to RunPod settings
5. Update `.env` with pod connection info

## Data Upload (First Time Only)
```bash
# On your laptop
./upload_to_cloud.sh
```

## Prepare Datasets (First Time Only)
```bash
# SSH into RunPod
ssh root@your-pod-id.runpod.io

cd /workspace/cp-swiss-german-asr

# Prepare Dutch metadata
python scripts/prepare_common_voice.py \
    --cv-root /workspace/data/raw/cv-corpus-23.0-2025-09-05 \
    --language-code nl \
    --output-dir /workspace/data/metadata/dutch

# Prepare German metadata
python scripts/prepare_common_voice.py \
    --cv-root /workspace/data/raw/cv-corpus-22.0-2025-06-20 \
    --language-code de \
    --output-dir /workspace/data/metadata/german

# Prepare Swiss German splits
python scripts/prepare_scripts.py \
    /workspace/data/raw/fhnw-swiss-german-corpus/public.tsv \
    /workspace/data/metadata \
    /workspace/data/raw/fhnw-swiss-german-corpus
```

## Run Training
```bash
# On your laptop (triggers remote training)
./train_on_cloud.sh
```

## Cost Management
- **While working:** Keep pod running (~$0.47/hr)
- **Between sessions:** Terminate pod (data persists on volume)
- **Storage cost:** ~$0.01/hr for network volume (always running)