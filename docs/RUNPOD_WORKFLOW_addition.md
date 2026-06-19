## German Common Voice Forgetting Benchmark — One-Time Setup

This section covers the one-time data preparation for the RQ2 catastrophic-forgetting
benchmark on German Common Voice 22.0. It assumes a dataset preparation step
has already been run, so `data/metadata/german/test.tsv` exists.

These steps are run **once per RunPod data volume**. The resulting seeded holdout file
persists on `/workspace` and is reused, unchanged, across every EWC λ condition — do not
regenerate it per run.

### Step 1 — Check Clip Availability

German CV 22.0 has ~1M clips; only the ones actually uploaded to the RunPod volume are
usable. Check coverage before doing anything else, since a partial upload silently
inflates `failed_samples` rather than erroring:

```bash
# On your laptop
source .env && ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST python3 - <<'EOF'
import os, pandas as pd
from pathlib import Path

clips_dir = Path('/workspace/data/raw/cv-corpus-22.0-2025-06-20/de/clips')
tsv_path  = Path('/workspace/data/metadata/german/test.tsv')

available = set(os.listdir(clips_dir))
df = pd.read_csv(tsv_path, sep='\t', low_memory=False, encoding='utf-8', quoting=3)

def clean(p):
    return str(p).strip().replace('\n','').replace('\r','').replace('\t','')

df['clean_path'] = df['path'].apply(clean)
found = df['clean_path'].apply(lambda p: p in available)
n_total = len(df)
n_found = found.sum()
print(f'test.tsv rows  : {n_total}')
print(f'clips on disk  : {len(available)}')
print(f'found          : {n_found} ({100*n_found/n_total:.1f}%)')
print(f'missing        : {n_total - n_found} ({100*(n_total-n_found)/n_total:.1f}%)')
EOF
```

- Reuses the same bulk `os.listdir()` + set-membership pattern as `compute_fisher.py`,
  avoiding per-file `Path.exists()` calls on the network volume (documented multi-minute
  freeze risk, see `KNOWN_ISSUES.md`).
- A high missing rate (we hit 55.8% on the first pass) means the upload was incomplete —
  proceed to Step 2 before generating the holdout sample.

### Step 2 — Rsync Missing Clips (If Needed)

Only the missing clips need to be transferred, not the full corpus:

```bash
# Generate the missing-files list on RunPod, pull it locally
source .env && ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST python3 - <<'EOF'
import os, pandas as pd
from pathlib import Path

clips_dir = Path('/workspace/data/raw/cv-corpus-22.0-2025-06-20/de/clips')
tsv_path  = Path('/workspace/data/metadata/german/test.tsv')

available = set(os.listdir(clips_dir))
df = pd.read_csv(tsv_path, sep='\t', low_memory=False, encoding='utf-8', quoting=3)

def clean(p):
    return str(p).strip().replace('\n','').replace('\r','').replace('\t','')

df['clean_path'] = df['path'].apply(clean)
missing = df.loc[~df['clean_path'].apply(lambda p: p in available), 'clean_path']
out = Path('/workspace/missing_test_clips.txt')
out.write_text('\n'.join(missing.tolist()))
print(f'Wrote {len(missing)} missing filenames to {out}')
EOF

scp -P $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST:/workspace/missing_test_clips.txt /tmp/missing_test_clips.txt

# Push only the missing clips from your local CV corpus.
# --no-owner --no-group avoids non-fatal chown warnings when rsyncing as a
# non-root local user into a root-owned RunPod filesystem (see upload_to_cloud.sh).
source .env && rsync -avz --progress \
    --no-owner --no-group \
    --files-from=/tmp/missing_test_clips.txt \
    -e "ssh -p $REMOTE_PORT" \
    data/raw/cv-corpus-22.0-2025-06-20/de/clips/ \
    $REMOTE_USER@$REMOTE_HOST:/workspace/data/raw/cv-corpus-22.0-2025-06-20/de/clips/
```

Re-run Step 1 afterwards to confirm 0 (or near-0) missing before proceeding.

### Step 3 — Generate the Seeded Holdout

The forgetting benchmark uses a fixed 1,000-sample holdout, drawn once with a fixed seed,
so every EWC λ condition is scored against an identical evaluation set:

```bash
# SSH into RunPod
cd /workspace/cp-swiss-german-asr

python3 scripts/sample_cv_german_holdout.py \
    --input /workspace/data/metadata/german/test.tsv \
    --output /workspace/data/metadata/german/test_1000_seed42.tsv \
    --n 1000 --seed 42
```

- `random_state=42`, deterministic — re-running with the same arguments reproduces the
  identical row selection.
- Adds a placeholder `accent` column (constant `"unknown"`) to the sampled TSV. This is
  required by `evaluator.py`'s schema check (`required_columns = {'path', 'sentence',
  'accent'}`, inherited from the FHNW per-canton dialect field) — German CV has no
  equivalent stratification axis, so `"unknown"` is a deliberate placeholder, not missing
  data. See the script's module docstring for the full rationale.
- Output: `/workspace/data/metadata/german/test_1000_seed42.tsv`. This file must persist
  across pod sessions — do not regenerate it for a new EWC condition. Regenerating with a
  different seed invalidates comparability across already-completed λ runs.

## See Also
- [CV German Forgetting Benchmark](CV_GERMAN_FORGETTING_BENCHMARK.md) — repeatable
  invocation, output conventions, and significance-test interpretation for the RQ2
  benchmark this section sets up.
