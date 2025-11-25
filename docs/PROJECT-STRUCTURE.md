# Project Structure

```text
cp-swiss-german-asr/
├── .dockerignore
├── .env                 # gitignored
├── .env.example
├── .env.example.local
├── .env.example.runpod
├── .gitignore
├── Dockerfile
├── LICENSE
├── PROJECT-STRUCTURE.md                          
├── README.md
├── docker-compose.yml
├── main.py
├── requirements.txt
├── .vscode/
│   └── settings.json
├── data/                # gitignored (the entire directory - large files)
│   ├── README.md
│   ├── metadata/
│   │   ├── test.tsv
│   │   ├── train.tsv
│   │   ├── val.tsv
│   │   ├── dutch/
│   │   │   ├── test.tsv
│   │   │   ├── train.tsv
│   │   │   └── val.tsv
│   │   └── german/
│   │       ├── test.tsv
│   │       ├── train.tsv
│   │       └── val.tsv
│   ├── processed/
│   └── raw/
│       ├── cv-corpus-22.0-2025-06-20/
│       │   └── de/
│       │       ├── clip_durations.tsv
│       │       ├── dev.tsv
│       │       ├── invalidated.tsv
│       │       ├── other.tsv
│       │       ├── reported.tsv
│       │       ├── test.tsv
│       │       ├── train.tsv
│       │       ├── unvalidated_sentences.tsv
│       │       ├── validated_sentences.tsv
│       │       ├── validated.tsv
│       │       └── clips/
│       ├── cv-corpus-23.0-2025-09-05/
│       │   └── nl/
│       │       ├── clip_durations.tsv
│       │       ├── dev.tsv
│       │       ├── invalidated.tsv
│       │       ├── other.tsv
│       │       ├── reported.tsv
│       │       ├── test.tsv
│       │       ├── train.tsv
│       │       ├── unvalidated_sentences.tsv
│       │       ├── validated_sentences.tsv
│       │       ├── validated.tsv
│       │       └── clips/
│       ├── fhnw-swiss-german-corpus/
│       │   ├── all.csv
│       │   ├── all.tsv
│       │   ├── example_submission.csv
│       │   ├── private.csv
│       │   ├── private.tsv
│       │   ├── public.csv
│       │   ├── public.tsv
│       │   ├── README.txt
│       │   └── clips/
│       └── sample_20utterances_sds200/
│           ├── export_20211220_sample_20utterances.tsv
│           ├── README_columns.txt
│           └── clips/
├── docs/
│   ├── DASHBOARD.md
│   ├── MIGRATION_GUIDE.md
│   ├── MODEL_SELECTION.md
│   └── RUNPOD_WORKFLOW.md
├── early-experiments/       # gitignored (the entire directory - large files)
│   ├── 20251104_152716/
│   │   ├── whisper-base_results.csv
│   │   └── whisper-base_results.json
│   ├── 20251104_153250/
│   │   ├── whisper-base_results.csv
│   │   ├── whisper-base_results.json
│   │   ├── whisper-small_results.csv
│   │   └── whisper-small_results.json
│   ├── 20251105_095956/
│   │   ├── wav2vec2-base_results.csv
│   │   └── wav2vec2-base_results.json
│   ├── 20251105_101429/
│   │   ├── wav2vec2-german_results.csv
│   │   └── wav2vec2-german_results.json
│   ├── 20251105_110723/
│   │   ├── wav2vec2-german_results.csv
│   │   ├── wav2vec2-german_results.json
│   │   ├── whisper-medium_results.csv
│   │   └── whisper-medium_results.json
│   ├── 20251106_182546/
│   │   ├── whisper-base_results.csv
│   │   ├── whisper-base_results.json
│   │   ├── whisper-tiny_results.csv
│   │   └── whisper-tiny_results.json
│   ├── 20251106_190948/
│   │   ├── whisper-tiny_results.csv
│   │   └── whisper-tiny_results.json
│   ├── 20251106_205056/
│   │   ├── whisper-medium_results.csv
│   │   └── whisper-medium_results.json
│   ├── 20251106_212449/
│   │   ├── whisper-small_results.csv
│   │   └── whisper-small_results.json
│   ├── 20251108_211806/
│   │   ├── whisper-base_results.csv
│   │   └── whisper-base_results.json
│   ├── 20251111_214302/
│   │   ├── mms-1b-all_results.csv
│   │   └── mms-1b-all_results.json
│   ├── 20251111_215303/
│   │   ├── mms-1b-l1107_results.csv
│   │   └── mms-1b-l1107_results.json
│   ├── 20251111_222101/
│   │   ├── whisper-large-v3-turbo_results.csv
│   │   └── whisper-large-v3-turbo_results.json
│   ├── 20251112_090508/
│   │   ├── whisper-large-v3-turbo_results.csv
│   │   └── whisper-large-v3-turbo_results.json
│   ├── 20251112_093435/
│   │   ├── whisper-medium_results.csv
│   │   └── whisper-medium_results.json
│   ├── 20251112_100255/
│   ├── 20251112_100358/
│   ├── 20251112_100456/
│   ├── 20251112_101241/
│   ├── 20251112_102158/
│   ├── 20251112_104414/
│   ├── 20251112_110035/
│   ├── 20251112_110226/
│   ├── 20251112_111618/
│   │   ├── whisper-large_results.csv
│   │   └── whisper-large_results.json
│   ├── 20251112_114058/
│   │   ├── whisper-large_results.csv
│   │   └── whisper-large_results.json
│   ├── 20251112_133439/
│   │   ├── whisper-large-v2_results.csv
│   │   ├── whisper-large-v2_results.json
│   │   ├── whisper-large-v3_results.csv
│   │   └── whisper-large-v3_results.json
│   ├── 20251113_111613/
│   ├── 20251113_113529/
│   ├── 20251113_113810/
│   ├── 20251113_113934/
│   ├── 20251113_114323/
│   ├── 20251113_115131/
│   │   └── ...
│   ├── 20251113_120550/
│   │   └── ...
│   ├── 20251113_124825/
│   ├── 20251113_154121/
│   ├── 20251113_160054/
│   │   └── ...
│   ├── 20251113_164317/
│   ├── 20251113_170834/
│   ├── 20251113_171334/
│   ├── 20251113_171835/
│   ├── 20251113_172257/
│   ├── 20251113_180429/
│   └── 20251113_181410/
├── logs/
├── personal-notes/     # gitignored (the entire directory - personal notes)
│   ├── week01_days1-2_docker_local_setup.md
│   ├── week01_days3-4_data_pipeline.md
│   ├── week01_days5-7_minimal_fastapi_and_analysis.md
│   ├── week02_days1-3_evaluation_framework.md
│   ├── week02_days4-7_streamlit_dashboard.md
│   ├── week03_days1-2_model_selection.md
│   ├── week03_days3-4_cloud_gpu_setup.md
│   └── week04_days1-2_dutch_pre-training.md
├── results/
│   └── metrics/
│       ├── 20251113_210648/
│       │   ├── whisper-large-v3-turbo_results.csv
│       │   ├── whisper-large-v3-turbo_results.json
│       │   ├── whisper-small_results.csv
│       │   └── whisper-small_results.json
│       ├── 20251113_214357/
│       │   ├── wav2vec2-german_results.csv
│       │   ├── wav2vec2-german_results.json
│       │   ├── wav2vec2-multi-56_results.csv
│       │   └── wav2vec2-multi-56_results.json
│       ├── 20251113_215305/
│       │   ├── mms-1b-all_results.csv
│       │   ├── mms-1b-all_results.json
│       │   ├── mms-1b-l1107_results.csv
│       │   └── mms-1b-l1107_results.json
│       ├── 20251113_221929/
│       │   ├── whisper-base_results.csv
│       │   ├── whisper-base_results.json
│       │   ├── whisper-tiny_results.csv
│       │   └── whisper-tiny_results.json
│       ├── 20251114_072930/
│       │   ├── whisper-large_results.csv
│       │   ├── whisper-large_results.json
│       │   ├── whisper-large-v2_results.csv
│       │   ├── whisper-large-v2_results.json
│       │   ├── whisper-large-v3_results.csv
│       │   └── whisper-large-v3_results.json
│       └── 20251114_113817/
│           ├── whisper-medium_results.csv
│           └── whisper-medium_results.json
├── scripts/
│   ├── evaluate_models.py
│   ├── inspect_results.py
│   ├── prepare_common_voice.py
│   ├── prepare_scripts.py
│   ├── train_dutch_pretrain.py
│   ├── train_german_adaptation.py
│   ├── train_on_cloud.sh 
│   └── upload_to_cloud.sh
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── __pycache__/
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── endpoints.py
│   │   ├── models.py
│   │   └── __pycache__/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── splitter.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   └── __pycache__/
│   ├── frontend/
│   │   ├── app.py
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── data_table.py
│   │   │   ├── dialect_breakdown.py
│   │   │   ├── model_comparison.py
│   │   │   ├── sidebar.py
│   │   │   ├── statistics_panel.py
│   │   │   └── __pycache__/
│   │   └── utils/
│   │       ├── __inip__.py
│   │       ├── data_loader.py
│   │       └── __pycache__/
│   ├── models/
│   │   ├── mms_model.py
│   │   ├── wav2vec2_model.py
│   │   └── __pycache__/
│   └── utils/
│       ├── __init__.py
│       ├── audio_utils.py
│       ├── file_utils.py
│       └── logging_config.py
└── tests/
    ├── __init__.py
    ├── test_evaluation.py
    └── __pycache__/
```