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
├── README.md
├── configs/
│   └── training/
│       ├── dutch_pretrain.yml
│       ├── german_adaptation.yml
│       └── wav2vec2_config.yml
├── docker-compose.yml
├── main.py
├── PROJECT_DISCUSSION.md
├── pytest.ini
├── requirements.txt
├── requirements_blackwell.txt
├── requirements_local.txt
├── .vscode/
│   └── settings.json
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
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
│   ├── metadata.backup_public_split/
│   │   ├── test.tsv
│   │   ├── train.tsv
│   │   └── val.tsv
│   ├── processed/
│   └── raw/
│       ├── cv-corpus-22.0-2025-06-20/
│       │   └── de/
│       │       ├── clip_durations.tsv
│       │       ├── dev.tsv
│       │       ├── invalidated.tsv
│       │       ├── other.tsv
│       │       ├── README-cv-de.txt
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
│       │       ├── README-cv-nl.txt
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
│   ├── COVERAGE_IMPROVEMENTS.md
│   ├── DASHBOARD.md
│   ├── ERROR_ANALYSIS_METHODOLOGY.md
│   ├── GPU_COMPATIBILITY.md
│   ├── HYPERPARAMETER_TUNING.md
│   ├── KNOWN_ISSUES.md
│   ├── MIGRATION_GUIDE.md
│   ├── MODEL_SELECTION.md
│   ├── PROJECT-STRUCTURE.md
│   ├── RUNPOD_POD_PERSISTENCE.md
│   ├── RUNPOD_WORKFLOW.md
│   ├── TEST_IMPROVEMENTS_SUMMARY.md
│   ├── TESTING.md
│   └── TRAINING_WORKFLOW.md
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
├── images/
│   ├── detailed-metrics.png
│   ├── dialect-analysis.png
│   ├── error-analysis.png
│   ├── main-dashboard-view.png
│   ├── multi-model-comparison-detailed-metrics.png
│   ├── multi-model-comparison-dialect-analysis.png
│   ├── multi-model-comparison-overview.png
│   ├── per-dialect-analysis.png
│   ├── sample-inspection-01.png
│   ├── sample-inspection-02.png
│   ├── sidebar-complete.png
│   └── sidebar-model-selection-dropdown-menu.png
├── logs/                # gitignored (*.log files)
│   └── evaluation.log
├── personal-notes/      # gitignored (the entire directory - personal notes)
│   ├── week01_days1-2_docker_local_setup.md
│   ├── week01_days3-4_data_pipeline.md
│   ├── week01_days5-7_minimal_fastapi_and_analysis.md
│   ├── week02_days1-3_evaluation_framework.md
│   ├── week02_days4-7_streamlit_dashboard.md
│   ├── week03_days1-2_model_selection.md
│   ├── week03_days3-4_cloud_gpu_setup.md
│   └── week04_days1-2_dutch_pre-training.md
├── results/
│   ├── error_analysis/
│   │   ├── ANALYSIS_NOTES.md
│   │   ├── error_analysis_config.yml
│   │   ├── 20251203_112924/
│   │   │   ├── README.txt
│   │   │   ├── analysis_wav2vec2-1b-german-cv11.json
│   │   │   ├── analysis_wav2vec2-german-with-lm.json
│   │   │   ├── analysis_whisper-large-v2.json
│   │   │   ├── analysis_whisper-large-v3-turbo.json
│   │   │   ├── analysis_whisper-large-v3.json
│   │   │   ├── analysis_whisper-medium.json
│   │   │   ├── model_comparison_summary.json
│   │   │   ├── worst_samples_wav2vec2-1b-german-cv11.csv
│   │   │   ├── worst_samples_wav2vec2-german-with-lm.csv
│   │   │   ├── worst_samples_whisper-large-v2.csv
│   │   │   ├── worst_samples_whisper-large-v3-turbo.csv
│   │   │   ├── worst_samples_whisper-large-v3.csv
│   │   │   └── worst_samples_whisper-medium.csv
│   │   └── bleu_integration_20251204_201238/
│   │       ├── README.txt
│   │       ├── analysis_wav2vec2-1b-german-cv11.json
│   │       ├── analysis_wav2vec2-german-with-lm.json
│   │       ├── analysis_whisper-large-v2.json
│   │       ├── analysis_whisper-large-v3-turbo.json
│   │       ├── analysis_whisper-large-v3.json
│   │       ├── analysis_whisper-medium.json
│   │       ├── model_comparison_summary.json
│   │       ├── worst_samples_wav2vec2-1b-german-cv11.csv
│   │       ├── worst_samples_wav2vec2-german-with-lm.csv
│   │       ├── worst_samples_whisper-large-v2.csv
│   │       ├── worst_samples_whisper-large-v3-turbo.csv
│   │       ├── worst_samples_whisper-large-v3.csv
│   │       └── worst_samples_whisper-medium.csv
│   └── metrics/
│       └── 20251202_171718/
│           ├── wav2vec2-1b-german-cv11_results.csv
│           ├── wav2vec2-1b-german-cv11_results.json
│           ├── wav2vec2-german-with-lm_results.csv
│           ├── wav2vec2-german-with-lm_results.json
│           ├── whisper-large-v2_results.csv
│           ├── whisper-large-v2_results.json
│           ├── whisper-large-v3_results.csv
│           ├── whisper-large-v3_results.json
│           ├── whisper-large-v3-turbo_results.csv
│           ├── whisper-large-v3-turbo_results.json
│           ├── whisper-medium_results.csv
│           └── whisper-medium_results.json
├── scripts/
│   ├── adapt_on_cloud.sh
│   ├── analyze_errors.py
│   ├── batch_evaluation.sh
│   ├── check_lm_vocab.py
│   ├── diagnose_lm_alignment.py
│   ├── download_lm.py
│   ├── evaluate_models.py
│   ├── inspect_results.py
│   ├── prepare_common_voice.py
│   ├── prepare_scripts.py
│   ├── runpod_analyze_errors.sh
│   ├── train_dutch_pretrain.py
│   ├── train_german_adaptation.py
│   ├── train_on_cloud.sh
│   └── upload_to_cloud.sh
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── __pycache__/         # gitignored
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── endpoints.py
│   │   ├── model_cache.py
│   │   ├── models.py
│   │   └── __pycache__/     # gitignored
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collator.py
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── splitter.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── error_analyzer.py
│   │   ├── evaluator.py
│   │   ├── evaluator.py.backup_pre_ssh_fix
│   │   ├── metrics.py
│   │   └── __pycache__/     # gitignored
│   ├── frontend/
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── data_table.py
│   │   │   ├── dialect_breakdown.py
│   │   │   ├── error_sample_viewer.py
│   │   │   ├── model_comparison.py
│   │   │   ├── plotly_charts.py
│   │   │   ├── sidebar.py
│   │   │   ├── statistics_panel.py
│   │   │   ├── terminology_panel.py
│   │   │   └── __pycache__/  # gitignored
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── data_loader.py
│   │       ├── error_data_loader.py
│   │       └── __pycache__/  # gitignored
│   ├── models/
│   │   ├── mms_model.py
│   │   ├── wav2vec2_model.py
│   │   └── __pycache__/      # gitignored
│   ├── training/
│   │   └── trainer.py
│   └── utils/
│       ├── __init__.py
│       ├── audio_utils.py
│       ├── checkpoint_manager.py
│       ├── file_utils.py
│       └── logging_config.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── README.md
    ├── __pycache__/          # gitignored
    ├── e2e/
    │   ├── __init__.py
    │   ├── conftest.py
    │   ├── test_api_workflow.py
    │   └── test_evaluation_workflow.py
    ├── fixtures/
    │   ├── __init__.py
    │   ├── audio/
    │   │   ├── sample_be_1.wav
    │   │   ├── sample_vs_3.wav
    │   │   └── sample_zh_2.wav
    │   └── data/
    │       ├── mock_results.json
    │       └── mock_swiss_german.tsv
    ├── integration/
    │   ├── __init__.py
    │   ├── conftest.py
    │   ├── test_backend_endpoints.py
    │   ├── test_data_pipeline.py
    │   ├── test_error_analysis_pipeline.py
    │   ├── test_frontend_data_loading.py
    │   └── test_model_evaluation.py
    └── unit/
        ├── __init__.py
        ├── conftest.py
        ├── test_config.py
        ├── test_edge_cases.py
        ├── test_parameterized.py
        ├── __pycache__/      # gitignored
        ├── backend/
        │   ├── __init__.py
        │   ├── test_model_cache.py
        │   └── test_pydantic_models.py
        ├── data_tests/
        │   ├── __init__.py
        │   ├── test_collator.py
        │   ├── test_loader.py
        │   ├── test_preprocessor.py
        │   └── test_splitter.py
        ├── evaluation/
        │   ├── __init__.py
        │   ├── test_error_analyzer.py
        │   ├── test_evaluator.py
        │   ├── test_metrics.py
        │   └── test_metrics_properties.py
        ├── frontend/
        │   ├── __init__.py
        │   ├── test_data_loader.py
        │   ├── test_error_data_loader.py
        │   ├── test_plotly_charts.py
        │   └── test_sidebar.py
        ├── model_tests/
        │   ├── __init__.py
        │   ├── test_mms_model.py
        │   └── test_wav2vec2_model.py
        ├── training/
        │   ├── __init__.py
        │   └── test_trainer.py
        └── utils/
            ├── __init__.py
            ├── test_audio_utils.py
            ├── test_checkpoint_manager.py
            ├── test_file_utils.py
            └── test_logging_config.py
```
