# Project Discussion Document
**Swiss German ASR Evaluation Platform**
Till Ermold | CODE University of Applied Sciences Berlin | December 2025

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Software Architecture & Technology Choices](#2-software-architecture--technology-choices)
3. [Machine Learning & Data Science Approach](#3-machine-learning--data-science-approach)
4. [Requirements Elicitation & Validation](#4-requirements-elicitation--validation)
5. [Results & Key Findings](#5-results--key-findings)
6. [Limitations & Future Work](#6-limitations--future-work)
7. [Conclusion](#7-conclusion)
8. [References](#8-references)


## 1. Executive Summary

This project developed a reproducible evaluation framework for comparing state-of-the-art ASR models on Swiss German dialect recognition. Swiss German presents unique challenges for ASR due to high dialectal variability, absence of standardised orthography, and limited training data compared to Standard German. The primary objective was to enable systematic comparison of ASR models across multiple Swiss German dialects, providing both quantitative metrics and qualitative error analysis to support practitioner decision-making.

The system comprises three integrated components: (1) a FastAPI backend providing evaluation endpoints for 6 ASR models including Whisper variants (large-v3, large-v2, medium, turbo) and Wav2Vec2-German models; (2) a Docker-based evaluation pipeline implementing WER, CER, and BLEU metrics with systematic error categorisation; and (3) an interactive Streamlit dashboard featuring multi-model comparison visualisations, per-dialect performance breakdowns, and word-level error alignment tools. All evaluation results are persisted in structured JSON and CSV formats, ensuring reproducibility and enabling offline analysis.

Evaluation on the Swiss German-to-Standard German translation task revealed systematic performance differences across models and dialects. Whisper models achieved 28.0–34.1% WER compared to 72.4–75.3% for Wav2Vec2 baselines, with Whisper large-v2 leading at 28.0% WER. Performance varied substantially by dialect, ranging from 5.8% (Glarus) to 39.7% (Zug), correlating with linguistic distance from Standard German. Error analysis of the 86 worst-performing samples (top 10% by WER) revealed that Whisper accurately transcribes Swiss German phonology but systematically fails to normalise to Standard German: 73% of errors exhibit perfect tense restructuring and 20.5% show dialectal article insertion patterns, producing Swiss German-influenced morphosyntax rather than the Standard German targets required by the task. BLEU analysis confirmed that only 1.4–2.1% of high-WER samples (WER ≥50%) preserved semantic meaning (BLEU ≥40%), validating WER as the appropriate metric for measuring translation quality on this corpus.

**Success Criteria Achievement:**

The project successfully met all exposé requirements:
- ✅ 6 models evaluated (requirement: ≥4)
- ✅ 17 dialects tested (requirement: ≥5)
- ✅ FastAPI backend with 10 endpoints
- ✅ Interactive Streamlit dashboard
- ✅ Docker-based evaluation pipeline
- ✅ Comprehensive error analysis with word-level alignment
- ✅ Technical documentation (12 documents covering methodology, workflows, architecture, and testing)

The evaluation framework provides a foundation for ASR model selection in Swiss German applications, with documented limitations including dialectal sample imbalance (1–203 samples per dialect) and zero-shot evaluation scope (no fine-tuning performed).

**Success Criteria Achievement:**

The project successfully met all exposé requirements:
- ✅ 6 models evaluated (requirement: ≥4)
- ✅ 17 dialects tested (requirement: ≥5)
- ✅ FastAPI backend with 10 endpoints
- ✅ Interactive Streamlit dashboard
- ✅ Docker-based evaluation pipeline
- ✅ Comprehensive error analysis with word-level alignment
- ✅ Technical documentation (12 documents covering methodology, workflows, architecture, and testing)

The evaluation framework provides a foundation for ASR model selection in Swiss German applications, with documented limitations including dialectal sample imbalance (1–203 samples per dialect) and zero-shot evaluation scope (no fine-tuning performed).


## 2. Software Architecture & Technology Choices

### 2.1 System Architecture

The system follows a modular, service-oriented architecture with clear separation of concerns across four primary components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER INTERACTION LAYER                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Streamlit Dashboard (Port 8501)                    │  │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │  │
│  │  │ Model Comparison │  │ Dialect Analysis │  │ Error Sample     │     │  │
│  │  │ Visualizations   │  │ Breakdowns       │  │ Viewer           │     │  │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘     │  │
│  │                                                                       │  │
│  │  Responsibilities:                                                    │  │
│  │  • Result visualisation (Plotly charts, data tables)                  │  │
│  │  • Multi-model comparison interface                                   │  │
│  │  • Interactive dialect filtering                                      │  │
│  │  • Word-level alignment display                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ HTTP REST API
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          API & EVALUATION LAYER                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      FastAPI Backend (Port 8000)                      │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐   │  │
│  │  │ Model Loading  │  │ Evaluation     │  │ Result Persistence     │   │  │
│  │  │ & Caching      │  │ Endpoints      │  │ (JSON/CSV)             │   │  │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘   │  │
│  │                                                                       │  │
│  │  Endpoints:                                                           │  │
│  │  • POST /evaluate - Trigger model evaluation                          │  │
│  │  • GET /results/{model} - Retrieve metrics                            │  │
│  │  • GET /available-models - List model registry                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Evaluation Pipeline                              │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐    │  │
│  │  │ ASR Evaluator   │  │ Metrics Module  │  │ Error Analyzer      │    │  │
│  │  │ (evaluator.py)  │  │ (metrics.py)    │  │ (error_analyzer.py) │    │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘    │  │
│  │          │                      │                      │              │  │
│  │          ├─ Whisper Models      ├─ WER calculation    ├─ Alignment    │  │
│  │          └─ Wav2Vec2 Models     ├─ CER calculation    ├─ Error types  │  │
│  │                                 └─ BLEU calculation   └─ Confusion    │  │
│  │                                                            patterns   │  │
│  │  Responsibilities:                                                    │  │
│  │  • Model inference (Whisper, Wav2Vec2)                                │  │
│  │  • Metric computation (WER, CER, BLEU)                                │  │
│  │  • Word-level alignment (jiwer)                                       │  │
│  │  • Error categorisation (substitution, deletion, insertion)           │  │
│  │  • Per-dialect aggregation                                            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Reads from
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA PROCESSING LAYER                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Data Pipeline                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐   │  │
│  │  │ DataLoader   │→ │ Preprocessor │→ │ Audio Utils  │→ │ Collator │   │  │
│  │  │ (loader.py)  │  │ (preproc.py) │  │ (utils.py)   │  │ (.py)    │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘   │  │
│  │                                                                       │  │
│  │  Responsibilities:                                                    │  │
│  │  • FHNW corpus loading (TSV → dataset)                                │  │
│  │  • Audio preprocessing (resampling, normalisation)                    │  │
│  │  • Text normalization (lowercasing, punctuation removal)              │  │
│  │  • Batch collation for evaluation                                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Reads from
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE LAYER                                     │
│                                                                             │
│  data/raw/                    results/                   models/            │
│  └─ fhnw-swiss-german-corpus/ ├─ metrics/               └─ (HuggingFace     │
│     ├─ clips/*.flac            │  └─ 20251202_171718/        cache)         │
│     └─ metadata/*.tsv          │     ├─ *_results.json                      │
│                                │     └─ *_results.csv                       │
│                                └─ error_analysis/                           │
│                                   └─ 20251203_112924/                       │
│                                      ├─ analysis_*.json                     │
│                                      └─ worst_samples_*.csv                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Architectural Decisions:**

1. **Separation of Concerns**: Dashboard, API, evaluation logic, and data processing are isolated, enabling independent testing and deployment.

2. **Result Persistence**: All metrics are written to timestamped JSON/CSV files before visualisation, allowing offline analysis and reproducibility.

3. **Model Registry Pattern**: Centralized model configuration in `scripts/evaluate_models.py` with model type abstractions (Whisper, Wav2Vec2).

4. **Stateless API**: FastAPI endpoints delegate to evaluation pipeline without maintaining session state, supporting horizontal scaling.

5. **Docker Containerisation**: All components packaged in unified Docker image with separate service definitions in `docker-compose.yml` for local/cloud deployment.


### 2.2 Technology Stack Decisions

The technology stack prioritises rapid development, reproducibility, and evaluator accessibility under a 9-week deadline with solo development constraints. Key infrastructure decisions (Docker, Python version) ensure consistent environments across development machines, cloud GPU resources, and evaluator systems. Framework selections (FastAPI, Streamlit) balance development velocity with production-grade capabilities, whilst library choices (jiwer, sacrebleu) ensure methodologically rigorous ASR evaluation aligned with academic standards.

| Technology | Selection Rationale | Alternatives & Trade-offs |
|------------|---------------------|---------------------------|
| **Python 3.11** | Latest stable version with performance improvements (up to 25% faster than 3.10) and enhanced error messages aiding rapid debugging. Supported by all required libraries (PyTorch 2.6.0, Transformers 4.36.0). | Python 3.10 (marginally slower), Python 3.12 (limited library support at project start). Trade-off: Requires explicit version specification in Docker to prevent future compatibility breaks. |
| **Docker** | Containerisation ensures identical runtime environments across local MacBook (Intel i9), RunPod GPU instances (RTX 3090/5090), and evaluator machines. Eliminates "works on my machine" issues critical for reproducible research. | Conda environments (platform-dependent, harder to replicate exact binary versions), virtual environments (no system-level dependency isolation). Trade-off: Additional complexity for local development, but reproducibility benefits essential for academic submission. |
| **Docker Compose** | Single-command deployment (`docker compose up`) orchestrates dashboard, API, and test services with shared network and volume configuration. Enables evaluators to run complete system without manual service coordination. | Kubernetes (excessive for 3-service local deployment), manual Docker commands (error-prone, poor documentation). Trade-off: Not suitable for large-scale production, but appropriate for evaluation/demonstration workload. |
| **PyTorch 2.6.0** | Industry-standard deep learning framework with comprehensive HuggingFace integration. Native support for MPS (Apple Silicon), CUDA, and CPU backends enables development on MacBook with production inference on RunPod GPUs. | TensorFlow (less prevalent in ASR research, weaker HuggingFace support), JAX (immature ecosystem for production ASR). Trade-off: Large dependency footprint (~2GB), but unavoidable for state-of-the-art models. |
| **FastAPI** | Native async support critical for handling concurrent evaluation requests without blocking (GPU inference takes 30-60s per batch). Auto-generated OpenAPI documentation (`/docs` endpoint) reduces manual API documentation burden—essential for solo developer timeline. Type hints enable request validation at runtime. | Flask (lacks native async needed for concurrent requests), aiohttp (less mature ecosystem, fewer middleware options), Django (unnecessary ORM/admin features for read-only evaluation workload). Trade-off: Newer framework with smaller community than Flask, but async benefits outweigh ecosystem maturity. |
| **Streamlit** | Python-only development (no JavaScript/CSS required) enables rapid dashboard prototyping with built-in widgets (st.selectbox, st.multiselect) matching requirements for multi-model comparison and dialect filtering. Deployment to Streamlit Cloud verified system fits within 2.7GB resource limits. | Dash by Plotly (callback-heavy architecture increases boilerplate, steeper learning curve), Gradio (less flexible layout control), custom React frontend (requires JavaScript expertise, infeasible for 9-week solo timeline). Trade-off: Limited customisation compared to React, but prototyping speed critical for iterative development. |
| **Plotly** | Interactive visualisations (hover tooltips, zoom, pan) essential for exploring per-dialect performance breakdown and error distributions. Native Streamlit integration via `st.plotly_chart()`. Produces publication-quality figures for thesis. | Matplotlib (static plots lack interactivity needed for exploration), Seaborn (built on Matplotlib, same limitations), Altair (declarative syntax steeper learning curve). Trade-off: Larger JavaScript bundle size (~3MB), acceptable for dashboard use case. |
| **HuggingFace Transformers** | Unified API for loading pre-trained Whisper, Wav2Vec2, and MMS models with automatic tokeniser handling. Model hub provides immediate access to 6 evaluated models without training overhead (infeasible given compute constraints). Caching system (`~/.cache/huggingface`) prevents repeated downloads. | Direct PyTorch model downloads (manual version management, no standardised API), TorchAudio models (limited pre-trained ASR coverage), OpenAI Whisper API (requires internet connectivity, per-request costs unsuitable for 863-sample evaluation). Trade-off: Dependency on external model hub availability, mitigated by local caching. |
| **jiwer** | Implements Wagner-Fischer alignment algorithm for word-level error categorisation (substitution, deletion, insertion) required for error analysis. Provides standardised WER/CER calculations matching academic benchmarks. Well-maintained with 500+ GitHub stars. | Manual Levenshtein distance implementation (error-prone, lacks alignment visualisation), editdistance library (character-level only, no word alignment), PER library (phoneme-level, not applicable to text evaluation). Trade-off: Black-box implementation, but established reliability and active maintenance outweigh transparency concerns. |
| **sacrebleu** | Reference implementation for BLEU score computation with standardised tokenisation ensuring reproducible results. Widely adopted in machine translation research (3000+ citations), providing methodological credibility for Swiss German→Standard German evaluation. Supports signature strings for exact reproduction. | NLTK BLEU (inconsistent tokenisation, deprecated in research), Moses multi-bleu.perl (requires Perl, harder to integrate), torchmetrics BLEU (less established in MT community). Trade-off: Primarily designed for MT evaluation rather than ASR, but semantic similarity metric applicable to translation task. |
| **pytest** | Fixture system enables model mocking (replacing GPU-intensive inference with dummy outputs) critical for running tests in CI without GPU access. Parameterisation reduces code duplication for testing multiple models. Comprehensive plugin ecosystem (pytest-cov for coverage, pytest-xdist for parallelisation). | unittest (more boilerplate, limited fixture support), nose2 (less actively maintained, smaller community), doctest (insufficient for integration testing). Trade-off: Additional dependency, but testing infrastructure essential for code quality assurance in solo development. |
| **Pandas** | Robust TSV/CSV handling for FHNW corpus metadata with built-in dialect grouping (`groupby`) and aggregation functions. Integrates seamlessly with Plotly for dataframe visualisation. Ubiquitous in data science, ensuring evaluator familiarity. | Polars (faster for large datasets, but 863 samples don't justify migration), CSV standard library (manual parsing error-prone), Dask (distributed computing unnecessary for dataset size). Trade-off: Heavy dependency (~100MB), acceptable for data-centric application. |
| **Git/GitHub** | Version control with feature branch workflow enables isolated development of dashboard, API, and evaluation components. GitHub Actions (if used) automates testing on push. Public repository satisfies assessment requirement for accessible codebase. | GitLab (self-hosting overhead), Bitbucket (smaller community), local Git only (no remote backup or collaboration capability). Trade-off: Public repository exposes code, mitigated by MIT licence permitting reuse. |

**Dependency Management & Security:**

All Python dependencies are pinned to specific versions in `requirements.txt` (e.g., `torch==2.6.0`, `transformers==4.36.0`, `streamlit==1.32.0`) to prevent unexpected breaking changes from upstream updates. Docker base image locked to `python:3.11-slim-bullseye` with explicit digest hash for cryptographic verification. Dependabot alerts enabled on GitHub for security vulnerability monitoring. Version updates are tested in isolated feature branches before merging to main, with regression tests validating metric calculations remain consistent. This approach balances reproducibility (strict pinning) with security (monitored updates).

**Testing Strategy & Quality Assurance:**

pytest's fixture system enables comprehensive test coverage without GPU requirements: a `FakeEvaluator` fixture mocks model inference by returning predetermined WER/CER/BLEU values, allowing integration tests to validate API response schemas and error handling without loading multi-gigabyte models. Parametrised tests cover all 6 models without code duplication (e.g., `@pytest.mark.parametrize("model", ["whisper-large-v3", "wav2vec2-german"])`). Test suite spans three levels: unit tests for metric calculations (jiwer integration, BLEU score validation), integration tests for FastAPI endpoints (using TestClient to simulate HTTP requests), and end-to-end tests verifying complete evaluation workflows (dataset loading → model inference → result persistence). Total test execution time under 30 seconds enables rapid development iteration. Coverage targets were not formally specified but critical paths (evaluation pipeline, API endpoints, metric computation) achieve >80% line coverage per pytest-cov reports.


### 2.3 Code Quality & Security Practices

**Clean Code Principles:** The codebase follows industry-standard Python conventions with snake_case function naming (e.g., `evaluate_dataset()`) and PascalCase class naming (e.g., `WhisperEvaluator`). All public functions include Google-style docstrings with type hints for parameters and return values, enabling IDE autocomplete and runtime validation. Project structure separates concerns: `src/` contains application code organised by layer (backend, frontend, evaluation), `tests/` mirrors this structure with unit/integration/e2e test suites, and `docs/` provides methodology documentation enabling reproduction. Separation of concerns is enforced through modular design: Pydantic schemas define API contracts (`src/backend/models.py`), evaluator classes encapsulate model inference logic (`src/evaluation/evaluator.py`), and FastAPI routes handle HTTP request/response cycles without business logic.

**Input Validation & Security:** API endpoints implement input sanitisation to prevent path traversal attacks. The `/api/results/{model}` endpoint validates model names against alphanumeric patterns before file system access:

```python
def validate_safe_path_component(component: str, name: str):
    """Prevent path traversal attacks in API path parameters."""
    if not re.match(r"^[a-zA-Z0-9_-]+$", component):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {name}: must contain only alphanumeric characters"
        )
```

This validation prevents malicious requests such as `/api/results/../../etc/passwd` from accessing files outside the results directory. No authentication is implemented as the system is designed for local deployment and evaluator access on trusted networks, with deployment documentation explicitly noting this architectural constraint. Security monitoring is provided through Dependabot alerts for dependency vulnerabilities, as documented in Section 2.2.


## 3. Machine Learning & Data Science Approach

### 3.1 Zero-Shot Evaluation Framework

This project employs a zero-shot evaluation methodology, assessing pre-trained ASR models on Swiss German without dialect-specific fine-tuning. This approach was chosen for three primary reasons: (1) **research validity**—zero-shot evaluation tests genuine generalization capabilities rather than task-specific fitting, providing insights relevant to practitioners deploying models on unseen dialects; (2) **methodological focus**—the 9-week timeline prioritised rigorous comparative evaluation over model training, aligning with the project's core objective of systematic performance analysis; and (3) **resource constraints**—fine-tuning large-scale models (up to 1.55B parameters) on limited Swiss German data (863 samples across 17 dialects) would require extensive hyperparameter tuning and cross-validation infrastructure beyond project scope.

An exploratory fine-tuning experiment was conducted in Week 6 to investigate Dutch→German→Swiss German transfer learning (leveraging linguistic proximity between Dutch and German). The experiment was discontinued due to two critical issues: (1) a deployment bug where the `adapt_on_cloud.sh` script downloaded and overwrote tokenizer vocabulary files from the base model, creating mismatches between the trained output layer and tokenizer vocabulary that resulted in 100% WER with `<unk>` token dominance; and (2) severe data imbalance between Dutch pre-training (100-300k samples × 10 epochs = 1-3M training examples) and German adaptation (30,708 samples × 3 epochs = 92k examples), creating a 10-30× training disparity. Despite implementing Elastic Weight Consolidation (lambda=0.4, fisher_samples=5000) to preserve Dutch knowledge, the German adaptation phase proved insufficient to restore German linguistic features given the limited training examples. These findings reinforced the strategic decision to focus project resources on zero-shot evaluation, which directly addresses the research question of out-of-the-box model performance for Swiss German ASR deployment scenarios without requiring complex multi-stage transfer learning infrastructure.

The zero-shot framework enables meaningful comparison of models' inherent multilingual capabilities and German-language transfer learning, providing actionable insights for practitioners selecting ASR systems for low-resource dialectal variants without access to fine-tuning infrastructure.

### 3.2 Model Selection Strategy

Six state-of-the-art ASR models were selected to represent diversity across three dimensions: (1) **architecture families** (Whisper encoder-decoder vs Wav2Vec2 self-supervised), (2) **parameter scales** (317M to 1.55B parameters), and (3) **training paradigms** (multilingual zero-shot vs German-specific fine-tuning). This selection strategy ensures comprehensive evaluation of approaches applicable to Swiss German ASR whilst maintaining computational feasibility within project constraints.

| Model | Parameters | Architecture | Training Data | Selection Rationale |
|-------|------------|--------------|---------------|---------------------|
| **whisper-large-v2** | 1.55B | Encoder-decoder transformer | 680k hours multilingual (v2 dataset) | Established baseline for Whisper family; widely deployed in production systems. Enables comparison with v3 training data improvements. |
| **whisper-large-v3** | 1.55B | Encoder-decoder transformer | 680k hours multilingual (v3 dataset, improved annotations) | Latest Whisper release; tests whether updated training data benefits Swiss German despite identical architecture to v2. |
| **whisper-large-v3-turbo** | 809M | Optimized encoder-decoder (reduced decoder layers) | Same as large-v3 | Efficiency-optimized variant released November 2024. Evaluates OpenAI's claim of similar accuracy with 8× faster inference for resource-constrained deployment. |
| **whisper-medium** | 769M | Encoder-decoder transformer | Same as large-v2/v3 | Resource-constrained baseline; tests performance degradation at ~50% parameter reduction. Smallest Whisper variant with documented reliability on low-resource languages (tiny/base/small excluded due to poor multilingual performance). |
| **wav2vec2-xls-r-1b-german-cv11** | 1.0B | XLS-R 1B fine-tuned on German | 1,700 hours German Common Voice v11 | Large-scale German-trained model providing upper bound for Standard German ASR transferred to Swiss German. Tests hypothesis that German-specific training aids dialectal recognition. |
| **wav2vec2-large-xlsr-53-german-with-lm** | 317M | XLSR-53 fine-tuned on German | 1,700 hours German Common Voice + 5-gram KenLM | Language model-enhanced decoding variant; isolates impact of explicit linguistic constraints vs pure neural prediction. Smallest model in evaluation set (resource-constrained comparison). |

**Architecture Family Rationale:**

**Whisper models** (4 variants) were prioritised due to their documented zero-shot translation capabilities, making them directly applicable to the Swiss German→Standard German task without modification. The inclusion of multiple Whisper variants enables analysis of three trade-off dimensions: (1) training data quality (v2 vs v3), (2) architectural efficiency (large-v3 vs turbo), and (3) parameter scale (large vs medium). This coverage addresses practitioner decision-making across accuracy, latency, and resource availability constraints.

**Wav2Vec2 models** (2 variants) provide a contrasting approach focused on direct transcription rather than translation. These German-trained models establish performance baselines for Standard German ASR applied to dialectal speech, quantifying the transfer gap and motivating future dialect-specific adaptation research. The language model-enhanced variant (with-lm) isolates the contribution of explicit linguistic constraints versus pure neural prediction, relevant for deployment scenarios where post-processing is feasible.

**Excluded alternatives:** Smaller Whisper variants (tiny: 39M, base: 74M, small: 244M) were excluded based on OpenAI documentation indicating poor performance on low-resource languages. MMS-1B-all (Massively Multilingual Speech, 1B parameters) was initially considered but deprioritised after preliminary evaluation showed 89.2% WER, indicating insufficient German-language coverage in training data.

### 3.3 Evaluation Metrics

Three complementary metrics assess model performance across accuracy and semantic dimensions, chosen for their established use in ASR and machine translation research whilst addressing the specific requirements of the Swiss German→Standard German translation task.

**Primary Metrics:**

1. **Word Error Rate (WER)**: Levenshtein distance at word level, calculated as (substitutions + deletions + insertions) / reference word count. WER serves as the primary metric due to its direct interpretability (percentage of words requiring correction) and widespread adoption in ASR literature, enabling comparison with existing Swiss German ASR benchmarks. For the translation task, WER measures lexical accuracy of Standard German output against ground-truth translations.

2. **Character Error Rate (CER)**: Levenshtein distance at character level, calculated identically to WER but on character sequences. CER provides finer-grained error assessment, particularly valuable for detecting systematic morphological errors (e.g., incorrect verb conjugations, article gender mismatches) that may inflate WER despite preserving semantic content. CER also captures partial word recognition, where models correctly transcribe word stems but fail on inflectional suffixes.

3. **BLEU Score** (Bilingual Evaluation Understudy): N-gram overlap metric (1-gram through 4-gram) with brevity penalty, computed using sacrebleu reference implementation. BLEU assesses semantic similarity by measuring phrase-level overlap rather than exact word matches, addressing the hypothesis that high WER might result from semantically equivalent paraphrases rather than genuine errors. BLEU is particularly relevant for translation tasks where multiple valid Standard German renderings exist for Swiss German utterances.

**Metric Validation:**

Post-evaluation integration of BLEU scores validated the appropriateness of WER as the primary metric for this task. Analysis of high-WER samples (WER ≥50%, representing the worst-performing 10% of transcriptions) revealed that only 1.4–2.1% achieved BLEU ≥40% (threshold for semantic preservation in MT literature). This finding demonstrates that high WER predominantly reflects genuine transcription errors rather than semantically valid paraphrases, confirming WER's suitability for measuring translation quality on the Swiss German→Standard German task. Detailed BLEU integration results are presented in Section 5.3.

**Alternatives Considered:**

- **chrF** (character n-gram F-score): Excludes word boundaries, potentially more robust to segmentation errors. Not adopted due to limited interpretability compared to CER and lack of established thresholds for "good" performance.
- **BERTScore**: Contextual embedding similarity using pre-trained language models. Excluded due to (1) absence of Swiss German BERT models, (2) computational cost (~10× slower than BLEU), and (3) lack of established benchmarks for ASR evaluation.

The selected three-metric combination balances comprehensiveness (word-level, character-level, semantic-level) with computational feasibility and result interpretability.

### 3.4 Statistical Validity & Aggregation Methodology

**Sample Size Documentation:**

The evaluation was conducted on the test split of the FHNW Swiss German corpus, comprising **863 samples across 17 Swiss German dialects**. This test set was created using stratified sampling (70% train, 15% validation, 15% test) from the public subset of the corpus (total 5,750 samples), ensuring proportional dialect representation whilst maintaining statistical independence from training data. The distribution exhibits substantial imbalance reflecting real-world Swiss German dialect prevalence:

| Dialect | Samples | Dialect | Samples | Dialect | Samples |
|---------|---------|---------|---------|---------|---------|
| BE (Bern) | 203 | ZH (Zürich) | 144 | SG (St. Gallen) | 116 |
| AG (Aargau) | 108 | BL (Basel-Land) | 54 | LU (Lucerne) | 51 |
| TG (Thurgau) | 50 | SO (Solothurn) | 36 | ZG (Zug) | 30 |
| VS (Valais) | 17 | UR (Uri) | 15 | GR (Graubünden) | 12 |
| SZ (Schwyz) | 9 | FR (Fribourg) | 7 | GL (Glarus) | 6 |
| SH (Schaffhausen) | 4 | NW (Nidwalden) | 1 | | |

**Total: 863 samples** (range: 1 sample [NW] to 203 samples [BE], median: 30 samples)

This distribution reflects the corpus design goal of approximating real-world Swiss German dialect prevalence rather than uniform sampling. However, it creates significant statistical challenges: dialects with fewer than 10 samples (NW, SH, GL, FR, SZ) cannot support robust statistical inference, whilst heavily represented dialects (BE, ZH, SG) dominate aggregate metrics.

**Aggregation Approach:**

Model performance is reported using **corpus-level WER**, calculated by aggregating all errors across the entire test set before computing the error rate: WER_corpus = (Σ errors) / (Σ reference words). This approach follows standard practice in ASR evaluation (e.g., Kaldi toolkit, LibriSpeech benchmark) and differs from averaging per-utterance WERs, which would give disproportionate weight to short utterances where a single error can yield 100% WER.

Corpus-level aggregation is appropriate for this evaluation because: (1) it reflects real-world deployment performance where total error count matters more than per-utterance variance, (2) it prevents artificially inflated error rates from short utterances (the FHNW corpus includes samples as brief as 2-3 words), and (3) it enables direct comparison with published ASR benchmarks that use identical methodology.

**Per-dialect WER** is computed separately for each of the 17 dialects using the same corpus-level approach within dialect subsets. This granularity reveals systematic performance patterns correlated with linguistic distance from Standard German whilst maintaining methodological consistency.

**Statistical Testing Limitations:**

Formal significance testing (e.g., paired t-tests, Wilcoxon signed-rank) was not conducted due to severe sample imbalance rendering such tests statistically invalid. With some dialects having as few as 1 sample, assumptions underlying parametric tests (normality, sufficient sample size) are violated. Non-parametric alternatives require minimum sample sizes (typically n ≥ 20 per group) that several dialects fail to meet.

This limitation is acknowledged as an inherent constraint of the FHNW corpus design rather than a methodological deficiency. The evaluation instead relies on **descriptive statistics** (mean WER/CER/BLEU, per-dialect breakdowns) and **effect size magnitude** (e.g., 23% absolute WER difference between Whisper and Wav2Vec2 models) to characterize performance differences. These descriptive measures provide actionable insights for practitioners whilst maintaining statistical honesty about inference limitations.

Future work with larger, balanced Swiss German corpora (e.g., SDS-200, SwissDial) could enable robust significance testing, but such datasets were unavailable or incompatible with the evaluation timeline.

### 3.5 Error Analysis Methodology

Beyond aggregate metrics, the evaluation framework incorporates systematic error analysis to identify linguistic patterns and model-specific failure modes. This analysis provides qualitative insights complementing quantitative WER/CER/BLEU scores.

**Word-Level Alignment:**

Error categorization employs the **Wagner-Fischer algorithm** implemented in the jiwer library, which computes minimum edit distance between hypothesis (model transcription) and reference (ground truth) at the word level. This alignment classifies each word as:

- **Correct (C)**: Exact match between hypothesis and reference
- **Substitution (S)**: Incorrect word predicted (e.g., "gross" → "groß")
- **Deletion (D)**: Reference word omitted in hypothesis
- **Insertion (I)**: Extra word added in hypothesis not present in reference

The alignment enables precise localization of errors within transcriptions, supporting targeted analysis of error types (phonological, morphological, syntactic) and extraction of confusion pairs (systematically misrecognized word mappings).

**Worst-Sample Analysis:**

The top 10% of samples by WER (86/863 samples, threshold: WER ≥ 50%) undergo detailed manual inspection to identify complex error patterns not captured by aggregate statistics. This 10% threshold balances two objectives: (1) sufficient sample size for pattern identification (86 samples provides statistical robustness), and (2) feasibility of manual linguistic analysis within project timeline (86 samples × 6 models = 516 transcriptions for review).

Worst-sample analysis revealed systematic phenomena including:

- **Perfect tense restructuring**: Swiss German auxiliary verb placement (e.g., "het verlore" → "hat verloren") transcribed accurately by Whisper but yielding high WER against Standard German word order references
- **Dialectal article patterns**: Bernese definite article insertion (e.g., "ds Huus" with dialectal article) transcribed with Standard German "das Haus", creating WER inflation despite semantic equivalence
- **Phonological substitutions**: Systematic consonant shifts (e.g., /k/ → /ch/) reflected in transcriptions, indicating models capture phonetic detail but mismatch orthographic conventions

These patterns are quantified in Section 5 and inform practitioner guidance on expected error types for Swiss German ASR deployment.

**Per-Dialect Confusion Patterns:**

For each dialect, the 10 most frequent word-pair confusions (substitutions with count ≥ 2 occurrences) are extracted to identify systematic misrecognitions. This analysis reveals dialect-specific phonological challenges (e.g., Valais /r/ → /ʀ/ substitutions) and guides future error correction strategies such as targeted fine-tuning or post-processing rules.

Confusion pattern extraction uses exact string matching on aligned word pairs, filtering for minimum occurrence frequency to distinguish systematic errors from random noise. Results are exported as CSV files (results/error_analysis/YYYYMMDD_HHMMSS/confusion_pairs_*.csv) for offline analysis and visualisation in the Streamlit dashboard.

**Methodological Justification:**

The combination of automated alignment (jiwer) and targeted manual inspection (top 10%) balances scalability with depth. Automated metrics provide comprehensive coverage across 863 samples, whilst manual analysis yields linguistic insights (e.g., morphosyntactic patterns) that purely algorithmic approaches cannot detect. This hybrid methodology is standard in ASR error analysis research, where quantitative metrics guide qualitative investigation of representative failure cases.

Per-dialect granularity (rather than global-only analysis) is essential because Swiss German exhibits substantial phonological and morphological variation across regions. Aggregate statistics can mask dialect-specific phenomena; for example, a model may excel on Zürich dialect (lexically closer to Standard German) whilst failing on Valais (Romance substrate influence), with these patterns only visible through per-dialect decomposition.