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

Evaluation on the Swiss German-to-Standard German translation task revealed systematic performance differences across models and dialects. Whisper models achieved 28.0–34.1% WER compared to 72.4–75.3% for Wav2Vec2 baselines, with Whisper large-v2 leading at 28.0% WER. Performance varied substantially by dialect, ranging from 5.8% (Glarus) to 46.3% (Zug), correlating with linguistic distance from Standard German. Error analysis of the 86 worst-performing samples (top 10% by WER) revealed that Whisper accurately transcribes Swiss German phonology but systematically fails to normalise to Standard German: 73% of errors exhibit perfect tense restructuring and 20.5% show dialectal article insertion patterns, producing Swiss German-influenced morphosyntax rather than the Standard German targets required by the task. BLEU analysis confirmed that only 1.4–2.1% of high-WER samples (WER ≥50%) preserved semantic meaning (BLEU ≥40%), validating WER as the appropriate metric for measuring translation quality on this corpus.

**Success Criteria Achievement:**

The project successfully met all exposé requirements:
- ✅ 6 models evaluated (requirement: ≥4)
- ✅ 15 dialects tested (requirement: ≥5)
- ✅ FastAPI backend with 10 endpoints
- ✅ Interactive Streamlit dashboard
- ✅ Docker-based evaluation pipeline
- ✅ Comprehensive error analysis with word-level alignment
- ✅ Technical documentation (12 documents covering methodology, workflows, architecture, and testing)

The evaluation framework provides a foundation for ASR model selection in Swiss German applications, with documented limitations including dialectal sample imbalance (6–203 samples per dialect) and zero-shot evaluation scope (no fine-tuning performed).


## 2. Software Architecture & Technology Choices

### 2.1 System Architecture

The system follows a modular, service-oriented architecture with clear separation of concerns across four primary components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER INTERACTION LAYER                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Streamlit Dashboard (Port 8501)                     │  │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │  │
│  │  │ Model Comparison │  │ Dialect Analysis │  │ Error Sample     │   │  │
│  │  │ Visualizations   │  │ Breakdowns       │  │ Viewer           │   │  │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘   │  │
│  │                                                                         │  │
│  │  Responsibilities:                                                      │  │
│  │  • Result visualisation (Plotly charts, data tables)                   │  │
│  │  • Multi-model comparison interface                                    │  │
│  │  • Interactive dialect filtering                                       │  │
│  │  • Word-level alignment display                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ HTTP REST API
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          API & EVALUATION LAYER                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      FastAPI Backend (Port 8000)                       │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐  │  │
│  │  │ Model Loading  │  │ Evaluation     │  │ Result Persistence    │  │  │
│  │  │ & Caching      │  │ Endpoints      │  │ (JSON/CSV)            │  │  │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘  │  │
│  │                                                                         │  │
│  │  Endpoints:                                                             │  │
│  │  • POST /evaluate - Trigger model evaluation                           │  │
│  │  • GET /results/{model} - Retrieve metrics                             │  │
│  │  • GET /available-models - List model registry                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Evaluation Pipeline                               │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │  │
│  │  │ ASR Evaluator   │  │ Metrics Module  │  │ Error Analyzer      │  │  │
│  │  │ (evaluator.py)  │  │ (metrics.py)    │  │ (error_analyzer.py) │  │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │  │
│  │          │                      │                      │               │  │
│  │          ├─ Whisper Models      ├─ WER calculation    ├─ Alignment   │  │
│  │          └─ Wav2Vec2 Models     ├─ CER calculation    ├─ Error types │  │
│  │                                 └─ BLEU calculation   └─ Confusion   │  │
│  │                                                            patterns    │  │
│  │  Responsibilities:                                                      │  │
│  │  • Model inference (Whisper, Wav2Vec2)                                 │  │
│  │  • Metric computation (WER, CER, BLEU)                                 │  │
│  │  • Word-level alignment (jiwer)                                        │  │
│  │  • Error categorisation (substitution, deletion, insertion)            │  │
│  │  • Per-dialect aggregation                                             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Reads from
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA PROCESSING LAYER                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Data Pipeline                                  │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │  │
│  │  │ DataLoader   │→ │ Preprocessor │→ │ Audio Utils  │→ │ Collator │  │  │
│  │  │ (loader.py)  │  │ (preproc.py) │  │ (utils.py)   │  │ (.py)    │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘  │  │
│  │                                                                         │  │
│  │  Responsibilities:                                                      │  │
│  │  • FHNW corpus loading (TSV → dataset)                                 │  │
│  │  • Audio preprocessing (resampling, normalisation)                     │  │
│  │  • Text normalization (lowercasing, punctuation removal)               │  │
│  │  • Batch collation for evaluation                                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Reads from
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE LAYER                                      │
│                                                                               │
│  data/raw/                    results/                   models/             │
│  └─ fhnw-swiss-german-corpus/ ├─ metrics/               └─ (HuggingFace     │
│     ├─ clips/*.flac            │  └─ 20251202_171718/        cache)          │
│     └─ metadata/*.tsv          │     ├─ *_results.json                       │
│                                │     └─ *_results.csv                         │
│                                └─ error_analysis/                             │
│                                   └─ 20251203_112924/                         │
│                                      ├─ analysis_*.json                       │
│                                      └─ worst_samples_*.csv                   │
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
| **HuggingFace Transformers** | Unified API for loading pre-trained Whisper, Wav2Vec2, and MMS models with automatic tokeniser handling. Model hub provides immediate access to 6 evaluated models without training overhead (infeasible given compute constraints). Caching system (`~/.cache/huggingface`) prevents repeated downloads. | Direct PyTorch model downloads (manual version management, no standardised API), TorchAudio models (limited pre-trained ASR coverage), OpenAI Whisper API (requires internet connectivity, per-request costs unsuitable for 867-sample evaluation). Trade-off: Dependency on external model hub availability, mitigated by local caching. |
| **jiwer** | Implements Wagner-Fischer alignment algorithm for word-level error categorisation (substitution, deletion, insertion) required for error analysis. Provides standardised WER/CER calculations matching academic benchmarks. Well-maintained with 500+ GitHub stars. | Manual Levenshtein distance implementation (error-prone, lacks alignment visualisation), editdistance library (character-level only, no word alignment), PER library (phoneme-level, not applicable to text evaluation). Trade-off: Black-box implementation, but established reliability and active maintenance outweigh transparency concerns. |
| **sacrebleu** | Reference implementation for BLEU score computation with standardised tokenisation ensuring reproducible results. Widely adopted in machine translation research (3000+ citations), providing methodological credibility for Swiss German→Standard German evaluation. Supports signature strings for exact reproduction. | NLTK BLEU (inconsistent tokenisation, deprecated in research), Moses multi-bleu.perl (requires Perl, harder to integrate), torchmetrics BLEU (less established in MT community). Trade-off: Primarily designed for MT evaluation rather than ASR, but semantic similarity metric applicable to translation task. |
| **pytest** | Fixture system enables model mocking (replacing GPU-intensive inference with dummy outputs) critical for running tests in CI without GPU access. Parameterisation reduces code duplication for testing multiple models. Comprehensive plugin ecosystem (pytest-cov for coverage, pytest-xdist for parallelisation). | unittest (more boilerplate, limited fixture support), nose2 (less actively maintained, smaller community), doctest (insufficient for integration testing). Trade-off: Additional dependency, but testing infrastructure essential for code quality assurance in solo development. |
| **Pandas** | Robust TSV/CSV handling for FHNW corpus metadata with built-in dialect grouping (`groupby`) and aggregation functions. Integrates seamlessly with Plotly for dataframe visualisation. Ubiquitous in data science, ensuring evaluator familiarity. | Polars (faster for large datasets, but 867 samples don't justify migration), CSV standard library (manual parsing error-prone), Dask (distributed computing unnecessary for dataset size). Trade-off: Heavy dependency (~100MB), acceptable for data-centric application. |
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