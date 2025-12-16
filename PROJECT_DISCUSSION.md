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

This project developed a reproducible evaluation framework for comparing state-of-the-art ASR models on Swiss German dialect recognition. Swiss German presents unique challenges for ASR due to high dialectal variability, absence of standardized orthography, and limited training data compared to Standard German. The primary objective was to enable systematic comparison of ASR models across multiple Swiss German dialects, providing both quantitative metrics and qualitative error analysis to support practitioner decision-making.

The system comprises three integrated components: (1) a FastAPI backend providing evaluation endpoints for 6 ASR models including Whisper variants (large-v3, large-v2, medium, turbo) and Wav2Vec2-German models; (2) a Docker-based evaluation pipeline implementing WER, CER, and BLEU metrics with systematic error categorization; and (3) an interactive Streamlit dashboard featuring multi-model comparison visualizations, per-dialect performance breakdowns, and word-level error alignment tools. All evaluation results are persisted in structured JSON and CSV formats, ensuring reproducibility and enabling offline analysis.

Evaluation on the Swiss German-to-Standard German translation task revealed systematic performance differences across models and dialects. Whisper models achieved 28.0–34.1% WER compared to 72.4–75.3% for Wav2Vec2 baselines, with Whisper large-v2 leading at 28.0% WER. Performance varied substantially by dialect, ranging from 5.8% (Glarus) to 46.3% (Zug), correlating with linguistic distance from Standard German. Error analysis of the 86 worst-performing samples (top 10% by WER) revealed that Whisper accurately transcribes Swiss German phonology but systematically fails to normalize to Standard German: 73% of errors exhibit perfect tense restructuring and 20.5% show dialectal article insertion patterns, producing Swiss German-influenced morphosyntax rather than the Standard German targets required by the task. BLEU analysis confirmed that only 1.4–2.1% of high-WER samples (WER ≥50%) preserved semantic meaning (BLEU ≥40%), validating WER as the appropriate metric for measuring translation quality on this corpus.

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