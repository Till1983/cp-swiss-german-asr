# Error Analysis Methodology

This document provides comprehensive technical documentation of the error analysis approach used in the Swiss German ASR project. It is intended for researchers and engineers who wish to understand, replicate, or extend this work.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Task Definition: Model Types and Evaluation Framework](#15-task-definition-model-types-and-evaluation-framework)
3. [Error Categorization Methodology](#2-error-categorization-methodology)
4. [Threshold Selection and Justification](#3-threshold-selection-and-justification)
5. [Confusion Pattern Calculation](#4-confusion-pattern-calculation)
6. [Statistical Aggregation Methods](#5-statistical-aggregation-methods)
7. [Annotated Alignment Examples](#6-annotated-alignment-examples)
8. [Limitations](#7-limitations)
9. [Reproduction Guide](#8-reproduction-guide)
10. [References](#9-references)

---

## 1. Overview

The error analysis pipeline transforms raw ASR evaluation results (reference/hypothesis pairs with WER/CER scores) into actionable insights about model behaviour. The analysis identifies:

- **Error type distributions**: Substitutions, deletions, and insertions
- **High-error samples**: Worst-performing utterances for qualitative inspection
- **Dialect-specific patterns**: Performance variations across Swiss German dialects
- **Confusion pairs**: Systematic word-level recognition errors

### Architecture
```
results/metrics/*_results.json
         │
         ▼
┌─────────────────────────┐
│   analyze_errors.py     │
│   ├─ Load results       │
│   ├─ ErrorAnalyzer      │
│   └─ Save outputs       │
└─────────────────────────┘
         │
         ▼
results/error_analysis/
├── analysis_*.json         (detailed hierarchical analysis)
├── worst_samples_*.csv     (high-error samples for inspection)
└── model_comparison_summary.json
```

### Quick Reference: Output Interpretation

| File | Error Rate Type | Denominator | Interpretation |
|------|----------------|-------------|----------------|
| `model_comparison_summary.json` | `sub_rate`, `del_rate`, `ins_rate` | **All operations** (incl. correct) | "% of all words that are substitutions/deletions/insertions" |
| `analysis_*.json` → `error_distribution_percent` | `substitution`, `deletion`, `insertion`, `correct` | **All operations** | Same as above (rates sum to 100%) |
| `analysis_*.json` → `dialect_analysis` → `error_distribution` | `sub_rate`, `del_rate`, `ins_rate` | **Errors only** (excl. correct) | "Among errors, % that are substitutions/deletions/insertions" |

**Rule of Thumb:**
- If rates sum to ~100% → calculated over all operations
- If rates sum to ~25-30% → calculated over all operations, correct rate is ~70-75%
- If rates sum to exactly 100% in `error_distribution` block → calculated over errors only

---

## 1.5 Task Definition: Model Types and Evaluation Framework

### Critical Context for Interpreting Results

This error analysis framework evaluates **multiple model architectures** performing **different ASR tasks** on Swiss German audio. Understanding these differences is essential for interpreting error patterns.

### Model Types and Tasks

#### Whisper Models: Speech Translation
**Task:** Cross-lingual Speech Translation  
**Input:** Swiss German audio (spoken dialectal German)  
**Output:** Standard German text (written standard)  
**Reference:** Standard German text
```
Audio (Swiss German): "ich gang hüt i d'stadt"
↓
Whisper Output (Standard German): "ich gehe heute in die stadt"
↓
Compared to Reference: "ich gehe heute in die stadt"
```

**Evaluation:** Measures translation quality from Swiss German dialects to Standard German.

---

#### Wav2Vec2 Models: Speech Recognition/Transcription
**Task:** Automatic Speech Recognition (with implicit normalisation pressure)  
**Input:** Swiss German audio (spoken dialectal German)  
**Output:** Mixed German text (dialectal features + Standard German elements)  
**Reference:** Standard German text
```
Audio (Swiss German): "ich gang hüt i d'stadt"
↓
Wav2Vec2 Output (Mixed): "ich gang heute in stadt"  // Preserves some dialect
↓
Compared to Reference: "ich gehe heute in die stadt"
```

**Evaluation:** Measures recognition accuracy AND implicitly measures "distance from Standard German."

---

### Why This Matters for Error Analysis

#### 1. Different Error Sources

**Whisper errors may indicate:**
- Translation failures (wrong lexical choice)
- Grammatical construction mismatches (tense/aspect)
- Valid translation variants counted as errors
- Conscious choice by the speaker to rephrase (dialectal influence or creative licence on the part of the speaker), not model error

**Wav2Vec2 errors may indicate:**
- Acoustic recognition failures (wrong phonemes heard)
- Dialectal transcription (correct Swiss German, wrong Standard German)
- Vocabulary gaps (model doesn't know Swiss German words)

#### 2. Metrics Interpretation Differs by Model Type

| Metric | Whisper Interpretation | Wav2Vec2 Interpretation |
|--------|----------------------|------------------------|
| **WER** | Translation quality to Standard German | Recognition accuracy + dialect normalisation |
| **Substitution** | Wrong word choice OR valid translation variant | Phonetic confusion OR dialectal lexicon |
| **Deletion** | Translation omission | Failed to hear/recognise word |
| **Insertion** | Over-generation/hallucination | Extra transcribed sounds |

#### 3. Confusion Pairs Reveal Different Patterns

**Whisper confusion pairs** (e.g., "war" → "ist...gewesen"):
- Often semantically valid (past tense vs. present perfect)
- Reflect translation choices, not recognition errors

**Wav2Vec2 confusion pairs** (e.g., "wurde" → "ist", "des" → "vom"):
- Reflect acoustic confusions in dialectal speech
- Indicate systematic phonetic recognition failures
- May preserve dialectal grammar (e.g., "gang" instead of "gehe")

#### 4. Performance Differences Explained

From actual results (FHNW corpus, 863 samples):

| Model | Mean WER | Primary Error Type | Task |
|-------|----------|-------------------|------|
| whisper-large-v2 | 28.00% | Substitution (18.98%) | Translation |
| whisper-large-v3 | 29.53% | Substitution (19.92%) | Translation |
| wav2vec2-german-with-lm | 75.28% | Substitution (55.18%) | Recognition |
| wav2vec2-1b-german-cv11 | 72.42% | Substitution (54.46%) | Recognition |

**Key Observation:** Wav2Vec2 models show 2.5-3x higher WER because:
1. They were not explicitly trained for Swiss German dialects
2. They attempt to transcribe dialectal pronunciation (not translate to Standard German)
3. Standard German references penalize correct dialectal transcriptions
4. They lack the multilingual pre-training that gives Whisper implicit translation capability

### Why This Framework Is Still Valid

Despite evaluating different tasks, the error analysis remains rigorous because:

1. **Consistent Reference Standard:** All models are evaluated against the same Standard German references, enabling direct comparison of "how close did each model get to Standard German output?"

2. **Complementary Insights:** Comparing translation vs. transcription models reveals:
  - Which approach (translate vs. recognise-then-normalise) works better for Swiss German
  - Whether dialectal pronunciation features are preserved or normalised
  - Relative difficulty of acoustic modelling vs. translation for low-resource dialects

3. **Practical Relevance:** Real-world Swiss German ASR applications need Standard German output regardless of the underlying approach, making this evaluation framework practically meaningful.

### Reference Material

For detailed discussion of different model behaviours:
- **Whisper translation:** Project Exposé, Plüss et al. (2023)
- **Wav2Vec2 recognition:** Analysis Notes (results/error_analysis/ANALYSIS_NOTES.md)
- **Human evaluation:** Project Exposé (4.36/5.0 meaning retention for Whisper)

---

## 2. Error Categorization Methodology

### 2.1 Alignment Algorithm

Word-level alignment between reference and hypothesis texts is computed using the `jiwer` library's `process_words()` function, which implements the Wagner-Fischer algorithm for edit distance with backtracking.
```python
# Core alignment extraction (from error_analyzer.py)
def get_alignment(self, reference: str, hypothesis: str) -> List[Dict[str, Optional[str]]]:
    # Normalise inputs using consistent preprocessing
    ref_norm = _normalize_text(reference)
    hyp_norm = _normalize_text(hypothesis)
    
    # Handle edge cases
    if not ref_norm and not hyp_norm:
        return []
    if not ref_norm:
        return [{'type': 'insertion', 'ref': None, 'hyp': w} for w in hyp_norm.split()]
    if not hyp_norm:
        return [{'type': 'deletion', 'ref': w, 'hyp': None} for w in ref_norm.split()]

    # Compute alignment via jiwer
    out = jiwer.process_words(ref_norm, hyp_norm)
    
    # Extract alignment chunks...
```

### 2.2 Error Type Definitions

| Error Type | Definition | Alignment Representation |
|------------|------------|--------------------------|
| **Correct** | Reference word matches hypothesis word | `{'type': 'correct', 'ref': 'word', 'hyp': 'word'}` |
| **Substitution** | Reference word replaced by different hypothesis word | `{'type': 'substitution', 'ref': 'word1', 'hyp': 'word2'}` |
| **Deletion** | Reference word missing from hypothesis | `{'type': 'deletion', 'ref': 'word', 'hyp': None}` |
| **Insertion** | Extra word in hypothesis not in reference | `{'type': 'insertion', 'ref': None, 'hyp': 'word'}` |

### 2.3 Text Normalisation

Before alignment, texts undergo normalisation via `_normalize_text()` from `src/evaluation/metrics.py`.

**Current Implementation (ASR-Fair Mode, Default):**
```python
def _normalize_text(text: str, mode: str = "asr_fair") -> str:
    text = text.lower()
    if mode == "asr_fair":
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())
```

Steps performed:
- Convert to lowercase
- Remove punctuation (in ASR-fair mode)
- Collapse multiple whitespace
- Strip leading/trailing whitespace

> **Correction Note (January 2026):** Prior to this commit, punctuation removal was documented but not implemented in the default normalisation. The original implementation only performed lowercase conversion and whitespace normalisation. This discrepancy was identified during cross-architecture reproducibility testing and corrected with the introduction of the `mode` parameter. Historical results (pre-January 2026) used what is now called "standard" mode.

**Note:** Normalisation is applied identically to both reference and hypothesis, ensuring consistent comparison across all model types.

### 2.4 Handling Complex Substitutions

When jiwer reports a substitution spanning multiple words (e.g., 2 reference words → 3 hypothesis words), the alignment logic pairs words sequentially and classifies residuals:

```python
elif type_ == 'substitute':
    max_len = max(len(ref_words), len(hyp_words))
    for i in range(max_len):
        r = ref_words[i] if i < len(ref_words) else None
        h = hyp_words[i] if i < len(hyp_words) else None
        
        if r and h:
            alignment.append({'type': 'substitution', 'ref': r, 'hyp': h})
        elif r:
            alignment.append({'type': 'deletion', 'ref': r, 'hyp': None})
        elif h:
            alignment.append({'type': 'insertion', 'ref': None, 'hyp': h})
```

---

## 3. Threshold Selection and Justification

### 3.1 Top Percent Threshold (Default: 10%)

The analysis extracts the **top 10%** of samples by WER for detailed inspection.

**Rationale:**

1. **Statistical relevance**: The 90th percentile of WER represents systematically problematic utterances rather than random variation.

2. **Practical inspection capacity**: For a typical test set of 1,000-5,000 samples, 10% yields 100-500 samples—sufficient for pattern identification while remaining manually reviewable.

3. **Error concentration hypothesis**: ASR errors often follow a long-tail distribution, where the worst-performing samples contribute disproportionately to the total error count.

4. **Literature precedent**: Similar thresholds used in ASR error analysis studies (see [References](#9-references)).

**Empirical Validation on FHNW Corpus:**

Analysis of the Swiss German FHNW corpus (863 test samples) demonstrates the effectiveness of the 10% threshold across model types:

| Model Type | Model | Total Samples | Top 10% Extracted | Mean WER (all) | Estimated Mean WER (worst 10%) |
|------------|-------|--------------|-------------------|----------------|-------------------------------|
| Translation | whisper-large-v2 | 863 | 86 | 28.00% | ~60-80% |
| Translation | whisper-large-v3 | 863 | 86 | 29.53% | ~65-85% |
| Recognition | wav2vec2-german-with-lm | 863 | 86 | 75.28% | ~95-120% |

**Key Finding:** The worst 10% of samples (86 utterances) provide sufficient diversity across dialects and error types for meaningful pattern analysis without overwhelming manual review capacity. This threshold is effective for both translation and recognition models.

**Configuration:**
```yaml
# error_analysis_config.yml
sampling:
  top_percent: 0.10  # 10% of worst samples
```

**Minimum Sample Guarantee:**

To ensure meaningful analysis even with small datasets, a minimum of 5 samples is always extracted:
```python
cutoff_index = int(len(samples) * top_percent)
cutoff_index = max(cutoff_index, min(5, len(samples)))
worst_samples = sorted_samples[:cutoff_index]
```

**Example:** For a test set of 30 samples, `0.10 * 30 = 3`, but the minimum guarantee ensures 5 samples are extracted.

### 3.2 Alternative: Absolute WER Threshold

For specific use cases, an absolute WER threshold can be configured:
```yaml
sampling:
  absolute_threshold: 60.0  # Only samples with WER > 60%
```

**Use Case:** When comparing models with different error distributions (e.g., Whisper at 28% WER vs. Wav2Vec2 at 75% WER), an absolute threshold provides a consistent error severity cutoff. For example, the top 10% of Whisper captures samples with ~60-80% WER, while the top 10% of Wav2Vec2 captures samples with ~95-120% WER. An absolute threshold of 60% ensures you're analyzing similarly difficult samples across model types.

---

## 4. Confusion Pattern Calculation

### 4.1 Confusion Pair Extraction

Confusion pairs identify systematic substitution patterns.
```python
def find_confusion_pairs(self, alignments: List[List[Dict]]) -> List[Tuple[Tuple[str, str], int]]:
    pairs = []
    for align in alignments:
        for item in align:
            if item['type'] == 'substitution':
                pairs.append((item['ref'], item['hyp']))
    
    return Counter(pairs).most_common()
```

### 4.2 Interpretation by Model Type

Confusion pairs are ranked by frequency. High-frequency pairs indicate different patterns depending on model type:

**Whisper (Translation) Confusion Pairs:**
- "wurde" → "ist" (tense/aspect variations)
- "war" → "ist...gewesen" (valid translation alternatives)
- "diese" → "die" (article/determiner choices)
- Often semantically similar or grammatically related

**Wav2Vec2 (Recognition) Confusion Pairs:**
- "des" → "vom" (genitive case misrecognition)
- "wurde" → "ist" (phonetic similarity in Swiss German)
- "den" → "der" (article confusion)
- "ich" → "i" (dialectal form preserved)
- Reflect acoustic/phonetic confusions in dialectal speech

### 4.3 Per-Dialect Analysis

Confusion pairs are computed separately for each dialect to identify dialect-specific patterns:
```python
for dialect, samples in by_dialect.items():
    alignments = [self.get_alignment(s['reference'], s['hypothesis']) for s in samples]
    analysis[dialect] = {
        # ...
        'top_confusions': self.find_confusion_pairs(alignments)[:10]
    }
```

**Example from Actual Results:**

Bern dialect (BE) confusion pairs for different model types:

| Whisper (Translation) | Wav2Vec2 (Recognition) |
|----------------------|------------------------|
| "wurde" → "ist" (4×) | "wurde" → "ist" (6×) |
| "diese" → "die" (3×) | "als" → "aus" (5×) |
| "war" → "ist" (2×) | "den" → "der" (5×) |

Both show similar high-frequency pairs, but Wav2Vec2 shows more acoustic confusions ("als" → "aus") while Whisper shows more grammatical variations.

---

## 5. Statistical Aggregation Methods

### 5.1 Error Rate Calculation

**Important:** This analysis uses **two different error rate calculation methods** depending on the context.

#### 5.1.1 Global Error Distribution (Script-Level)

The main analysis script (`analyze_errors.py`) calculates error rates as proportions of **all alignment operations** (including correct matches):
```python
# From analyze_errors.py
global_counts = {'substitution': 0, 'deletion': 0, 'insertion': 0, 'correct': 0}

# Count all operations
for align in global_alignments:
    counts = analyzer.categorize_errors(align)
    for k in global_counts:
        global_counts[k] += counts.get(k, 0)

# Calculate rates over ALL operations
total_ops = sum(global_counts.values())
global_dist = {
    k: (v / total_ops * 100) if total_ops > 0 else 0.0 
    for k, v in global_counts.items()
}
```

**Result Structure:**
```json
{
  "error_distribution_percent": {
    "substitution": 18.98,
    "deletion": 2.44,
    "insertion": 4.51,
    "correct": 74.07
  }
}
```

**Note:** Rates sum to 100% because denominator includes correct matches.

**Use Case:** High-level model comparison, understanding overall accuracy.

---

#### 5.1.2 Per-Dialect Error Distribution (Analyzer-Level)

The `ErrorAnalyzer` class calculates error rates as proportions of **errors only** (excluding correct matches):
```python
# From error_analyzer.py - analyze_by_dialect()
total_sub = sum(counts['substitution'] for counts in all_counts)
total_del = sum(counts['deletion'] for counts in all_counts)
total_ins = sum(counts['insertion'] for counts in all_counts)
total_cor = sum(counts['correct'] for counts in all_counts)

total_errs = total_sub + total_del + total_ins  # Note: excludes total_cor

analysis[dialect] = {
    'error_distribution': {
        'substitution': total_sub,
        'deletion': total_del,
        'insertion': total_ins,
        'correct': total_cor,
        'sub_rate': total_sub / total_errs if total_errs > 0 else 0.0,  # Over errors only
        'del_rate': total_del / total_errs if total_errs > 0 else 0.0,
        'ins_rate': total_ins / total_errs if total_errs > 0 else 0.0,
    }
}
```

**Result Structure:**
```json
{
  "BE": {
    "error_distribution": {
      "substitution": 245,
      "deletion": 25,
      "insertion": 58,
      "correct": 982,
      "sub_rate": 0.747,    // 245/(245+25+58) = 74.7% of errors are substitutions
      "del_rate": 0.076,    // 25/(245+25+58) = 7.6% of errors are deletions
      "ins_rate": 0.177     // 58/(245+25+58) = 17.7% of errors are insertions
    }
  }
}
```

**Note:** Rates sum to 100% because denominator excludes correct matches.

**Use Case:** Understanding error type composition—useful for diagnosing systematic model failures (e.g., "Does this model mostly substitute words, or does it delete them?").

---

#### 5.1.3 Rationale for Dual Methods

**Why two different calculations?**

1. **Global Distribution (with correct):**
   - Answers: "What percentage of all words are errors?"
   - Enables comparison: "Model A has 25% errors vs Model B's 75% errors"
   - Maps directly to WER (Word Error Rate)

2. **Error Type Distribution (errors only):**
   - Answers: "Among the errors, what types dominate?"
   - Enables diagnosis: "75% of errors are substitutions → phonetic confusion issues"
   - Independent of overall accuracy (useful for comparing error patterns across models with different WER)

**Example Interpretation:**
```json
// Global: "Out of 1000 words, 250 were errors (25% error rate)"
"error_distribution_percent": {
  "substitution": 18.75,  // 187.5 words
  "deletion": 3.75,       // 37.5 words  
  "insertion": 2.50,      // 25 words
  "correct": 75.00        // 750 words
}

// Per-Dialect: "Of those 250 errors, 75% were substitutions"
"error_distribution": {
  "sub_rate": 0.75,  // 187.5/250 = 75%
  "del_rate": 0.15,  // 37.5/250 = 15%
  "ins_rate": 0.10   // 25/250 = 10%
}
```

### 5.2 Aggregate Statistics
```python
def calculate_aggregate_stats(self, results: List[Dict]) -> Dict[str, float]:
    wers = [r['wer'] for r in results]
    cers = [r['cer'] for r in results]
    
    return {
        'mean_wer': statistics.mean(wers),
        'median_wer': statistics.median(wers),
        'std_wer': statistics.stdev(wers) if len(wers) > 1 else 0.0,
        'mean_cer': statistics.mean(cers),
        'median_cer': statistics.median(cers),
        'std_cer': statistics.stdev(cers) if len(cers) > 1 else 0.0,
    }
```

**Important:** WER and CER statistics are computed **per-sample**, not per-word. A 100-word utterance with 20% WER contributes one data point (20.0) to the mean calculation, not 100 data points.

### 5.3 Dialect-Level Aggregation

For each dialect, we compute:

| Metric | Description |
|--------|-------------|
| `sample_count` | Number of utterances |
| `mean_wer` | Average WER across samples |
| `std_wer` | Standard deviation of WER |
| `mean_cer` | Average CER across samples |
| `error_distribution` | Counts and rates for S/D/I (calculated over errors only) |
| `top_confusions` | Top 10 (ref, hyp) substitution pairs |

---

## 6. Annotated Alignment Examples

### 6.1 Readable Alignment Format

The `format_alignment_readable()` method produces human-inspectable alignments:
```
REF:  das    ist    ein    test
HYP:  das    isch   ei     test   extra
TYPE: C      S      S      C      I
```

Legend:
- **C** = Correct
- **S** = Substitution
- **D** = Deletion
- **I** = Insertion

### 6.2 Example Alignments by Model Type

This section provides examples for **both model types** to illustrate different error patterns.

---

#### Whisper Examples: Speech Translation

##### Example 1: Perfect Translation

**Swiss German Audio (spoken):** *"ich gang hüt i d'stadt"*  
**Standard German Reference:** "ich gehe heute in die stadt"  
**Whisper Hypothesis:** "ich gehe heute in die stadt"
```
REF:  ich    gehe   heute  in     die    stadt
HYP:  ich    gehe   heute  in     die    stadt
TYPE: C      C      C      C      C      C
```

**Error Counts:**
- Substitutions: 0
- Deletions: 0
- Insertions: 0
- Correct: 6

**WER:** 0% (perfect translation)

---

##### Example 2: Minor Translation Error

**Swiss German Audio (spoken):** *"ich gang hüt i d'stadt"*  
**Standard German Reference:** "ich gehe heute in die stadt"  
**Whisper Hypothesis:** "ich gehe heute in der stadt"
```
REF:  ich    gehe   heute  in     die    stadt
HYP:  ich    gehe   heute  in     der    stadt
TYPE: C      C      C      C      S      C
```

**Error Counts:**
- Substitutions: 1 ("die" → "der"; wrong article gender)
- Deletions: 0
- Insertions: 0
- Correct: 5

**WER:** 16.67% (1 error / 6 words)

**Analysis:** Whisper correctly translated the Swiss German lexical items (*"gang"* → *"gehe"*, *"hüt"* → *"heute"*, *"i d'"* → *"in die"*) but made a grammatical error with article gender.

---

##### Example 3: Deletion-Heavy Error (Whisper)

**Swiss German Audio (spoken):** *"mir gönd morn go schaffe"*  
**Standard German Reference:** "wir gehen morgen zur arbeit"  
**Whisper Hypothesis:** "wir gehen arbeit"
```
REF:  wir    gehen  morgen  zur    arbeit
HYP:  wir    gehen  *****   ***    arbeit
TYPE: C      C      D       D      C
```

**Error Counts:**
- Substitutions: 0
- Deletions: 2 ("morgen", "zur")
- Insertions: 0
- Correct: 3

**WER:** 40% (2 errors / 5 words)

**Analysis:** Whisper correctly identified key content words (*"mir"* → *"wir"*, *"gönd"* → *"gehen"*, *"schaffe"* → *"arbeit"*) but failed to translate the temporal marker *"morn"* → *"morgen"* and the directional preposition *"go"* → *"zur"*.

---

##### Example 4: Substitution with Semantic Drift (Whisper)

**Swiss German Audio (spoken):** *"das isch es guets buch gsi"*  
**Standard German Reference:** "das ist ein gutes buch gewesen"  
**Whisper Hypothesis:** "das war ein gutes buch"
```
REF:  das    ist    ein    gutes  buch   gewesen
HYP:  das    war    ein    gutes  buch   *******
TYPE: C      S      C      C      C      D
```

**Error Counts:**
- Substitutions: 1 ("ist" → "war"; present tense → past tense)
- Deletions: 1 ("gewesen" missing)
- Insertions: 0
- Correct: 4

**WER:** 33.33% (2 errors / 6 words)

**Analysis:** Whisper translated the Swiss German past perfect construction (*"isch...gsi"*) into German past tense (*"war"*) instead of the reference present perfect (*"ist...gewesen"*). While semantically similar, this is counted as errors by edit-distance metrics.

---

#### Wav2Vec2 Examples: Speech Recognition

##### Example 5: Dialectal Preservation (Wav2Vec2)

**Swiss German Audio (spoken):** *"ich gang hüt i d'stadt"*  
**Standard German Reference:** "ich gehe heute in die stadt"  
**Wav2Vec2 Hypothesis:** "ich gang heute in stadt"
```
REF:  ich    gehe   heute  in     die    stadt
HYP:  ich    gang   heute  in     ***    stadt
TYPE: C      S      C      C      D      C
```

**Error Counts:**
- Substitutions: 1 ("gehe" → "gang")
- Deletions: 1 ("die" missing)
- Insertions: 0
- Correct: 4

**WER:** 33.33% (2 errors / 6 words)

**Analysis:** Model preserved dialectal verb form "gang" instead of normalising to Standard German "gehe". Also failed to recognise article "die". This pattern shows Wav2Vec2 tends to transcribe what it acoustically hears rather than normalising to Standard German.

---

##### Example 6: Severe Recognition Errors (Wav2Vec2)

**Swiss German Audio (spoken):** *"das isch es guets buch gsi"*  
**Standard German Reference:** "das ist ein gutes buch gewesen"  
**Wav2Vec2 Hypothesis:** "das ich es guets buch"
```
REF:  das    ist    ein    gutes  buch   gewesen
HYP:  das    ich    es     guets  buch   *******
TYPE: C      S      S      S      C      D
```

**Error Counts:**
- Substitutions: 3 ("ist" → "ich", "ein" → "es", "gutes" → "guets")
- Deletions: 1 ("gewesen" missing)
- Insertions: 0
- Correct: 2

**WER:** 66.67% (4 errors / 6 words)

**Analysis:** Multiple recognition failures - "ist" → "ich" (phonetic confusion), "ein" → "es" (article error), preserved Swiss German "guets" instead of normalising to "gutes", failed to recognise "gewesen" completely. Shows Wav2Vec2's difficulty with Swiss German phonetics and lack of Standard German normalisation.

---

##### Example 7: Article Confusion (Wav2Vec2)

**Swiss German Audio (spoken):** *[Utterance with genitive case]*  
**Standard German Reference:** "... des jahres ..." (genitive article + noun)  
**Wav2Vec2 Hypothesis:** "... vom jahr ..." (dative preposition + noun)
```
REF:  ...  des    jahres  ...
HYP:  ...  vom    jahr    ...
TYPE: ...  S      S       ...
```

**Error Counts (fragment):**
- Substitutions: 2 ("des" → "vom", "jahres" → "jahr")

**Analysis:** Common Wav2Vec2 pattern - substitutes genitive construction with dative prepositional phrase. This appears frequently in confusion pairs ("des" → "vom" is a top confusion across multiple dialects). Indicates systematic difficulty with Swiss German case system.

---

##### Example 8: Catastrophic Wav2Vec2 Failure (Actual Result)

**Swiss German Audio (spoken):** *[Longer utterance with multiple clauses]*  
**Standard German Reference:** "allerdings sind diese ergebnisse umstritten" (7 words)  
**Wav2Vec2 Hypothesis:** "man muss aber auch sagen dass diese ergebnisse umstritten sind" (10 words)
```
REF:  allerdings  sind   diese  ergebnisse  umstritten  ***  ***   ***   ***
HYP:  man         muss   aber   auch        sagen       dass diese ergebnisse umstritten sind
TYPE: S           S      S      S           S           I    I     I     I
```

**Error Counts:**
- Substitutions: 5
- Deletions: 0
- Insertions: 4
- Correct: 0

**WER:** 129% (9 errors / 7 reference words = more errors than reference length)

**Analysis:** Complete semantic paraphrase - Wav2Vec2 produced a grammatically correct Standard German sentence with similar meaning but completely different wording. This illustrates the fundamental challenge: the model recognised the audio and generated plausible German text, but not the specific words in the reference. This is common in Swiss German ASR where the model has learned German language patterns but lacks Swiss-German-to-Standard-German translation capability.

---

#### Key Observations from Examples:

**Whisper (Translation Task):**
1. Successfully translates Swiss German → Standard German in most cases
2. Errors typically involve grammatical details (articles, tense)
3. Lower overall error rates (28-34% WER)
4. Human evaluators rate output highly (4.36/5.0 meaning retention)

**Wav2Vec2 (Recognition Task):**
1. Often preserves dialectal pronunciation features ("gang" vs "gehe")
2. Struggles with Standard German normalisation
3. Much higher error rates (72-75% WER) due to dialect/standard mismatch
4. May produce semantically valid German that differs from reference wording
5. Lacks translation capability - attempts transcription of dialectal speech
6. Shows systematic acoustic confusions ("des" → "vom", "den" → "der")

---

## 7. Limitations

### 7.1 Word-Level Granularity

Alignments operate at word level, not sub-word or phoneme level. This means:

- **Compound word errors** are treated as single units (e.g., "Bahnhof" → "Banhof" is one substitution, not a character-level edit)
- **Morphological variations** (e.g., "laufen" vs "läuft") are substitutions, not partial matches

### 7.2 Normalisation Effects

Text normalisation affects metric calculation differently depending on mode:

**Standard Mode** (lowercase + whitespace only):
- Preserves punctuation differences
- "Hallo!" vs "Hallo" are treated as different (punctuation counts as characters)
- May penalize CTC models that don't output punctuation

**ASR-Fair Mode** (lowercase + punctuation removal + whitespace):
- "Hallo!" vs "Hallo" are treated as identical
- "Berlin" → "berlin" (capitalization masked)
- Enables fair comparison between seq2seq (Whisper) and CTC (Wav2Vec2) models

### 7.3 Reference Quality Dependency

Error analysis assumes **gold-standard references**. In reality:

- Swiss German → Standard German references may have multiple valid translations
- Dialect transcriptions may have spelling variations
- References may contain annotation errors

### 7.4 Different Metric Meanings for Different Tasks

WER and CER have different interpretations depending on the model type:

**For Translation Models (Whisper):**
- WER measures translation quality
- May penalize valid translation alternatives
- Should be complemented with human evaluation

**For Recognition Models (Wav2Vec2):**
- WER measures recognition accuracy + implicit normalisation
- Penalizes correct dialectal transcriptions
- Does not distinguish between acoustic errors and dialect preservation

**Mitigation:** Model-specific interpretation guidelines provided in Section 1.5.

### 7.5 No Acoustic Feature Analysis

This analysis is purely text-based. It cannot identify:

- Signal-to-noise ratio effects
- Speaker accent variations
- Microphone quality issues

### 7.6 Cross-Task Comparison Challenges

Comparing models that perform different tasks (translation vs. recognition) requires careful interpretation:

- Higher WER in Wav2Vec2 does not necessarily mean "worse" - it may indicate dialectal preservation
- Lower WER in Whisper may mask translation errors that are semantically problematic
- Direct WER comparison should be supplemented with qualitative error analysis

---

### 7.7 Cross-Architecture Evaluation Variation

Evaluation metrics may vary slightly when computed on different GPU architectures:

| Architecture | Whisper Variation | Wav2Vec2 Variation |
|--------------|-------------------|---------------------|
| Same GPU, multiple runs | 0.000pp | 0.000pp |
| Ampere vs Blackwell | ~0.002pp | ~0.2pp |

**Implications:**
- Results from RTX 3090 and RTX 5090 are not directly comparable at high precision
- CTC-based models (Wav2Vec2) show 100× larger variation than seq2seq models (Whisper)
- Rankings and conclusions are unaffected (differences are below statistical significance)

**Recommendation:** Report hardware configuration in methodology; use single architecture for all evaluations.

**Reference:** See `docs/GPU_COMPATIBILITY.md` for detailed cross-architecture testing results.

---

### 7.8 Normalisation Mode Selection

This project supports two text normalisation modes that affect metric calculation:

#### Standard Mode
```python
def _normalize_text(text: str, mode: str = "standard") -> str:
    text = text.lower()
    # No punctuation removal
    return " ".join(text.split())
```
- Lowercases text, collapses whitespace
- **Preserves punctuation**
- Use for: Comparability with published benchmarks using standard WER

#### ASR-Fair Mode (Default)
```python
def _normalize_text(text: str, mode: str = "asr_fair") -> str:
    text = text.lower()
    import string
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())
```
- Lowercases text, **removes punctuation**, collapses whitespace
- Use for: Fair cross-architecture comparison (CTC vs seq2seq models)

#### Why This Matters

CTC-based models (Wav2Vec2, MMS) typically do not output punctuation, while seq2seq models (Whisper) do. Under standard normalisation, CTC models are penalized for missing punctuation they were never trained to produce.

**Empirical impact (863-sample test set):**

| Model | Standard WER | ASR-Fair WER | Δ |
|-------|--------------|--------------|---|
| whisper-large-v3-turbo | 30.94% | 28.23% | -2.71pp |
| wav2vec2-german-with-lm | 75.28% | 70.05% | -5.23pp |
| wav2vec2-1b-german-cv11 | 72.42% | 70.97% | -1.45pp |

CTC models with LM decoding benefit most from fair normalisation.

#### Recommendation

Report **both** metrics in research for transparency. Document which mode was used for each comparison.

---

## 8. Reproduction Guide

### 8.1 Prerequisites
```bash
# Required packages
pip install jiwer pandas tqdm psutil

# Verify jiwer version (tested with 3.0+)
python -c "import jiwer; print(jiwer.__version__)"
```

### 8.2 Input Format

Evaluation results must be JSON with this structure:
```json
{
  "model_name": "whisper-large-v3",
  "timestamp": "2025-12-03T11:29:24",
  "results": {
    "samples": [
      {
        "dialect": "BE",
        "reference": "das ist ein test",
        "hypothesis": "das isch ei test",
        "wer": 50.0,
        "cer": 25.0
      }
    ]
  }
}
```

Alternative flat structure also supported:
```json
{
  "model_name": "model-name",
  "samples": [...]
}
```

**Note:** The error analysis pipeline is **model-agnostic** and works with results from any ASR model (Whisper, Wav2Vec2, or custom models) as long as they follow the above format.

### 8.3 Running the Analysis

**Basic usage:**
```bash
python scripts/analyze_errors.py \
    --input_dir results/metrics \
    --output_dir results/error_analysis \
    --top_percent 0.1
```

**Using configuration file:**
```bash
# Edit configs as needed
cat results/error_analysis/error_analysis_config.yml

# Run with defaults (reads from config)
python scripts/analyze_errors.py
```

### 8.4 Output Files

| File | Description |
|------|-------------|
| `analysis_<model>.json` | Full hierarchical analysis |
| `worst_samples_<model>.csv` | Flat CSV of high-error samples |
| `model_comparison_summary.json` | Cross-model metrics comparison |

### 8.5 Example Analysis Output
```json
{
  "meta": {
    "model_name": "whisper-large-v3",
    "source_file": "whisper-large-v3_results.json",
    "total_samples": 863
  },
  "global_metrics": {
    "mean_wer": 29.53,
    "median_wer": 25.20,
    "std_wer": 22.97
  },
  "error_distribution_percent": {
    "substitution": 19.92,
    "deletion": 2.12,
    "insertion": 5.14,
    "correct": 72.82
  },
  "dialect_analysis": {
    "BE": {
      "sample_count": 203,
      "mean_wer": 28.50,
      "error_distribution": {
        "sub_rate": 0.717,
        "del_rate": 0.061,
        "ins_rate": 0.222
      },
      "top_confusions": [
        [["wurde", "ist"], 4],
        [["diese", "die"], 3]
      ]
    }
  }
}
```

### 8.6 Memory Considerations

For large result files (>10,000 samples), monitor memory usage:
```python
# Built-in memory logging (requires psutil)
# Logs every 200 samples during alignment computation
```

Expected memory: ~500MB for 10,000 samples with full alignment computation.

---

## 9. References

### ASR Error Analysis Literature

1. **Jurafsky, D., & Martin, J. H. (2024).** *Speech and Language Processing* (3rd ed., draft). Chapter 15: Automatic Speech Recognition.
   - Standard reference for WER computation and error analysis methodology.

2. **Manohar, V., et al. (2018).** "A Systematic Comparison of Grapheme and Phoneme-Based Speech Recognition." *ICASSP*.
   - Discusses error analysis methodologies for comparing ASR systems.

3. **Park, D. S., et al. (2019).** "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition." *Interspeech*.
   - Uses similar error categorization for evaluating augmentation effects.

### Swiss German ASR

4. **Scherrer, Y., & Rambow, O. (2010).** "Word-based dialect identification with georeferenced rules." *EMNLP*.
   - Background on Swiss German dialect variation relevant to error patterns.

5. **Plüss, M., et al. (2021).** "Swiss German ASR: Developing Resources and Models."
   - Related work on Swiss German speech recognition challenges.

6. **Plüss, M., et al. (2023).** "SDS-200: A Swiss German Speech to Standard German Text Corpus."
   - Defines the translation task and provides human evaluation benchmarks for Whisper.
   - Key finding: Whisper achieves 4.36/5.0 meaning retention and 4.39/5.0 fluency ratings from native speakers.

### Technical References

7. **jiwer library documentation:** https://github.com/jitsi/jiwer
   - Implementation details for word/character error rate computation.

8. **Baevski, A., et al. (2020).** "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." *NeurIPS*.
   - Base model architecture for Wav2Vec2 models evaluated in this project.

---

## Appendix A: Complete Error Type Counting Code
```python
def categorize_errors(self, alignment: List[Dict[str, Optional[str]]]) -> Dict[str, int]:
    """
    Count occurrences of each error type in an alignment.
    
    Args:
        alignment: Output from get_alignment()
        
    Returns:
        Dictionary with counts for correct, substitution, deletion, 
        insertion, and total_errors.
    """
    counts = Counter(item['type'] for item in alignment)
    return {
        'correct': counts['correct'],
        'substitution': counts['substitution'],
        'deletion': counts['deletion'],
        'insertion': counts['insertion'],
        'total_errors': counts['substitution'] + counts['deletion'] + counts['insertion']
    }
```

---

## Appendix B: Configuration Schema
```yaml
# error_analysis_config.yml - Full schema with defaults

paths:
  input_dir: "metrics"          # Relative to RESULTS_DIR
  output_dir: "analysis"        # Relative to RESULTS_DIR

sampling:
  top_percent: 0.10             # Float in [0, 1]
  # absolute_threshold: null    # Optional: WER threshold (0-100)

analysis:
  dialect_breakdown: true       # Compute per-dialect stats
  error_distribution: true      # Compute S/D/I rates
  top_confusion_pairs: 10       # Number of pairs to extract

output:
  save_json: true               # Detailed analysis
  save_csv: true                # Worst samples flat format
  comparison_summary: true      # Cross-model comparison

logging:
  level: "INFO"                 # DEBUG, INFO, WARNING, ERROR
  log_to_file: false
  log_filename: "error_analysis.log"
```

---

**Document Version:** 2.1  
**Last Updated:** 2026-01-05  
**Authors:** Till Ermold  
**Revisions:** 
- Documented normalisation mode changes in Section 7.8 and their impact on WER calculations.