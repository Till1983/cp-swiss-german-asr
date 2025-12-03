```markdown
# Error Analysis Notes

## Session: 2024-12-03

### Overview

This document contains qualitative observations and patterns from the ASR error analysis conducted on Swiss German dialect data. Observations are organized by model and dialect, with links to specific samples for reference.

---

## Model-Specific Observations

### Whisper Large v3

**Systematic Patterns:**
- **Verb restructuring to perfect tense:** Model frequently converts simple past/present tense to Swiss German-style perfect tense constructions (e.g., "war" → "ist...gewesen", "verdiente" → "hat...verdient", "gehörte" → "hat...gehört")
- **Article insertions:** Frequent insertion of articles ("der", "die", "das") at sentence beginnings where reference lacks them
- **Word order changes:** Model often restructures sentences with different word order, particularly moving verbs to end position (e.g., "verlor...Selbstständigkeit" → "hat...Selbstständigkeit verloren")
- **Compound word splitting:** Hyphenated/compound words sometimes split (e.g., "landsberg-velen" → "landsberg velen", "atom-u-boot" → "atom-aubau")
- **High insertion rate in BE/SO/SG dialects:** BE (20.5%), SO (25.7%), SG (23.1%) insertion rates significantly higher than GR (5%) and GL (0%)

**Dialect Performance:**
| Dialect | Mean WER | Sample Count | Notable Pattern |
|---------|----------|--------------|-----------------|
| ZG | 46.3% | 30 | Highest WER, many restructuring errors |
| SO | 34.8% | 36 | High variance (std: 41.0), highest insertion rate |
| BE | 31.7% | 203 | Most samples, consistent restructuring patterns |
| GR | 17.2% | 12 | Lowest WER, minimal insertions |
| GL | 10.7% | 6 | Best performance, no insertions |

**Questions:**
- Does Whisper normalize Swiss German to Standard German spellings?
    - **Answer:** Yes, but in reverse - the model appears to produce Swiss German-influenced output (perfect tense constructions, different word order) when processing Swiss German audio, even when the reference is in Standard German form

**Hypotheses:**
- The model has learned Swiss German speech patterns and generates output reflecting Swiss German syntax (verb-final, analytic perfect tense) rather than Standard German text conventions
- Higher error rates in ZG/SO/BE may reflect more distinct dialectal phonology that triggers non-standard output structures
- Low error rates in GR/GL suggest these dialects may be closer to training data phonology or have less distinctive prosodic patterns
- Many "errors" may actually be semantically correct paraphrases in Swiss German style rather than true recognition failures (e.g., "Trotz dieses Wachstums verlor der Ort..." → "...hat der Ort...verloren" preserves meaning)

**Worst Samples to Review:**
| Priority | File | Dialect | WER | Issue |
|----------|------|---------|-----|-------|
| High | `699a7e10-2912-427f-9947-8c73e60394c1.flac` | SO | 200% | Complete semantic paraphrase |
| High | `0bfb5d1c-e804-424c-a796-9efe12a8e390.flac` | BE | 150% | Compound word fusion ("Mal Etappenort" → "MAU Etappe Ort") |
| High | `ae929da4-e560-4812-b0a6-a9ddc74d50ce.flac` | BE | 146% | Massive sentence restructuring |
| High | `8708ff28-66f0-4445-82a5-b14fb55f21cb.flac` | ZH | 133% | Perfect tense + article insertion |
| High | `68974798-4b20-4a31-89e2-eb3a913e79f6.flac` | ZG | 100% | Complete word order inversion |
| Medium | `d3535ea7-e4cc-40b9-8e56-88ce68bdac27.flac` | BE | 100% | Perfect tense conversion ("wuchs auf" → "ist aufgewachsen") |
| Medium | `1cccaffd-b3ea-442d-adea-85be608c4883.flac` | UR | 100% | Dialectal phonetic transcription |
| Medium | `254ad4f9-bf85-4b08-a6d6-1d324d53300e.flac` | ZH | 100% | Sentence structure inversion |
| Medium | `9143a06d-8892-42a0-82c9-a27c0b24319a.flac` | BE | 100% | Catastrophic failure ("Kopf hält dir") |
| Medium | `332d2e69-5674-4dcb-bb80-9900901a5ca8.flac` | ZG | 100% | Article insertion + compound handling |

---

### Whisper Large v3 Turbo

**Systematic Patterns:**
- **Perfect tense restructuring (identical to v3):** Consistent conversion of simple past to analytic perfect tense (e.g., "verlor" → "hat...verloren", "wuchs auf" → "ist aufgewachsen")
- **Semantic paraphrasing:** Produces semantically equivalent but structurally different output (e.g., "Allerdings sind diese Ergebnisse umstritten" → "Man muss aber auch sagen, dass die Ergebnisse sehr umstritten sind" - 200% WER)
- **Swiss German phonetic transcription:** Sometimes transcribes dialectal pronunciation rather than Standard German (e.g., "Wohin führt dieser?" → "Wo hi fährt der?", "Aber woran liegt das?" → "A prvo roll ist das.")
- **Article insertions at sentence start:** Frequent "der", "die", "das" insertions (e.g., "Landsberg-Velen war..." → "Der Landsberg Wehlen war ein...")
- **Compound word splitting:** Similar to v3 (e.g., "Ventimiglia-Sanremo" → "Ventimilia San Remo")
- **Occasional code-switching to English:** Rare cases of English output (e.g., "Späth Orgelbau erbaut" → "Spread all about")

**Dialect Performance:**
| Dialect | Mean WER | Sample Count | Insertion Rate | Notable Pattern |
|---------|----------|--------------|----------------|-----------------|
| GL | 10.7% | 6 | 0% | Best performance, clean output |
| GR | 18.8% | 12 | 13.6% | Low errors, minimal restructuring |
| ZH | 28.2% | 144 | 16.1% | Moderate restructuring |
| BE | 32.2% | 203 | 20.9% | High perfect tense conversion |
| SG | 30.5% | 116 | 22.3% | High insertion rate |
| SO | 36.0% | 36 | 22.7% | Highest variance (std: 38.4) |
| ZG | 41.6% | 30 | 16.7% | Worst WER, heavy restructuring |

**Questions:**
- How does speed optimization affect Swiss German recognition?
    - **Answer:** Minimal impact - only +1.4% WER increase compared to v3. Error distribution proportions remain nearly identical (sub: 21.2% vs 19.9%, ins: 5.3% vs 5.1%, del: 2.3% vs 2.1%). The Turbo optimization does not fundamentally change Swiss German processing patterns.

**Hypotheses:**
- Speed-accuracy tradeoff is linear: Turbo's efficiency gains from reduced decoder complexity cause uniform ~10% relative increase across all error types
- Same Swiss German training data influence as v3: Model generates output reflecting Swiss German syntax when processing dialectal audio
- Many high-WER samples preserve semantics: BLEU scores remain non-zero even for 100%+ WER samples, indicating paraphrasing rather than recognition failure
- Dialect phonology determines error rate: GL/GR (low WER) may have phonology closer to Standard German training data; ZG/SO (high WER) have more distinctive features

**Worst Samples to Review:**
| Priority | File | Dialect | WER | Issue |
|----------|------|---------|-----|-------|
| High | `699a7e10-2912-427f-9947-8c73e60394c1.flac` | SO | 200% | Complete semantic paraphrase |
| High | `ae929da4-e560-4812-b0a6-a9ddc74d50ce.flac` | BE | 146% | Massive sentence restructuring |
| High | `49a11f9d-d3cc-439a-a56c-5e585550c754.flac` | UR | 133% | Dialectal phonetic transcription ("Wohin" → "Wo hi") |
| High | `98b2381e-bde4-49b2-97e3-5b867a9be815.flac` | SZ | 125% | Dialectal transcription ("Aber woran" → "A prvo roll") |
| High | `63daf2e7-9d35-48aa-b718-3e45e7e51a96.flac` | SG | 114% | Perfect tense + restructuring |
| Medium | `68974798-4b20-4a31-89e2-eb3a913e79f6.flac` | ZG | 107% | Complete word order inversion |
| Medium | `d3535ea7-e4cc-40b9-8e56-88ce68bdac27.flac` | BE | 100% | Perfect tense conversion |
| Medium | `3950d3dc-4b1f-4de6-8742-aa9306c85f65.flac` | BL | 100% | Code-switch to English ("Spread all about") |
| Medium | `acaaa975-240e-4265-9be2-7bd80b2d9398.flac` | AG | 100% | Perfect tense + restructuring |
| Medium | `633f3365-3884-462d-aa3a-fbfb12fb2202.flac` | LU | 100% | Catastrophic failure ("Hut Langetasnummer") |

---

### Whisper Large v2

**Systematic Patterns:**
- **Best overall Whisper performance:** Mean WER 28.0% (vs v3: 29.5%, v3-Turbo: 30.9%) - surprisingly outperforms newer versions on Swiss German
- **Same perfect tense restructuring:** Converts simple past to analytic perfect tense (e.g., "unterschrieb" → "hat...unterschrieben", "arbeitete" → "hat...gearbeitet")
- **Semantic paraphrasing identical to v3:** Same complete restructuring pattern (e.g., "Allerdings sind diese Ergebnisse umstritten" → "Man muss aber auch sagen..." - 200% WER)
- **Compound word handling:** Both fusion ("Mal Etappenort" → "Maletappenort") and splitting ("Landsberg-Velen" → "Landsberg Wählen")
- **Word order inversion:** Entire sentences restructured while preserving meaning (e.g., "Im Norden liegt die Bandasee" → "Der Bandasee ist im Norden")
- **Swiss spelling conventions:** Converts ß to ss (e.g., "Außerdem" → "Ausserdem", "Bußmann" → "Bussmann")
- **Lower insertion rate than v3/Turbo:** 4.5% vs 5.1-5.3%, suggesting slightly more conservative decoding

**Dialect Performance:**
| Dialect | Mean WER | Sample Count | Insertion Rate | Notable Pattern |
|---------|----------|--------------|----------------|-----------------|
| GL | 5.8% | 6 | 0% | Best performance, minimal errors |
| GR | 11.3% | 12 | 0% | Very low errors, no insertions |
| SZ | 14.6% | 9 | 0% | Low WER, no insertions |
| UR | 21.2% | 15 | 3.8% | Lower than v3/Turbo |
| ZH | 23.5% | 144 | 14.6% | Moderate restructuring |
| BL | 26.6% | 54 | 16.7% | Moderate errors |
| AG | 27.8% | 108 | 15.8% | Consistent performance |
| LU | 28.9% | 51 | 16.4% | Near average |
| SG | 29.2% | 116 | 20.7% | High insertion rate |
| BE | 29.9% | 203 | 19.4% | Most samples, consistent patterns |
| VS | 31.0% | 17 | 5.6% | Low insertions despite high WER |
| SO | 34.9% | 36 | 24.3% | Highest insertion rate |
| FR | 37.0% | 7 | 16.7% | Limited samples |
| ZG | 39.7% | 30 | 17.8% | Highest WER, heavy restructuring |

**Questions:**
- Why does v2 outperform v3 and v3-Turbo on Swiss German?
    - **Answer:** Possibly due to differences in training data or decoder architecture. V2 may have been trained on more German dialectal data, or v3's larger training set may have introduced more Standard German bias. The lower insertion rate (4.5% vs 5.1-5.3%) suggests v2 is more conservative in generating additional words.

**Hypotheses:**
- V2's training data may have included more Swiss German or dialectal content, leading to better adaptation
- V3/Turbo's expanded multilingual training may have diluted German dialect recognition
- The consistent pattern of semantic paraphrasing across all Whisper versions suggests this is a fundamental model behavior, not version-specific
- Lower insertion rates in v2 correlate with better WER, suggesting over-generation is a key error source in v3/Turbo
- Dialects with 0% insertion rate (GL, GR, SZ) consistently achieve best performance across all Whisper versions

**Worst Samples to Review:**
| Priority | File | Dialect | WER | Issue |
|----------|------|---------|-----|-------|
| High | `699a7e10-2912-427f-9947-8c73e60394c1.flac` | SO | 200% | Complete semantic paraphrase (identical to v3/Turbo) |
| High | `3d9f579c-8c8d-4b3f-b7de-a8ba97c7c77d.flac` | BE | 150% | Perfect tense + word splitting ("Dreijahresvertrag" → "drei Jahre lang...Vertrag") |
| High | `ae929da4-e560-4812-b0a6-a9ddc74d50ce.flac` | BE | 138% | Massive restructuring with name confusion ("Trencavel" → "Trencabel hörte") |
| High | `332d2e69-5674-4dcb-bb80-9900901a5ca8.flac` | ZG | 133% | Compound handling + article insertion |
| High | `8708ff28-66f0-4445-82a5-b14fb55f21cb.flac` | ZH | 133% | Perfect tense + article insertion (identical pattern to v3) |
| Medium | `11e2d4d7-b7f6-4b78-a9e7-261731da2c58.flac` | BL | 122% | Complex restructuring with insertions |
| Medium | `e483cfbd-e8b4-4b3a-ba2f-dbada24ab04f.flac` | BE | 110% | Compound splitting + phonetic confusion ("Ventimiglia-Sanremo" → "Milia san Remo") |
| Medium | `5da5e669-2f95-4efa-862b-b6cb3bead8f7.flac` | BE | 109% | Multiple substitutions + perfect tense |
| Medium | `68974798-4b20-4a31-89e2-eb3a913e79f6.flac` | ZG | 107% | Complete word order inversion (same as v3/Turbo) |
| Medium | `3950d3dc-4b1f-4de6-8742-aa9306c85f65.flac` | BL | 100% | Compound fusion + phonetic confusion ("Späth Orgelbau" → "Spätogelbau abbaut") |

---

### Whisper Large v3 Turbo

**Systematic Patterns:**
- **Perfect tense restructuring (identical to v3):** Consistent conversion of simple past to analytic perfect tense (e.g., "verlor" → "hat...verloren", "wuchs auf" → "ist aufgewachsen")
- **Semantic paraphrasing:** Produces semantically equivalent but structurally different output (e.g., "Allerdings sind diese Ergebnisse umstritten" → "Man muss aber auch sagen, dass die Ergebnisse sehr umstritten sind" - 200% WER)
- **Swiss German phonetic transcription:** Sometimes transcribes dialectal pronunciation rather than Standard German (e.g., "Wohin führt dieser?" → "Wo hi fährt der?", "Aber woran liegt das?" → "A prvo roll ist das.")
- **Article insertions at sentence start:** Frequent "der", "die", "das" insertions (e.g., "Landsberg-Velen war..." → "Der Landsberg Wehlen war ein...")
- **Compound word splitting:** Similar to v3 (e.g., "Ventimiglia-Sanremo" → "Ventimilia San Remo")
- **Occasional code-switching to English:** Rare cases of English output (e.g., "Späth Orgelbau erbaut" → "Spread all about")

**Dialect Performance:**
| Dialect | Mean WER | Sample Count | Insertion Rate | Notable Pattern |
|---------|----------|--------------|----------------|-----------------|
| GL | 10.7% | 6 | 0% | Best performance, clean output |
| GR | 18.8% | 12 | 13.6% | Low errors, minimal restructuring |
| ZH | 28.2% | 144 | 16.1% | Moderate restructuring |
| BE | 32.2% | 203 | 20.9% | High perfect tense conversion |
| SG | 30.5% | 116 | 22.3% | High insertion rate |
| SO | 36.0% | 36 | 22.7% | Highest variance (std: 38.4) |
| ZG | 41.6% | 30 | 16.7% | Worst WER, heavy restructuring |

**Questions:**
- How does speed optimization affect Swiss German recognition?
    - **Answer:** Minimal impact - only +1.4% WER increase compared to v3. Error distribution proportions remain nearly identical (sub: 21.2% vs 19.9%, ins: 5.3% vs 5.1%, del: 2.3% vs 2.1%). The Turbo optimization does not fundamentally change Swiss German processing patterns.

**Hypotheses:**
- Speed-accuracy tradeoff is linear: Turbo's efficiency gains from reduced decoder complexity cause uniform ~10% relative increase across all error types
- Same Swiss German training data influence as v3: Model generates output reflecting Swiss German syntax when processing dialectal audio
- Many high-WER samples preserve semantics: BLEU scores remain non-zero even for 100%+ WER samples, indicating paraphrasing rather than recognition failure
- Dialect phonology determines error rate: GL/GR (low WER) may have phonology closer to Standard German training data; ZG/SO (high WER) have more distinctive features

**Worst Samples to Review:**
| Priority | File | Dialect | WER | Issue |
|----------|------|---------|-----|-------|
| High | `699a7e10-2912-427f-9947-8c73e60394c1.flac` | SO | 200% | Complete semantic paraphrase |
| High | `ae929da4-e560-4812-b0a6-a9ddc74d50ce.flac` | BE | 146% | Massive sentence restructuring |
| High | `49a11f9d-d3cc-439a-a56c-5e585550c754.flac` | UR | 133% | Dialectal phonetic transcription ("Wohin" → "Wo hi") |
| High | `98b2381e-bde4-49b2-97e3-5b867a9be815.flac` | SZ | 125% | Dialectal transcription ("Aber woran" → "A prvo roll") |
| High | `63daf2e7-9d35-48aa-b718-3e45e7e51a96.flac` | SG | 114% | Perfect tense + restructuring |
| Medium | `68974798-4b20-4a31-89e2-eb3a913e79f6.flac` | ZG | 107% | Complete word order inversion |
| Medium | `d3535ea7-e4cc-40b9-8e56-88ce68bdac27.flac` | BE | 100% | Perfect tense conversion |
| Medium | `3950d3dc-4b1f-4de6-8742-aa9306c85f65.flac` | BL | 100% | Code-switch to English ("Spread all about") |
| Medium | `acaaa975-240e-4265-9be2-7bd80b2d9398.flac` | AG | 100% | Perfect tense + restructuring |
| Medium | `633f3365-3884-462d-aa3a-fbfb12fb2202.flac` | LU | 100% | Catastrophic failure ("Hut Langetasnummer") |

---

### Whisper Large v2

**Systematic Patterns:**
- **Best overall Whisper performance:** Mean WER 28.0% (vs v3: 29.5%, v3-Turbo: 30.9%) - surprisingly outperforms newer versions on Swiss German
- **Same perfect tense restructuring:** Converts simple past to analytic perfect tense (e.g., "unterschrieb" → "hat...unterschrieben", "arbeitete" → "hat...gearbeitet")
- **Semantic paraphrasing identical to v3:** Same complete restructuring pattern (e.g., "Allerdings sind diese Ergebnisse umstritten" → "Man muss aber auch sagen..." - 200% WER)
- **Compound word handling:** Both fusion ("Mal Etappenort" → "Maletappenort") and splitting ("Landsberg-Velen" → "Landsberg Wählen")
- **Word order inversion:** Entire sentences restructured while preserving meaning (e.g., "Im Norden liegt die Bandasee" → "Der Bandasee ist im Norden")
- **Swiss spelling conventions:** Converts ß to ss (e.g., "Außerdem" → "Ausserdem", "Bußmann" → "Bussmann")
- **Lower insertion rate than v3/Turbo:** 4.5% vs 5.1-5.3%, suggesting slightly more conservative decoding

**Dialect Performance:**
| Dialect | Mean WER | Sample Count | Insertion Rate | Notable Pattern |
|---------|----------|--------------|----------------|-----------------|
| GL | 5.8% | 6 | 0% | Best performance, minimal errors |
| GR | 11.3% | 12 | 0% | Very low errors, no insertions |
| SZ | 14.6% | 9 | 0% | Low WER, no insertions |
| UR | 21.2% | 15 | 3.8% | Lower than v3/Turbo |
| ZH | 23.5% | 144 | 14.6% | Moderate restructuring |
| BL | 26.6% | 54 | 16.7% | Moderate errors |
| AG | 27.8% | 108 | 15.8% | Consistent performance |
| LU | 28.9% | 51 | 16.4% | Near average |
| SG | 29.2% | 116 | 20.7% | High insertion rate |
| BE | 29.9% | 203 | 19.4% | Most samples, consistent patterns |
| VS | 31.0% | 17 | 5.6% | Low insertions despite high WER |
| SO | 34.9% | 36 | 24.3% | Highest insertion rate |
| FR | 37.0% | 7 | 16.7% | Limited samples |
| ZG | 39.7% | 30 | 17.8% | Highest WER, heavy restructuring |

**Questions:**
- Why does v2 outperform v3 and v3-Turbo on Swiss German?
    - **Answer:** Possibly due to differences in training data or decoder architecture. V2 may have been trained on more German dialectal data, or v3's larger training set may have introduced more Standard German bias. The lower insertion rate (4.5% vs 5.1-5.3%) suggests v2 is more conservative in generating additional words.

**Hypotheses:**
- V2's training data may have included more Swiss German or dialectal content, leading to better adaptation
- V3/Turbo's expanded multilingual training may have diluted German dialect recognition
- The consistent pattern of semantic paraphrasing across all Whisper versions suggests this is a fundamental model behavior, not version-specific
- Lower insertion rates in v2 correlate with better WER, suggesting over-generation is a key error source in v3/Turbo
- Dialects with 0% insertion rate (GL, GR, SZ) consistently achieve best performance across all Whisper versions

**Worst Samples to Review:**
| Priority | File | Dialect | WER | Issue |
|----------|------|---------|-----|-------|
| High | `699a7e10-2912-427f-9947-8c73e60394c1.flac` | SO | 200% | Complete semantic paraphrase (identical to v3/Turbo) |
| High | `3d9f579c-8c8d-4b3f-b7de-a8ba97c7c77d.flac` | BE | 150% | Perfect tense + word splitting ("Dreijahresvertrag" → "drei Jahre lang...Vertrag") |
| High | `ae929da4-e560-4812-b0a6-a9ddc74d50ce.flac` | BE | 138% | Massive restructuring with name confusion ("Trencavel" → "Trencabel hörte") |
| High | `332d2e69-5674-4dcb-bb80-9900901a5ca8.flac` | ZG | 133% | Compound handling + article insertion |
| High | `8708ff28-66f0-4445-82a5-b14fb55f21cb.flac` | ZH | 133% | Perfect tense + article insertion (identical pattern to v3) |
| Medium | `11e2d4d7-b7f6-4b78-a9e7-261731da2c58.flac` | BL | 122% | Complex restructuring with insertions |
| Medium | `e483cfbd-e8b4-4b3a-ba2f-dbada24ab04f.flac` | BE | 110% | Compound splitting + phonetic confusion ("Ventimiglia-Sanremo" → "Milia san Remo") |
| Medium | `5da5e669-2f95-4efa-862b-b6cb3bead8f7.flac` | BE | 109% | Multiple substitutions + perfect tense |
| Medium | `68974798-4b20-4a31-89e2-eb3a913e79f6.flac` | ZG | 107% | Complete word order inversion (same as v3/Turbo) |
| Medium | `3950d3dc-4b1f-4de6-8742-aa9306c85f65.flac` | BL | 100% | Compound fusion + phonetic confusion ("Späth Orgelbau" → "Spätogelbau abbaut") |

---

### Whisper Medium

**Systematic Patterns:**
- **Highest insertion rate among Whisper models:** 6.4% (vs v2: 4.5%, v3: 5.1%, Turbo: 5.3%) - more prone to generating extra words
- **Hallucination/repetition loops:** Catastrophic failure case with 469% WER where model repeated the same sentence 5+ times ("In der Vergangenheit wurden Temples...")
- **Same perfect tense restructuring:** Converts simple past to analytic perfect tense (e.g., "absolvierte" → "hat...absolviert", "gehörte" → "hat...gehört")
- **Semantic paraphrasing identical to larger models:** Same 200% WER sample ("Allerdings sind diese Ergebnisse umstritten" → "Man muss aber auch sagen...")
- **Swiss German phonetic transcription:** More frequent than larger models (e.g., "Sie fehlen im Mittelmeer" → "Se fala e me tu meir" - 150% WER)
- **Word order inversion:** Same pattern as v2/v3 (e.g., "Im Norden liegt die Bandasee" → "Der Bandasee ist im Norden")
- **Compound word handling:** Both fusion ("Mal Etappenort" → "MAU-Etappe ORT") and splitting ("Landsberg-Velen" → "Landsberg Wählen")
- **Complete sentence deletion:** Rare but present (e.g., "Aber woran liegt das?" → "" - 100% WER with 4 deletions)
- **Higher variance than larger models:** std_wer 30.1% vs v2: 25.1%, v3: 25.7%

**Dialect Performance:**
| Dialect | Mean WER | Sample Count | Insertion Rate | Notable Pattern |
|---------|----------|--------------|----------------|-----------------|
| GL | 15.0% | 6 | 42.9% | Best performance but high insertion rate |
| GR | 22.0% | 12 | 11.5% | Low errors, low insertions |
| ZH | 30.3% | 144 | 15.8% | Near average |
| UR | 30.5% | 15 | 12.8% | Moderate errors |
| SG | 33.0% | 116 | 21.3% | High insertion rate |
| VS | 33.6% | 17 | 11.1% | Low insertions despite moderate WER |
| BE | 35.2% | 203 | 21.2% | Most samples, high insertions |
| BL | 35.2% | 54 | 17.1% | Moderate errors |
| LU | 35.0% | 51 | 18.2% | Near average |
| TG | 35.3% | 50 | 17.2% | Consistent errors |
| SO | 36.0% | 36 | 22.5% | High insertion rate |
| AG | 36.8% | 108 | 30.2% | Highest insertion rate, includes hallucination case |
| ZG | 38.3% | 30 | 16.7% | High WER |
| SZ | 40.8% | 9 | 10.7% | Highest WER, low insertions (includes deletion case) |
| FR | 44.0% | 7 | 13.9% | Limited samples, worst dialect |

**Questions:**
- Does reduced model capacity cause more hallucinations?
    - **Answer:** Yes, the Medium model shows a unique catastrophic failure pattern (469% WER) with repetition loops not seen in larger models. This suggests the smaller decoder has less robust stopping criteria.
- How does Medium compare to the Large models overall?
    - **Answer:** Medium has +6.1% WER compared to v2 (34.1% vs 28.0%), with the gap primarily driven by higher insertion rates and occasional hallucinations. The core error patterns (perfect tense, word order) remain identical.

**Hypotheses:**
- Reduced model capacity leads to less stable decoding, causing repetition loops and hallucinations
- The 6.4% insertion rate (vs 4.5-5.3% in Large models) indicates the smaller model is more prone to over-generation
- Swiss German phonetic transcription errors are more severe in Medium (e.g., "Sie fehlen" → "Se fala") suggesting weaker acoustic modeling
- Despite capacity reduction, the fundamental Swiss German syntax patterns (perfect tense, word order) are preserved, indicating these are learned at the architecture level
- Complete deletion failures (100% WER with all deletions) suggest occasional audio processing failures unique to Medium
- AG dialect's 30.2% insertion rate (highest) may indicate specific speaker characteristics triggering hallucinations

**Worst Samples to Review:**
| Priority | File | Dialect | WER | Issue |
|----------|------|---------|-----|-------|
| Critical | `55f9b670-5cac-4dbd-b837-82646ee1f274.flac` | AG | 469% | Repetition loop hallucination (5x repeated sentence) |
| High | `699a7e10-2912-427f-9947-8c73e60394c1.flac` | SO | 200% | Complete semantic paraphrase (identical to v2/v3) |
| High | `ae929da4-e560-4812-b0a6-a9ddc74d50ce.flac` | BE | 162% | Massive restructuring + name confusion |
| High | `991aad32-e695-42c8-a480-a329a3d78d90.flac` | BE | 150% | Swiss German phonetic transcription ("Sie fehlen" → "Se fala") |
| High | `8708ff28-66f0-4445-82a5-b14fb55f21cb.flac` | ZH | 133% | Perfect tense + article insertion (identical to v2/v3) |
| Medium | `98b2381e-bde4-49b2-97e3-5b867a9be815.flac` | SZ | 100% | Complete deletion (empty output) |
| Medium | `1cccaffd-b3ea-442d-adea-85be608c4883.flac` | UR | 100% | Dialectal transcription ("Tötet" → "Du hättest") |
| Medium | `633f3365-3884-462d-aa3a-fbfb12fb2202.flac` | LU | 83% | Swiss German transcription ("Heute reicht" → "Hüt langet") |
| Medium | `9143a06d-8892-42a0-82c9-a27c0b24319a.flac` | BE | 83% | Catastrophic failure ("Mensch, schön" → "Kopf hätt") |
| Medium | `68974798-4b20-4a31-89e2-eb3a913e79f6.flac` | ZG | 107% | Complete word order inversion (same as v2/v3) |

---

### Wav2Vec2 (German CV11)

**Systematic Patterns:**
- 

**Questions:**
- Why does Wav2Vec2 have significantly more deletions than Whisper models?

**Hypotheses:**
- Possible lack of Swiss German data in CommonVoice 11 training set
- CTC-based decoding may struggle with unfamiliar phoneme sequences

---

### Wav2Vec2 with LM

**Systematic Patterns:**
- 

**Questions:**
- Does the language model help or hurt Swiss German recognition?
- Is the LM trained on Standard German, causing normalization errors?

**Hypotheses:**
- 

---

## Dialect-Specific Observations

### Bern Dialect

| Model | Observation | Sample Count | Example File |
|-------|-------------|--------------|--------------|
| | | | |

---

### Zurich Dialect

| Model | Observation | Sample Count | Example File |
|-------|-------------|--------------|--------------|
| | | | |

---

### Basel Dialect

| Model | Observation | Sample Count | Example File |
|-------|-------------|--------------|--------------|
| | | | |

---

## Cross-Cutting Themes

### Phonetic Confusions
- 

### Orthographic Ambiguity
- Swiss German lacks standardized orthography
- 

### Training Data Gaps
- 

---

## Samples to Review

| Priority | File Path | Issue | Notes |
|----------|-----------|-------|-------|
| High | | | |
| Medium | | | |
| Low | | | |

---

## Next Steps

- [ ] Listen to top 10 worst samples per model
- [ ] Categorize errors by type (substitution, deletion, insertion)
- [ ] Identify dialect-specific patterns
- [ ] Compare error patterns across model families (Whisper vs Wav2Vec2)

---

## References

- Analysis files: `./20251203_112924/`
- Model comparison summary: `./20251203_112924/model_comparison_summary.json`