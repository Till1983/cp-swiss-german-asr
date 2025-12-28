```markdown
# Error Analysis Notes

## Session: 2025-12-03

### Overview

This document contains qualitative observations and patterns from the ASR error analysis conducted on Swiss German dialect data. Observations are organised by model and dialect, with links to specific samples for reference.

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
- Does Whisper normalise Swiss German to Standard German spellings?
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
- How does speed optimisation affect Swiss German recognition?
    - **Answer:** Minimal impact - only +1.4% WER increase compared to v3. Error distribution proportions remain nearly identical (sub: 21.2% vs 19.9%, ins: 5.3% vs 5.1%, del: 2.3% vs 2.1%). The Turbo optimisation does not fundamentally change Swiss German processing patterns.

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
- The consistent pattern of semantic paraphrasing across all Whisper versions suggests this is a fundamental model behaviour, not version-specific
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
- **Semantic paraphrasing identical to larger models:** Same 200% WER sample ("Allerdings sind diese Ergebnisse umstritten" → "Man muss aber auch sagen..." - 200% WER)
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
- Swiss German phonetic transcription errors are more severe in Medium (e.g., "Sie fehlen" → "Se fala") suggesting weaker acoustic modelling
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

### Wav2Vec2 with LM (German)

**Systematic Patterns:**
- **Worst overall performance:** Mean WER 75.3% (vs Wav2Vec2 without LM: 72.4%, Whisper v2: 28.0%) - language model aides Swiss German recognition poorly. Base of this model is Wav2Vec2 Large XLSR 53 German by Jonatas Grosman (WER ~79% without LM)
- **Highest insertion rate among all models:** 7.1% (vs Wav2Vec2 without LM: 4.6%, Whisper: 4.5-6.4%) - LM over-generates words
- **Lower deletion rate than without LM:** 6.9% (vs 9.7%) - LM reduces word dropping but increases substitutions
- **Substitution-dominated errors:** 55.2% substitution rate (vs 54.5% without LM) - marginally worse
- **Same semantic paraphrasing as all other models:** Identical 200% WER sample ("Allerdings sind diese Ergebnisse umstritten" → "Man muss aber auch sagen...")
- **Uppercase output convention:** All hypotheses in uppercase (e.g., "MAN MUSS ABER AUCH SAGEN DASS DIE ERGEBNISSE SEHR UMSTRITTEN SIND")
- **Swiss German phonetic transcription:** Severe phonetic confusion (e.g., "Lebensjahr vollendet" → "SLABENSJAUR ISCH VOLLENDET" - 150% WER)
- **Perfect tense conversion (similar to Whisper):** Converts simple past to perfect tense (e.g., "gehörte" → "HAT...KURT", "unterschrieb" → "HÄTTE...SCHRIEBEN")
- **Compound word splitting:** Consistent splitting (e.g., "Bioley-Magnoux" → "BIORE MANU", "Landsberg-Velen" → "LANDSBERG WÄHLEN")
- **Pronoun confusion:** Systematic "wir"→"mir" substitution (Swiss German influence)
- **Article/case confusion:** Identical patterns to Wav2Vec2 without LM ("den"→"der", "des"→"vom")

**Dialect Performance:**
| Dialect | Mean WER | Sample Count | Sub Rate | Del Rate | Ins Rate | Notable Pattern |
|---------|----------|--------------|----------|----------|----------|-----------------|
| NW | 50.0% | 1 | 50.0% | 0% | 50.0% | Single sample, unreliable |
| GL | 57.8% | 6 | 79.2% | 0% | 20.8% | Best major dialect, no deletions |
| GR | 63.6% | 12 | 73.4% | 17.7% | 8.9% | High deletion rate |
| SH | 65.4% | 4 | 76.7% | 6.7% | 16.7% | Limited samples |
| VS | 67.3% | 17 | 82.1% | 14.5% | 3.4% | Very low insertions |
| ZH | 70.0% | 144 | 79.7% | 8.8% | 11.4% | Near average |
| BL | 70.8% | 54 | 83.0% | 8.9% | 8.1% | High substitution rate |
| FR | 71.6% | 7 | 81.0% | 17.2% | 1.7% | Lowest insertion rate |
| TG | 71.8% | 50 | 81.5% | 8.0% | 10.4% | Consistent errors |
| SG | 72.9% | 116 | 79.6% | 9.9% | 10.5% | Near average |
| AG | 73.2% | 108 | 80.3% | 9.7% | 10.1% | Consistent errors |
| LU | 75.0% | 51 | 81.1% | 10.8% | 8.1% | High deletions |
| SZ | 75.6% | 9 | 79.4% | 11.1% | 9.5% | Consistent errors |
| SO | 76.1% | 36 | 75.2% | 8.8% | 16.0% | Highest insertion rate |
| ZG | 83.3% | 30 | 80.4% | 6.3% | 13.3% | High WER, high insertions |
| UR | 84.4% | 15 | 80.8% | 13.3% | 5.8% | Worst major dialect |
| BE | 84.0% | 203 | 78.9% | 11.0% | 10.1% | Most samples, worst performance |

**Top Confusion Pairs (across dialects):**
| Reference | Hypothesis | Count | Pattern |
|-----------|------------|-------|---------|
| "wurde" | "ist" | 22 | Tense conversion (same as without LM) |
| "den" | "der" | 20+ | Article confusion |
| "diese" | "die" | 14 | Demonstrative simplification |
| "des" | "vom" | 13 | Case confusion |
| "wir" | "mir" | 7 | Swiss German pronoun |
| "von" | "vor" | 7 | Phonetic similarity |
| "waren" | "sind" | 6 | Tense + number |
| "werden" | "werdet/werde" | 9 | Inflection errors |

**Questions:**
- Does the language model help or hurt Swiss German recognition?
    - **Answer:** The LM **hurts** performance (+2.9% WER compared to without LM). The German LM is trained on Standard German text, creating a mismatch with Swiss German phonology. The LM increases insertions (7.1% vs 4.6%) while only marginally reducing deletions (6.9% vs 9.7%), suggesting it's generating plausible but incorrect Standard German words.
- Why is BE dialect performance so poor (84.0% WER)?
    - **Answer:** BE (Bernese German) has the most distinctive phonology among major dialects. The Standard German LM actively "corrects" Bernese patterns to Standard German, amplifying errors rather than reducing them.

**Hypotheses:**
- **LM creates domain mismatch:** The Standard German language model expects Standard German word sequences, but receives Swiss German acoustic features, causing systematic errors
- **LM increases insertions due to over-completion:** When acoustics are ambiguous, LM "completes" sequences with Standard German words that weren't spoken
- **Deletion reduction is misleading:** LM reduces deletions by inserting incorrect words rather than truly recognizing missing content
- **BE dialect suffers most from LM:** Bernese German's unique phonology is furthest from Standard German, making LM corrections maximally harmful
- **Same semantic paraphrasing as all models:** The 200% WER sample appears identically across all 6 models, suggesting this is a fundamental property of the Swiss German speech content, not model architecture
- **Uppercase output is a tokenization artifact:** The LM may use uppercase-only vocabulary, explaining output convention
- **Pronoun confusion ("wir"→"mir") is Swiss German influence:** This substitution reflects actual Swiss German pronunciation being correctly recognised acoustically but "corrected" by LM
- **GR/GL dialects perform better** because they may have phonology closer to Standard German, making LM corrections less harmful

**Worst Samples to Review:**
| Priority | File | Dialect | WER | Issue |
|----------|------|---------|-----|-------|
| Critical | `699a7e10-2912-427f-9947-8c73e60394c1.flac` | SO | 200% | Complete semantic paraphrase (identical across all models) |
| High | `8708ff28-66f0-4445-82a5-b14fb55f21cb.flac` | ZH | 167% | Perfect tense + compound splitting + gibberish ("GUES-PSIZIRXIE") |
| High | `28c1daae-a56b-4741-95a4-2dc56b8f2844.flac` | BE | 160% | Complete word confusion ("Bioley-Magnoux" → "BIORE MANU KIRGIN") |
| High | `ae929da4-e560-4812-b0a6-a9ddc74d50ce.flac` | BE | 154% | Massive restructuring + phonetic confusion ("Trencavel" → "TRENGA WEL") |
| High | `0bfb5d1c-e804-424c-a796-9efe12a8e390.flac` | BE | 150% | Compound splitting (identical to all other models) |
| Medium | `3d9f579c-8c8d-4b3f-b7de-a8ba97c7c77d.flac` | BE | 150% | Perfect tense restructuring + word splitting |
| Medium | `1c0e7e9d-c6f9-4ef1-8f89-a551136551a7.flac` | AG | 150% | Severe phonetic confusion ("Lebensjahr" → "SLABENSJAUR ISCH") |
| Medium | `81498cb3-d406-4aa5-a519-66550adda99c.flac` | SO | 140% | Complete sentence restructuring |
| Medium | `81ad1bd1-007a-481c-a2ca-97d43ccf959b.flac` | SG | 133% | Short utterance with compound splitting ("Überwachung" → "ÜBER WACHING") |
| Medium | `b938a223-23f7-443d-880b-4bf243c26af7.flac` | ZG | 133% | Perfect tense + word order restructuring |

---

### Wav2Vec2 (German CV11)

**Systematic Patterns:**
- **Dramatically higher error rates than Whisper:** Mean WER 72.4% (vs Whisper v2: 28.0%) - more than 2.5x worse performance
- **Substitution-dominated errors:** 54.5% substitution rate (vs Whisper: 19-23%) - model struggles with Swiss German phonemes
- **High deletion rate:** 9.7% (vs Whisper: 2.1-2.4%) - frequently drops words entirely
- **Swiss German phonetic transcription:** Transcribes dialectal pronunciation rather than Standard German (e.g., "Allerdings" → "Man muss aber auch sagen" - 220% WER, identical semantic paraphrase to Whisper)
- **Severe word boundary confusion:** Compounds split incorrectly (e.g., "Schicksal führt" → "SSchicksau führt", "Landsberg-Velen" → "Landsberg Wehlen")
- **Vowel/consonant substitutions:** Swiss German vowel shifts poorly handled (e.g., "wuchs" → "ish", "Lebensjahr" → "Slavens jar")
- **Case/article confusion:** Consistent errors with articles (e.g., "den" → "der", "des" → "vom", "die" → "wo")
- **Tense conversion (less systematic than Whisper):** Some perfect tense patterns but more chaotic (e.g., "wurde" → "ist", "war" → "ist")
- **Complete sentence restructuring:** Similar to Whisper but with higher error density (e.g., "Allerdings sind diese Ergebnisse umstritten" → "Man muss aber auch sagen, dass die Ergeibniss sehr umstritten sind" - 220% WER)

**Dialect Performance:**
| Dialect | Mean WER | Sample Count | Sub Rate | Del Rate | Ins Rate | Notable Pattern |
|---------|----------|--------------|----------|----------|----------|-----------------|
| NW | 50.0% | 1 | 50.0% | 0% | 50.0% | Single sample, unreliable |
| GL | 57.8% | 6 | 79.2% | 0% | 20.8% | Best major dialect, no deletions |
| GR | 63.6% | 12 | 73.4% | 17.7% | 8.9% | High deletion rate |
| SH | 65.4% | 4 | 76.7% | 6.7% | 16.7% | Limited samples |
| VS | 67.3% | 17 | 82.1% | 14.5% | 3.4% | Very low insertions |
| ZH | 70.0% | 144 | 79.7% | 8.8% | 11.4% | Near average |
| BL | 70.8% | 54 | 83.0% | 8.9% | 8.1% | High substitution rate |
| FR | 71.6% | 7 | 81.0% | 17.2% | 1.7% | Lowest insertion rate |
| TG | 71.8% | 50 | 81.5% | 8.0% | 10.4% | Consistent errors |
| SG | 72.9% | 116 | 79.6% | 9.9% | 10.5% | Near average |
| AG | 73.2% | 108 | 80.3% | 9.7% | 10.1% | Consistent errors |
| LU | 75.0% | 51 | 81.1% | 10.8% | 8.1% | High deletions |
| SZ | 75.6% | 9 | 79.4% | 11.1% | 9.5% | Consistent errors |
| SO | 76.1% | 36 | 75.2% | 8.8% | 16.0% | Highest insertion rate |
| ZG | 83.3% | 30 | 80.4% | 6.3% | 13.3% | High WER, high insertions |
| UR | 84.4% | 15 | 80.8% | 13.3% | 5.8% | Worst major dialect |
| BE | 84.0% | 203 | 78.9% | 11.0% | 10.1% | Most samples, worst performance |

**Top Confusion Pairs (across dialects):**
| Reference | Hypothesis | Count | Pattern |
|-----------|------------|-------|---------|
| "wurde" | "ist" | 22 | Tense conversion (same as without LM) |
| "den" | "der" | 20+ | Article confusion |
| "diese" | "die" | 14 | Demonstrative simplification |
| "des" | "vom" | 13 | Case confusion |
| "wir" | "mir" | 7 | Swiss German pronoun |
| "von" | "vor" | 7 | Phonetic similarity |
| "waren" | "sind" | 6 | Tense + number |
| "werden" | "werdet/werde" | 9 | Inflection errors |

**Questions:**
- Does the language model help or hurt Swiss German recognition?
    - **Answer:** The LM **hurts** performance (+2.9% WER compared to without LM). The German LM is trained on Standard German text, creating a mismatch with Swiss German phonology. The LM increases insertions (7.1% vs 4.6%) while only marginally reducing deletions (6.9% vs 9.7%), suggesting it's generating plausible but incorrect Standard German words.
- Why is BE dialect performance so poor (84.0% WER)?
    - **Answer:** BE (Bernese German) has the most distinctive phonology among major dialects. The Standard German LM actively "corrects" Bernese patterns to Standard German, amplifying errors rather than reducing them.

**Hypotheses:**
- **LM creates domain mismatch:** The Standard German language model expects Standard German word sequences, but receives Swiss German acoustic features, causing systematic errors
- **LM increases insertions due to over-completion:** When acoustics are ambiguous, LM "completes" sequences with Standard German words that weren't spoken
- **Deletion reduction is misleading:** LM reduces deletions by inserting incorrect words rather than truly recognizing missing content
- **BE dialect suffers most from LM:** Bernese German's unique phonology is furthest from Standard German, making LM corrections maximally harmful
- **Same semantic paraphrasing as all models:** The 200% WER sample appears identically across all 6 models, suggesting this is a fundamental property of the Swiss German speech content, not model architecture
- **Uppercase output is a tokenization artifact:** The LM may use uppercase-only vocabulary, explaining output convention
- **Pronoun confusion ("wir"→"mir") is Swiss German influence:** This substitution reflects actual Swiss German pronunciation being correctly recognised acoustically but "corrected" by LM
- **GR/GL dialects perform better** because they may have phonology closer to Standard German, making LM corrections less harmful

**Worst Samples to Review:**
| Priority | File | Dialect | WER | Issue |
|----------|------|---------|-----|-------|
| Critical | `699a7e10-2912-427f-9947-8c73e60394c1.flac` | SO | 200% | Complete semantic paraphrase (identical across all models) |
| High | `8708ff28-66f0-4445-82a5-b14fb55f21cb.flac` | ZH | 167% | Perfect tense + compound splitting + gibberish ("GUES-PSIZIRXIE") |
| High | `28c1daae-a56b-4741-95a4-2dc56b8f2844.flac` | BE | 160% | Complete word confusion ("Bioley-Magnoux" → "BIORE MANU KIRGIN") |
| High | `ae929da4-e560-4812-b0a6-a9ddc74d50ce.flac` | BE | 154% | Massive restructuring + phonetic confusion ("Trencavel" → "TRENGA WEL") |
| High | `0bfb5d1c-e804-424c-a796-9efe12a8e390.flac` | BE | 150% | Compound splitting (identical to all other models) |
| Medium | `3d9f579c-8c8d-4b3f-b7de-a8ba97c7c77d.flac` | BE | 150% | Perfect tense restructuring + word splitting |
| Medium | `1c0e7e9d-c6f9-4ef1-8f89-a551136551a7.flac` | AG | 150% | Severe phonetic confusion ("Lebensjahr" → "SLABENSJAUR ISCH") |
| Medium | `81498cb3-d406-4aa5-a519-66550adda99c.flac` | SO | 140% | Complete sentence restructuring |
| Medium | `81ad1bd1-007a-481c-a2ca-97d43ccf959b.flac` | SG | 133% | Short utterance with compound splitting ("Überwachung" → "ÜBER WACHING") |
| Medium | `b938a223-23f7-443d-880b-4bf243c26af7.flac` | ZG | 133% | Perfect tense + word order restructuring |

---

## Session: 2024-12-04

## Dialect-Specific Observations

### Summary Table: Mean WER by Dialect Across All Models

| Dialect | Sample Count | Whisper v2 | Whisper v3 | Whisper v3-Turbo | Whisper Medium | Wav2Vec2 CV11 | Wav2Vec2 + LM |
|---------|--------------|------------|------------|------------------|----------------|---------------|---------------|
| **GL** | 6 | **5.8%** | 10.7% | 10.7% | 15.0% | 52.1% | 57.8% |
| **GR** | 12 | 11.3% | 17.2% | 18.8% | 22.0% | 66.9% | 63.6% |
| **SZ** | 9 | 14.6% | ~25% | ~30% | 40.8% | 70.0% | 75.6% |
| **SH** | 4 | ~20% | ~25% | ~28% | ~30% | 57.8% | 65.4% |
| **UR** | 15 | 21.2% | ~28% | ~30% | 30.5% | 72.7% | 84.4% |
| **ZH** | 144 | 23.5% | ~28% | 28.2% | 30.3% | 65.8% | 70.0% |
| **BL** | 54 | 26.6% | ~28% | ~30% | 35.2% | 68.7% | 70.8% |
| **AG** | 108 | 27.8% | ~30% | ~31% | 36.8% | 69.2% | 73.2% |
| **LU** | 51 | 28.9% | ~32% | ~33% | 35.0% | 76.0% | 75.0% |
| **SG** | 116 | 29.2% | ~31% | 30.5% | 33.0% | 72.5% | 72.9% |
| **TG** | 50 | ~27% | ~30% | ~32% | 35.3% | 68.2% | 71.8% |
| **BE** | 203 | 29.9% | 31.7% | 32.2% | 35.2% | **80.3%** | **84.0%** |
| **VS** | 17 | 31.0% | ~33% | ~34% | 33.6% | 69.1% | 67.3% |
| **SO** | 36 | 34.9% | 34.8% | 36.0% | 36.0% | 72.9% | 76.1% |
| **FR** | 7 | 37.0% | ~40% | ~42% | 44.0% | 68.2% | 71.6% |
| **ZG** | 30 | 39.7% | **46.3%** | **41.6%** | 38.3% | **81.4%** | **83.3%** |

**Key:** Best performers in **bold green**, worst performers in **bold red**

---

### Dialect Rankings

#### Best Performing Dialects (Consistently Low WER)

| Rank | Dialect | Region | Avg WER (Whisper) | Avg WER (Wav2Vec2) | Key Characteristics |
|------|---------|--------|-------------------|--------------------|--------------------|
| 1 | **GL** (Glarus) | Central-East | 10.5% | 55.0% | Lowest WER across all models, 0% insertion rate in Whisper v2/v3 |
| 2 | **GR** (Graubünden) | Southeast | 17.3% | 65.3% | Low errors, minimal restructuring, 0% insertion rate in v2 |
| 3 | **SZ** (Schwyz) | Central | 27.4% | 72.8% | Low WER in v2, but high variance across models |

#### Worst Performing Dialects (Consistently High WER)

| Rank | Dialect | Region | Avg WER (Whisper) | Avg WER (Wav2Vec2) | Key Characteristics |
|------|---------|--------|-------------------|--------------------|--------------------|
| 1 | **ZG** (Zug) | Central | 41.5% | 82.4% | Highest WER, heavy sentence restructuring |
| 2 | **BE** (Bern) | West-Central | 32.3% | 82.2% | Most samples (203), consistent restructuring patterns |
| 3 | **SO** (Solothurn) | Northwest | 35.4% | 74.5% | Highest variance (std: 38-41%), highest insertion rates |

---

### Bern Dialect (BE) - 203 Samples

**Characteristics:** Most represented dialect, highest sample count. Distinctive phonology with unique vowel qualities.

| Model | Mean WER | Insertion Rate | Top Error Patterns | Example File |
|-------|----------|----------------|-------------------|--------------|
| Whisper v2 | 29.9% | 19.4% | Perfect tense conversion, compound splitting | `3d9f579c-8c8d-4b3f-b7de-a8ba97c7c77d.flac` |
| Whisper v3 | 31.7% | 20.5% | Sentence restructuring, article insertion | `ae929da4-e560-4812-b0a6-a9ddc74d50ce.flac` |
| Whisper v3-Turbo | 32.2% | 20.9% | Perfect tense, semantic paraphrasing | `991aad32-e695-42c8-a480-a329a3d78d90.flac` |
| Whisper Medium | 35.2% | 21.2% | Phonetic transcription ("Sie fehlen" → "Se fala") | `991aad32-e695-42c8-a480-a329a3d78d90.flac` |
| Wav2Vec2 CV11 | **80.3%** | 6.2% | Severe phonetic confusion, high deletion rate | `d3535ea7-e4cc-40b9-8e56-88ce68bdac27.flac` |
| Wav2Vec2 + LM | **84.0%** | 10.1% | LM "corrections" amplify errors | `28c1daae-a56b-4741-95a4-2dc56b8f2844.flac` |

**Observations:**
- BE has the **worst Wav2Vec2 performance** among major dialects (80-84% WER)
- Consistent restructuring patterns across all Whisper models (perfect tense, word order)
- High insertion rates in Whisper (19-21%) suggest over-generation
- Distinctive phonology (unique vowel qualities) causes maximum domain mismatch with Standard German training data

**Notable Error Examples:**
| File | Reference | Hypothesis (Whisper v3) | WER | Issue |
|------|-----------|------------------------|-----|-------|
| `ae929da4-e560-4812-b0a6-a9ddc74d50ce.flac` | "Trencavel seiner Besitzungen für verlustig erklärt..." | "Es wurde erklärt, dass das, was Trencabel gehört hat..." | 146% | Complete sentence restructuring |
| `3d9f579c-8c8d-4b3f-b7de-a8ba97c7c77d.flac` | "Er unterschrieb einen Dreijahresvertrag." | "Er hat drei Jahre lang einen Vertrag unterschrieben." | 150% | Perfect tense + compound splitting |
| `0bfb5d1c-e804-424c-a796-9efe12a8e390.flac` | "Mal Etappenort." | "MAU Etappe Ort" | 150% | Compound word fusion |

---

### Zurich Dialect (ZH) - 144 Samples

**Characteristics:** Second-largest sample count. Urban dialect with broad speaker variation.

| Model | Mean WER | Insertion Rate | Top Error Patterns | Example File |
|-------|----------|----------------|-------------------|--------------|
| Whisper v2 | 23.5% | 14.6% | Perfect tense, article insertion | `8708ff28-66f0-4445-82a5-b14fb55f21cb.flac` |
| Whisper v3 | ~28% | ~16% | Word order inversion | `254ad4f9-bf85-4b08-a6d6-1d324d53300e.flac` |
| Whisper v3-Turbo | 28.2% | 16.1% | Compound splitting | `8708ff28-66f0-4445-82a5-b14fb55f21cb.flac` |
| Whisper Medium | 30.3% | 15.8% | Moderate restructuring | `8708ff28-66f0-4445-82a5-b14fb55f21cb.flac` |
| Wav2Vec2 CV11 | 65.8% | 7.0% | Case/article confusion | `254ad4f9-bf85-4b08-a6d6-1d324d53300e.flac` |
| Wav2Vec2 + LM | 70.0% | 11.4% | Near average performance | `8708ff28-66f0-4445-82a5-b14fb55f21cb.flac` |

**Observations:**
- ZH shows **moderate performance** across all models (middle of pack)
- Consistent ~133% WER on `8708ff28` sample across all models (perfect tense + article insertion)
- Lower insertion rate than BE/SO/SG suggests cleaner acoustic signal

**Notable Error Examples:**
| File | Reference | Hypothesis (Whisper v3) | WER | Issue |
|------|-----------|------------------------|-----|-------|
| `8708ff28-66f0-4445-82a5-b14fb55f21cb.flac` | "Landsberg-Velen war promovierter Jurist und Rittergutsbesitzer." | "Der Landsberg Velen ist ein promovierter Jurist und ein Rittergutsbesitzer gewesen." | 133% | Perfect tense + article insertion |
| `254ad4f9-bf85-4b08-a6d6-1d324d53300e.flac` | "Im Norden liegt die Bandasee." | "Der Bandasee ist im Norden." | 100% | Complete word order inversion |

---

### Basel Dialect (BL) - 54 Samples

**Characteristics:** Northwestern dialect with French influence.

| Model | Mean WER | Insertion Rate | Top Error Patterns | Example File |
|-------|----------|----------------|-------------------|--------------|
| Whisper v2 | 26.6% | 16.7% | Complex restructuring | `11e2d4d7-b7f6-4b78-a9e7-261731da2c58.flac` |
| Whisper v3 | ~28% | ~17% | Compound handling | `3950d3dc-4b1f-4de6-8742-aa9306c85f65.flac` |
| Whisper v3-Turbo | ~30% | ~17% | Code-switching to English | `3950d3dc-4b1f-4de6-8742-aa9306c85f65.flac` |
| Whisper Medium | 35.2% | 17.1% | Moderate errors | `3950d3dc-4b1f-4de6-8742-aa9306c85f65.flac` |
| Wav2Vec2 CV11 | 68.7% | 5.1% | **Highest deletion rate (18.6%)** | `11e2d4d7-b7f6-4b78-a9e7-261731da2c58.flac` |
| Wav2Vec2 + LM | 70.8% | 8.1% | High substitution rate (83%) | `11e2d4d7-b7f6-4b78-a9e7-261731da2c58.flac` |

**Observations:**
- BL has **highest Wav2Vec2 deletion rate** (18.6%) suggesting acoustic model failure
- Occasional code-switching to English in Whisper models
- French influence may contribute to phonetic confusion

**Notable Error Examples:**
| File | Reference | Hypothesis (Whisper v3-Turbo) | WER | Issue |
|------|-----------|------------------------------|-----|-------|
| `3950d3dc-4b1f-4de6-8742-aa9306c85f65.flac` | "Späth Orgelbau erbaut." | "Spread all about" | 100% | Code-switch to English |
| `11e2d4d7-b7f6-4b78-a9e7-261731da2c58.flac` | "Wie anderswo waren auch hier zunächst nur Männer teilnahmeberechtigt." | "Wie auch andere, die hier waren, waren zunächst nur..." | 122% | Complex restructuring |

---

### Zug Dialect (ZG) - 30 Samples

**Characteristics:** Central Swiss dialect with **worst overall WER** across most models.

| Model | Mean WER | Insertion Rate | Top Error Patterns | Example File |
|-------|----------|----------------|-------------------|--------------|
| Whisper v2 | 39.7% | 17.8% | Heavy restructuring | `68974798-4b20-4a31-89e2-eb3a913e79f6.flac` |
| Whisper v3 | **46.3%** | ~18% | Word order inversion | `68974798-4b20-4a31-89e2-eb3a913e79f6.flac` |
| Whisper v3-Turbo | 41.6% | 16.7% | Complete restructuring | `68974798-4b20-4a31-89e2-eb3a913e79f6.flac` |
| Whisper Medium | 38.3% | 16.7% | High WER | `b938a223-23f7-443d-880b-4bf243c26af7.flac` |
| Wav2Vec2 CV11 | **81.4%** | 10.3% | Worst overall | `b938a223-23f7-443d-880b-4bf243c26af7.flac` |
| Wav2Vec2 + LM | 83.3% | 13.3% | High insertions | `332d2e69-5674-4dcb-bb80-9900901a5ca8.flac` |

**Observations:**
- ZG consistently shows **worst WER** among all dialects
- Heavy sentence restructuring is the dominant error pattern
- Perfect tense conversion + word order inversion occurs together

**Notable Error Examples:**
| File | Reference | Hypothesis (Whisper v3) | WER | Issue |
|------|-----------|------------------------|-----|-------|
| `68974798-4b20-4a31-89e2-eb3a913e79f6.flac` | "Trotz dieses Wachstums verlor der Ort seine Selbstständigkeit..." | "...hat der Ort seine Selbstständigkeit verloren..." | 100-107% | Complete word order inversion + perfect tense |
| `b938a223-23f7-443d-880b-4bf243c26af7.flac` | "Dort gehörte er zur ursprünglichen Macintosh-Entwicklungsmannschaft." | "Det hett e...khürt." | 150% | Swiss German phonetic transcription |
| `332d2e69-5674-4dcb-bb80-9900901a5ca8.flac` | "Staats-Obergymnasium unterrichtete, kennen." | "Staatsopergymnasium unterrichtet – Kennen" | 133% | Compound handling + article insertion |

---

### Solothurn Dialect (SO) - 36 Samples

**Characteristics:** Northwestern dialect with **highest variance** and **insertion rates**.

| Model | Mean WER | Std WER | Insertion Rate | Example File |
|-------|----------|---------|----------------|--------------|
| Whisper v2 | 34.9% | ~38% | **24.3%** | `699a7e10-2912-427f-9947-8c73e60394c1.flac` |
| Whisper v3 | 34.8% | 41.0% | **25.7%** | `699a7e10-2912-427f-9947-8c73e60394c1.flac` |
| Whisper v3-Turbo | 36.0% | 38.4% | **22.7%** | `699a7e10-2912-427f-9947-8c73e60394c1.flac` |
| Whisper Medium | 36.0% | ~35% | 22.5% | `699a7e10-2912-427f-9947-8c73e60394c1.flac` |
| Wav2Vec2 CV11 | 72.9% | ~25% | 12.8% | `81498cb3-d406-4aa5-a519-66550adda99c.flac` |
| Wav2Vec2 + LM | 76.1% | ~24% | **16.0%** | `699a7e10-2912-427f-9947-8c73e60394c1.flac` |

**Observations:**
- SO has **highest insertion rates** across Whisper models (22-26%)
- **Highest variance** (std: 35-41%) indicates inconsistent performance
- Contains the infamous **200% WER sample** (`699a7e10`) that appears identically across ALL models

**The Universal Worst Sample:**
| File | Dialect | Reference | Hypothesis (ALL models) | WER |
|------|---------|-----------|------------------------|-----|
| `699a7e10-2912-427f-9947-8c73e60394c1.flac` | SO | "Allerdings sind diese Ergebnisse umstritten." | "Man muss aber auch sagen, dass die Ergebnisse sehr umstritten sind." | 200-220% |

This sample produces **identical semantic paraphrasing** across all 6 models, suggesting the Swiss German speech content itself triggers this restructuring, not the model architecture.

---

### Glarus Dialect (GL) - 6 Samples

**Characteristics:** Best performing dialect with **lowest WER** and **0% insertion rate**.

| Model | Mean WER | Insertion Rate | Top Error Patterns | Example File |
|-------|----------|----------------|-------------------|--------------|
| Whisper v2 | **5.8%** | **0%** | Minimal errors | `f10c5233-e2db-4089-a981-30e2c9facf1c.flac` |
| Whisper v3 | 10.7% | 0% | Clean output | `51ce3042-9109-432d-a2ae-00be2ba5e764.flac` |
| Whisper v3-Turbo | 10.7% | 0% | Clean output | `92028d5d-c6f2-4b22-b898-d1e928d018e0.flac` |
| Whisper Medium | 15.0% | 42.9% | Higher insertions (anomaly) | `679e27f2-93be-47ee-ba28-5ef510a2047d.flac` |
| Wav2Vec2 CV11 | 52.1% | 16.7% | Still 5x worse than Whisper | `f10c5233-e2db-4089-a981-30e2c9facf1c.flac` |
| Wav2Vec2 + LM | 57.8% | 20.8% | No deletions | `2afe940b-25c4-48ee-8f42-4462578cc04c.flac` |

**Observations:**
- GL has **best Whisper performance** (5.8-15% WER)
- **0% insertion rate** in Whisper v2/v3/Turbo suggests no over-generation
- May have phonology **closest to Standard German** training data
- Limited sample size (6) affects reliability

---

### Graubünden Dialect (GR) - 12 Samples

**Characteristics:** Second-best performing dialect, southeastern region.

| Model | Mean WER | Insertion Rate | Top Error Patterns | Example File |
|-------|----------|----------------|-------------------|--------------|
| Whisper v2 | 11.3% | **0%** | Minimal errors | `3313b955-b2de-4f80-ba17-ab48883fd2d3.flac` |
| Whisper v3 | 17.2% | 5% | Minimal restructuring | `3030a452-3e56-4ae6-81e7-e0d31ed0bd11.flac` |
| Whisper v3-Turbo | 18.8% | 13.6% | Low errors | `efdaf2cd-c654-4b30-8f6f-20fc28040d55.flac` |
| Whisper Medium | 22.0% | 11.5% | Low errors | `428ac188-e9b4-4e31-b247-8d14d352769c.flac` |
| Wav2Vec2 CV11 | 66.9% | 4.8% | Moderate errors | `9a0892dd-3293-4ba1-bd08-39f62e92f4f8.flac` |
| Wav2Vec2 + LM | 63.6% | 8.9% | High deletion rate (17.7%) | `f0835949-d4be-41d3-9c3a-854c14e65d27.flac` |

**Observations:**
- GR has **second-best Whisper performance** (11-22% WER)
- **0% insertion rate** in Whisper v2 (same as GL)
- Romansh influence may create cleaner phonetic patterns

---

### Uri Dialect (UR) - 15 Samples

**Characteristics:** Central Swiss dialect with **distinctive phonetic transcription errors**.

| Model | Mean WER | Insertion Rate | Top Error Patterns | Example File |
|-------|----------|----------------|-------------------|--------------|
| Whisper v2 | 21.2% | 3.8% | Lower than v3/Turbo | `f27034de-622d-4ee7-a156-0240a4ac23c6.flac` |
| Whisper v3 | ~28% | ~10% | Phonetic transcription | `1cccaffd-b3ea-442d-adea-85be608c4883.flac` |
| Whisper v3-Turbo | ~30% | ~12% | Dialectal phonetic output | `49a11f9d-d3cc-439a-a56c-5e585550c754.flac` |
| Whisper Medium | 30.5% | 12.8% | Moderate errors | `1cccaffd-b3ea-442d-adea-85be608c4883.flac` |
| Wav2Vec2 CV11 | 72.7% | 2.9% | Low insertions | `6127bcc2-b837-41f9-a79e-2c623150607a.flac` |
| Wav2Vec2 + LM | **84.4%** | 5.8% | Worst major dialect | `0257414a-4e39-41c0-8d9b-83df216c1e67.flac` |

**Observations:**
- UR shows **distinctive Swiss German phonetic transcription** errors
- Models transcribe dialectal pronunciation rather than Standard German
- Wav2Vec2 + LM shows **worst UR performance** (84.4% WER)

**Notable Error Examples:**
| File | Reference | Hypothesis (Whisper v3-Turbo) | WER | Issue |
|------|-----------|------------------------------|-----|-------|
| `49a11f9d-d3cc-439a-a56c-5e585550c754.flac` | "Wohin führt dieser?" | "Wo hi fährt der?" | 133% | Dialectal phonetic transcription |
| `1cccaffd-b3ea-442d-adea-85be608c4883.flac` | "Tötet nicht den Boten." | "Du hättest..." | 100% | Catastrophic phonetic failure |

---

### St. Gallen Dialect (SG) - 116 Samples

**Characteristics:** Eastern Swiss dialect with **high insertion rates**.

| Model | Mean WER | Insertion Rate | Top Error Patterns | Example File |
|-------|----------|----------------|-------------------|--------------|
| Whisper v2 | 29.2% | 20.7% | High insertion rate | `63daf2e7-9d35-48aa-b718-3e45e7e51a96.flac` |
| Whisper v3 | ~31% | **23.1%** | High insertions | `81ad1bd1-007a-481c-a2ca-97d43ccf959b.flac` |
| Whisper v3-Turbo | 30.5% | **22.3%** | High insertions | `63daf2e7-9d35-48aa-b718-3e45e7e51a96.flac` |
| Whisper Medium | 33.0% | 21.3% | High insertions | `81ad1bd1-007a-481c-a2ca-97d43ccf959b.flac` |
| Wav2Vec2 CV11 | 72.5% | 7.8% | Near average | `e7256fd7-499b-4afb-9a9a-cd15d3f46d08.flac` |
| Wav2Vec2 + LM | 72.9% | 10.5% | Near average | `81ad1bd1-007a-481c-a2ca-97d43ccf959b.flac` |

**Observations:**
- SG has **second-highest insertion rates** after SO (21-23%)
- Third-largest sample count (116) provides reliable statistics
- Consistent restructuring patterns across models

---

### Valais Dialect (VS) - 17 Samples

**Characteristics:** Southwestern dialect with **very low insertion rates** despite moderate WER.

| Model | Mean WER | Insertion Rate | Top Error Patterns | Example File |
|-------|----------|----------------|-------------------|--------------|
| Whisper v2 | 31.0% | **5.6%** | Low insertions | `45d84bf7-849e-499d-92dd-ff707b0a8497.flac` |
| Whisper v3 | ~33% | ~6% | Low insertions | `60e68b17-a7fb-49ec-8947-f49f1d7b5265.flac` |
| Whisper v3-Turbo | ~34% | ~6% | Low insertions | `8b5bfb26-381c-471c-ab0e-100c554d2267.flac` |
| Whisper Medium | 33.6% | 11.1% | Low insertions | `bab8e94d-9bc1-4b1c-b7fb-5d2b51e46e4a.flac` |
| Wav2Vec2 CV11 | 69.1% | **2.5%** | **Extreme deletions (20.5%)** | `45d84bf7-849e-499d-92dd-ff707b0a8497.flac` |
| Wav2Vec2 + LM | 67.3% | **3.4%** | Very low insertions | `97f1b02c-5c40-4d78-bbd0-74a94dc753dc.flac` |

**Observations:**
- VS has **lowest insertion rates** across most models (2.5-11%)
- Wav2Vec2 shows **extreme deletion rate (20.5%)** - highest across all dialects
- French influence from border region may create unique phonetic patterns

---

### Aargau Dialect (AG) - 108 Samples

**Characteristics:** Northern dialect with **hallucination issues** in Whisper Medium.

| Model | Mean WER | Insertion Rate | Top Error Patterns | Example File |
|-------|----------|----------------|-------------------|--------------|
| Whisper v2 | 27.8% | 15.8% | Consistent performance | `1c0e7e9d-c6f9-4ef1-8f89-a551136551a7.flac` |
| Whisper v3 | ~30% | ~16% | Moderate errors | `55f9b670-5cac-4dbd-b837-82646ee1f274.flac` |
| Whisper v3-Turbo | ~31% | ~17% | Moderate errors | `acaaa975-240e-4265-9be2-7bd80b2d9398.flac` |
| Whisper Medium | 36.8% | **30.2%** | **Hallucination loops** | `55f9b670-5cac-4dbd-b837-82646ee1f274.flac` |
 | Wav2Vec2 CV11 | 69.2% | 6.5% | High deletion rate (14.6%) | `1c0e7e9d-c6f9-4ef1-8f89-a551136551a7.flac` |
| Wav2Vec2 + LM | 73.2% | 10.1% | Consistent errors | `1c0e7e9d-c6f9-4ef1-8f89-a551136551a7.flac` |

**Observations:**
- AG shows **highest insertion rate (30.2%)** in Whisper Medium - includes hallucination case
- Contains the **469% WER catastrophic failure** (repetition loop)

**Catastrophic Failure Example:**
| File | Reference | Hypothesis (Whisper Medium) | WER |
|------|-----------|----------------------------|-----|
| `55f9b670-5cac-4dbd-b837-82646ee1f274.flac` | "Die Samples wurden in der Vergangenheit von verschiedenen Kommentatoren..." | "In der Vergangenheit wurden Temples... (repeated 5+ times)" | **469%** |

---

### Schwyz Dialect (SZ) - 9 Samples

**Characteristics:** Central Swiss dialect with **complete deletion failures**.

| Model | Mean WER | Insertion Rate | Top Error Patterns | Example File |
|-------|----------|----------------|-------------------|--------------|
| Whisper v2 | 14.6% | **0%** | Low WER, no insertions | `7842c9b1-b176-4995-b6e6-99f79d207401.flac` |
| Whisper v3 | ~25% | ~5% | Moderate errors | `98b2381e-bde4-49b2-97e3-5b867a9be815.flac` |
| Whisper v3-Turbo | ~30% | ~8% | Dialectal phonetic transcription | `98b2381e-bde4-49b2-97e3-5b867a9be815.flac` |
| Whisper Medium | **40.8%** | 10.7% | **Complete deletion case** | `98b2381e-bde4-49b2-97e3-5b867a9be815.flac` |
| Wav2Vec2 CV11 | 70.0% | 6.7% | Consistent errors | `d66268be-5331-413d-8eb5-339cdc6dfd60.flac` |
| Wav2Vec2 + LM | 75.6% | 9.5% | Consistent errors | `423eeb98-9c38-47aa-98d1-1b2edcbfd2ff.flac` |

**Observations:**
- SZ has **0% insertion rate** in Whisper v2 (same as GL/GR)
- Contains a **complete deletion failure** in Whisper Medium (empty output)

**Notable Error Example:**
| File | Reference | Hypothesis (Whisper v3-Turbo) | WER | Issue |
|------|-----------|------------------------------|-----|-------|
| `98b2381e-bde4-49b2-97e3-5b867a9be815.flac` | "Aber woran liegt das?" | "A prvo roll ist das." | 125% | Dialectal phonetic transcription |

---

## Cross-Cutting Themes

### Phonetic Confusions
- **Swiss German vowel shifts** poorly handled by all models
- Common confusions: "wir"→"mir", "war"→"ist", "wurde"→"ist"
- Dialectal phonetic transcription occurs when models transcribe pronunciation rather than Standard German

### Orthographic Ambiguity
- Swiss German lacks standardized orthography
- Models sometimes produce Swiss German spellings (ß→ss, e.g., "Außerdem"→"Ausserdem")
- Compound word handling varies: both fusion and splitting occur

### Training Data Gaps
- Wav2Vec2 models trained on different versions of CommonVoice (with LM - CV6, without LM - CV11, CV11 likely contains some Swiss German, hence better performance)
- Whisper has some Swiss German exposure but still restructures to Swiss German syntax
- LM trained on Standard German text creates domain mismatch

### Dialect Phonology Patterns
| Dialect Group | Characteristics | Whisper WER | Wav2Vec2 WER |
|---------------|-----------------|-------------|--------------|
| **Eastern (GL, GR)** | Closer to Standard German | 10-18% | 52-67% |
| **Central (ZH, SZ, ZG)** | Moderate divergence | 23-42% | 66-81% |
| **Western (BE, SO, BL, VS)** | High divergence, French influence | 27-35% | 68-84% |

---