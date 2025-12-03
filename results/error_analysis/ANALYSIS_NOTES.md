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
- **Pronoun confusion ("wir"→"mir") is Swiss German influence:** This substitution reflects actual Swiss German pronunciation being correctly recognized acoustically but "corrected" by LM
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
| GL | 52.1% | 6 | 75.0% | 8.3% | 16.7% | Best performance, but still 5x worse than Whisper |
| SH | 57.8% | 4 | 88.9% | 7.4% | 3.7% | High substitution rate |
| ZH | 65.8% | 144 | 78.6% | 14.4% | 7.0% | Moderate errors |
| GR | 66.9% | 12 | 85.7% | 9.5% | 4.8% | Higher than Whisper's 11.3% |
| TG | 68.2% | 50 | 85.0% | 10.1% | 4.9% | Consistent high sub rate |
| FR | 68.2% | 7 | 89.3% | 8.9% | 1.8% | Very low insertions |
| AG | 69.2% | 108 | 78.9% | 14.6% | 6.5% | High deletion rate |
| BL | 68.7% | 54 | 76.3% | 18.6% | 5.1% | Highest deletion rate |
| VS | 69.1% | 17 | 77.0% | 20.5% | 2.5% | Extreme deletions |
| SZ | 70.0% | 9 | 80.0% | 13.3% | 6.7% | Consistent errors |
| SG | 72.5% | 116 | 79.5% | 12.7% | 7.8% | Near average |
| UR | 72.7% | 15 | 82.5% | 14.6% | 2.9% | Low insertions |
| SO | 72.9% | 36 | 77.6% | 9.6% | 12.8% | High insertion rate |
| LU | 76.0% | 51 | 75.9% | 18.6% | 5.5% | High deletions |
| BE | 80.3% | 203 | 79.0% | 14.8% | 6.2% | Worst major dialect |
| ZG | 81.4% | 30 | 80.2% | 9.5% | 10.3% | Worst overall |

**Top Confusion Pairs (across dialects):**
| Reference | Hypothesis | Count | Pattern |
|-----------|------------|-------|---------|
| "wurde" | "ist" | 27 | Tense conversion |
| "den" | "der" | 24 | Article confusion |
| "des" | "vom" | 17 | Case confusion |
| "war" | "ist" | 16 | Tense conversion |
| "wir" | "mir" | 4 | Pronoun confusion (Swiss German influence) |
| "als" | "aus" | 4 | Phonetic similarity |
| "werden" | "werde" | 7 | Inflection errors |
| "wurden" | "sind" | 10 | Tense + number |

**Questions:**
- Why does Wav2Vec2 have significantly more deletions than Whisper models?
    - **Answer:** The CTC-based architecture lacks the language model capabilities of Whisper's encoder-decoder design. When encountering unfamiliar Swiss German phoneme sequences, CTC tends to skip (delete) rather than hallucinate. The 9.7% deletion rate (vs Whisper's 2.1-2.4%) reflects the model's inability to map dialectal sounds to Standard German words.
- Why is BE dialect performance so poor (80.3% WER)?
    - **Answer:** Bernese German has distinctive phonological features (e.g., unique vowel qualities, different consonant patterns) that deviate significantly from the Standard German training data in CommonVoice 11. The model has no mechanism to adapt to these patterns.

**Hypotheses:**
- **CTC architecture fundamentally unsuited for dialect ASR:** The frame-by-frame classification approach cannot handle the phoneme-to-grapheme mapping differences between Swiss German and Standard German
- **CommonVoice 11 training data lacks Swiss German:** The model was trained exclusively on Standard German speakers, creating a significant domain mismatch
- **Deletion errors indicate acoustic model failure:** When Swiss German phonemes don't match any training distribution, CTC outputs blank tokens, causing word deletions
- **Substitution patterns reveal phonetic confusion:** The high substitution rate (54.5%) suggests the model is attempting transcription but mapping Swiss German sounds to wrong Standard German words
- **Article/case errors reflect German grammar challenges:** The systematic confusion of "den"/"der", "des"/"vom" suggests the model lacks grammatical context that Whisper's decoder provides
- **GL dialect's relative success (52.1% WER)** may indicate closer phonological similarity to Standard German, though still far worse than Whisper (10.7%)
- **The model exhibits similar semantic paraphrasing to Whisper** (e.g., 220% WER sample), suggesting this is a property of Swiss German speech patterns, not model architecture

**Worst Samples to Review:**
| Priority | File | Dialect | WER | Issue |
|----------|------|---------|-----|-------|
| Critical | `699a7e10-2912-427f-9947-8c73e60394c1.flac` | SO | 220% | Complete semantic paraphrase (same pattern as all Whisper models) |
| High | `1c0e7e9d-c6f9-4ef1-8f89-a551136551a7.flac` | AG | 200% | Severe phonetic confusion ("Lebensjahr" → "Slavens jar") |
| High | `0bfb5d1c-e804-424c-a796-9efe12a8e390.flac` | BE | 150% | Compound splitting (identical to Whisper) |
| High | `b938a223-23f7-443d-880b-4bf243c26af7.flac` | ZG | 150% | Perfect tense + word order restructuring |
| High | `81498cb3-d406-4aa5-a519-66550adda99c.flac` | SO | 150% | Complete sentence restructuring |
| Medium | `254ad4f9-bf85-4b08-a6d6-1d324d53300e.flac` | ZH | 140% | Word order inversion ("Im Norden liegt" → "Der Ban dar sehe isch") |
| Medium | `ae929da4-e560-4812-b0a6-a9ddc74d50ce.flac` | BE | 138% | Massive restructuring (same sample as Whisper worst cases) |
| Medium | `8708ff28-66f0-4445-82a5-b14fb55f21cb.flac` | ZH | 117% | Article insertion + compound splitting |
| Medium | `d3535ea7-e4cc-40b9-8e56-88ce68bdac27.flac` | BE | 120% | Perfect tense + phonetic confusion ("wuchs" → "ish") |
| Medium | `68974798-4b20-4a31-89e2-eb3a913e79f6.flac` | ZG | 100% | Complete word order inversion (same as Whisper) |

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