# Fine-Tuning Summary: Whisper Large-v2 on FHNW Swiss German

This note compares the zero-shot Whisper Large-v2 baseline against every fine-tuned Whisper Large-v2 variant that was run on the FHNW test split. The test set contains 863 samples across 17 dialect labels, but per-dialect conclusions only hold where the sample count is large enough to support them. I therefore exclude dialects with $n \le 10$ from the ranking tables below and treat low-count dialects with caution.

## Overall Effect Of Fine-Tuning

All fine-tuned runs improve the corpus-level metrics relative to zero-shot Whisper Large-v2. The gain is real, but it is modest: the best WER drop is a little under 4.5 points, and the BLEU gain stays in a narrow band of about 5.1 to 5.6 points. In other words, fine-tuning helps consistently, but the checkpoints mostly converge on the same performance range instead of changing the picture dramatically.

| Model | WER | Delta WER | CER | Delta CER | BLEU | Delta BLEU |
|---|---:|---:|---:|---:|---:|---:|
| Whisper Large-v2 | 24.98 | 0.00 | 12.18 | 0.00 | 59.81 | 0.00 |
| Baseline | 20.80 | -4.17 | 9.13 | -3.05 | 64.96 | +5.15 |
| Baseline-step1260 | 20.80 | -4.17 | 9.13 | -3.05 | 64.96 | +5.15 |
| EWC lambda 3000 | 20.50 | -4.48 | 9.01 | -3.18 | 65.42 | +5.61 |
| EWC lambda 3000-step1260 | 20.51 | -4.47 | 9.01 | -3.18 | 65.44 | +5.62 |
| EWC lambda 30000 | 20.49 | -4.49 | 9.22 | -2.96 | 65.31 | +5.50 |
| EWC lambda 30000-step1260 | 20.49 | -4.49 | 9.22 | -2.97 | 65.30 | +5.48 |
| EWC lambda 300000 | 20.99 | -3.98 | 9.61 | -2.57 | 65.05 | +5.24 |
| EWC lambda 300000-step1260 | 21.22 | -3.76 | 9.63 | -2.55 | 64.97 | +5.16 |

Two details matter here. First, the baseline and baseline-step1260 checkpoints are numerically identical on this test split. Second, the EWC runs with lambda 3000 and 30000 do the best job overall; the 300000 runs still beat zero-shot, but they give back some of the improvement.

## Per-Dialect Rankings

The tables below rank dialects from best to worst WER for each fine-tuned model. I only include dialects with $n > 10$. The change column shows the difference versus zero-shot Whisper Large-v2, so negative values mean improvement.
Small 0.01-point differences can appear if a reader recomputes the change from the rounded WER values shown in the tables. Those are rounding artifacts, not arithmetic errors; the change column is calculated from the raw metric values before rounding.

### Zero-Shot Whisper Large-v2

| Rank | Dialect | n | WER |
|---:|---|---:|---:|
| 1 | GR | 12 | 9.68 |
| 2 | UR | 15 | 18.44 |
| 3 | ZH | 144 | 21.39 |
| 4 | BL | 54 | 23.08 |
| 5 | TG | 50 | 23.99 |
| 6 | AG | 108 | 24.82 |
| 7 | LU | 51 | 25.78 |
| 8 | SG | 116 | 26.05 |
| 9 | BE | 203 | 27.28 |
| 10 | VS | 17 | 28.32 |
| 11 | SO | 36 | 30.40 |
| 12 | ZG | 30 | 33.79 |

### Baseline

| Rank | Dialect | n | WER | Change vs zero-shot |
|---:|---|---:|---:|---:|
| 1 | GR | 12 | 12.90 | +3.23 |
| 2 | BL | 54 | 17.40 | -5.68 |
| 3 | ZH | 144 | 18.09 | -3.30 |
| 4 | TG | 50 | 19.96 | -4.03 |
| 5 | SG | 116 | 20.86 | -5.19 |
| 6 | LU | 51 | 21.00 | -4.78 |
| 7 | VS | 17 | 21.39 | -6.94 |
| 8 | AG | 108 | 21.60 | -3.22 |
| 9 | BE | 203 | 22.15 | -5.13 |
| 10 | UR | 15 | 24.11 | +5.67 |
| 11 | SO | 36 | 25.28 | -5.11 |
| 12 | ZG | 30 | 30.34 | -3.45 |

### Baseline-step1260

| Rank | Dialect | n | WER | Change vs zero-shot |
|---:|---|---:|---:|---:|
| 1 | GR | 12 | 12.90 | +3.23 |
| 2 | BL | 54 | 17.40 | -5.68 |
| 3 | ZH | 144 | 18.09 | -3.30 |
| 4 | TG | 50 | 19.96 | -4.03 |
| 5 | SG | 116 | 20.86 | -5.19 |
| 6 | LU | 51 | 21.00 | -4.78 |
| 7 | VS | 17 | 21.39 | -6.94 |
| 8 | AG | 108 | 21.60 | -3.22 |
| 9 | BE | 203 | 22.15 | -5.13 |
| 10 | UR | 15 | 24.11 | +5.67 |
| 11 | SO | 36 | 25.28 | -5.11 |
| 12 | ZG | 30 | 30.34 | -3.45 |

### EWC Lambda 3000

| Rank | Dialect | n | WER | Change vs zero-shot |
|---:|---|---:|---:|---:|
| 1 | GR | 12 | 9.68 | +0.00 |
| 2 | BL | 54 | 16.12 | -6.96 |
| 3 | LU | 51 | 18.30 | -7.48 |
| 4 | ZH | 144 | 18.36 | -3.03 |
| 5 | SG | 116 | 19.61 | -6.45 |
| 6 | TG | 50 | 20.38 | -3.61 |
| 7 | UR | 15 | 21.28 | +2.84 |
| 8 | AG | 108 | 21.78 | -3.03 |
| 9 | BE | 203 | 22.45 | -4.83 |
| 10 | VS | 17 | 24.28 | -4.05 |
| 11 | SO | 36 | 25.28 | -5.11 |
| 12 | ZG | 30 | 29.66 | -4.14 |

### EWC Lambda 3000-step1260

| Rank | Dialect | n | WER | Change vs zero-shot |
|---:|---|---:|---:|---:|
| 1 | GR | 12 | 9.68 | +0.00 |
| 2 | BL | 54 | 16.12 | -6.96 |
| 3 | LU | 51 | 18.09 | -7.69 |
| 4 | ZH | 144 | 18.36 | -3.03 |
| 5 | SG | 116 | 19.79 | -6.27 |
| 6 | TG | 50 | 20.38 | -3.61 |
| 7 | UR | 15 | 21.28 | +2.84 |
| 8 | AG | 108 | 21.78 | -3.03 |
| 9 | BE | 203 | 22.45 | -4.83 |
| 10 | VS | 17 | 24.28 | -4.05 |
| 11 | SO | 36 | 25.28 | -5.11 |
| 12 | ZG | 30 | 29.66 | -4.14 |

### EWC Lambda 30000

| Rank | Dialect | n | WER | Change vs zero-shot |
|---:|---|---:|---:|---:|
| 1 | GR | 12 | 12.10 | +2.42 |
| 2 | BL | 54 | 17.22 | -5.86 |
| 3 | ZH | 144 | 18.23 | -3.16 |
| 4 | LU | 51 | 19.54 | -6.24 |
| 5 | SG | 116 | 20.14 | -5.91 |
| 6 | UR | 15 | 20.57 | +2.13 |
| 7 | TG | 50 | 21.02 | -2.97 |
| 8 | AG | 108 | 21.14 | -3.68 |
| 9 | VS | 17 | 21.39 | -6.94 |
| 10 | BE | 203 | 22.45 | -4.83 |
| 11 | SO | 36 | 24.15 | -6.25 |
| 12 | ZG | 30 | 26.55 | -7.24 |

### EWC Lambda 30000-step1260

| Rank | Dialect | n | WER | Change vs zero-shot |
|---:|---|---:|---:|---:|
| 1 | GR | 12 | 12.10 | +2.42 |
| 2 | BL | 54 | 17.22 | -5.86 |
| 3 | ZH | 144 | 18.23 | -3.16 |
| 4 | LU | 51 | 19.54 | -6.24 |
| 5 | UR | 15 | 19.86 | +1.42 |
| 6 | SG | 116 | 20.14 | -5.91 |
| 7 | TG | 50 | 21.02 | -2.97 |
| 8 | AG | 108 | 21.14 | -3.68 |
| 9 | VS | 17 | 21.39 | -6.94 |
| 10 | BE | 203 | 22.50 | -4.78 |
| 11 | SO | 36 | 24.15 | -6.25 |
| 12 | ZG | 30 | 26.55 | -7.24 |

### EWC Lambda 300000

| Rank | Dialect | n | WER | Change vs zero-shot |
|---:|---|---:|---:|---:|
| 1 | GR | 12 | 11.29 | +1.61 |
| 2 | ZH | 144 | 17.74 | -3.65 |
| 3 | BL | 54 | 18.86 | -4.21 |
| 4 | SG | 116 | 18.89 | -7.16 |
| 5 | UR | 15 | 19.15 | +0.71 |
| 6 | VS | 17 | 19.65 | -8.67 |
| 7 | LU | 51 | 21.00 | -4.78 |
| 8 | TG | 50 | 21.02 | -2.97 |
| 9 | AG | 108 | 22.98 | -1.84 |
| 10 | BE | 203 | 23.20 | -4.08 |
| 11 | SO | 36 | 25.28 | -5.11 |
| 12 | ZG | 30 | 31.38 | -2.41 |

### EWC Lambda 300000-step1260

| Rank | Dialect | n | WER | Change vs zero-shot |
|---:|---|---:|---:|---:|
| 1 | GR | 12 | 10.48 | +0.81 |
| 2 | UR | 15 | 14.18 | -4.26 |
| 3 | ZH | 144 | 17.54 | -3.85 |
| 4 | SG | 116 | 19.43 | -6.62 |
| 5 | VS | 17 | 19.65 | -8.67 |
| 6 | BL | 54 | 19.78 | -3.30 |
| 7 | TG | 50 | 21.44 | -2.55 |
| 8 | LU | 51 | 22.87 | -2.91 |
| 9 | AG | 108 | 23.53 | -1.29 |
| 10 | BE | 203 | 23.69 | -3.58 |
| 11 | SO | 36 | 24.15 | -6.25 |
| 12 | ZG | 30 | 27.59 | -6.21 |

## What Fine-Tuning Rewards And Punishes

Across the full fine-tuned sweep, the clear winners are BL, SG, VS, LU, SO, BE, ZG, ZH, TG, and AG. Each of these dialects improves every time relative to the zero-shot baseline, so fine-tuning acts as a stable gain on them rather than a gamble. VS stands out as the strongest consistent winner, and SG, LU, and BL also gain strongly.

The clear losers are GR and UR. GR is especially unstable: it sits above the baseline in most fine-tuned runs and only matches it in the two EWC lambda 3000 checkpoints. UR is mixed, but it still trends worse than the baseline on average, so it does not benefit reliably from fine-tuning.

No dialect with $n > 10$ looks truly unaffected. The closest cases are GR and UR, but both move enough across checkpoints that they still show a real response to fine-tuning. The evidence does not support a claim of neutrality for any filtered dialect.

## Caution On Low Sample Dialects

I exclude GL, SZ, NW, SH, and FR because their sample counts are at or below 10. That leaves too little evidence to decide whether a gain or loss reflects the model or just sampling noise. These dialects can look extreme on paper, but the numbers are too fragile to support a stable conclusion.

Dialects just above the threshold still need care. GR has 12 samples and UR has 15, so both sit in the danger zone where a few utterances can move the score noticeably. VS is less fragile than those two, but 17 samples is still not enough to treat its rank as settled.

## Conclusion

Fine-tuning Whisper Large-v2 helps on the FHNW corpus overall, but it helps unevenly. It reliably lowers WER on most dialects, yet it does not rescue every dialect equally and it can leave a few low-sample dialects looking worse. The safest reading is simple: fine-tuning improves the model, but the gains cluster by dialect and the small-count results remain provisional.