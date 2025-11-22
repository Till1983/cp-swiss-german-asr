import argparse
from pathlib import Path
import re

def load_unigrams_from_arpa(arpa_path: Path, max_unigrams: int = None):
    unigrams = []
    in_unigrams = False
    with open(arpa_path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("\\1-grams") or line.lower().startswith("\\1-grams"):
                in_unigrams = True
                continue
            if in_unigrams and line.startswith("\\"):
                break
            if in_unigrams:
                parts = line.split()
                if len(parts) >= 2:
                    unigram = parts[1]
                    unigrams.append(unigram)
                    if max_unigrams and len(unigrams) >= max_unigrams:
                        break
    return unigrams

def load_labels_from_processor(model_name: str):
    from transformers import AutoProcessor, AutoConfig
    proc = AutoProcessor.from_pretrained(model_name)
    tokenizer = getattr(proc, "tokenizer", None)
    if tokenizer is None:
        if hasattr(proc, "get_vocab"):
            vocab = proc.get_vocab()
        else:
            raise RuntimeError("Processor does not expose tokenizer/get_vocab")
    else:
        vocab = tokenizer.get_vocab()
    labels = [k for k, v in sorted(vocab.items(), key=lambda it: it[1])]
    # try to get expected model vocab size from config (if available)
    try:
        config = AutoConfig.from_pretrained(model_name)
        model_vocab_size = getattr(config, "vocab_size", None)
    except Exception:
        model_vocab_size = None
    return labels, model_vocab_size

def normalize_token(tok: str):
    # Heuristic normalization to catch common mismatches:
    # - lowercase
    # - strip angle-bracket tokens like <pad>, <unk>
    # - remove leading byte markers and special whitespace markers (e.g. '▁')
    # - replace '|' with space
    t = tok
    t = t.replace("|", " ")
    t = t.replace("▁", "")  # common sentencepiece marker
    t = re.sub(r"^[\x00-\x1f]+", "", t)   # drop control prefix bytes
    t = re.sub(r"[\x00-\x1f]+$", "", t)
    t = re.sub(r"^<(.+)>$", r"\1", t)     # <pad> -> pad
    return t.lower().strip()

def analyze(labels, unigrams, top_n=30):
    labels_set = set(labels)
    unigrams_set = set(unigrams)

    unigrams_lower_set = {u.lower() for u in unigrams_set}
    labels_normalized = [normalize_token(l) for l in labels]
    unigrams_normalized_set = {normalize_token(u) for u in unigrams_set}

    exact_matches = [l for l in labels if l in unigrams_set]
    ci_matches = [l for l in labels if l.lower() in unigrams_lower_set and l not in exact_matches]
    norm_matches = [l for l in labels if normalize_token(l) in unigrams_normalized_set and l not in exact_matches and l not in ci_matches]

    only_in_labels = sorted(list(labels_set - unigrams_set))
    only_in_unigrams = sorted(list(unigrams_set - labels_set))

    multi_char_labels = [l for l in labels if len(l) > 1 and not (l.startswith("<") and l.endswith(">"))]
    special_token_labels = [l for l in labels if re.match(r"^<.*>$", l)]

    print("=== SUMMARY ===")
    print(f"Labels total: {len(labels)}")
    print(f"ARPA unigrams parsed: {len(unigrams)}")
    print(f"Exact matches (label ∈ ARPA): {len(exact_matches)}")
    print(f"Case-insensitive matches (but not exact): {len(ci_matches)}")
    print(f"Heuristic-normalized matches (but not exact/ci): {len(norm_matches)}")
    matched_total = len(set(exact_matches) | set(ci_matches) | set(norm_matches))
    print(f"Labels matched by any rule: {matched_total} / {len(labels)} ({matched_total/len(labels):.3%})")
    print()
    print("=== TOKEN STYLE ===")
    print(f"Labels with length > 1 (excluding explicit <...> tokens): {len(multi_char_labels)} (sample: {multi_char_labels[:10]})")
    print(f"Special tokens (angle brackets): {len(special_token_labels)} (sample: {special_token_labels[:10]})")
    print()
    print("=== MISMATCH EXAMPLES ===")
    print(f"Labels only in labels (sample {min(len(only_in_labels), top_n)}): {only_in_labels[:top_n]}")
    print(f"Labels matched only by lowercase (sample {min(len(ci_matches), top_n)}): {ci_matches[:top_n]}")
    print(f"Labels matched only by normalization (sample {min(len(norm_matches), top_n)}): {norm_matches[:top_n]}")
    print()
    print(f"ARPA unigrams not present in labels (sample {min(len(only_in_unigrams), top_n)}): {only_in_unigrams[:top_n]}")
    print()
    # show intersection ratio relative to label set and to arpa set
    print(f"Fraction of labels present exactly in ARPA: {len(exact_matches)/len(labels):.3%}")
    print(f"Fraction of labels matched after heuristics: {matched_total/len(labels):.3%}")
    # show a small normalized mapping table
    print("\n=== LABEL -> NORMALIZED -> IN_ARPA? ===")
    for i, lab in enumerate(labels[:min(40, len(labels))]):
        norm = normalize_token(lab)
        in_arpa = lab in unigrams_set
        in_arpa_ci = lab.lower() in unigrams_lower_set
        in_arpa_norm = norm in unigrams_normalized_set
        print(f"{i:03d}: '{lab}' -> '{norm}' | exact={in_arpa} ci={in_arpa_ci} norm={in_arpa_norm}")

def main():
    parser = argparse.ArgumentParser(description="Diagnose LM <> tokenizer label alignment")
    parser.add_argument("--model", required=True, help="HF model id or local path for processor (e.g. aware-ai/wav2vec2-large-xlsr-53-german-with-lm)")
    parser.add_argument("--lm-path", required=True, help="Path to kenLM .arpa file")
    parser.add_argument("--max-unigrams", type=int, default=None, help="Limit unigrams parsed (useful for very large ARPA)")
    args = parser.parse_args()

    model_name = args.model
    lm_path = Path(args.lm_path)
    if not lm_path.exists():
        print(f"❌ LM file not found: {lm_path}")
        return

    print(f"Loading labels for processor '{model_name}' ...")
    labels, model_vocab_size = load_labels_from_processor(model_name)
    print(f"Loaded {len(labels)} labels. Model config reported vocab_size={model_vocab_size}")

    print(f"Parsing ARPA unigrams from '{lm_path}' ... (this may take a while for large files)")
    unigrams = load_unigrams_from_arpa(lm_path, max_unigrams=args.max_unigrams)
    print(f"Parsed {len(unigrams)} unigrams.")

    analyze(labels, unigrams)

if __name__ == "__main__":
    main()