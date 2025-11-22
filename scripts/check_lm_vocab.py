import argparse
from pathlib import Path

def load_unigrams_from_arpa(arpa_path: Path):
    unigrams = []
    in_unigrams = False
    with open(arpa_path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            # Start of 1-gram section (ARPA uses "\1-grams:")
            if line.startswith("\\1-grams") or line.lower().startswith("\\1-grams"):
                in_unigrams = True
                continue
            # If we hit another section, stop
            if in_unigrams and line.startswith("\\"):
                break
            if in_unigrams:
                parts = line.split()
                # standard ARPA unigram line: <prob> <token> [<backoff>]
                if len(parts) >= 2:
                    unigram = parts[1]
                    unigrams.append(unigram)
    return unigrams

def load_labels_from_processor(model_name: str):
    try:
        from transformers import AutoProcessor
    except Exception as e:
        raise RuntimeError(f"transformers not available: {e}")
    proc = AutoProcessor.from_pretrained(model_name)
    # get vocab dictionary from processor.tokenizer or processor if present
    tokenizer = getattr(proc, "tokenizer", None)
    if tokenizer is None:
        # some processors expose .tokenizer differently; try proc if it has get_vocab
        if hasattr(proc, "get_vocab"):
            vocab = proc.get_vocab()
        else:
            raise RuntimeError("Processor does not expose tokenizer/get_vocab")
    else:
        vocab = tokenizer.get_vocab()
    # sort by id -> labels list
    labels = [k for k, v in sorted(vocab.items(), key=lambda it: it[1])]
    return labels

def report_diff(labels, unigrams, top_n=20):
    labels_set = set(labels)
    unigrams_set = set(unigrams)

    only_in_labels = sorted(list(labels_set - unigrams_set))
    only_in_unigrams = sorted(list(unigrams_set - labels_set))
    intersection = labels_set & unigrams_set

    print(f"Labels count: {len(labels)}")
    print(f"Unigrams count: {len(unigrams)}")
    print(f"Intersection: {len(intersection)}")
    print(f"Only in labels (sample {min(len(only_in_labels), top_n)}): {only_in_labels[:top_n]}")
    print(f"Only in unigrams (sample {min(len(only_in_unigrams), top_n)}): {only_in_unigrams[:top_n]}")
    if len(only_in_labels) == 0 and len(only_in_unigrams) == 0:
        print("✅ Perfect match between labels and LM unigrams.")
    else:
        overlap_ratio = len(intersection) / max(1, len(labels_set | unigrams_set))
        print(f"Overlap ratio: {overlap_ratio:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Compare model tokenizer labels vs KenLM unigrams")
    parser.add_argument("--model", required=True, help="HF model id or local path for processor (e.g. aware-ai/wav2vec2-large-xlsr-53-german-with-lm)")
    parser.add_argument("--lm-path", required=True, help="Path to kenLM .arpa file")
    parser.add_argument("--lowercase-lm", action="store_true", help="Compare using lowercased LM tokens as well (helpful if LM was lowercased)")
    args = parser.parse_args()

    model_name = args.model
    lm_path = Path(args.lm_path)
    if not lm_path.exists():
        print(f"❌ LM file not found: {lm_path}")
        return

    print(f"Loading labels from model processor: {model_name} ...")
    labels = load_labels_from_processor(model_name)
    print(f"Loaded {len(labels)} labels.")

    print(f"Parsing unigrams from ARPA: {lm_path} ...")
    unigrams = load_unigrams_from_arpa(lm_path)
    print(f"Parsed {len(unigrams)} unigram tokens from ARPA.")

    print("\n--- Direct comparison ---")
    report_diff(labels, unigrams)

    if args.lowercase_lm:
        print("\n--- Comparison against lowercased LM unigrams ---")
        report_diff(labels, [u.lower() for u in unigrams])

if __name__ == "__main__":
    main()