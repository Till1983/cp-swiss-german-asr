import json
import pandas as pd
import jiwer
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm


class ErrorAnalyzer:
    def __init__(self, results_dir: str = "results/metrics"):
        """
        Initialize the Error Analyzer.
        
        Args:
            results_dir: Directory containing evaluation result JSON/CSV files.
        """
        self.results_dir = Path(results_dir)
        self.models_data = {}

    def load_results(self):
        """
        Recursively load all result JSON files from the results directory.
        """
        print(f"Scanning {self.results_dir} for evaluation results...")
        json_files = list(self.results_dir.rglob("*_results.json"))
        
        if not json_files:
            print("No result files found.")
            return

        for file_path in json_files:
            # Extract model name from filename (e.g., 'whisper-large-v3_results.json' -> 'whisper-large-v3')
            model_name = file_path.stem.replace("_results", "")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Ensure we have the samples list
                if 'samples' in data:
                    # If samples are truncated in JSON (some evaluators do this), try loading CSV
                    csv_path = file_path.with_suffix('.csv')
                    if len(data['samples']) < data.get('total_samples', 0) and csv_path.exists():
                        print(f"Loading full samples from CSV for {model_name}...")
                        df = pd.read_csv(csv_path)
                        # Convert DataFrame back to list of dicts to match JSON structure
                        data['samples'] = df.to_dict('records')
                    
                    self.models_data[model_name] = data
                    print(f"Loaded {len(data['samples'])} samples for {model_name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    def _normalize(self, text: str) -> str:
        """Basic normalization wrapper."""
        if not isinstance(text, str):
            return ""
        return " ".join(text.lower().split())

    def analyze_model_errors(self, model_name: str, top_n_percent: float = 0.10) -> Dict[str, Any]:
        """
        Analyze errors for a specific model.
        
        Args:
            model_name: Name of the model to analyze.
            top_n_percent: Percentage of worst samples to analyze (0.0 to 1.0).
            
        Returns:
            Dictionary containing error statistics and patterns.
        """
        if model_name not in self.models_data:
            raise ValueError(f"Model {model_name} not found.")

        samples = self.models_data[model_name]['samples']
        
        # Filter out empty references
        valid_samples = [s for s in samples if self._normalize(s.get('reference', ''))]
        
        # Sort by WER descending to find worst performers
        sorted_samples = sorted(valid_samples, key=lambda x: x.get('wer', 0), reverse=True)
        cutoff_index = int(len(sorted_samples) * top_n_percent)
        worst_samples = sorted_samples[:cutoff_index]
        
        print(f"Analyzing top {len(worst_samples)} worst samples for {model_name}...")

        # Initialize counters
        error_counts = {
            'substitutions': 0,
            'deletions': 0,
            'insertions': 0,
            'hits': 0
        }
        
        dialect_stats = defaultdict(lambda: {
            'substitutions': 0, 'deletions': 0, 'insertions': 0, 'hits': 0, 'count': 0
        })
        
        # Word confusion matrix: (reference_word, hypothesis_word) -> count
        substitutions_map = Counter()
        deletions_map = Counter()
        insertions_map = Counter()
        
        # Dialect-specific confusion
        dialect_confusions = defaultdict(Counter)

        for sample in tqdm(worst_samples, desc=f"Processing {model_name}"):
            ref = self._normalize(sample['reference'])
            hyp = self._normalize(sample['hypothesis'])
            dialect = sample.get('dialect', 'unknown')
            
            if not ref: continue

            # Use jiwer to get alignment
            out = jiwer.process_words(ref, hyp)
            
            # Update global counts
            error_counts['substitutions'] += out.substitutions
            error_counts['deletions'] += out.deletions
            error_counts['insertions'] += out.insertions
            error_counts['hits'] += out.hits
            
            # Update dialect stats
            dialect_stats[dialect]['substitutions'] += out.substitutions
            dialect_stats[dialect]['deletions'] += out.deletions
            dialect_stats[dialect]['insertions'] += out.insertions
            dialect_stats[dialect]['hits'] += out.hits
            dialect_stats[dialect]['count'] += 1

            # Analyze alignments for specific word patterns
            for alignment in out.alignments[0]:
                if alignment.type == 'substitute':
                    ref_word = out.references[alignment.ref_start_idx]
                    hyp_word = out.hypothesis[alignment.hyp_start_idx]
                    substitutions_map[(ref_word, hyp_word)] += 1
                    dialect_confusions[dialect][(ref_word, hyp_word)] += 1
                elif alignment.type == 'delete':
                    ref_word = out.references[alignment.ref_start_idx]
                    deletions_map[ref_word] += 1
                elif alignment.type == 'insert':
                    hyp_word = out.hypothesis[alignment.hyp_start_idx]
                    insertions_map[hyp_word] += 1

        # Calculate percentages
        total_ops = sum(error_counts.values())
        error_distribution = {k: (v / total_ops * 100) if total_ops > 0 else 0 for k, v in error_counts.items()}

        # Format top confusions
        top_substitutions = [
            {"ref": k[0], "hyp": k[1], "count": v} 
            for k, v in substitutions_map.most_common(20)
        ]
        
        # Format dialect specific patterns
        dialect_patterns = {}
        for dialect, confusions in dialect_confusions.items():
            dialect_patterns[dialect] = [
                {"ref": k[0], "hyp": k[1], "count": v}
                for k, v in confusions.most_common(10)
            ]

        return {
            "model": model_name,
            "samples_analyzed": len(worst_samples),
            "error_distribution": error_distribution,
            "dialect_stats": dict(dialect_stats),
            "top_substitutions": top_substitutions,
            "top_deletions": [
                {"word": k, "count": v} for k, v in deletions_map.most_common(20)
            ],
            "top_insertions": [
                {"word": k, "count": v} for k, v in insertions_map.most_common(20)
            ],
            "dialect_patterns": dialect_patterns,
            "worst_samples_examples": [
                {
                    "ref": s['reference'],
                    "hyp": s['hypothesis'],
                    "wer": s['wer'],
                    "dialect": s.get('dialect')
                } for s in worst_samples[:5]
            ]
        }

    def generate_summary_report(self, output_path: str = "results/error_analysis_report.json"):
        """
        Generate a comprehensive report for all loaded models.
        """
        if not self.models_data:
            self.load_results()
            
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "models": {}
        }
        
        for model_name in self.models_data:
            try:
                analysis = self.analyze_model_errors(model_name)
                report["models"][model_name] = analysis
            except Exception as e:
                print(f"Failed to analyze {model_name}: {e}")
                
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"\nAnalysis complete. Report saved to {output_path}")
        
        # Print brief summary to console
        self._print_console_summary(report)

    def _print_console_summary(self, report: Dict):
        """Print a human-readable summary to the console."""
        print("\n" + "="*60)
        print("ERROR ANALYSIS SUMMARY")
        print("="*60)
        
        for model_name, data in report["models"].items():
            print(f"\nModel: {model_name}")
            dist = data["error_distribution"]
            print(f"  Error Distribution (Top 10% worst samples):")
            print(f"    - Substitutions: {dist['substitutions']:.1f}%")
            print(f"    - Deletions:     {dist['deletions']:.1f}%")
            print(f"    - Insertions:    {dist['insertions']:.1f}%")
            
            print("  Top 3 Confusions:")
            for item in data["top_substitutions"][:3]:
                print(f"    - '{item['ref']}' → '{item['hyp']}' ({item['count']} times)")

if __name__ == "__main__":
    analyzer = ErrorAnalyzer()
    analyzer.generate_summary_report()