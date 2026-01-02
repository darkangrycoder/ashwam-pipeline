# -*- coding: utf-8 -*-
"""CLI Entry Point for Ashwam Evidence-Grounded Extraction & Evaluation"""

import argparse
import json
from datetime import datetime
import time
from pathlib import Path

from pipeline import FixedProductionPipeline
from extractor import FixedProductionRuleBasedExtractor
from evaluator import EnhancedProductionEvaluator
from utils import DataLoader, SemanticObject


class AshwamEvalCLI:
    """Command Line Interface for the pipeline"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Ashwam Evidence-Grounded Extraction & Evaluation Pipeline',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python ashwam_eval.py run --data ./data --out ./results
  python ashwam_eval.py extract --journals ./data/journals.jsonl --out ./predictions.jsonl
  python ashwam_eval.py evaluate --gold ./data/gold.jsonl --pred ./predictions.jsonl --out ./scores
            """
        )
        self.setup_parser()

    def setup_parser(self):
        subparsers = self.parser.add_subparsers(dest='command', help='Command to execute')

        # Run command
        run_parser = subparsers.add_parser('run', help='Run full pipeline')
        run_parser.add_argument('--data', type=str, default='./data',
                               help='Path to data directory')
        run_parser.add_argument('--out', type=str, default='./output',
                               help='Output directory')
        run_parser.add_argument('--debug', action='store_true',
                               help='Enable debug mode')
        run_parser.add_argument('--similarity', type=float, default=0.6,
                               help='Similarity threshold for matching (default: 0.6)')

        # Extract command
        extract_parser = subparsers.add_parser('extract', help='Extract only')
        extract_parser.add_argument('--journals', type=str, required=True,
                                   help='Path to journals.jsonl file')
        extract_parser.add_argument('--out', type=str, required=True,
                                   help='Output file for predictions')
        extract_parser.add_argument('--debug', action='store_true',
                                   help='Enable debug mode')

        # Evaluate command
        evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate only')
        evaluate_parser.add_argument('--gold', type=str, required=True,
                                    help='Path to gold.jsonl file')
        evaluate_parser.add_argument('--pred', type=str, required=True,
                                    help='Path to predictions.jsonl file')
        evaluate_parser.add_argument('--out', type=str, default='./scores',
                                    help='Output directory for scores')
        evaluate_parser.add_argument('--debug', action='store_true',
                                    help='Enable debug mode')
        evaluate_parser.add_argument('--similarity', type=float, default=0.6,
                                    help='Similarity threshold for matching (default: 0.6)')

    def run(self):
        args = self.parser.parse_args()

        if args.command == 'run':
            self.run_pipeline(args)
        elif args.command == 'extract':
            self.run_extraction(args)
        elif args.command == 'evaluate':
            self.run_evaluation(args)
        else:
            self.parser.print_help()

    def run_pipeline(self, args):
        """Run full pipeline"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting pipeline...")
        start_time = time.time()

        # Construct file paths
        data_dir = Path(args.data)
        journals_file = data_dir / 'journals.jsonl'
        gold_file = data_dir / 'gold.jsonl'

        if not journals_file.exists():
            print(f"Error: journals.jsonl not found at {journals_file}")
            return

        if not gold_file.exists():
            print(f"Error: gold.jsonl not found at {gold_file}")
            return

        # Run pipeline
        config = {
            'debug': args.debug,
            'similarity_threshold': args.similarity
        }

        pipeline = FixedProductionPipeline(config=config)
        results = pipeline.run_full_pipeline(
            journals_file=str(journals_file),
            gold_file=str(gold_file),
            output_dir=args.out
        )

        elapsed = time.time() - start_time
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Pipeline completed in {elapsed:.1f} seconds")

        return results

    def run_extraction(self, args):
        """Run extraction only"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting extraction...")
        start_time = time.time()

        # Load journals
        data_loader = DataLoader()
        journals = data_loader.load_journals(args.journals)

        # Extract
        extractor = FixedProductionRuleBasedExtractor(debug=args.debug)
        predictions = {}

        for journal_id, text in journals.items():
            if args.debug:
                print(f"Extracting from {journal_id}...")
            predictions[journal_id] = extractor.extract(text, journal_id)

        # Save predictions
        output_path = Path(args.out)
        with open(output_path, 'w', encoding='utf-8') as f:
            for journal_id, objects in predictions.items():
                entry = {
                    'journal_id': journal_id,
                    'items': objects  # Already in dict format with 'text' field
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        elapsed = time.time() - start_time
        total_objects = sum(len(objs) for objs in predictions.values())
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Extraction completed in {elapsed:.1f} seconds")
        print(f"Extracted {total_objects} objects from {len(predictions)} journals")
        print(f"Saved to: {output_path}")

    def run_evaluation(self, args):
        """Run evaluation only"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting evaluation...")
        start_time = time.time()

        # Load gold and predictions
        data_loader = DataLoader()
        evaluator = EnhancedProductionEvaluator(
            similarity_threshold=args.similarity,
            debug=args.debug
        )

        gold_objects = data_loader.load_gold_objects(args.gold)

        # Load predictions
        predictions = {}
        pred_data = data_loader.load_jsonl(args.pred)
        for entry in pred_data:
            journal_id = entry.get('journal_id')
            objects = []
            for item in entry.get('items', []):
                try:
                    obj = SemanticObject.from_dict(item)
                    objects.append(obj)
                except:
                    continue
            predictions[journal_id] = objects

        # Evaluate
        all_metrics = []
        for journal_id in gold_objects:
            if journal_id in predictions:
                metrics = evaluator.evaluate_journal(
                    gold_objects[journal_id],
                    predictions[journal_id]
                )
                metrics['journal_id'] = journal_id
                all_metrics.append(metrics)

        # Save results
        output_dir = Path(args.out)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save per-journal metrics
        per_journal_file = output_dir / "per_journal_scores.jsonl"
        with open(per_journal_file, 'w', encoding='utf-8') as f:
            for metric in all_metrics:
                f.write(json.dumps(metric, ensure_ascii=False) + '\n')

        # Compute and save aggregate
        if all_metrics:
            total_tp = sum(m["object_level"]["tp"] for m in all_metrics)
            total_fp = sum(m["object_level"]["fp"] for m in all_metrics)
            total_fn = sum(m["object_level"]["fn"] for m in all_metrics)

            micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

            aggregate = {
                "micro": {
                    "precision": round(micro_precision, 4),
                    "recall": round(micro_recall, 4),
                    "f1": round(micro_f1, 4),
                    "tp": total_tp,
                    "fp": total_fp,
                    "fn": total_fn
                },
                "total_journals": len(all_metrics)
            }

            summary_file = output_dir / "score_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(aggregate, f, indent=2, ensure_ascii=False)

            print(f"\nAggregate Metrics:")
            print(f"  Precision: {micro_precision:.3f}")
            print(f"  Recall:    {micro_recall:.3f}")
            print(f"  F1 Score:  {micro_f1:.3f}")
            print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")

        elapsed = time.time() - start_time
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Evaluation completed in {elapsed:.1f} seconds")
        print(f"Results saved to: {output_dir}")


def main():
    """Main CLI entry point"""
    cli = AshwamEvalCLI()
    cli.run()


if __name__ == "__main__":
    main()