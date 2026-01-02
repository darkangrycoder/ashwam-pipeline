# -*- coding: utf-8 -*-
"""Evaluation engine for evidence-grounded extraction"""

from typing import List, Dict
from difflib import SequenceMatcher
import numpy as np

from utils import SemanticObject, Domain


class EnhancedProductionEvaluator:
    """
    High-performance evaluator for evidence-grounded extraction with fixed tests
    """

    def __init__(self, similarity_threshold: float = 0.6, debug: bool = False):
        self.similarity_threshold = similarity_threshold
        self.debug = debug

    def evaluate_journal(self, gold_objects: List[SemanticObject],
                        pred_objects: List[SemanticObject]) -> Dict:
        """Evaluate a single journal"""
        if not gold_objects and not pred_objects:
            return self._create_empty_metrics()

        # Match objects
        matches = self._match_objects_optimized(gold_objects, pred_objects)

        # Compute metrics
        metrics = {
            "object_level": self._compute_object_metrics(matches),
            "polarity_accuracy": self._compute_polarity_accuracy(matches["tp"]),
            "bucket_accuracy": self._compute_bucket_accuracy(matches["tp"]),
            "time_accuracy": self._compute_time_accuracy(matches["tp"]),
            "evidence_coverage": self._compute_evidence_coverage(pred_objects),
            "matches": {
                "tp_count": len(matches["tp"]),
                "fp_count": len(matches["fp"]),
                "fn_count": len(matches["fn"])
            }
        }

        if self.debug and matches["tp"]:
            self._debug_matches(matches["tp"])

        return metrics

    def _match_objects_optimized(self, gold: List[SemanticObject],
                               pred: List[SemanticObject]) -> Dict:
        """Optimized matching algorithm"""
        tp = []
        fp = pred.copy()
        fn = gold.copy()

        # Track matches
        matched_gold = set()
        matched_pred = set()

        # Pre-process evidence spans
        gold_evidence = [g.evidence_span.lower().strip() for g in gold]
        pred_evidence = [p.evidence_span.lower().strip() for p in pred]

        # First pass: Exact and substring matches
        for i, g_ev in enumerate(gold_evidence):
            if i in matched_gold:
                continue

            for j, p_ev in enumerate(pred_evidence):
                if j in matched_pred:
                    continue

                # Check domain match
                if gold[i].domain != pred[j].domain:
                    continue

                # Check for exact match or substring
                if g_ev == p_ev or g_ev in p_ev or p_ev in g_ev:
                    tp.append((gold[i], pred[j]))
                    matched_gold.add(i)
                    matched_pred.add(j)
                    break

        # Second pass: Fuzzy matches with similarity threshold
        remaining_gold = [i for i in range(len(gold)) if i not in matched_gold]
        remaining_pred = [j for j in range(len(pred)) if j not in matched_pred]

        for i in remaining_gold:
            best_match_idx = -1
            best_similarity = 0

            for j in remaining_pred:
                if gold[i].domain != pred[j].domain:
                    continue

                # Calculate similarity
                g_ev = gold_evidence[i]
                p_ev = pred_evidence[j]
                similarity = SequenceMatcher(None, g_ev, p_ev).ratio()

                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match_idx = j

            if best_match_idx != -1:
                tp.append((gold[i], pred[best_match_idx]))
                matched_gold.add(i)
                matched_pred.add(best_match_idx)
                # Remove from remaining pred
                remaining_pred = [j for j in remaining_pred if j != best_match_idx]

        # Identify remaining false positives and false negatives
        fp = [pred[j] for j in range(len(pred)) if j not in matched_pred]
        fn = [gold[i] for i in range(len(gold)) if i not in matched_gold]

        return {"tp": tp, "fp": fp, "fn": fn}

    def _compute_object_metrics(self, matches: Dict) -> Dict:
        """Compute precision, recall, F1"""
        tp_count = len(matches["tp"])
        fp_count = len(matches["fp"])
        fn_count = len(matches["fn"])

        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp_count,
            "fp": fp_count,
            "fn": fn_count
        }

    def _compute_polarity_accuracy(self, tp_pairs: List) -> float:
        if not tp_pairs:
            return 0.0
        correct = sum(1 for gold, pred in tp_pairs if gold.polarity == pred.polarity)
        return round(correct / len(tp_pairs), 4)

    def _compute_bucket_accuracy(self, tp_pairs: List) -> float:
        if not tp_pairs:
            return 0.0
        correct = 0
        for gold, pred in tp_pairs:
            if gold.domain == Domain.EMOTION:
                if gold.arousal_bucket == pred.arousal_bucket:
                    correct += 1
            else:
                if gold.intensity_bucket == pred.intensity_bucket:
                    correct += 1
        return round(correct / len(tp_pairs), 4)

    def _compute_time_accuracy(self, tp_pairs: List) -> float:
        if not tp_pairs:
            return 0.0
        correct = sum(1 for gold, pred in tp_pairs if gold.time_bucket == pred.time_bucket)
        return round(correct / len(tp_pairs), 4)

    def _compute_evidence_coverage(self, pred_objects: List[SemanticObject]) -> float:
        if not pred_objects:
            return 0.0
        valid = sum(1 for obj in pred_objects if obj.evidence_span and len(obj.evidence_span) > 5)
        return round(valid / len(pred_objects), 4)

    def _create_empty_metrics(self) -> Dict:
        return {
            "object_level": {"precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": 0, "fn": 0},
            "polarity_accuracy": 0,
            "bucket_accuracy": 0,
            "time_accuracy": 0,
            "evidence_coverage": 0,
            "matches": {"tp_count": 0, "fp_count": 0, "fn_count": 0}
        }

    def _debug_matches(self, tp_pairs: List):
        """Debug information for matches"""
        print("\n" + "="*60)
        print(f"DEBUG MATCHES (TP: {len(tp_pairs)}):")
        for i, (gold, pred) in enumerate(tp_pairs[:5], 1):
            print(f"\nMatch {i}:")
            print(f"  Gold: [{gold.domain.value}] '{gold.evidence_span[:60]}...'")
            print(f"  Pred: [{pred.domain.value}] '{pred.evidence_span[:60]}...'")
            print(f"  Polarity: Gold={gold.polarity.value}, Pred={pred.polarity.value}")
            print(f"  Time: Gold={gold.time_bucket.value}, Pred={pred.time_bucket.value}")
        if len(tp_pairs) > 5:
            print(f"\n... and {len(tp_pairs) - 5} more matches")
        print("="*60)