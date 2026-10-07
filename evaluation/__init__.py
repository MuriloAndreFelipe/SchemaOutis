"""
Scoring runs, and the retrieval sweep that feeds them.

Separate from `schemaoracle`, which produces rankings and traces; this package
judges them. It imports `schemaoracle`, never the reverse.

  matching       recall@k, MRR, and precision/recall with declining as silence
  blocking       the paged arm as a blocking scheme (Pair Completeness,
                 Reduction Ratio, block decisions)
  scoring        runs into the numbers the paper reports, pooled over studies
  competitors    Magneto and Valentine's matchers, run as baselines
  ablation       the retrieval sweep: encoder, serialiser, sampler, sample size
"""

from .blocking import BlockingReport, blocking_report
from .matching import (
    OUTCOMES,
    classify_columns,
    gold_beyond_k,
    gold_by_source,
    mrr,
    mrr_forfeited,
    precision_recall,
    recall_at_k,
    recall_at_ground_truth,
    recall_at_ground_truth_bounds,
    top1_accuracy,
)

__all__ = [
    "BlockingReport", "blocking_report",
    "OUTCOMES", "classify_columns", "gold_beyond_k", "gold_by_source",
    "mrr", "mrr_forfeited", "precision_recall", "recall_at_k",
    "recall_at_ground_truth", "recall_at_ground_truth_bounds",
    "top1_accuracy",
]
