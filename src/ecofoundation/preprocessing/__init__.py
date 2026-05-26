"""Preprocessing: QC, normalization, HVG selection, optional batch correction.

Step 1 ships descriptive QC stats. Filtering / normalization / HVG selection
are added in later steps when needed (the user's example dataset is already
scVI-processed).
"""

from ecofoundation.preprocessing.qc_stats import QCStats, compute_qc_stats

__all__ = ["QCStats", "compute_qc_stats"]
