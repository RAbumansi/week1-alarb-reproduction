"""ALARB benchmark harness.

An independent implementation of the tasks defined in "ALARB: An Arabic Legal
Argument Reasoning Benchmark" (arXiv:2510.00694). The authors released the
dataset but not the benchmark code; this package rebuilds it from the prompts
published in the paper's appendices. See DEVIATIONS.md for where this
implementation necessarily departs from the paper.
"""

__all__ = ["corpus", "data", "outcomes"]
