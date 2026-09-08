"""CLI entry point for the loader benchmarking workflow.

Measures slides/s through the exact training read stack (packed store ->
dataset -> collator -> DataLoader -> optional H2D) across worker counts and
store residency modes. See oceanpath.workflows.benchmarking for options.
"""

from oceanpath.workflows.benchmarking import run_loader_benchmark

if __name__ == "__main__":
    run_loader_benchmark()
