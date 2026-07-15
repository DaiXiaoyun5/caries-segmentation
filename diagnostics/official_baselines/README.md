# Official baseline failure reports

The official comparison runners write a small `*_failure.json` file here only
when a run fails. Unlike `runs/` and Slurm output, this directory is not
ignored, so a failure report can be committed normally and inspected after a
GitHub pull. Reports contain the Python traceback, last completed startup
stage, PyTorch/CUDA versions, GPU name, and non-secret Slurm identifiers.

Successful reruns remove their stale failure report.
