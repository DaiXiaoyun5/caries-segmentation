"""Pure scheduling helpers for iteration- and epoch-budget baselines."""

from typing import Optional


def training_segment_update_limit(
    global_step: int,
    max_iterations: Optional[int],
    val_interval_iterations: Optional[int],
    next_val_step: int,
) -> Optional[int]:
    """Limit a training segment at the next hard iteration boundary."""

    limits = []
    if max_iterations is not None:
        limits.append(max(int(max_iterations) - int(global_step), 0))
    if val_interval_iterations is not None:
        if int(val_interval_iterations) <= 0:
            raise ValueError("val_interval_iterations must be positive")
        limits.append(max(int(next_val_step) - int(global_step), 0))
    return min(limits) if limits else None
