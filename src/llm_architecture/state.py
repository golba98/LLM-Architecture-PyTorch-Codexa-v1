"""Serializable training counters shared by checkpoint readers and writers."""
from dataclasses import asdict, dataclass
import math

def _json_value(value: object, field_name: str) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    return value

@dataclass
class TrainingState:
    """Mutable counters for a training run."""

    micro_step: int = 0
    optimizer_step: int = 0
    tokens_seen: int = 0
    completed_epochs: int = 0
    batches_in_epoch: int = 0
    best_validation_loss: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            name: _json_value(value, name)
            for name, value in asdict(self).items()
        }
