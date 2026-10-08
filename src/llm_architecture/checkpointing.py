"""Checksummed native model loading, independent of optimizers and trainers."""
from dataclasses import dataclass
import hashlib
import hmac
import math
from pathlib import Path
import torch
from torch import nn
from .state import TrainingState
from .model import ModelConfig
CHECKPOINT_FORMAT_VERSION = 1

@dataclass(frozen=True)
class InferenceCheckpoint:
    """Configuration and references restored for inference."""

    path: Path
    config: dict[str, object]
    training_state: TrainingState
    run_name: str
    run_id: str
    tokenizer_reference: str | None
    tokenizer_sha256: str | None

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def verify_checkpoint_checksum(path: str | Path) -> str:
    """Validate a checkpoint against its required SHA-256 sidecar."""

    checkpoint_path = Path(path)
    checksum_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".sha256"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not checksum_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint checksum does not exist: {checksum_path}"
        )
    fields = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != checkpoint_path.name:
        raise ValueError(f"Malformed checkpoint checksum file: {checksum_path}")
    expected = fields[0]
    actual = file_sha256(checkpoint_path)
    if not hmac.compare_digest(expected, actual):
        raise ValueError(
            f"Checkpoint checksum mismatch for {checkpoint_path}: "
            f"expected {expected}, got {actual}."
        )
    return actual

def _validate_training_state(value: object) -> TrainingState:
    if not isinstance(value, dict):
        raise ValueError("Checkpoint training state must be an object.")
    try:
        state = TrainingState(**value)
    except TypeError as error:
        raise ValueError("Checkpoint training state is invalid.") from error
    integer_fields = (
        state.micro_step,
        state.optimizer_step,
        state.tokens_seen,
        state.completed_epochs,
        state.batches_in_epoch,
    )
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in integer_fields
    ):
        raise ValueError("Checkpoint training counters must be non-negative.")
    if state.best_validation_loss is not None and (
        not math.isfinite(state.best_validation_loss)
        or state.best_validation_loss < 0
    ):
        raise ValueError("Checkpoint best validation loss is invalid.")
    return state

def load_model_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    map_location: str | torch.device = "cpu",
) -> InferenceCheckpoint:
    """Load only model weights and inference metadata from a trusted checkpoint."""

    checkpoint_path = Path(path)
    verify_checkpoint_checksum(checkpoint_path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except Exception as error:
        raise ValueError(
            f"Failed to deserialize checkpoint {checkpoint_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be an object.")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format {payload.get('format_version')!r}."
        )
    model_state = payload.get("model_state_dict")
    config = payload.get("config")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint model state is invalid.")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint configuration is missing.")
    state = _validate_training_state(payload.get("training_state"))
    run_name = payload.get("run_name")
    run_id = payload.get("run_id")
    tokenizer_reference = payload.get("tokenizer_reference")
    tokenizer_sha256 = payload.get("tokenizer_sha256")
    if not isinstance(run_name, str) or not isinstance(run_id, str):
        raise ValueError("Checkpoint run identity is invalid.")
    if tokenizer_reference is not None and not isinstance(
        tokenizer_reference,
        str,
    ):
        raise ValueError("Checkpoint tokenizer reference is invalid.")
    if tokenizer_sha256 is not None and not isinstance(tokenizer_sha256, str):
        raise ValueError("Checkpoint tokenizer checksum is invalid.")
    model.load_state_dict(model_state, strict=True)
    return InferenceCheckpoint(
        path=checkpoint_path,
        config=config,
        training_state=state,
        run_name=run_name,
        run_id=run_id,
        tokenizer_reference=tokenizer_reference,
        tokenizer_sha256=tokenizer_sha256,
    )


def read_native_checkpoint(path: str | Path) -> dict:
    """Verify and map a trusted full, legacy SFT, or native export checkpoint."""
    verify_checkpoint_checksum(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be an object.")
    if payload.get("format_version") not in (None, CHECKPOINT_FORMAT_VERSION):
        raise ValueError("Unsupported checkpoint format.")
    config = payload.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Checkpoint model configuration is missing.")
    if not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("Checkpoint model state is missing.")
    return payload


def checkpoint_model_config(path: str | Path) -> ModelConfig:
    """Resolve architecture from a verified native checkpoint."""
    return ModelConfig(**read_native_checkpoint(path)["config"]["model"])
