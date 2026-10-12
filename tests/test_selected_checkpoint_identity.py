"""Selected checkpoint identity is checked before tensor deserialization."""
import hashlib
from pathlib import Path

import pytest
import torch

from llm_architecture import checkpointing


def test_selected_identity_uses_one_full_hash(tmp_path: Path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    torch.save({"config": {"model": {}}, "model_state_dict": {"weight": torch.arange(8)}}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".pt.sha256").write_text(f"{digest} {path.name}\n")
    original = checkpointing.file_sha256
    calls = []

    def measured(candidate):
        calls.append(candidate)
        return original(candidate)

    monkeypatch.setattr(checkpointing, "file_sha256", measured)
    result = checkpointing.read_native_checkpoint(path, expected_sha256=digest)
    assert torch.equal(result["model_state_dict"]["weight"], torch.arange(8))
    assert len(calls) == 1
    with pytest.raises(ValueError, match="selected model"):
        checkpointing.read_native_checkpoint(path, expected_sha256="0" * 64)
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError, match="checksum mismatch"):
        checkpointing.read_native_checkpoint(path, expected_sha256=digest)
