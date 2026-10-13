"""Read-only native checkpoint storage evidence without allocating model tensors."""

from collections import Counter
from pathlib import Path
import time
import zipfile

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from .checkpointing import verify_checkpoint_checksum


def _storage_usage(value: object) -> dict:
    """Count unique serialized tensor storages in one checkpoint component."""
    counts, seen = Counter(), set()

    def visit(child: object) -> None:
        if isinstance(child, torch.Tensor):
            storage = child.untyped_storage()
            offset = getattr(storage, '_checkpoint_offset', None)
            if offset is None:
                raise ValueError(
                    'Runtime does not expose verified serialized storage offsets.'
                )
            key = (offset, storage.nbytes())
            if key not in seen:
                seen.add(key)
                counts[str(child.dtype)] += storage.nbytes()
        elif isinstance(child, dict):
            for item in child.values():
                visit(item)
        elif isinstance(child, (list, tuple)):
            for item in child:
                visit(item)

    visit(value)
    return dict(bytes=sum(counts.values()), by_dtype=dict(counts),
                unique_storages=len(seen))


def inspect_checkpoint_storage(path: Path) -> dict:
    """Verify native bytes and inspect unique serialized storages using fake tensors.

    Offsets, rather than fake storage identities, account correctly for tied weights.
    This inspects trusted local torch serialization; no optimizer or CUDA is created.
    """
    path = Path(path).absolute()
    if path.name.endswith('.ckpt.json'):
        raise ValueError('This evidence inspector expects an existing native checkpoint.')
    started = time.monotonic()
    fingerprint = verify_checkpoint_checksum(path)
    with FakeTensorMode():
        value = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    groups = {
        name: _storage_usage(value[name])
        for name in ('model_state_dict', 'optimizer_state_dict')
    }
    with zipfile.ZipFile(path) as archive:
        serialized = sum(
            record.file_size for record in archive.infolist()
            if '/data/' in record.filename
        )
    info = path.stat()
    optimizer = value['optimizer_state_dict']
    ids = [
        param
        for group in optimizer.get('param_groups', [])
        for param in group.get('params', [])
    ]
    complete = (
        bool(ids)
        and len(set(ids)) == len(ids)
        and set(ids) == set(optimizer.get('state', {}))
    )
    return dict(
        format='codexa-storage-evidence-v1',
        path=str(path),
        sha256=fingerprint,
        file_bytes=info.st_size,
        allocated_bytes=info.st_blocks * 512,
        serialized_tensor_bytes=serialized,
        components=groups,
        training_state=value['training_state'],
        run_id=value['run_id'],
        config=value['config'],
        inspection_seconds=time.monotonic() - started,
        optimizer_parameter_count=len(ids),
        complete_optimizer_state=complete,
        cuda_initialized=torch.cuda.is_initialized(),
        precision_conversion=False,
    )
