"""Immutable checkpoint writer and stable component-checkpoint import surface.

Only trusted local torch payloads are supported. The trainer owns reclamation
and role publication; this package has no training dependency.
"""

from collections.abc import Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import torch

from ._component_files import (
    FORMAT, PARTS, HEX, LIFECYCLE, _ReadPaths, _plain_path,
    atomic_json, digest_file, durable_directory, reader_lock_path,
    store_entries, store_usage, sync_directory,
)
from ._component_reader import (
    _LeasedPayload, _ReaderLease, _lease_payload, _read_components_admitted,
    manifest_document, open_components, read_components, reader_admission,
)


def _tensor_storages(value) -> set:
    found = set()
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError('Optimized storage currently requires strided tensors.')
        if value.numel():
            found.add((str(value.device), value.untyped_storage()._cdata))
        if value.is_floating_point() or value.is_complex():
            if not torch.isfinite(value).all().item():
                raise ValueError('Nonfinite checkpoint tensor.')
    elif isinstance(value, Mapping):
        for child in value.values():
            found.update(_tensor_storages(child))
    elif isinstance(value, (tuple, list)):
        for child in value:
            found.update(_tensor_storages(child))
    return found


class _HashWriter:
    def __init__(self, stream, max_bytes=None):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.max_bytes = max_bytes
        self.written = 0

    def write(self, raw):
        if self.max_bytes is not None and self.written + len(raw) > self.max_bytes:
            raise OSError('Checkpoint store byte quota exhausted during serialization.')
        written = self.stream.write(raw)
        self.written += written
        if written != len(raw):
            raise OSError('Short checkpoint write.')
        self.digest.update(raw)
        return written

    def flush(self):
        self.stream.flush()


class ComponentStore:
    """A single-writer experiment-group object store with no garbage collection."""

    def __init__(
        self,
        root: Path,
        *,
        reserve_bytes: int,
        retention_policy_sha256: str | None = None,
        continuous_retention: bool = False,
    ):
        self.root = Path(root).absolute()
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError('Checkpoint reserve must be a nonnegative integer.')
        self.reserve_bytes = reserve_bytes
        self.retention_policy_sha256 = retention_policy_sha256
        self.serialization_reserve_bytes = 0
        self.component_size_limit = None
        self.continuous_retention = continuous_retention
        # Granted by the recovery controller after capacity and authority checks.
        self.transient_snapshot_allowance = 0

    @contextmanager
    def writer(self):
        """Serialize writers; refuse adoption of non-store historical directories."""
        if self.root.exists() and not (self.root / 'store.json').exists():
            if any(p.name != '.writer.lock' for p in self.root.iterdir()):
                lock_path = _plain_path(self.root, '.writer.lock')
                # Another legitimate initializer may be publishing its marker.
                # Wait read-only; never create a lock inside a foreign directory.
                if lock_path.is_file():
                    with lock_path.open('rb') as initializing:
                        fcntl.flock(initializing, fcntl.LOCK_EX)
                if not (self.root / 'store.json').exists():
                    raise FileExistsError(
                        'Optimized storage requires a fresh experiment store.'
                    )
        durable_directory(self.root)
        _plain_path(self.root, '.writer.lock')
        with (self.root / '.writer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            new_marker = not (self.root / 'store.json').exists()
            if new_marker:
                marker = dict(format=FORMAT, store_id=uuid.uuid4().hex)
                if self.retention_policy_sha256:
                    marker['retention_policy_sha256'] = self.retention_policy_sha256
                if self.continuous_retention:
                    gate = self.root / '.reader-admission.lock'
                    with gate.open('xb') as stream:
                        stream.flush()
                        os.fsync(stream.fileno())
                    sync_directory(self.root)
                    marker['lifecycle'] = LIFECYCLE
                atomic_json(self.root / 'store.json', marker, immutable=True)
            store = json.loads(_plain_path(self.root, 'store.json').read_text())
            if not isinstance(store, dict) or store.get('format') != FORMAT:
                raise ValueError('Unsupported object store.')
            if not new_marker and store.get('retention_policy_sha256'):
                if (
                    digest_file(_plain_path(self.root, 'retention-policy.json'))
                    != store['retention_policy_sha256']
                ):
                    raise ValueError('Retention policy identity mismatch.')
            if (
                self.retention_policy_sha256
                and store.get('retention_policy_sha256') != self.retention_policy_sha256
            ):
                raise ValueError('Checkpoint retention policy drift.')
            if self.continuous_retention and store.get('lifecycle') != LIFECYCLE:
                raise ValueError('Continuous retention requires a fresh enrolled store.')
            yield

    def _object(self, value, *, verified_objects: set | None = None) -> dict:
        descriptor, name = tempfile.mkstemp(dir=self.root, prefix='.object-')
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                limit = None
                policy_path = self.root / 'retention-policy.json'
                if policy_path.exists():
                    policy = json.loads(policy_path.read_text())
                    # Keep 8 MiB for filesystem/manifest overhead; count existing orphans.
                    limit = (
                        policy['max_bytes']
                        - store_usage(self.root)['accounted_bytes']
                        - 8 * 1024**2
                        - self.serialization_reserve_bytes
                    )
                    if self.component_size_limit is not None:
                        limit = min(limit, self.component_size_limit)
                writer = _HashWriter(stream, max_bytes=limit)
                # File-like serialization fixes the archive root independently of staging names.
                torch.save(value, writer)
                writer.flush()
                os.fsync(stream.fileno())
                digest = writer.digest.hexdigest()
            size = temporary.stat().st_size
            relative = f'objects/{digest[:2]}/{digest}.pt'
            target = _plain_path(self.root, relative)
            durable_directory(target.parent)
            cache = verified_objects if verified_objects is not None else set()
            key = (str(self.root.resolve()), digest)
            if target.exists():
                if target.stat().st_size != size or (
                    key not in cache and digest_file(target) != digest
                ):
                    raise ValueError('Existing immutable checkpoint object is corrupt.')
            else:
                # Read/deserialize the staged bytes before any snapshot can reference them.
                if digest_file(temporary) != digest:
                    raise ValueError('Staged checkpoint digest mismatch.')
                torch.load(temporary, map_location='cpu', weights_only=False, mmap=True)
                os.link(temporary, target)
                sync_directory(target.parent)
            cache.add(key)
            return dict(path=relative, sha256=digest, bytes=size)
        finally:
            temporary.unlink(missing_ok=True)

    def snapshot(self, payload: Mapping, *, verified_objects: set | None = None) -> Path:
        """Publish objects and an immutable manifest while the writer lock is held."""
        run_id = payload.get('run_id')
        if (
            not isinstance(run_id, str)
            or not run_id
            or Path(run_id).name != run_id
            or run_id in ('.', '..')
        ):
            raise ValueError('Checkpoint run ID is not path-safe.')
        required = {
            'model_state_dict', 'optimizer_state_dict', 'scheduler_state',
            'random_states', 'grad_scaler_state_dict', 'training_state', 'config',
            'run_name', 'run_id', 'tokenizer_sha256', 'resume_context',
            'format_version',
        }
        if required - payload.keys() or payload['format_version'] != 1:
            raise ValueError('Incomplete or unsupported native recovery payload.')
        policy_path = self.root / 'retention-policy.json'
        if policy_path.exists():
            policy = json.loads(policy_path.read_text())
            limit = policy['max_snapshots'] + self.transient_snapshot_allowance
            if len(list((self.root / 'snapshots').glob('*/*.ckpt.json'))) >= limit:
                raise RuntimeError('Snapshot cap exhausted; no checkpoint published.')
        context = payload.get('resume_context') or {}
        if context.get('microstep_within_gradient_accumulation', 0) != 0:
            raise ValueError('Optimized v1 checkpoints require an optimizer boundary.')
        parts = dict(
            model=payload['model_state_dict'],
            optimizer=payload['optimizer_state_dict'],
            recovery={
                key: value for key, value in payload.items()
                if key not in ('model_state_dict', 'optimizer_state_dict')
            },
        )
        storages = [_tensor_storages(value) for value in parts.values()]
        if any(storages[i] & storages[j] for i in range(3) for j in range(i)):
            raise ValueError('Cross-component tensor aliases cannot be split losslessly.')
        # Upper bound includes staging and newly retained bytes; no duplicate reuse is assumed.
        tensor_bytes = sum(_storage_bytes(value) for value in parts.values())
        needed = self.reserve_bytes + 2 * tensor_bytes + 8 * 1024**2
        free = shutil.disk_usage(self.root).free
        if free < needed:
            raise OSError(
                f'Checkpoint capacity insufficient: free={free}, '
                f'required={needed}; no state published.'
            )
        cache = verified_objects if verified_objects is not None else set()
        records = {}
        for name, value in parts.items():
            self.component_size_limit = (
                policy['max_checkpoint_bytes']
                - sum(record['bytes'] for record in records.values())
                if policy_path.exists() else None
            )
            records[name] = self._object(value, verified_objects=cache)
        policy_path = self.root / 'retention-policy.json'
        if policy_path.exists():
            policy = json.loads(policy_path.read_text())
            if sum(r['bytes'] for r in records.values()) > policy['max_checkpoint_bytes']:
                raise OSError('Serialized checkpoint exceeds declared checkpoint byte bound.')
        document = dict(
            format=FORMAT,
            store_id=json.loads((self.root / 'store.json').read_text())['store_id'],
            run_id=run_id,
            training_state=payload['training_state'],
            components=records,
        )
        raw = (
            json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False)
            + '\n'
        ).encode()
        digest = hashlib.sha256(raw).hexdigest()
        path = _plain_path(self.root, f'snapshots/{run_id}/{digest}.ckpt.json')
        if (self.root / 'retention-policy.json').exists():
            lock = reader_lock_path(path)
            durable_directory(path.parent)
            with lock.open('ab') as stream:
                stream.flush()
                os.fsync(stream.fileno())
            sync_directory(path.parent)
        atomic_json(path, document, immutable=True)
        manifest_document(path, verified_objects=cache)
        return path


def _storage_bytes(value) -> int:
    storages = {}

    def visit(child):
        if isinstance(child, torch.Tensor):
            storage = child.untyped_storage()
            storages[(str(child.device), storage._cdata)] = storage.nbytes()
        elif isinstance(child, Mapping):
            for item in child.values():
                visit(item)
        elif isinstance(child, (list, tuple)):
            for item in child:
                visit(item)
    visit(value)
    return sum(storages.values())
