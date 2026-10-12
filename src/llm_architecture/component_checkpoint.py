"""Immutable lossless checkpoint objects and versioned portable manifests.

Only trusted local torch payloads are supported. The trainer owns reclamation.
The trainer owns role publication; this module has no training dependency.
"""

from collections.abc import Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid
import weakref

import torch

FORMAT = 'codexa-component-checkpoint-v1'
PARTS = {'model', 'optimizer', 'recovery'}
HEX = re.compile(r'[0-9a-f]{64}')
LIFECYCLE = 'codexa-continuous-retention-v1'


def digest_file(path: Path) -> str:
    """Hash a file using bounded buffers."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_directory(path: Path) -> None:
    """Create directories and synchronize each new parent entry."""
    if path.is_dir():
        return
    durable_directory(path.parent)
    path.mkdir(exist_ok=True)
    sync_directory(path.parent)


def atomic_json(path: Path, value: Mapping, *, immutable: bool = False,
                temporary_prefix: str = '.publish-') -> None:
    """Publish synchronized JSON; immutable names are never replaced."""
    durable_directory(path.parent)
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if Path(temporary_prefix).name != temporary_prefix:
        raise ValueError('Atomic metadata staging prefix must be path-safe.')
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=temporary_prefix)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != raw:
                    raise ValueError('Immutable manifest identity collision.')
        else:
            temporary.replace(path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _plain_path(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Checkpoint component path escapes its store.')
    for parent in (path, *path.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError('Checkpoint store paths cannot contain symlinks.')
    return path


class _ReadPaths:
    """Validate each shared ancestor once, within one read operation."""

    def __init__(self, root: Path):
        self.root = root
        self.checked = set()

    def path(self, relative: str) -> Path:
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Checkpoint component path escapes its store.')
        path = self.root / rel
        for parent in (path, *path.parents):
            if parent == self.root.parent:
                break
            if parent in self.checked:
                continue
            if parent.is_symlink():
                raise ValueError('Checkpoint store paths cannot contain symlinks.')
            self.checked.add(parent)
        return path


def manifest_document(path: Path, *, verified_objects: set | None = None,
                      _paths: _ReadPaths | None = None) -> tuple[dict, Path]:
    """Validate all immutable bytes; cache only within this operation."""
    path = Path(path).absolute()
    if not path.name.endswith('.ckpt.json') or not HEX.fullmatch(path.name[:-10]):
        raise ValueError('Invalid component checkpoint manifest name.')
    root = path.parents[2]
    if path.parent.parent.name != 'snapshots':
        raise ValueError('Manifest must remain inside its portable experiment store.')
    paths = _paths or _ReadPaths(root)
    if paths.root != root:
        raise ValueError('Read operation store identity mismatch.')
    paths.path(str(path.relative_to(root)))
    if path.stat().st_size > 1024**2:
        raise ValueError('Checkpoint manifest exceeds the metadata safety bound.')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != path.name[:-10]:
        raise ValueError('Checkpoint manifest digest mismatch.')
    value = json.loads(raw)
    store = json.loads(paths.path('store.json').read_text())
    if not isinstance(value, dict) or not isinstance(store, dict):
        raise ValueError('Checkpoint manifest and store marker must be objects.')
    if value.get('format') != FORMAT or store.get('format') != FORMAT:
        raise ValueError('Unsupported component checkpoint format.')
    if store.get('retention_policy_sha256'):
        policy_raw = paths.path('retention-policy.json').read_bytes()
        if hashlib.sha256(policy_raw).hexdigest() != store['retention_policy_sha256']:
            raise ValueError('Retention policy identity mismatch.')
        policy = json.loads(policy_raw)
        if not isinstance(policy, dict):
            raise ValueError('Invalid checkpoint retention policy.')
        if policy.get('continuous') is True and store.get('lifecycle') != LIFECYCLE:
            raise ValueError('Continuous-store reader protocol is missing or unsupported.')
    if value.get('store_id') != store.get('store_id'):
        raise ValueError('Checkpoint store identity mismatch.')
    if value.get('run_id') != path.parent.name:
        raise ValueError('Checkpoint manifest run identity mismatch.')
    if not isinstance(value.get('components'), dict) or set(value['components']) != PARTS:
        raise ValueError('Checkpoint components are incomplete.')
    checked = verified_objects if verified_objects is not None else set()
    root_key = str(root.resolve())
    for record in value['components'].values():
        if not isinstance(record, dict):
            raise ValueError('Invalid component record.')
        digest, size = record.get('sha256'), record.get('bytes')
        if not isinstance(digest, str) or not HEX.fullmatch(digest) or type(size) is not int or size <= 0:
            raise ValueError('Invalid component identity or size.')
        expected = f'objects/{digest[:2]}/{digest}.pt'
        if record.get('path') != expected:
            raise ValueError('Invalid component path.')
        component = paths.path(expected)
        if component.stat().st_size != size:
            raise ValueError('Checkpoint component size mismatch.')
        key = (root_key, digest)
        if key not in checked and digest_file(component) != digest:
            raise ValueError('Checkpoint component digest mismatch.')
        checked.add(key)
    return value, root


class _LeasedPayload(dict):
    """Dictionary whose mapping/storage lifetimes hold a reader lease."""


class _ReaderLease:
    def __init__(self, path: Path):
        self.stream = path.open('rb')
        fcntl.flock(self.stream, fcntl.LOCK_SH)
        self.references = 1

    def retain(self, value):
        self.references += 1
        weakref.finalize(value, self.release)

    def release(self):
        self.references -= 1
        if self.references == 0:
            self.stream.close()


def reader_lock_path(path: Path) -> Path:
    return path.with_name('.' + path.name + '.reader.lock')


@contextmanager
def reader_admission(root: Path, *, exclusive: bool = False):
    """Close admission while reclamation checks existing mapped-storage leases.

    Lock order is writer -> admission -> snapshot. Readers never take the writer
    lock. Missing gates in enrolled stores fail closed; old stores are unchanged.
    """
    marker = json.loads(_plain_path(root, 'store.json').read_text())
    if not isinstance(marker, dict) or marker.get('format') != FORMAT:
        raise ValueError('Invalid reader-admission store marker.')
    if marker.get('lifecycle') not in (None, LIFECYCLE):
        raise ValueError('Unsupported reader-admission protocol.')
    if marker.get('lifecycle') == LIFECYCLE and not marker.get('retention_policy_sha256'):
        raise ValueError('Enrolled store is missing its policy identity.')
    if marker.get('lifecycle') != LIFECYCLE:
        if exclusive:
            raise ValueError('Reclamation requires a newly enrolled continuous store.')
        yield
        return
    gate = _plain_path(root, '.reader-admission.lock')
    with gate.open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def _lease_payload(value: dict, lease: _ReaderLease) -> dict:
    result = _LeasedPayload(value)
    lease.retain(result)
    seen = set()
    def visit(child):
        if type(child) in (int, float, str, bool, bytes, type(None)):
            return
        if isinstance(child, torch.Tensor):
            storage = child.untyped_storage()
            if storage._cdata not in seen:
                seen.add(storage._cdata)
                lease.retain(storage)
        elif isinstance(child, Mapping):
            for item in child.values():
                visit(item)
        elif isinstance(child, (tuple, list)):
            for item in child:
                visit(item)
    visit(result)
    return result


def read_components(path: Path, *, model_only: bool = False, metadata_only: bool = False,
                    map_location='cpu', verified_objects: set | None = None) -> dict:
    """Verify once and reconstruct state; bounded-store mmap storage pins a lease."""
    path = Path(path).absolute()
    root = path.parents[2]
    with reader_admission(root):
        return _read_components_admitted(path, model_only=model_only,
            metadata_only=metadata_only, map_location=map_location,
            verified_objects=verified_objects)


def _read_components_admitted(path: Path, *, model_only: bool = False,
                             metadata_only: bool = False, map_location='cpu',
                             verified_objects: set | None = None) -> dict:
    """Internal primitive: caller must hold the store reader-admission gate."""
    root = path.parents[2]
    lease = None
    paths = None
    # Only fresh policy-enabled stores have lock sidecars. Old stores remain read-only.
    if (root / 'retention-policy.json').exists():
        paths = _ReadPaths(root)
        paths.path(str(path.relative_to(root)))
        lock = paths.path(str(reader_lock_path(path).relative_to(root)))
        lease = _ReaderLease(lock)
    try:
        manifest, root = manifest_document(path, verified_objects=verified_objects, _paths=paths)
        components = manifest['components']
        def read(name):
            return torch.load(root / components[name]['path'], map_location=map_location,
                              weights_only=False, mmap=True)
        recovery = read('recovery')
        if not isinstance(recovery, dict):
            raise ValueError('Invalid recovery component.')
        if recovery.get('run_id') != manifest['run_id'] or recovery.get('training_state') != manifest['training_state']:
            raise ValueError('Recovery metadata differs from manifest identity.')
        if 'model_state_dict' in recovery or 'optimizer_state_dict' in recovery:
            raise ValueError('Recovery component overlaps tensor components.')
        if not metadata_only:
            recovery['model_state_dict'] = read('model')
            if not model_only:
                recovery['optimizer_state_dict'] = read('optimizer')
        return _lease_payload(recovery, lease) if lease is not None else recovery
    finally:
        if lease is not None:
            lease.release()


@contextmanager
def open_components(path: Path, **kwargs):
    """Managed read; escaped mapped tensors continue holding their storage leases."""
    value = read_components(path, **kwargs)
    try:
        yield value
    finally:
        del value


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



def store_entries(root: Path, *, paths: bool = True):
    """Walk fresh inode metadata once, without following internal symlinks.

    Callers hold the store writer/admission lock. This is not a trust cache.
    """
    pending = [Path(root)]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ValueError('Unknown symlink inside checkpoint store.')
                info = entry.stat(follow_symlinks=False)
                path = Path(entry.path) if paths else entry.path
                yield path, info
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)


def store_usage(root: Path) -> dict:
    """Conservative inode accounting; includes temporary/orphan files and directories."""
    logical = allocated = 0
    seen = set()
    if root.is_symlink():
        raise ValueError('Unknown symlink inside checkpoint store.')
    paths = [(root, root.stat()), *store_entries(root, paths=False)]
    registry = root / 'retention-runs.json'
    if registry.exists():
        records = json.loads(registry.read_text())
        if not isinstance(records, dict):
            raise ValueError('Invalid checkpoint run registry.')
        for entry in records.values():
            index = root / entry['index']
            if index.exists():
                if index.is_symlink():
                    raise ValueError('Unknown symlink run index.')
                paths.append((index, index.stat()))
    for path, info in paths:
        key = (info.st_dev, info.st_ino)
        if key in seen:
            continue
        seen.add(key)
        logical += info.st_size
        allocated += info.st_blocks * 512
    return dict(logical_bytes=logical, allocated_bytes=allocated,
                accounted_bytes=max(logical, allocated))


class ComponentStore:
    """A single-writer experiment-group object store with no garbage collection."""

    def __init__(self, root: Path, *, reserve_bytes: int, retention_policy_sha256: str | None = None,
                 continuous_retention: bool = False):
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
                    raise FileExistsError('Optimized storage requires a fresh experiment store.')
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
                if digest_file(_plain_path(self.root, 'retention-policy.json')) != store['retention_policy_sha256']:
                    raise ValueError('Retention policy identity mismatch.')
            if self.retention_policy_sha256 and store.get('retention_policy_sha256') != self.retention_policy_sha256:
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
                    limit = policy['max_bytes'] - store_usage(self.root)['accounted_bytes'] - 8 * 1024**2 - self.serialization_reserve_bytes
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
                if target.stat().st_size != size or (key not in cache and digest_file(target) != digest):
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
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
            raise ValueError('Checkpoint run ID is not path-safe.')
        required = {'model_state_dict', 'optimizer_state_dict', 'scheduler_state',
                    'random_states', 'grad_scaler_state_dict', 'training_state', 'config',
                    'run_name', 'run_id', 'tokenizer_sha256', 'resume_context', 'format_version'}
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
        parts = dict(model=payload['model_state_dict'], optimizer=payload['optimizer_state_dict'],
                     recovery={k: v for k, v in payload.items() if k not in ('model_state_dict', 'optimizer_state_dict')})
        storages = [_tensor_storages(value) for value in parts.values()]
        if any(storages[i] & storages[j] for i in range(3) for j in range(i)):
            raise ValueError('Cross-component tensor aliases cannot be split losslessly.')
        # Upper bound includes staging and newly retained bytes; no duplicate reuse is assumed.
        tensor_bytes = sum(_storage_bytes(value) for value in parts.values())
        needed = self.reserve_bytes + 2 * tensor_bytes + 8 * 1024**2
        free = shutil.disk_usage(self.root).free
        if free < needed:
            raise OSError(f'Checkpoint capacity insufficient: free={free}, required={needed}; no state published.')
        cache = verified_objects if verified_objects is not None else set()
        records = {}
        for name, value in parts.items():
            self.component_size_limit = (policy['max_checkpoint_bytes']
                - sum(r['bytes'] for r in records.values())) if policy_path.exists() else None
            records[name] = self._object(value, verified_objects=cache)
        policy_path = self.root / 'retention-policy.json'
        if policy_path.exists():
            policy = json.loads(policy_path.read_text())
            if sum(r['bytes'] for r in records.values()) > policy['max_checkpoint_bytes']:
                raise OSError('Serialized checkpoint exceeds declared checkpoint byte bound.')
        document = dict(format=FORMAT, store_id=json.loads((self.root / 'store.json').read_text())['store_id'],
                        run_id=run_id, training_state=payload['training_state'], components=records)
        raw = (json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
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
