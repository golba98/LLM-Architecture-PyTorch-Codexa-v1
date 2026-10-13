"""Verify manifests and protect mapped checkpoint tensors with reader leases."""

from collections.abc import Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import weakref

import torch

from ._component_files import (
    FORMAT, HEX, LIFECYCLE, PARTS, _ReadPaths, _plain_path,
    digest_file, reader_lock_path,
)


def manifest_document(
    path: Path,
    *,
    verified_objects: set | None = None,
    _paths: _ReadPaths | None = None,
) -> tuple[dict, Path]:
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
        if (
            not isinstance(digest, str)
            or not HEX.fullmatch(digest)
            or type(size) is not int
            or size <= 0
        ):
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


def read_components(
    path: Path,
    *,
    model_only: bool = False,
    metadata_only: bool = False,
    map_location='cpu',
    verified_objects: set | None = None,
) -> dict:
    """Verify once and reconstruct state; bounded-store mmap storage pins a lease."""
    path = Path(path).absolute()
    root = path.parents[2]
    with reader_admission(root):
        return _read_components_admitted(
            path,
            model_only=model_only,
            metadata_only=metadata_only,
            map_location=map_location,
            verified_objects=verified_objects,
        )


def _read_components_admitted(
    path: Path,
    *,
    model_only: bool = False,
    metadata_only: bool = False,
    map_location='cpu',
    verified_objects: set | None = None,
) -> dict:
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
        manifest, root = manifest_document(
            path, verified_objects=verified_objects, _paths=paths
        )
        components = manifest['components']

        def read(name):
            return torch.load(
                root / components[name]['path'],
                map_location=map_location,
                weights_only=False,
                mmap=True,
            )

        recovery = read('recovery')
        if not isinstance(recovery, dict):
            raise ValueError('Invalid recovery component.')
        if (
            recovery.get('run_id') != manifest['run_id']
            or recovery.get('training_state') != manifest['training_state']
        ):
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
