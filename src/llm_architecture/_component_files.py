"""Durable paths, immutable metadata, and disk accounting for component stores."""

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

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


def atomic_json(
    path: Path,
    value: Mapping,
    *,
    immutable: bool = False,
    temporary_prefix: str = '.publish-',
) -> None:
    """Publish synchronized JSON; immutable names are never replaced."""
    durable_directory(path.parent)
    raw = (
        json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
        + '\n'
    ).encode()
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


def reader_lock_path(path: Path) -> Path:
    return path.with_name('.' + path.name + '.reader.lock')


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
    return dict(
        logical_bytes=logical,
        allocated_bytes=allocated,
        accounted_bytes=max(logical, allocated),
    )
