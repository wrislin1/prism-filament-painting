"""Publish a completed sibling copy without replacing a source or a different result."""

from contextlib import contextmanager
import os
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def staged_copy(source, destination, *, equivalent):
    source, destination = Path(source), Path(destination)
    if destination.is_symlink() or (destination.exists() and destination.samefile(source)):
        raise ValueError("migration requires a separate destination")
    with TemporaryDirectory(dir=destination.parent, prefix=".migration-") as directory:
        staged = Path(directory) / "copy"
        staged.touch(mode=0o600)
        yield staged
        # Atomic create-if-absent. A competing migrator can only see a complete file.
        try:
            os.link(staged, destination)
        except FileExistsError:
            if destination.is_symlink() or not equivalent(staged, destination):
                raise FileExistsError("destination contains a different result") from None
