"""Streaming, verified installation of explicitly listed data files (stdlib only)."""
from pathlib import Path, PurePosixPath
import hashlib
import shutil
import stat
import tempfile
import zipfile


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def destination(root, name):
    parts = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or parts.is_absolute()
            or any(p in {"", ".", ".."} for p in name.split("/"))):
        raise ValueError(f"Unsafe archive path: {name}")
    root = Path(root).resolve()
    path = root.joinpath(*parts.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Destination escapes data directory: {name}")
    # Do not follow an existing symlink, including dangling links.
    if any(p.is_symlink() for p in [path, *path.parents] if p != root and p.is_relative_to(root)):
        raise ValueError(f"Symlink in destination: {name}")
    return path


def installed(root, files):
    complete = True
    for row in files:
        path = destination(root, row["path"])
        if path.exists():
            if not path.is_file() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError(f"Existing file differs; preserved without overwriting: {path}")
        else:
            complete = False
    return complete


def install_zip(archive, root, files):
    """Validate every member before installing; retain existing matching files."""
    root = Path(root).resolve()
    if installed(root, files):
        return "already verified"
    expected = {r["path"]: r for r in files}
    if len(expected) != len(files):
        raise ValueError("Duplicate catalog entries")
    with tempfile.TemporaryDirectory(prefix="worcap-data-") as temporary, zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if len(members) != len(expected) or {m.filename for m in members} != set(expected):
            raise ValueError("Archive members differ from the recorded file list")
        for member in members:
            row = expected[member.filename]
            target = destination(temporary, member.filename)
            mode = member.external_attr >> 16
            if member.is_dir() or stat.S_ISLNK(mode) or member.file_size != row["bytes"]:
                raise ValueError(f"Invalid member type or size: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as source, target.open("xb") as out:
                shutil.copyfileobj(source, out, 1024 * 1024)
            if digest(target) != row["sha256"]:
                raise ValueError(f"Member hash mismatch: {member.filename}")
        # Check again before writing in case the destination changed during validation.
        installed(root, files)
        for row in files:
            target = destination(root, row["path"])
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            created = False
            try:
                with target.open("xb") as out, destination(temporary, row["path"]).open("rb") as source:
                    created = True
                    shutil.copyfileobj(source, out, 1024 * 1024)
            except BaseException:
                # Only remove an incomplete file created by this invocation.
                if created:
                    target.unlink()
                raise
    if not installed(root, files):
        raise ValueError("Installation did not complete")
    return "installed and verified"
