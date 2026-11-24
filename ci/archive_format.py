"""Reproducible distribution headers and compression."""

from datetime import datetime, timezone
from gzip import GzipFile
from io import BytesIO
import tarfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def canonical_archive(path, files, epoch):
    """Write only inventoried regular files, with fixed order and metadata."""
    buf = BytesIO()
    if path.suffix == ".whl":
        stamp = datetime.fromtimestamp(epoch, timezone.utc).timetuple()[:6]
        with ZipFile(buf, "w") as archive:
            for name, data in sorted(files.items()):
                info = ZipInfo(name, stamp)
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data, compress_type=ZIP_DEFLATED, compresslevel=9)
    else:
        prefix = path.name.removesuffix(".tar.gz")
        with GzipFile(filename="", fileobj=buf, mode="wb", mtime=epoch) as compressed:
            with tarfile.open(mode="w", format=tarfile.USTAR_FORMAT, fileobj=compressed) as archive:
                for name, data in sorted(files.items()):
                    info = tarfile.TarInfo(prefix + "/" + name)
                    info.size = len(data)
                    info.mtime = epoch
                    info.mode = 0o644
                    archive.addfile(info, BytesIO(data))
    return buf.getvalue()
