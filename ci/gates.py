"""Audit the clean checkout and build both reviewed distributions offline."""

from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from zipfile import ZipFile

from archive_format import canonical_archive
import archive_inventory as audit
from naming import check_members, check_names

MANIFEST = "DESTINATION-SHA256.json"


def git(root, *args):
    return subprocess.check_output(["git", "-c", "safe.directory=*", "-C", root, *args])


def selection(root):
    manifest = json.loads((root / MANIFEST).read_bytes())
    expected = set(manifest) | {MANIFEST}
    actual = set()
    for directory, dirs, names in os.walk(root):
        if Path(directory) == root:
            dirs.remove(".git")
        for name in dirs + names:
            audit.require(not (Path(directory) / name).is_symlink(), "symlink in checkout")
        actual.update((Path(directory) / n).relative_to(root).as_posix() for n in names)
    audit.require(actual == expected, "unlisted or missing source file")
    for name in sorted(actual):
        data = (root / name).read_bytes()
        check_names(name, data)
        if name != MANIFEST:
            audit.require(sha256(data).hexdigest() == manifest[name], "source digest differs")
    return manifest


def public_policy(root, manifest):
    files = {name: {"destination": name, "sha256": digest} for name, digest in manifest.items()}
    ledger = json.loads((root / "ci/credits.json").read_text())
    for credit in ledger["credits"].values():
        if credit["kind"] == "published equations":
            credit["source"] = credit["publication_url"]
    audit.require(ledger["files"].keys() <= files.keys(), "credit refers to a missing file")
    ledger["files"] = {n: ledger["files"].get(n, "original") for n in files}
    return {"allowlist": {"files": files}, "provenance": ledger,
            "builder-metadata": json.loads((root / "ci/builder-metadata.json").read_text())}.__getitem__


def secrets(scanner, directory, expected=0):
    # A finding has its own status. Scanner errors can never satisfy a control.
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "rules.toml"
        config.write_text("[extend]\nuseDefault = true\n")
        result = subprocess.run([scanner, "dir", "--redact=100", "--no-banner", "--exit-code=23",
                                 "--config", config, "--gitleaks-ignore-path", os.devnull,
                                 "--ignore-gitleaks-allow", "--max-archive-depth=6", directory],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    audit.require(result.returncode == expected, "secret scan rejected or failed")


def scan_history(root, scanner, scratch, expected=0):
    objects = scratch / "objects"
    objects.mkdir(parents=True)
    (objects / "refs.txt").write_bytes(git(root, "for-each-ref", "--format=%(refname)"))
    for oid in git(root, "rev-list", "--objects", "--no-object-names", "--all", "HEAD").decode().splitlines():
        kind = git(root, "cat-file", "-t", oid).decode().strip()
        data = git(root, "cat-file", "-p", oid)
        (objects / (oid + "." + kind)).write_bytes(data)
    secrets(scanner, objects, expected)


def seeded_checks(scanner, scratch, archive, files, inspect):
    word = "For" + "Ge"
    plants = {"route": ("routes.py", f'@get("/api/{word}")'),
              "heading": ("help.md", f"# {word}"),
              "selector": ("style.css", ".h" + "f-score {}"),
              "filename": (word + ".txt", "content"),
              "JSON value": ("item.json", json.dumps({"value": word}))}
    for label, (name, content) in plants.items():
        planted = scratch / "names.zip"
        with ZipFile(planted, "w") as output:
            output.writestr(name, content)
        try:
            with ZipFile(planted) as archive_members:
                check_members({n: archive_members.read(n) for n in archive_members.namelist()})
        except ValueError as error:
            if str(error) != "restricted public name":
                raise
        else:
            raise AssertionError("naming control passed: " + label)
        print("Rejected " + label)

    for label, member, content, reason in (
        ("unlisted file", "extra.txt", b"invented", "unlisted"),
        ("altered source digest", "README.md", b"changed", "digest differs"),
        ("altered licence", "licenses/layertone-MIT.txt", b"changed", "digest differs"),
    ):
        path = scratch / archive.name
        path.write_bytes(canonical_archive(path, files | {member: content}, 1700000000))
        try:
            inspect(path)
        except ValueError as error:
            if reason not in str(error):
                raise
        else:
            raise AssertionError("archive control passed: " + label)
        print("Rejected " + label)

    # The scanner receives the actual ZIP and a real commit object separately.
    sample = "gh" + "p_" + ("A1bcD2efG" * 4)
    target = scratch / "sample.zip"
    with ZipFile(target, "w") as planted:
        planted.writestr("item.txt", sample)
    secrets(scanner, target, expected=23)
    print("Rejected synthetic secret in archive")
    planted = scratch / "history"
    git(scratch, "init", "-q", "--template=", str(planted))
    subprocess.run(["git", "-C", planted, "-c", "user.name=wrislin1",
                    "-c", "user.email=60682090+wrislin1@users.noreply.github.com",
                    "commit", "--allow-empty", "-qm", sample], check=True)
    # Test the same all-ref object traversal used on the real checkout.
    scan_history(planted, scanner, scratch / "seed", expected=23)
    print("Rejected synthetic secret in commit message")


def main(root, out, scanner):
    manifest = selection(root)
    print("Candidate names, file list and SHA-256 verified:", len(manifest), "listed files")
    config = tomllib.loads((root / "pyproject.toml").read_text())
    rows = audit.dependencies((root / "THIRD_PARTY_NOTICES.md").read_text(), config)
    audit.verify_installed(rows)
    print("Direct dependency declarations and installed licences verified")
    inspect = partial(audit.inventory, config=config, rows=rows, policy=public_policy(root, manifest))
    out.mkdir()
    epoch = int(git(root, "show", "-s", "--format=%ct", "HEAD"))
    os.environ["SOURCE_DATE_EPOCH"] = str(epoch)
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        scan_history(root, scanner, scratch)
        secrets(scanner, root)
        print("Redacted secret scans passed for the checkout and all reachable Git objects")
        source = scratch / "build"
        shutil.copytree(root, source, ignore=shutil.ignore_patterns(".git"))
        for command, pattern in (("build_sdist", "*.tar.gz"), ("build_wheel", "*.whl")):
            subprocess.run([sys.executable, "-I", "-B", "-c",
                            f"from setuptools import build_meta as b; import sys; b.{command}(sys.argv[1])",
                            str(out)], cwd=source, stdout=subprocess.DEVNULL, check=True)
            path, = out.glob(pattern)
            files, members = inspect(path)
            check_members(files)
            path.write_bytes(canonical_archive(path, files, epoch))
            secrets(scanner, path)
            report = audit.spdx(path, config, files, members)
            (out / (path.name + ".spdx.json")).write_text(json.dumps(report, indent=2) + "\n")
            print(path.name, len(files), "members", sha256(path.read_bytes()).hexdigest())
            if command == "build_sdist":
                seeded_checks(scanner, scratch, path, files, inspect)
                with tarfile.open(path) as archive:
                    archive.extractall(scratch / "sdist", filter="data")
                source, = (scratch / "sdist").iterdir()
    (out / "tree.txt").write_bytes(git(root, "rev-parse", "HEAD^{tree}"))
    shutil.copyfile(root / "CHANGELOG.md", out / "CHANGELOG.md")


if __name__ == "__main__":
    main(*(Path(arg).resolve() for arg in sys.argv[1:]))
