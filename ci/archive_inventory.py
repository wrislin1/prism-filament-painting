"""Verify Prism archives against reviewed source and emit one SPDX 2.3 document per artifact."""

import base64
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from hashlib import sha1, sha256
from importlib.metadata import distribution
from io import StringIO
from pathlib import PurePosixPath
import stat
import tarfile
from uuid import uuid4
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PACKAGE = "prism_filament_painting"
OWNER = "Copyright (c) 2026 wrislin1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dependencies(notices, config):
    """Read the human-facing table and compare every declaration, including markers."""
    section = notices.split("## Direct dependencies\n", 1)[-1]
    rows = {}
    for line in section.splitlines():
        if line.startswith("|") and not line.startswith(("| Scope ", "| ---")):
            scope, spec, license_id = (s.strip() for s in line.strip("|").split("|"))
            key = (scope, Requirement(spec))
            require(key not in rows and license_id, "duplicate dependency or missing licence")
            rows[key] = license_id
    project = config["project"]
    scopes = {"base": project["dependencies"], "build": config["build-system"]["requires"],
              **project["optional-dependencies"]}
    declared = {(scope, Requirement(s)) for scope, specs in scopes.items() for s in specs}
    require(rows.keys() == declared, "dependency table differs from pyproject declarations")
    return rows


def installed_licence(dist):
    meta = dist.metadata
    if expr := meta.get("License-Expression"):
        return expr
    legacy = (meta.get("License") or "").strip()
    if legacy in {"MIT", "BSD-3-Clause"}:
        return legacy
    classifiers = meta.get_all("Classifier", [])
    if "License :: OSI Approved :: MIT License" in classifiers and legacy.startswith("The MIT License (MIT)"):
        return "MIT"
    # Jinja2 3.1.6 declares only the BSD family. Verify its actual three-clause text.
    if "License :: OSI Approved :: BSD License" in classifiers:
        for file in dist.files or ():
            if str(file).endswith(".dist-info/licenses/LICENSE.txt"):
                if sha256(file.read_binary()).hexdigest() == "3b49dcee4105eb37bac10faf1be260408fe85d252b8e9df2e0979fc1e094437b":
                    return "BSD-3-Clause"
    raise ValueError(f"unrecognised licence metadata for {meta['Name']}")


def verify_installed(rows, lookup=distribution):
    for (_, req), licence in rows.items():
        dist = lookup(req.name)
        require(canonicalize_name(dist.metadata["Name"]) == canonicalize_name(req.name),
                f"installed name differs for {req.name}")
        require(installed_licence(dist) == licence, f"installed licence differs for {req.name}")
        if req.marker is None or req.marker.evaluate():
            require(req.specifier.contains(dist.version), f"installed version differs for {req.name}")


def check_metadata(data, config, rows, texts):
    meta = BytesParser().parsebytes(data)
    project = config["project"]
    expected = []
    for scope, req in rows:
        if scope == "build":
            continue
        if scope == "base":
            expected.append(req)
            continue
        declaration = str(req).split(";", 1)[0]
        platform = f"({req.marker}) and " if req.marker else ""
        expected.append(Requirement(f'{declaration}; {platform}extra == "{scope}"'))
    require(Counter(map(Requirement, meta.get_all("Requires-Dist", []))) == Counter(expected),
            "archive dependency metadata differs from notices")
    require(set(meta.get_all("Provides-Extra", [])) == set(project["optional-dependencies"]),
            "archive extras differ")
    require(meta["Name"] == project["name"] and meta["Version"] == project["version"]
            and meta["License-Expression"] == "MIT" and set(meta.get_all("License-File", [])) == texts,
            "archive identity or licence declarations differ")


@dataclass(frozen=True)
class Member:
    origin: str
    copyright: str
    digest: str | None
    licence: str = "MIT"
    source: str = ""


def expected_members(wheel, prefix, policy):
    allow = policy("allowlist")["files"]
    ledger = policy("provenance")
    adaptation = ledger["origins"]["layertone-events"]
    upstream = [adaptation, *(s for c in ledger["credits"].values() for s in c.get("sources", []))]
    by_text = {s["destination"]: s for s in upstream}
    texts = {"LICENSE", "THIRD_PARTY_NOTICES.md", *by_text}
    members = {}
    for source, row in allow.items():
        dest = row["destination"]
        key = ledger["files"].get(source)
        require(key in ledger["origins"], "unclassified source: " + source)
        origin = ledger["origins"][key]
        holder = OWNER
        label = "Prism"
        upstream_url = ""
        if dest in by_text:
            credit = by_text[dest]
            require(origin["kind"] == "licence-text" and row["sha256"] == credit["license_sha256"],
                    "upstream notice digest differs")
            label, holder = credit["repository"], credit["copyright"]
            upstream_url = "https://github.com/{repository}/blob/{revision}/LICENSE".format(**credit)
        elif origin["kind"] == "adaptation":
            label = "Prism + " + origin["repository"]
            holder += "\n" + origin["copyright"]
            upstream_url = "https://github.com/{repository}/blob/{revision}/{file}".format(**origin)
        else:
            require(origin["kind"] == "original", "unclassified source")
        require(origin["license"] == "MIT", "unreviewed member licence")
        if wheel and dest in texts:
            dest = prefix + ".dist-info/licenses/" + dest
        elif wheel and not dest.startswith(PACKAGE + "/"):
            continue
        members[dest] = Member(label, holder, row["sha256"], source=upstream_url)
    for name in texts:
        require((prefix + ".dist-info/licenses/" + name if wheel else name) in members,
                "required licence or notice missing from selection")
    generated = policy("builder-metadata")["wheel" if wheel else "sdist"]
    for name, digest in generated.items():
        require(name not in members, "generated member shadows source")
        members[name] = Member("Prism packaging metadata (setuptools)", OWNER, digest)
    return members, texts


def read_archive(path, wheel, prefix, expected):
    files = {}
    directories = {str(p) for name in expected for p in PurePosixPath(name).parents}

    def add(name, reader):
        require(name in expected and name not in files, "unlisted or duplicate archive member: " + repr(name))
        files[name] = reader()

    if wheel:
        with ZipFile(path) as archive:
            for info in archive.infolist():
                require(stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG), "nonregular wheel member")
                add(info.filename, lambda: archive.read(info))
    else:
        with tarfile.open(path, "r:gz") as archive:
            for item in archive:
                if item.isdir() and item.name == prefix:
                    continue
                require(item.name.startswith(prefix + "/"), "unexpected sdist root")
                name = item.name[len(prefix) + 1:]
                if item.isdir():
                    require(name in directories and name != ".", "unlisted sdist directory")
                else:
                    require(item.isfile(), "nonregular sdist member")
                    add(name, lambda: archive.extractfile(item).read())
    require(files.keys() == expected.keys(), "missing archive members: " + repr(sorted(expected.keys() - files.keys())))
    return files


def check_record(files, record):
    rows = list(csv.reader(StringIO(files[record].decode()), strict=True))
    expected = []
    for name, data in files.items():
        encoded = base64.urlsafe_b64encode(sha256(data).digest()).rstrip(b"=").decode()
        expected.append([name, "sha256=" + encoded, str(len(data))] if name != record else [name, "", ""])
    require(sorted(rows) == sorted(expected), "invalid wheel RECORD")


def check_credits(files, root, policy):
    """Require the actual scope, attribution and linked full texts even after a re-pin."""
    ledger = policy("provenance")
    notices = files[root + "THIRD_PARTY_NOTICES.md"].decode()
    adaptation = ledger["origins"]["layertone-events"]
    mapping = policy("allowlist")["files"]
    for source, key in ledger["files"].items():
        if key == "layertone-events":
            require(mapping[source]["destination"] in notices, "missing adaptation scope")
    sources = [adaptation]
    for source, credit in ledger["credits"].items():
        require(mapping[source]["destination"] in notices, "missing credit scope")
        if credit["kind"] == "published equations":
            require(credit["source"] in notices, "missing equation credit")
        sources.extend(credit.get("sources", []))
    for source in sources:
        require(all(value in notices for value in (source["repository"], source["revision"],
                                                   f'({source["destination"]})')) and source["copyright"].casefold() in notices.casefold(),
                "incomplete upstream credit")
    original = files[root + adaptation["destination"]].replace(adaptation["copyright"].encode(), OWNER.encode())
    require(files[root + "LICENSE"] == original, "incomplete project MIT licence")


def inventory(path, config, rows, policy):
    wheel = path.suffix == ".whl"
    prefix = PACKAGE + "-" + config["project"]["version"]
    expected, texts = expected_members(wheel, prefix, policy)
    files = read_archive(path, wheel, prefix, expected)
    metadata = [prefix + ".dist-info/METADATA"] if wheel else ["PKG-INFO", PACKAGE + ".egg-info/PKG-INFO"]
    for name in metadata:
        check_metadata(files[name], config, rows, texts)
    for name, member in expected.items():
        if wheel and name == prefix + ".dist-info/RECORD":
            check_record(files, name)
        else:
            require(sha256(files[name]).hexdigest() == member.digest, "archive digest differs: " + name)
    check_credits(files, prefix + ".dist-info/licenses/" if wheel else "", policy)
    return files, expected


def spdx(path, config, files, expected):
    archive_sha = sha256(path.read_bytes()).hexdigest()
    entries = []
    archive_root = "" if path.suffix == ".whl" else path.name.removesuffix(".tar.gz") + "/"
    for i, (name, data) in enumerate(sorted(files.items()), 1):
        member = expected[name]
        entries.append({
            "SPDXID": f"SPDXRef-File-{i}", "fileName": "./" + archive_root + name,
            "checksums": [{"algorithm": "SHA1", "checksumValue": sha1(data).hexdigest()},
                          {"algorithm": "SHA256", "checksumValue": sha256(data).hexdigest()}],
            "licenseConcluded": member.licence, "licenseInfoInFiles": ["NOASSERTION"],
            "copyrightText": member.copyright, "comment": " ".join(filter(None, (member.origin, member.source))),
        })
    hashes = sorted(f["checksums"][0]["checksumValue"] for f in entries)
    project = config["project"]
    package_id = "SPDXRef-Prism"
    return {
        "spdxVersion": "SPDX-2.3", "dataLicense": "CC0-1.0", "SPDXID": "SPDXRef-DOCUMENT",
        "name": path.name, "documentNamespace": f"urn:uuid:{uuid4()}",
        "creationInfo": {"created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "creators": ["Tool: Prism archive inventory"]},
        "documentDescribes": [package_id],
        "packages": [{"SPDXID": package_id, "name": project["name"], "versionInfo": project["version"],
                      "packageFileName": path.name, "downloadLocation": "NOASSERTION", "filesAnalyzed": True,
                      "packageVerificationCode": {"packageVerificationCodeValue": sha1("".join(hashes).encode()).hexdigest()},
                      "licenseDeclared": "MIT", "licenseConcluded": "MIT", "copyrightText": "NOASSERTION",
                      "licenseInfoFromFiles": ["NOASSERTION"],
                      "checksums": [{"algorithm": "SHA256", "checksumValue": archive_sha}]}],
        "files": entries,
        "relationships": [{"spdxElementId": package_id, "relationshipType": "CONTAINS",
                           "relatedSpdxElement": f["SPDXID"]} for f in entries],
    }
