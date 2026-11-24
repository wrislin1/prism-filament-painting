# Contributing to Prism

Open a GitHub issue for a reproducible bug or a feature request. Use invented
spools, generated images and a temporary data directory. Include the version,
steps, expected result and actual result. A maintainer checks for duplicates,
tries the example, labels the report and records the outcome on the issue.
Report security problems using the instructions in SECURITY.md.

Pull requests are welcome. By submitting original code, tests or documentation,
you agree to license your contribution under the project's MIT licence.
Declare whether you borrowed any code, data, equations or assets. For each
borrowed item, identify the affected files, upstream URL and exact revision,
licence, copyright holder, changes and any required notice. Explain the rights
to redistribute data separately from the licence of its reader. Use synthetic
test inputs you may distribute.

A maintainer reviews new dependencies, assets and copied blocks, checks the
upstream licence at the stated revision and verifies the credits and full texts.
Restricted material and material with unexplained origin cannot be accepted.
Update THIRD_PARTY_NOTICES.md and the archive inventory for changed obligations.

## Versions and releases

Use semantic versions (MAJOR.MINOR.PATCH). Increment MAJOR for an incompatible
public API, command or file-format change, MINOR for compatible features and
PATCH for compatible fixes. During 0.x development, flag breaking changes in
release notes and increment MINOR. Record user-visible changes in CHANGELOG.md.
Release candidates use an rc suffix such as 0.2.0rc1 in package metadata.

The maintainer reviews each changed destination and its origin before updating
DESTINATION-SHA256.json. Include every regular source file except the manifest
itself. For backend or file-list changes, review generated distribution members
and update ci/builder-metadata.json. Run the distribution checks on the proposed
commit. Check both installation reports, skip counts, licence inventories, tree
ID and distribution hashes. Publish release notes identifying the reviewed
version, changes and artifact hashes. A failed check holds the release.
