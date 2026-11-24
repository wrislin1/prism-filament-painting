# Rehearse a release candidate

Start only after the personal owner authorises the repository name and a private
GitHub Actions run. For an initial rehearsal, use the verified checkout and the expected
commit, tree, manifest and archive digests from its independent verification.
Keep that verification record on the maintainer's machine. The hosted jobs
need only the checkout. Git, GitHub CLI, jq and sha256sum are required locally.

The following Bash commands assume CANDIDATE is the verified checkout,
RECEIPT is the local JSON verification record, and STAGING_NAME is the
approved, unused staging slug. Run them from a separate scratch directory.
The personal owner's authenticated CLI session needs repository-creation
access, permission to push a workflow and access to Actions runs and artifacts.
Hosted Linux runners must be enabled for this repository.

```bash
set -euo pipefail
: "${CANDIDATE:?}" "${RECEIPT:?}" "${STAGING_NAME:?}"
[[ "$STAGING_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9-]*$ ]]
repo="wrislin1/$STAGING_NAME"
test "$(gh api user --jq .login)" = wrislin1
test "$(jq -r .attestation.result "$RECEIPT")" = pass
sha=$(jq -er .attestation.commit "$RECEIPT")
tree=$(jq -er .attestation.tree "$RECEIPT")
test "$(git -C "$CANDIDATE" rev-parse HEAD)" = "$sha"
test "$(git -C "$CANDIDATE" rev-parse 'HEAD^{tree}')" = "$tree"
test "$(git -C "$CANDIDATE" rev-list --count --all)" = "$(jq -er .objects.commit "$RECEIPT")"
test "$(sha256sum "$CANDIDATE/DESTINATION-SHA256.json" | cut -d' ' -f1)" = \
     "$(jq -er .attestation.manifest_sha256 "$RECEIPT")"

gh repo create "$repo" --private --disable-wiki
test "$(gh repo view "$repo" --json nameWithOwner,visibility \
  --jq '.nameWithOwner + " " + .visibility')" = "$repo PRIVATE"
git -C "$CANDIDATE" -c credential.helper= -c 'credential.helper=!gh auth git-credential' \
  push "https://github.com/$repo.git" "$sha:refs/heads/main"
```

Creation must succeed on a new, empty repository. Stop on any error. The push
sends only the candidate's main branch. When the push run appears in Actions,
run the following. If registration is delayed, repeat the run lookup.

```bash
run=$(gh run list -R "$repo" --workflow distributions.yml --event push --commit "$sha" \
  --limit 1 --json databaseId --jq '.[0].databaseId // empty')
test -n "$run"
gh run watch "$run" -R "$repo" --interval 60 --exit-status
test "$(gh run view "$run" -R "$repo" --json headSha --jq .headSha)" = "$sha"
results=$(mktemp -d)
gh run download "$run" -R "$repo" --pattern 'verified-*' --dir "$results"
jq -er '.archives | to_entries[] | .value.sha256 + "  " + .key' "$RECEIPT" > "$results/expected"
for kind in wheel sdist; do
  bundle="$results/verified-$kind"
  (cd "$bundle" && sha256sum --check "$results/expected")
  test "$(cat "$bundle/tree.txt")" = "$tree"
  jq -e --arg kind "$kind" '.installed_from == $kind and .passed > 0 and .skipped == 0' \
    "$bundle/installation.json"
done
```

Read both test reports, SPDX inventories and changelogs. Open Issues → New
issue and confirm both forms show the synthetic-reproducer prompt. Preview a
synthetic report, then discard the draft. Check that the contribution guide,
issue triage, semantic versions and Unreleased section render on the default
branch. Open SECURITY.md and check the reporting instructions. GitHub enables
the vulnerability submission form on public repositories. After separately
authorised publication the owner enables private vulnerability reporting and
verifies that form from a reporter account. [GitHub reporting guide](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/report-privately)

Record the run URL and comparison in the verification record. Read-only checks
need no rollback. To stop a running job, use `gh run cancel "$run" -R "$repo"`.
If keeping staging, leave its visibility private. To remove just a finished run
and its hosted artifacts, use `gh run delete "$run" -R "$repo"` with Actions
write access. To undo repository creation and the push, obtain the owner's
disposition, confirm the full name and private visibility again, then use
`gh repo delete "$repo" --yes` with repository deletion permission. Deletion
also removes issues and run history. The local candidate and downloaded
evidence remain available for a new rehearsal.
