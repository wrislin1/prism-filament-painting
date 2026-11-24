#!/bin/sh
# Fetch reviewed Linux x86-64 wheels and the scanner into the scratch directory.
set -eu
out=$1
mkdir -p "$out/wheels"
python -m pip --isolated download --disable-pip-version-check --no-cache-dir \
    --index-url https://pypi.org/simple --only-binary=:all: --no-deps \
    --require-hashes -r "$(dirname "$0")/requirements.lock" -d "$out/wheels"
archive=gitleaks_8.28.0_linux_x64.tar.gz
curl --fail --silent --show-error --location --retry 3 \
    "https://github.com/gitleaks/gitleaks/releases/download/v8.28.0/$archive" -o "$out/$archive"
(cd "$out" && echo "a65b5253807a68ac0cafa4414031fd740aeb55f54fb7e55f386acb52e6a840eb  $archive" | sha256sum -c -)
tar -xzf "$out/$archive" -C "$out" gitleaks
