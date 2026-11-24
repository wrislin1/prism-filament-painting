#!/bin/sh
# Run in an isolated Python container with read-only source and wheel inputs.
set -eu
source_dir=$1
downloads=$2
out=$3
kind=$4
case "$kind" in wheel|sdist) ;; *) exit 2 ;; esac
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
cd "$scratch"
export HOME="$scratch"
python -m venv env
py="$scratch/env/bin/python"
"$py" -m pip install --quiet --no-index --no-deps "$downloads"/wheels/*.whl
"$py" -B "$source_dir/ci/gates.py" "$source_dir" "$out" "$downloads/gitleaks"
case "$kind" in
    wheel) set -- "$out"/*.whl ;;
    sdist) set -- "$out"/*.tar.gz ;;
esac
# A fresh environment for each matrix entry. For the sdist pip calls the
# backend here a second time, outside the checkout and without a build index.
"$py" -m pip install --no-cache-dir --no-index --no-build-isolation --find-links "$downloads/wheels" "$1[mcp,catalogues,optimize]"
"$py" -m pip check
cp -R "$source_dir/tests" tests
"$py" -I -B -m pytest -q tests --junitxml="$out/tests.xml"
"$py" -B "$source_dir/ci/installed.py" "$out/tests.xml" "$kind" "$scratch/tests"
