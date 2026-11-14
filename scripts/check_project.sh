#!/usr/bin/env bash
# Project health report: tests, lint, clean tree, no data or large files tracked.
set -u
cd "$(dirname "$0")/.."

status=0
check() {
    local name=$1
    shift
    if "$@" >/dev/null 2>&1; then
        echo "[ok]   $name"
    else
        echo "[FAIL] $name"
        status=1
    fi
}

no_large_tracked() {
    local f
    while IFS= read -r -d '' f; do
        [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 1048576 ] && return 1
    done < <(git ls-files -z)
    return 0
}

readme_matches_csvs() {
    local line
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        sed 's/^ *//' README.md | grep -qxF -- "$line" || return 1  # list items are indented
    done < <(python scripts/results_table.py)
}

check "make test" make test
check "make lint" make lint
check "working tree clean" test -z "$(git status --porcelain)"
check "data/ not tracked" test -z "$(git ls-files data)"
check "no tracked file above 1 MB" no_large_tracked
check "README tables match docs/*.csv" readme_matches_csvs

exit $status
