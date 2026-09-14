#!/usr/bin/env bash
# Regenerate bom.json -- a CycloneDX SBOM of cvkit's full dependency closure,
# for ingestion by Dependency-Track (or any CycloneDX consumer).
#
#   ./scripts/make_bom.sh          regenerate bom.json
#   ./scripts/make_bom.sh --check  fail if bom.json is out of date (used by CI)
#
# The BOM is built from a real, fully resolved environment with every extra
# installed, so each component carries a concrete version and PackageURL --
# which is what Dependency-Track matches advisories against. Declared ranges
# (">=1.24") would not be matchable.
#
# The environment is created WITHOUT pip and setuptools: they are venv
# bootstrap, not dependencies of cvkit, and listing them would raise
# advisories in Dependency-Track for packages this project does not ship.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="bom.json"
VENV="$(mktemp -d)/bomenv"
trap 'rm -rf "$(dirname "$VENV")"' EXIT

command -v python3 >/dev/null || { echo "python3 not found" >&2; exit 1; }
python3 -m venv --without-pip "$VENV"

# Install into that interpreter from outside it, so no pip lands inside.
python3 -m pip --python "$VENV/bin/python" install --quiet ".[all]"

# cyclonedx-py comes from the dev extra; run whichever copy is on PATH.
CDX="$(command -v cyclonedx-py || true)"
[ -n "$CDX" ] || { echo "cyclonedx-py not found -- pip install '.[dev]'" >&2; exit 1; }

TMP="$(mktemp)"
"$CDX" environment "$VENV" \
    --of JSON --sv 1.6 \
    --output-reproducible \
    --pyproject pyproject.toml \
    -o "$TMP"

if [ "${1:-}" = "--check" ]; then
    if diff -q "$TMP" "$OUT" >/dev/null 2>&1; then
        echo "$OUT is up to date"
    else
        echo "$OUT is out of date -- run ./scripts/make_bom.sh and commit the result" >&2
        diff -u "$OUT" "$TMP" | head -40 >&2 || true
        exit 1
    fi
else
    mv "$TMP" "$OUT"
    echo "wrote $OUT ($(grep -c '"purl"' "$OUT") components)"
fi
