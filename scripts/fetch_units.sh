#!/usr/bin/env bash
# The ten practice units (and the exemplar) from the official Track 4 repository, at the commit this agent was
# developed against, linked as ./units. The units are the organisers' data and are not redistributed here.
set -euo pipefail
REPO=https://github.com/Agenthon-2026/track4-analysis-public
COMMIT=1c744e1d6725340643a533f436517d72b53ca0e1
cd "$(dirname "$0")/.."
if [ ! -d .cache/track4 ]; then
  mkdir -p .cache/track4
  git -C .cache/track4 init -q
  git -C .cache/track4 remote add origin "$REPO"
fi
git -C .cache/track4 fetch -q --depth 1 origin "$COMMIT"
git -C .cache/track4 checkout -q FETCH_HEAD
ln -sfn .cache/track4/units units
echo "units: $(ls units | wc -l | tr -d ' ') from $REPO at ${COMMIT:0:7}"
