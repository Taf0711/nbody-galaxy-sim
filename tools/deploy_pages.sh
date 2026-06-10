#!/bin/sh
# Publish web/ to the gh-pages branch (GitHub Pages serves it at
# https://taf0711.github.io/nbody-galaxy-sim/). Run from the repo root.
set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
WT=$(mktemp -d)

git worktree add --detach "$WT" >/dev/null
cd "$WT"
git checkout --orphan gh-pages >/dev/null 2>&1 || git checkout gh-pages >/dev/null 2>&1
git rm -rfq --cached . 2>/dev/null || true
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R "$REPO_ROOT/web/." .
touch .nojekyll
git add -A
git commit -qm "Deploy web demo to GitHub Pages"
git push -f origin gh-pages

cd "$REPO_ROOT"
git worktree remove --force "$WT"
echo "deployed: https://taf0711.github.io/nbody-galaxy-sim/"
