#!/usr/bin/env bash

# if any command inside script returns error, exit and return that error
set -e

# magic line to ensure that we're always inside the root of our application,
# no matter from which directory we'll run script
# thanks to it we can just enter `./scripts/run-tests.bash`
cd "${0%/*}/.."

# generate the changelog
echo "Generate changelog"
git --no-pager log --no-merges --format="### %s%d%n>%aD%n%n>Author: %aN (%aE)%n%n>Commiter: %cN (%cE)%n%n%b%n%N%n" > CHANGELOG

# prevent recursion!
# since a 'commit --amend' will trigger the post-commit script again
# we have to check if the changelog file has changed or not
res=$(git status --porcelain | grep -c CHANGELOG)
if [ "$res" -gt 0 ]; then
  git add CHANGELOG
  git commit --amend --no-edit --no-verify
  echo "Populated Changelog in CHANGELOG"
fi
