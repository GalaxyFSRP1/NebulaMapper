#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
DEFAULT_URL="https://github.com/GalaxyFSRP1/NebulaMapper.git"
echo "========================================================"
echo "  Push NebulaMapper to GitHub ($DEFAULT_URL)"
echo "========================================================"
read -p "Press ENTER to push to $DEFAULT_URL (or paste another URL): " REPO_URL
REPO_URL="${REPO_URL:-$DEFAULT_URL}"
rm -rf .git
git init
git branch -M main
git add .
git commit -m "Release NebulaMapper — PhantomVoid Neural BeatStudio"
git remote add origin "$REPO_URL"
git push -u origin main --force
echo "Done! Live at https://github.com/GalaxyFSRP1/NebulaMapper"
echo "Colab GPU link: https://colab.research.google.com/github/GalaxyFSRP1/NebulaMapper/blob/main/colab.ipynb"
