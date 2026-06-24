#!/usr/bin/env bash
# new-worktree.sh — create a git worktree for isolated development.
#
# Usage:    ./scripts/new-worktree.sh <branch-name> [prompt]
# Example:  ./scripts/new-worktree.sh feat-new-feature
# Example:  ./scripts/new-worktree.sh feat-fix-bug "please fix the login bug"

set -euo pipefail

BRANCH="${1:-}"
PROMPT="${2:-}"
if [[ -z "$BRANCH" ]]; then
	echo "Usage: $0 <branch-name> [prompt]" >&2
	exit 1
fi

# Resolve the main worktree root (first entry in worktree list)
MAIN_ROOT=$(git worktree list --porcelain | head -1 | sed 's/^worktree //')
PROJECT_NAME=$(basename "$MAIN_ROOT")
WORKTREE_DIR="$MAIN_ROOT/.worktrees/$BRANCH"

# Ensure .worktrees/ is gitignored
if ! grep -qxF '.worktrees/' "$MAIN_ROOT/.gitignore" 2>/dev/null; then
	echo '.worktrees/' >> "$MAIN_ROOT/.gitignore"
	echo "Added .worktrees/ to .gitignore"
fi

echo "Creating worktree at: $WORKTREE_DIR"
git worktree add "$WORKTREE_DIR" -b "$BRANCH"

# Copy environment files if present
for ENV_FILE in .env .env.local; do
	if [[ -f "$MAIN_ROOT/$ENV_FILE" ]]; then
		cp "$MAIN_ROOT/$ENV_FILE" "$WORKTREE_DIR/$ENV_FILE"
		echo "Copied $ENV_FILE from main worktree"
	fi
done

# Copy .vscode from main worktree, then strip Peacock color so each worktree
# gets its own color (or none) rather than inheriting the parent's.
if [[ -d "$MAIN_ROOT/.vscode" ]]; then
	cp -r "$MAIN_ROOT/.vscode" "$WORKTREE_DIR/.vscode"
	echo "Copied .vscode from main worktree"
	VSCODE_SETTINGS="$WORKTREE_DIR/.vscode/settings.json"
	if [[ -f "$VSCODE_SETTINGS" ]] && command -v jq &>/dev/null; then
		jq 'del(
          ."peacock.color",
          ."workbench.colorCustomizations"."titleBar.activeBackground",
          ."workbench.colorCustomizations"."titleBar.activeForeground",
          ."workbench.colorCustomizations"."titleBar.inactiveBackground",
          ."workbench.colorCustomizations"."titleBar.inactiveForeground",
          ."workbench.colorCustomizations"."commandCenter.border"
        )
        | if (."workbench.colorCustomizations" // {} | length) == 0
          then del(."workbench.colorCustomizations") else . end' \
			"$VSCODE_SETTINGS" >"${VSCODE_SETTINGS}.tmp" &&
			mv "${VSCODE_SETTINGS}.tmp" "$VSCODE_SETTINGS"
		echo "Removed Peacock color from .vscode/settings.json"
	fi
fi

# Copy node_modules if present to save full reinstall time
if [[ -d "$MAIN_ROOT/node_modules" ]]; then
	echo "Copying node_modules from main worktree (this may take a moment)..."
	cp -r "$MAIN_ROOT/node_modules" "$WORKTREE_DIR/node_modules"
	echo "Copied node_modules"
elif [[ -f "$MAIN_ROOT/package.json" ]]; then
	echo "No node_modules in main worktree — running npm install..."
	cd "$WORKTREE_DIR"
	npm install --silent
fi

# Add this worktree as a folder in a .code-workspace file inside the project.
# Paths are relative to the workspace file (which lives at the project root),
# so the whole tree survives moves.
WORKSPACE_FILE="$MAIN_ROOT/${PROJECT_NAME}.code-workspace"
# Path from workspace file to this worktree (e.g. .worktrees/feat-x)
NEW_FOLDER=".worktrees/$BRANCH"
if command -v jq &>/dev/null; then
	if [[ ! -f "$WORKSPACE_FILE" ]]; then
		jq -n --arg new "$NEW_FOLDER" --arg branch "$BRANCH" \
			'{folders: [{name: "(main)", path: "."}, {name: $branch, path: $new}], settings: {}}' \
			>"$WORKSPACE_FILE"
		echo "Created $WORKSPACE_FILE"
	elif ! jq -e --arg p "$NEW_FOLDER" '.folders[]? | select(.path == $p)' "$WORKSPACE_FILE" >/dev/null; then
		jq --arg name "$BRANCH" --arg path "$NEW_FOLDER" \
			'.folders += [{name: $name, path: $path}]' \
			"$WORKSPACE_FILE" >"${WORKSPACE_FILE}.tmp" && mv "${WORKSPACE_FILE}.tmp" "$WORKSPACE_FILE"
		echo "Added $BRANCH to $WORKSPACE_FILE"
	else
		echo "$BRANCH already present in $WORKSPACE_FILE"
	fi
else
	echo "jq not found — skipped .code-workspace update (install jq to auto-register worktrees)"
fi

# Go to worktree and start tools
cd "$WORKTREE_DIR"
if command -v code &>/dev/null; then
	code .
fi

if [[ -n "$PROMPT" ]] && command -v omp &>/dev/null; then
	omp "$PROMPT"
fi
