## Git Worktrees

This project uses `scripts/new-worktree.sh` as its **native worktree tool**.

When the `using-git-worktrees` skill applies, use this script (Step 1a) — do **not** fall through to `git worktree add`:

```bash
./scripts/new-worktree.sh <branch-name> [optional-prompt]
```

The script handles: branch creation, `.env`/`.env.local` copying, `.vscode` copying (stripping Peacock colors), `node_modules` copying or `npm install`, `.code-workspace` registration, and opening VS Code.
