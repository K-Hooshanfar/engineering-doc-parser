# Pre-commit Guide

This document explains how we use **pre-commit** to keep our codebase clean and consistent *before* changes reach CI and pull requests.

> TL;DR: Install pre-commit once, and it will automatically run formatters and linters on staged files at commit time.


## Why pre-commit?

- Catches issues early (formatting, linting, simple security checks)
- Auto-fixes common problems (imports, styling) so reviews focus on logic
- Keeps diffs tidy and enforces project standards consistently


## What runs locally

Our pre-commit configuration (see `.pre-commit-config.yaml` at repo root) includes fast checks:

- **Black** — code formatting
- **Ruff** — linting + import sorting (with autofix)
- **Detect-secrets** — light secret scanning (with baseline)
- *(Optional, manual hooks)* **Mypy**/**Bandit** — available to run locally on demand; enforced in CI

> CI re-runs these plus heavier checks (Pylint with score gate, full type checking, coverage, etc.) in a clean environment.


## One-time setup

```bash
# From your repo root:
pip install pre-commit

# Install Git hook
pre-commit install

# (Recommended) Run on the entire repository once
pre-commit run --all-files
````

If you use multiple Python versions, install pre-commit into the Python you use for development (e.g., your project venv).


## Everyday workflow

1. Edit files.
2. `git add ...`
3. `git commit` → pre-commit runs on staged files.

   * If a hook **fails** or **autofixes** files, review changes, `git add` the fixes, and commit again.


## Running hooks explicitly

* Run against all files:

  ```bash
  pre-commit run --all-files
  ```
* Run a single hook by id (examples):

  ```bash
  pre-commit run black --all-files
  pre-commit run ruff --all-files
  pre-commit run detect-secrets --all-files
  ```
* Run the optional/manual hooks locally:

  ```bash
  pre-commit run mypy --all-files
  pre-commit run bandit --all-files
  ```


## Secrets baseline (detect-secrets)

We keep a baseline to reduce false positives while still blocking accidental secrets.

Create/update the baseline:

```bash
detect-secrets scan > .secrets.baseline
git add .secrets.baseline
```

If a hook flags something:

* Verify it’s **not** a real secret. If it is, **rotate it immediately** and remove from code.
* For allowed false positives, update the baseline and commit it.

> Never commit real credentials, tokens, or certificates. Use env vars and `.env.example` patterns.


## Policy

* Commits must pass all **pre-commit** hooks locally.
* Do **not** commit code formatted differently from project standards.
* Do **not** bypass hooks except in emergencies; if you must:

  ```bash
  git commit -n -m "your message"
  ```

  Then **fix and follow up** promptly with a correcting commit. Repeated bypasses are not allowed.


## Troubleshooting

* **“Command not found” / missing tools**
  Ensure you installed pre-commit: `pip install pre-commit` and `pre-commit install`.
  Some hooks manage their own environments; pre-commit will auto-install them on first run.

* **Hook is slow on Windows**
  Ensure your repo is on a local NTFS path (not network/WSL cross-mount). Consider running pre-commit inside WSL for better performance.

* **Ruff/Black disagreement**
  Our config treats Black as the source of truth for formatting; Ruff handles lint & import order. Run:

  ```bash
  pre-commit run black --all-files
  pre-commit run ruff --all-files
  ```

* **Virtualenv confusion**
  Pre-commit creates its own isolated envs per hook. That’s expected. Your project venv is still used for development.


## Updating hooks

Keep hooks fresh to benefit from fixes and speed improvements.

```bash
pre-commit autoupdate
# Review diffs in .pre-commit-config.yaml
git add .pre-commit-config.yaml
git commit -m "chore(pre-commit): autoupdate hooks"
```

If an update introduces new rules causing failures, either:

* Fix the flagged issues, or
* Adjust configuration (e.g., in `pyproject.toml` / `.ruff.toml`) and document the choice.


## Adding a new hook

1. Edit `.pre-commit-config.yaml` and add the repo + hook id.
2. Run:

   ```bash
   pre-commit install
   pre-commit run --all-files
   ```
3. Document any developer steps in `CONTRIBUTING.md` (e.g., tool installation, config).

**Rule of thumb:** keep hooks **fast** and **autofix-friendly**. Heavy checks belong in CI.


## Relationship to CI

* **pre-commit**: fast, incremental, runs locally on changed files → developer feedback loop.
* **CI** (GitHub Actions): authoritative, runs full checks on the entire project across Python versions, blocks merges on failure.

Both are required. Passing pre-commit locally reduces CI churn; CI ensures consistency for everyone.


## FAQ

* **Can I commit with failing hooks?**
  Only as a last resort using `git commit -n`, and you must follow up with fixes. Persistent bypassing is not allowed.

* **Do I need to run pre-commit on CI?**
  Not required—CI runs equivalent/larger checks. You may add a “pre-commit” step for parity, but it’s optional.

* **Where are hook settings?**

  * Hook list: `.pre-commit-config.yaml`
  * Tool settings (line length, ignores, etc.): `pyproject.toml` (preferred) or tool-specific files.


## Quick commands reference

```bash
# Install & initialize
pip install pre-commit
pre-commit install
pre-commit run --all-files

# Fix common issues
pre-commit run black --all-files
pre-commit run ruff --all-files

# Secrets baseline
detect-secrets scan > .secrets.baseline
git add .secrets.baseline

# Update hooks
pre-commit autoupdate
```

---

Happy committing! ✨
