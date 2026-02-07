# Self-Healing CI Pipeline

## What This Document Is For

**New to PROTEUS?** This document explains how our self-healing CI pipeline works: when the nightly CI fails, an AI agent automatically diagnoses the failure, attempts a fix, and opens a PR for human review.

**Key concepts:**
- **Self-healing** means the CI pipeline can automatically attempt to fix its own failures
- **AI agent** (Claude Code via OpenRouter) reads the codebase, diagnoses issues, and implements fixes
- **Human review** is always required — the agent never merges to main directly

For test markers and CI pipelines, see [Test Categorization](test_categorization.md). For coverage workflows, see [Test Infrastructure](test_infrastructure.md). For Docker architecture, see [Docker CI Architecture](docker_ci_architecture.md).

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Trigger Conditions](#trigger-conditions)
3. [Workflow Steps](#workflow-steps)
4. [Environmental Memory](#environmental-memory)
5. [Issue and PR Management](#issue-and-pr-management)
6. [Cost Management](#cost-management)
7. [Configuration](#configuration)
8. [Reviewing Self-Heal PRs](#reviewing-self-heal-prs)
9. [Scope and Limitations](#scope-and-limitations)
10. [Manual Controls](#manual-controls)
11. [Troubleshooting](#troubleshooting)
12. [References](#references)

---

## How It Works

```
ci-nightly.yml ─────┐  scheduled runs only
docker-build.yml ───┘  (not workflow_dispatch/push)
        │ failure
        ▼
ci-self-heal.yml
        │
        ├─ 1. Triage: extract failure context from artifacts
        ├─ 2. Create Issue: tracked on Project #7
        ├─ 3. AI Agent: diagnose and implement fix
        ├─ 4. Validate: run tests + lint
        └─ 5. Open PR (if fix passes) or update Issue (if not)
```

When the nightly Docker build (`docker-build.yml`) or nightly science validation (`ci-nightly.yml`) fails during its **scheduled** run, the self-healing workflow automatically:

1. Downloads artifacts from the failed run (JUnit XML, logs)
2. Extracts and classifies the failure (test failure, build failure, or infrastructure failure)
3. Creates a GitHub Issue with labels, assignees, and adds it to the project board
4. Runs an AI coding agent that reads the codebase and attempts a minimal fix
5. Validates the fix against the existing test suite and linter
6. Opens a PR if validation passes, or updates the Issue with a diagnosis if not

---

## Trigger Conditions

The self-healing pipeline **only** triggers when ALL of these conditions are met:

| Condition | Value | Why |
|-----------|-------|-----|
| Workflow | `CI - Nightly Science Validation` or `Docker Build and Push` | Only nightly workflows |
| Conclusion | `failure` | Only on actual failures |
| Branch | `main` | Not feature branches |
| Original trigger | `schedule` | Not `workflow_dispatch` or `push` |

**Explicitly excluded:**
- Manual workflow runs (`workflow_dispatch`) — e.g., when testing CI changes
- Push-triggered runs — e.g., when dependency files change on main
- PR check failures — handled by developers directly
- Successful runs — no action needed

---

## Workflow Steps

### Job 1: Triage

- Downloads artifacts from the failed workflow run
- Fetches workflow logs via GitHub API
- Runs `selfheal/extract_failures.py` to parse JUnit XML and log files
- Classifies failure as: `test_failure`, `build_failure`, or `infrastructure_failure`
- Checks for duplicate: skips if an open `selfheal/*` PR already exists for the same commit

### Job 2: Create Issue

- Creates a GitHub Issue with:
    - **Labels:** `Bug`, `Priority 1: critical`
    - **Assignees:** nichollsh, timlichtenberg, egpbos
    - **Project:** Added to [FormingWorlds Project #7](https://github.com/orgs/FormingWorlds/projects/7) with Status "In Progress"
- This happens **regardless** of whether the agent can fix the issue

### Job 3: Auto-Fix

- Checks out the code from `main`
- Runs the Claude Code agent (via OpenRouter) with failure context
- Agent reads `AGENTS.md` and `MEMORY.md` first for project context
- Agent attempts to diagnose and implement a minimal fix
- Validates the fix:
    - `pytest -m "unit and not skip" --ignore=tests/examples`
    - `ruff check src/ tests/` and `ruff format --check src/ tests/`
- If validation passes: creates a PR
- If validation fails: updates the Issue with a diagnosis comment

---

## Environmental Memory

The AI agent is given access to two key files that serve as its "environmental memory":

- **`AGENTS.md`**: Project coding standards, build commands, test conventions, ecosystem structure
- **`MEMORY.md`**: Architectural decisions, active context, known debt, recent lessons learned

The prompt explicitly instructs the agent to read these files first. This means the agent understands:
- How tests are structured (markers, conftest fixtures, mirror layout)
- Physical validity constraints (T > 0K, P > 0, no float ==)
- Known fragile areas (AGNI integration, data downloads, config system)
- Recent infrastructure changes and their rationale

---

## Issue and PR Management

### Issue

Every nightly failure creates a tracked Issue:
- Added to [Project #7](https://github.com/orgs/FormingWorlds/projects/7/views/1) with Status "In Progress"
- Labels: `Bug` + `Priority 1: critical`
- Assignees: nichollsh, timlichtenberg, egpbos
- Updated with agent results (fix PR link or diagnosis)

### Pull Request (if fix succeeds)

- Branch: `selfheal-<commit-sha>`
- Reviewers: `@FormingWorlds/proteus-maintainer`
- Labels: `Bug` + `Priority 1: critical`
- Assignees: nichollsh, timlichtenberg, egpbos
- Added to Project #7 with Status "In Progress"
- Body links to failed run and tracking Issue
- Auto-closes the tracking Issue when merged

---

## Cost Management

| Parameter | Value |
|-----------|-------|
| **AI provider** | OpenRouter (Claude Code) |
| **Max turns per run** | 15 |
| **Estimated cost per run** | $1–8 |
| **Monthly budget cap** | $20 (set in OpenRouter dashboard) |

**Cost breakdown:**
- Simple fix (1-2 files, ~10 turns): $1–3
- Complex fix (multi-file, 15 turns): $3–8
- Agent gives up (diagnosis only, ~5 turns): $0.50–2
- No failures (workflow skipped): $0

If the OpenRouter budget cap is reached, the agent step will fail gracefully — the Issue is still created (it runs before the agent), so failures are always tracked.

---

## Configuration

### Required GitHub Secrets

| Secret | Purpose | Who creates it |
|--------|---------|---------------|
| `OPENROUTER_API_KEY` | API key for the AI agent | Repo admin (from [openrouter.ai](https://openrouter.ai)) |
| `PROJECT_PAT` | Personal Access Token for org Project V2 board access | Org admin (classic PAT with `project`, `repo`, `issues` scopes) |

### Required GitHub Settings

- **Repo → Settings → Actions → General**: "Allow GitHub Actions to create and approve pull requests" must be enabled
- **OpenRouter dashboard**: Set monthly budget cap to $20

### Workflow File

**Location:** `.github/workflows/ci-self-heal.yml`

### Support Files

| File | Purpose |
|------|---------|
| `selfheal/extract_failures.py` | Parses JUnit XML + logs into structured failure context |
| `selfheal/prompt-template.txt` | PROTEUS-specific instructions for the AI agent |

---

## Reviewing Self-Heal PRs

When you see a `[Self-Heal]` PR in the morning:

1. **Read the PR body** — it links to the failed run and describes the failure
2. **Check the diff** — changes should be minimal and targeted
3. **Verify the fix is correct**, not just a workaround:
    - Source code fix preferred over test weakening
    - No float `==` comparisons introduced
    - Physical validity maintained (T > 0K, P > 0)
    - No unrelated changes
4. **Check the linked Issue** for additional context
5. **Merge or request changes** as you would for any PR
6. The tracking Issue auto-closes when the PR is merged

---

## Scope and Limitations

### What the agent CAN fix
- Python test failures (unit, smoke, integration)
- Import errors and missing dependencies
- Configuration issues in TOML files
- Simple logic bugs in `src/proteus/`
- Ruff lint/format violations

### What the agent CANNOT fix (opens Issue with diagnosis instead)
- Docker build failures (Dockerfile, system packages)
- Compiled binary issues (SOCRATES, AGNI, SPIDER compilation)
- Julia/Fortran/C code changes
- External service outages (Zenodo, OSF, network)
- Timeout failures (usually need config tuning)
- Cross-repository breaks (upstream module API changes)
- Flaky tests (may misdiagnose; human review catches this)

---

## Manual Controls

### Disable self-healing temporarily

Disable the workflow in GitHub:
**Actions → CI - Self-Healing (Nightly Failures) → ⋯ → Disable workflow**

### Re-enable

**Actions → CI - Self-Healing (Nightly Failures) → Enable workflow**

### Force a self-heal run

The workflow only triggers on `schedule` events, so you cannot manually trigger it via `workflow_dispatch`. To test, you can temporarily change the trigger condition in the workflow file.

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Workflow never triggers | No scheduled nightly failures | Working as intended — no failures to fix |
| Workflow triggers but skips | `should_skip=true` | An open `selfheal/*` PR already exists; close or merge it |
| Issue created but no PR | Agent couldn't fix or validation failed | Check Issue comments for diagnosis |
| `PROJECT_PAT` errors | Token expired or missing scopes | Regenerate PAT with `project`, `repo`, `issues` scopes |
| `OPENROUTER_API_KEY` errors | Key missing or budget exceeded | Check OpenRouter dashboard; add/rotate key |
| Agent makes wrong fix | Misdiagnosis | Close PR, fix manually, update `MEMORY.md` with lesson |
| Agent loops or times out | Complex failure beyond 15 turns | 30-min timeout kills the job; check Issue for partial diagnosis |

---

## References

### PROTEUS Documentation
- [Test Infrastructure](test_infrastructure.md) — Coverage workflows, thresholds, troubleshooting
- [Test Categorization](test_categorization.md) — Test markers, CI pipelines, fixtures
- [Docker CI Architecture](docker_ci_architecture.md) — Docker image, CI pipelines
- [AGENTS.md](../AGENTS.md) — Project coding standards (read by the AI agent)
- [MEMORY.md](../MEMORY.md) — Architectural decisions (read by the AI agent)

### External Resources
- [Claude Code Action](https://github.com/anthropics/claude-code-action) — GitHub Action for Claude Code
- [OpenRouter](https://openrouter.ai) — AI model routing with budget controls
- [GitHub Projects V2 API](https://docs.github.com/en/issues/planning-and-tracking-with-projects/automating-your-project/using-the-api-to-manage-projects) — Project board automation
