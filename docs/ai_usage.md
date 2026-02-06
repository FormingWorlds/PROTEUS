# AI-Assisted Development

## How PROTEUS Uses AI

PROTEUS leverages AI assistants for **software engineering tasks**, not scientific content:

- **Test implementation** — Generating unit tests, expanding coverage, writing fixtures
- **Code security** — Identifying vulnerabilities, reviewing for unsafe patterns
- **Code refactoring** — Improving consistency, streamlining framework structure
- **Automated code reviews** — PR reviews via GitHub Copilot and Cursor Bugbot

**AI is not used for:** Scientific algorithms, physics implementations, or research decisions. These require domain expertise and human judgment.

---

## What This Document Is For

**New to AI coding assistants?** This guide explains how to use AI tools (GitHub Copilot, Cursor, Windsurf) safely and effectively with PROTEUS. AI assistants can significantly accelerate development, but require careful use to maintain code quality and security.

**Key principle:** AI is a powerful tool, not a replacement for understanding. Always review AI-generated code before committing.

---

## Quick Start

1. **Set up an AI assistant**: Install [GitHub Copilot](https://github.com/features/copilot), [Cursor](https://cursor.sh/), or [Windsurf](https://codeium.com/windsurf)
2. **Provide context**: Point the assistant to `AGENTS.md` (coding guidelines) and `MEMORY.md` (project state)
3. **Generate code**: Use prompts from [Test Building](test_building.md) for tests
4. **Review thoroughly**: Check all AI output before committing
5. **Run tests**: `pytest -m "unit and not skip"` and `ruff check`

---

## AGENTS.md and MEMORY.md

PROTEUS uses two special files to provide AI assistants with project context:

### AGENTS.md — Coding Guidelines

**Purpose:** Instructions for AI agents on how to write PROTEUS-compliant code.

**Contains:**
- Project structure and architecture
- Coding standards and style rules
- Testing requirements and markers
- Build commands and validation steps
- Common patterns and anti-patterns

**How to use:** Add `AGENTS.md` to your AI assistant's context window or reference it in prompts.

### MEMORY.md — Project State

**Purpose:** Living document capturing current project state and decisions.

**Contains:**
- Recent architectural decisions
- Current sprint focus and priorities
- Known issues and workarounds
- Coverage thresholds and CI status
- Lessons learned from past work

**How to use:** Reference when you need context about *why* things are done a certain way.

---

## IDE Setup

### VS Code with GitHub Copilot

1. **Install**: [VS Code Copilot Extension](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot)
2. **Enable Copilot Chat**: Install [Copilot Chat Extension](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat)
3. **Add context files**: In chat, use `@workspace` to reference project files, or drag `AGENTS.md` into the chat
4. **Use `.github/copilot-instructions.md`**: This file automatically provides Copilot with PROTEUS guidelines

**Tutorials:**
- [Getting Started with GitHub Copilot](https://docs.github.com/en/copilot/using-github-copilot/getting-started-with-github-copilot)
- [Using Copilot Chat in VS Code](https://docs.github.com/en/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide)
- [Copilot Best Practices](https://docs.github.com/en/copilot/using-github-copilot/best-practices-for-using-github-copilot)

**Academic License:** Students and educators can apply for free GitHub Copilot access:
[GitHub Education](https://education.github.com/benefits)

### Cursor

1. **Install**: Download from [cursor.sh](https://cursor.sh/)
2. **Open PROTEUS**: `cursor /path/to/PROTEUS`
3. **Add rules**: Cursor reads `.cursorrules` if present; alternatively, add `AGENTS.md` content to Settings → Rules
4. **Reference files**: Use `@AGENTS.md` or `@MEMORY.md` in chat to include context

### Windsurf (Cascade)

1. **Install**: Download from [codeium.com/windsurf](https://codeium.com/windsurf)
2. **Open PROTEUS**: Windsurf automatically reads `AGENTS.md` from the workspace
3. **Memory system**: Windsurf maintains persistent memory across sessions
4. **Reference files**: Use `@file` mentions to include specific files in context

---

## AI for Test Implementation

AI assistants excel at generating tests when given proper context. PROTEUS has standardized prompts for this purpose.

### Workflow

1. **Open the source file** you want to test
2. **Open `tests/conftest.py`** to show available fixtures
3. **Use the Master Prompt** from [Test Building](test_building.md):

```
Act as a Senior Scientific Software Engineer for PROTEUS.
I need robust unit tests for the open file. Follow these strict guidelines:
- Use @pytest.mark.unit marker
- Mock all external dependencies
- Use pytest.approx() for float comparisons
- Add docstrings explaining physical scenarios
```

4. **Review the generated tests** for:
   - Correct markers (`@pytest.mark.unit`)
   - Proper mocking (no real I/O or network calls)
   - Physically valid test inputs
   - Clear docstrings

5. **Run and verify**: `pytest tests/<module>/test_<file>.py -v`

### Why AI + Tests Work Well

- **Repetitive patterns**: Test structures are predictable; AI handles boilerplate
- **Coverage expansion**: AI can suggest edge cases you might miss
- **Fixture awareness**: AI learns your fixture patterns from `conftest.py`
- **Consistency**: AI applies the same style across all tests

See [Test Building](test_building.md) for detailed prompts and examples.

---

## AI for Code Review

Use AI to review your changes *before* pushing a PR. This catches issues early and reduces review cycles.

### Local Review Workflow

1. **Stage your changes**: `git add -p` (interactive staging)

2. **Generate a diff**: `git diff --staged > changes.diff`

3. **Ask AI to review**:
   ```
   Review this diff for PROTEUS. Check for:
   - Style violations (should pass ruff)
   - Missing tests for new code
   - Incorrect float comparisons (must use pytest.approx)
   - Security issues (hardcoded paths, secrets)
   - Breaking changes to public APIs
   ```

4. **Address feedback** before committing

### Automated PR Reviews (GitHub)

PROTEUS uses automated AI reviewers on pull requests:

- **GitHub Copilot** — Reviews code for bugs, security issues, and style
- **Cursor Bugbot** — Analyzes code for potential bugs and improvements

**When you open a PR**, these bots automatically comment with suggestions. Here's how to handle them:

#### Reviewing Bot Comments

1. **Read each comment carefully** — Bots highlight specific lines with potential issues
2. **Evaluate relevance** — Not all suggestions apply; use your judgment
3. **Check for false positives** — AI may flag valid code as problematic (especially physics-specific patterns)

#### Responding to Suggestions

| Action | When to Use |
|--------|-------------|
| **Accept & implement** | Suggestion is valid and improves code |
| **Dismiss with reason** | False positive; explain why in a reply |
| **Ask for clarification** | Unclear suggestion; reply to the bot comment |
| **Defer to reviewer** | Uncertain; tag a human reviewer for input |

#### Common Bot Suggestions

- **"Consider adding error handling"** — Valid if function can fail; dismiss if errors are handled upstream
- **"Magic number detected"** — Consider using a named constant; dismiss if value is obvious (e.g., `0`, `1`)
- **"Function too complex"** — Consider refactoring; may be acceptable for physics calculations
- **"Missing docstring"** — Add docstring for public functions; internal helpers may not need one
- **"Potential security issue"** — Always investigate; err on the side of caution

#### Best Practices

- **Don't ignore all suggestions** — Bots catch real issues
- **Don't accept all suggestions** — Bots make mistakes, especially with scientific code
- **Document dismissals** — Reply explaining why you're not implementing a suggestion
- **Batch responses** — Address all bot comments before requesting human review

### What AI Can Catch

- **Style issues**: Inconsistent formatting, missing docstrings
- **Common bugs**: Off-by-one errors, unhandled edge cases
- **Test gaps**: New functions without corresponding tests
- **Security issues**: Hardcoded credentials, unsafe file operations
- **API breaks**: Changes to function signatures without migration

### What AI Cannot Replace

- **Domain expertise**: AI doesn't understand planetary physics
- **Architectural decisions**: Humans decide system design
- **Security audits**: Critical security requires human review
- **Final approval**: A human must approve all PRs

---

## Safety and Security

### ⚠️ Critical Rules

1. **Never share secrets**: Don't paste API keys, passwords, or credentials into AI prompts
2. **Review all output**: AI can generate plausible-looking but incorrect code
3. **Verify physics**: AI doesn't understand scientific validity—check equations manually
4. **Check file operations**: AI may suggest destructive file operations (rm, overwrite)
5. **Validate external calls**: AI may add network requests or subprocess calls

### Security Checklist

Before committing AI-generated code:

- [ ] No hardcoded paths, credentials, or secrets
- [ ] No unexpected network requests
- [ ] No file operations outside expected directories
- [ ] All tests pass (`pytest -m "unit and not skip"`)
- [ ] Linting passes (`ruff check src/ tests/`)
- [ ] You understand what every line does

### Maintaining Code Quality

```bash
# Before committing AI-generated code:
ruff check src/ tests/                    # Check style
ruff format src/ tests/                   # Format code
pytest -m "unit and not skip"             # Run tests
bash tools/validate_test_structure.sh     # Validate structure
git diff --staged                         # Review changes yourself
```

---

## Best Practices

### Do

- **Provide context**: Include `AGENTS.md` and relevant source files
- **Be specific**: "Write a unit test for `calculate_flux` that tests edge case when T=0"
- **Iterate**: Ask AI to refine based on your feedback
- **Learn from output**: Use AI suggestions to improve your understanding
- **Attribute appropriately**: Note significant AI contributions in commit messages if relevant

### Don't

- **Blindly accept**: Never commit without understanding the code
- **Skip tests**: AI-generated code needs testing like any other code
- **Ignore warnings**: If AI says "this might need adjustment," investigate
- **Share sensitive data**: Keep credentials and private data out of prompts
- **Over-rely**: AI is a tool, not a substitute for expertise

---

## Troubleshooting

### AI generates incorrect markers

**Problem:** AI uses `@pytest.mark.test` instead of `@pytest.mark.unit`

**Solution:** Include `AGENTS.md` in context; it specifies valid markers

### AI doesn't know about fixtures

**Problem:** AI creates fixtures that already exist in `conftest.py`

**Solution:** Always include `tests/conftest.py` in the context window

### AI suggests outdated patterns

**Problem:** AI uses deprecated APIs or old coding patterns

**Solution:** Reference `MEMORY.md` for current patterns; specify Python version (3.12)

### AI generates code that fails CI

**Problem:** Generated code passes locally but fails in CI

**Solution:** Run full pre-commit checklist before pushing:
```bash
ruff check src/ tests/ && ruff format src/ tests/
pytest -m "unit and not skip"
bash tools/validate_test_structure.sh
```

---

## References

- [AGENTS.md](../AGENTS.md) — AI coding guidelines for PROTEUS
- [MEMORY.md](../MEMORY.md) — Project state and decisions
- [Test Building](test_building.md) — Test generation prompts
- [Test Categorization](test_categorization.md) — Test markers and CI
- [Test Infrastructure](test_infrastructure.md) — Coverage and workflows
- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [GitHub Education (Academic License)](https://education.github.com/benefits)
