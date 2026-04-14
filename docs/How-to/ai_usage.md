# AI-Assisted Development

## Philosophy

PROTEUS is a scientific simulation framework where correctness matters more than convenience.
AI coding assistants can accelerate development, but every contribution — human or AI-generated — must meet the same standards for physical validity, numerical robustness, and test coverage.

**AI is not used for:** Scientific algorithms, physics implementations, or research decisions. These require domain expertise and human judgment.

---

## What This Document Is For

This guide explains how to use AI coding assistants safely and effectively with PROTEUS. AI assistants can significantly accelerate development, but require careful use to maintain code quality and security.

**Key principle:** AI is a powerful tool, not a replacement for understanding. Always review AI-generated code before committing.

---

## Quick Start

1. **Set up an AI assistant**: Install [GitHub Copilot for VS Code](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot), or use a CLI-based tool such as [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
2. **Provide context**: Point the assistant to `.github/copilot-instructions.md` (coding guidelines) and `.github/copilot-memory.md` (project state)
3. **Generate code**: Use prompts from [Test Building](test_building.md) for tests
4. **Review thoroughly**: Check all AI output before committing
5. **Run tests**: `pytest -m "unit and not skip"` and `ruff check`

---

## Project Context Files

PROTEUS uses two special files to provide AI assistants with project context:

### .github/copilot-instructions.md — Coding Guidelines

**Purpose:** Instructions for AI agents on how to write PROTEUS-compliant code. GitHub Copilot automatically discovers this file; other tools access it via the `CLAUDE.md` symlink at the project root.

**Contains:**

- Project structure and architecture
- Coding standards and style rules
- Testing requirements and markers
- Build commands and validation steps
- Common patterns and anti-patterns

**How to use:** GitHub Copilot reads this file automatically. For other AI assistants, add it to the context window or reference it in prompts.

### .github/copilot-memory.md — Project State

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
3. **Add context files**: In chat, use `@workspace` to reference project files, or drag `.github/copilot-instructions.md` into the chat
4. **Use `.github/copilot-instructions.md`**: This file automatically provides Copilot with PROTEUS guidelines

**Tutorials:**

- [Getting Started with GitHub Copilot](https://docs.github.com/en/copilot/using-github-copilot/getting-started-with-github-copilot)
- [Using Copilot Chat in VS Code](https://docs.github.com/en/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide)
- [Copilot Best Practices](https://docs.github.com/en/copilot/using-github-copilot/best-practices-for-using-github-copilot)

**Academic License:** Students and educators can apply for free GitHub Copilot access:
[GitHub Education](https://education.github.com/benefits)

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

### What AI Can Catch

- **Style issues**: Inconsistent formatting, missing docstrings
- **Common bugs**: Off-by-one errors, unhandled edge cases
- **Test gaps**: New functions without corresponding tests
- **Security issues**: Hardcoded credentials, unsafe file operations
- **API breaks**: Changes to function signatures without migration

### What AI Cannot Replace

- **Domain expertise**: AI does not understand planetary physics
- **Architectural decisions**: Humans decide system design
- **Security audits**: Critical security requires human review
- **Final approval**: A human must approve all PRs

---

## Safety and Security

### Critical Rules

1. **Never share secrets**: Do not paste API keys, passwords, or credentials into AI prompts
2. **Review all output**: AI can generate plausible-looking but incorrect code
3. **Verify physics**: AI does not understand scientific validity — check equations, units, and boundary conditions manually
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

- **Provide context**: Include `.github/copilot-instructions.md` and relevant source files
- **Be specific**: "Write a unit test for `calculate_flux` that tests edge case when T=0"
- **Iterate**: Ask AI to refine based on your feedback
- **Learn from output**: Use AI suggestions to improve your understanding

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

**Solution:** Include `.github/copilot-instructions.md` in context; it specifies valid markers

### AI doesn't know about fixtures

**Problem:** AI creates fixtures that already exist in `conftest.py`

**Solution:** Always include `tests/conftest.py` in the context window

### AI suggests outdated patterns

**Problem:** AI uses deprecated APIs or old coding patterns

**Solution:** Reference `.github/copilot-memory.md` for current patterns; specify Python version (3.12)

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

- [.github/copilot-instructions.md](https://github.com/FormingWorlds/PROTEUS/blob/main/.github/copilot-instructions.md) — AI coding guidelines for PROTEUS
- [.github/copilot-memory.md](https://github.com/FormingWorlds/PROTEUS/blob/main/.github/copilot-memory.md) — Project state and decisions
- [Test Building](test_building.md) — Test generation prompts
- [Test Categorization](test_categorization.md) — Test markers and CI
- [Test Infrastructure](test_infrastructure.md) — Coverage and workflows
- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [GitHub Education (Academic License)](https://education.github.com/benefits)
