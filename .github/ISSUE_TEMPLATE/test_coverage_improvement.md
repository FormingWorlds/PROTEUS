---
name: Test Coverage Improvement
about: Track test coverage improvements for specific folders
title: 'Improve test coverage for [FOLDER]'
labels: 'testing, enhancement'
assignees: ''
---

## Folder
<!-- e.g., proteus.config, proteus.interior, etc. -->

## Current Coverage
<!-- Run: pytest --cov=src/proteus/[folder] --cov-report=term-missing -->
```
Current: X%
Target: Y%
```

## Uncovered Lines
<!-- From coverage report --show-missing -->
```
file.py: 10, 25-30, 45
```

## Test Strategy

### Unit Tests Needed
- [ ] Function: `function_name()` (lines X-Y)
- [ ] Function: `another_function()` (lines X-Y)
- [ ] Class: `ClassName` (lines X-Y)

### Integration Tests Needed
- [ ] Integration point: description
- [ ] Workflow: description

### Edge Cases
- [ ] Error handling for X
- [ ] Boundary conditions for Y
- [ ] Invalid input handling

## Implementation Plan

1. [ ] Create test file: `tests/[folder]/test_[feature].py`
2. [ ] Add fixtures in `conftest.py` (if needed)
3. [ ] Write unit tests
4. [ ] Write integration tests
5. [ ] Run locally: `pytest tests/[folder]/`
6. [ ] Verify coverage: `pytest --cov=src/proteus/[folder] --cov-report=html`
7. [ ] Update documentation

## Success Criteria
- [ ] Coverage increases to target %
- [ ] All new tests pass
- [ ] No regressions in existing tests
- [ ] CI pipeline passes

## Notes
<!-- Additional context, challenges, or considerations -->
