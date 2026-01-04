# PROTEUS Documentation Restructure Plan

## Overview

This document outlines the plan to restructure PROTEUS documentation according to the [Diataxis framework](https://diataxis.fr/), a systematic approach to technical documentation that organizes content into four distinct categories based on user needs.

**Plan created:** January 4, 2026  
**Current branch:** `tl/enhance_docs`

---

## The Diataxis Framework

Diataxis organizes documentation into four quadrants based on two axes:

1. **Practical vs. Theoretical knowledge**
2. **Learning vs. Working (study vs. work)**

### The Four Documentation Types

| Type | Purpose | User Mode | Content Style |
|------|---------|-----------|---------------|
| **Tutorials** | Learning-oriented | Study | Practical steps, lessons |
| **How-to Guides** | Task-oriented | Work | Practical steps, goals |
| **Reference** | Information-oriented | Work | Theoretical knowledge, descriptions |
| **Explanation** | Understanding-oriented | Study | Theoretical knowledge, discussion |

**Key Principles:**

- **Tutorials** are lessons that take the user by the hand through a series of steps to complete a project
- **How-to guides** are directions that guide the user through solving a real-world problem
- **Reference** is technical description of the machinery and how it operates
- **Explanation** is discussion that clarifies and illuminates a particular topic

---

## Current PROTEUS Documentation Analysis

### Existing Files (as of January 2026)

Located in `docs/`:

- `index.md` - Home page
- `model.md` - Model description (mixed: explanation + some reference)
- `installation.md` - Installation guide (how-to)
- `local_machine_guide.md` - Local setup (how-to)
- `kapteyn_cluster_guide.md` - Cluster setup (how-to)
- `snellius_cluster_guide.md` - Cluster setup (how-to)
- `habrok_cluster_guide.md` - Cluster setup (how-to)
- `usage.md` - Using PROTEUS (mixed: tutorial + how-to + reference)
- `config.md` - Configuration (reference)
- `test_infrastructure.md` - Testing infrastructure (mixed: reference + how-to + explanation)
- `test_building.md` - Building tests (how-to)
- `troubleshooting.md` - Troubleshooting (how-to)
- `data.md` - Reference data (reference)
- `bibliography.md` - Bibliography (reference)
- `inference.md` - Bayesian inference (how-to)
- `contact.md` - Contact information
- `funding.md` - Funding information
- `CONTRIBUTING.md` - Contributing guidelines (mixed: how-to + reference + explanation)
- `CODE_OF_CONDUCT.md` - Code of conduct

### Current Issues

1. **No clear tutorials** - No beginner-friendly learning journey exists
2. **Mixed content types** - Many documents combine different Diataxis types
3. **Limited explanation** - Conceptual/theoretical discussions are scattered
4. **Unclear navigation** - Not organized by user intent

---

## Proposed New Structure

### Directory Organization

```
docs/
├── old_structure/           # Archive of original docs (preserved)
│   ├── model.md
│   ├── installation.md
│   ├── usage.md
│   ├── test_infrastructure.md
│   └── ... (all original files)
├── tutorials/               # NEW: Learning-oriented guides
│   ├── getting-started.md
│   ├── first-simulation.md
│   └── understanding-results.md
├── how-to/                  # Task-oriented guides (reorganized)
│   ├── installation/
│   │   ├── install.md
│   │   ├── local-setup.md
│   │   ├── kapteyn-setup.md
│   │   ├── snellius-setup.md
│   │   └── habrok-setup.md
│   ├── simulations/
│   │   ├── run-simulation.md
│   │   ├── configure-planet.md
│   │   ├── run-grids.md
│   │   └── bayesian-inference.md
│   ├── development/
│   │   ├── dev-setup.md
│   │   ├── write-tests.md
│   │   ├── run-tests.md
│   │   └── contribute.md
│   └── troubleshoot.md
├── reference/               # Information-oriented docs
│   ├── config-options.md
│   ├── data-sources.md
│   ├── test-structure.md
│   ├── ci-workflows.md
│   ├── api/                 # API documentation
│   └── bibliography.md
├── explanation/             # NEW: Understanding-oriented docs
│   ├── architecture.md
│   ├── coupling.md
│   ├── ecosystem.md
│   ├── scientific-background/
│   └── design-decisions/
├── index.md                 # Updated home page with Diataxis structure
├── contact.md
├── funding.md
└── CODE_OF_CONDUCT.md
```

---

## Implementation Plan

### Phase 1: Preserve Existing Documentation ✅

**Action:** Move all current docs to `docs/old_structure/`

```bash
mkdir -p docs/old_structure
# Original files will be moved here for reference
```

**Status:** COMPLETED

### Phase 2: Create Directory Structure ✅

**Action:** Create new Diataxis-aligned directories

```bash
mkdir -p docs/tutorials
mkdir -p docs/how-to/installation
mkdir -p docs/how-to/simulations
mkdir -p docs/how-to/development
mkdir -p docs/reference/api
mkdir -p docs/explanation/scientific-background
mkdir -p docs/explanation/design-decisions
```

**Status:** COMPLETED

### Phase 3: Create Stub Files and Extract Existing Content

**Strategy:** Create stub files for each new document with comments indicating what content from `old_structure/` should be extracted and placed there. Extract existing sections from original docs instead of writing new material.

**Stub files to create:**

#### Tutorials (learning-oriented)
- `tutorials/getting-started.md` 
  - *Stub:* Extract from `usage.md` "Running PROTEUS from the terminal" section
  - Add from `installation.md` user install steps
  
- `tutorials/first-simulation.md`
  - *Stub:* Extract from `usage.md` "Running PROTEUS from the terminal" section
  - Add from `config.md` basic parameter explanation
  
- `tutorials/understanding-results.md`
  - *Stub:* Extract from `usage.md` "Output and results" section

#### How-to Guides (task-oriented)

**Installation how-tos:**
- `how-to/installation/install.md`
  - *Stub:* Extract from `installation.md` "Setup a Python environment" and "User install" sections

- `how-to/installation/local-setup.md`
  - *Stub:* Entire content from `local_machine_guide.md`

- `how-to/installation/kapteyn-setup.md`
  - *Stub:* Entire content from `kapteyn_cluster_guide.md`

- `how-to/installation/snellius-setup.md`
  - *Stub:* Entire content from `snellius_cluster_guide.md`

- `how-to/installation/habrok-setup.md`
  - *Stub:* Entire content from `habrok_cluster_guide.md`

**Simulation how-tos:**
- `how-to/simulations/run-simulation.md`
  - *Stub:* Extract from `usage.md` sections on running and grids

- `how-to/simulations/grid-simulations.md`
  - *Stub:* Extract from `usage.md` "Running grids of simulations" section

- `how-to/simulations/remote-clusters.md`
  - *Stub:* Extract from `usage.md` "Running PROTEUS on remote machines" section

- `how-to/simulations/bayesian-inference.md`
  - *Stub:* Entire content from `inference.md`

**Development how-tos:**
- `how-to/development/write-tests.md`
  - *Stub:* Extract from `test_infrastructure.md` the writing tests sections

- `how-to/development/run-tests.md`
  - *Stub:* Extract from `test_infrastructure.md` "Quick Start" section

- `how-to/development/contribute.md`
  - *Stub:* Entire content from `CONTRIBUTING.md`

**General troubleshooting:**
- `how-to/troubleshoot.md`
  - *Stub:* Entire content from `troubleshooting.md`

#### Reference (information-oriented, pure description)

- `reference/config-options.md`
  - *Stub:* Entire content from `config.md`

- `reference/data-formats.md`
  - *Stub:* Entire content from `data.md`

- `reference/test-structure.md`
  - *Stub:* Extract from `test_infrastructure.md` "Architecture Overview" section

- `reference/ci-workflows.md`
  - *Stub:* Extract from `test_infrastructure.md` CI/CD Pipeline section

- `reference/bibliography.md`
  - *Stub:* Entire content from `bibliography.md`

#### Explanation (understanding-oriented, conceptual)

- `explanation/architecture.md`
  - *Stub:* Extract from `model.md` overview and schematic

- `explanation/scientific-background/planetary-evolution.md`
  - *Stub:* Create with conceptual explanations from `model.md`

- `explanation/design-decisions/modular-approach.md`
  - *Stub:* Create from philosophical discussions in `model.md`

### Phase 4: Reorganize How-to Guides

**CONTENT EXTRACTION:** Split existing files by Diataxis type

#### From `installation.md` → `how-to/installation/install.md`
**Keep:**
- Installation procedures
- Command sequences
- Dependency installation steps

**Move elsewhere:**
- Why certain dependencies → explanation/
- Troubleshooting → how-to/troubleshoot.md
- Technical details → reference/

#### From `local_machine_guide.md` → `how-to/installation/local-setup.md`
**Keep:**
- Platform-specific setup steps
- Environment configuration
- Path setup

**Move elsewhere:**
- System architecture explanations → explanation/
- Technical specifications → reference/

#### From `usage.md` → Split into multiple how-tos
**Extract to:**
- `how-to/simulations/run-simulation.md` - Basic execution
- `how-to/simulations/configure-planet.md` - Parameter configuration  
- `how-to/simulations/run-grids.md` - Running parameter grids

**Move elsewhere:**
- Conceptual explanations → explanation/
- Configuration reference → reference/config-options.md
- Examples that teach → tutorials/

#### From `test_infrastructure.md` → Split
**Extract to:**
- `how-to/development/write-tests.md` - How to write tests
- `how-to/development/run-tests.md` - How to run tests
- `reference/test-structure.md` - Test structure requirements
- `reference/test-config.md` - pytest/coverage configuration
- `reference/ci-workflows.md` - CI/CD workflow specs
- `explanation/testing-strategy.md` - Why this approach

#### From `CONTRIBUTING.md` → Split
**Extract to:**
- `how-to/development/contribute.md` - Contribution workflow
- `how-to/development/add-data.md` - Adding input data
- `reference/code-style.md` - Style guidelines
- `explanation/licensing.md` - Licensing philosophy



---

## Content Migration Guidelines

### For Each Existing Document

**Step 1: Identify Content Types**

Read through and classify each paragraph/section:
- ☑️ Tutorial content: Learning journey, first experiences
- ☑️ How-to content: Specific tasks, problem-solving
- ☑️ Reference content: Technical specifications, API docs
- ☑️ Explanation content: Concepts, context, "why" discussions

**Step 2: Extract and Reorganize**

Move content to appropriate new location:
- Tutorial → `tutorials/`
- How-to → `how-to/`
- Reference → `reference/`
- Explanation → `explanation/`

**Step 3: Adapt Writing Style**

Adjust tone and style for Diataxis category:

| Category | Style | Language |
|----------|-------|----------|
| Tutorial | Instructive, encouraging | "We will...", "You've learned..." |
| How-to | Directive, clear | "Do this...", "If X, then Y..." |
| Reference | Neutral, factual | "X is...", "Parameter Y accepts..." |
| Explanation | Discursive, contextual | "This is because...", "Consider..." |

**Step 4: Add Cross-Links**

Link between document types:
- Tutorials link to reference and explanation (for later learning)
- How-tos link to reference (for details) and tutorials (for background)
- Reference links to how-tos (for usage) and explanation (for context)
- Explanation links to all others as appropriate

---

## Quality Checks

### Diataxis Compliance Checklist

For each document, verify:

**Tutorials:**
- [ ] Takes user through a complete learning experience
- [ ] Focuses on doing, not explaining
- [ ] Shows expected results at each step
- [ ] Suitable for absolute beginners
- [ ] Learning-oriented language

**How-to Guides:**
- [ ] Addresses a specific task or problem
- [ ] Assumes user knows what they want
- [ ] Provides clear steps to achieve goal
- [ ] Task-oriented language
- [ ] No teaching or explanation

**Reference:**
- [ ] Pure description, no instruction
- [ ] Technically accurate and complete
- [ ] Organized by code/system structure
- [ ] Neutral, factual language
- [ ] No "how to" content

**Explanation:**
- [ ] Provides context and background
- [ ] Discusses "why" not "how"
- [ ] Can be read away from the product
- [ ] Understanding-oriented language
- [ ] No step-by-step instructions

---

## Timeline Estimate

**Phase 1 (Preserve):** 1 hour ✅  
**Phase 2 (Structure):** 1 hour ✅  
**Phase 3 (Stub files & extract):** 4-5 days
**Phase 4 (Review & refine):** 2-3 days
**Phase 5 (Navigation updates):** 1 day
**Phase 6 (Testing & validation):** 1-2 days

**Total estimated time:** 8-12 days

---

## Success Criteria

After restructure, documentation should:

✅ Follow Diataxis principles consistently  
✅ Help new users get started (tutorials)  
✅ Help experienced users solve problems (how-tos)  
✅ Provide authoritative technical information (reference)  
✅ Build understanding of concepts (explanation)  
✅ Maintain all information from original docs  
✅ Have clear, intuitive navigation  
✅ Work correctly with mkdocs/Material theme  
✅ All internal links functional  
✅ Pass documentation review

---

## Notes and Considerations

### Original Documentation Preservation

All original documentation files remain in `docs/old_structure/` directory:
- Available for reference during migration
- Ensures no information is lost
- Can be compared with new structure
- May be removed after successful migration and review

### Gradual Migration Strategy

This restructure can be done incrementally:
1. Start with one category (e.g., tutorials)
2. Test with users and gather feedback
3. Iterate and improve
4. Move to next category
5. Update navigation progressively

### Links and References

- Diataxis framework: https://diataxis.fr/
- Material for MkDocs: https://squidfunk.github.io/mkdocs-material/
- PROTEUS GitHub: https://github.com/FormingWorlds/PROTEUS

---

**Document Status:** Draft Plan  
**Next Steps:** Review plan with team, begin Phase 1  
**Maintained by:** Tim Lichtenberg  
**Last updated:** January 4, 2026
