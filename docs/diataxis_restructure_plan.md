# PROTEUS Documentation Restructure Plan

## Overview

This document outlines the plan to restructure PROTEUS documentation according to the [Diataxis framework](https://diataxis.fr/), a systematic approach to technical documentation that organizes content into four distinct categories based on user needs.

**Plan created:** January 4, 2026
**Last updated:** January 4, 2026
**Current branch:** `tl/enhance_docs`
**Status:** In Progress - Phase 4

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

**Current Status (as of January 4, 2026):**

```
docs/
├── setup/                   # ✅ Created - Platform-specific setup guides
│   ├── local-setup.md      # ✅ Content extracted from local_machine_guide.md
│   ├── kapteyn-setup.md    # ✅ Content extracted from kapteyn_cluster_guide.md
│   ├── habrok-setup.md     # ✅ Content extracted from habrok_cluster_guide.md
│   └── snellius-setup.md   # ✅ Content extracted from snellius_cluster_guide.md
├── reference/               # ✅ Created - Technical reference docs
│   ├── bibliography.md     # ✅ Content extracted from old bibliography.md
│   ├── data-formats.md     # ✅ Content extracted from old data.md
│   ├── test-structure.md   # ✅ Content extracted from test_infrastructure.md
│   ├── test-config.md      # ✅ Content extracted from test_infrastructure.md
│   ├── ci-workflows.md     # ✅ Content extracted from test_infrastructure.md
│   └── api/                # Directory for API docs
├── old_structure/          # ✅ Created - Archived original docs
│   ├── README.md           # ✅ Migration guide and deprecation notice
│   ├── bibliography.md     # Original file preserved
│   ├── data.md            # Original file preserved
│   ├── local_machine_guide.md
│   ├── kapteyn_cluster_guide.md
│   ├── habrok_cluster_guide.md
│   ├── snellius_cluster_guide.md
│   └── test_infrastructure.md
├── tutorials/              # 🚧 Stub files exist, need content
│   ├── getting-started.md
│   ├── first-simulation.md
│   └── understanding-results.md
├── how-to/                 # 🚧 Partial - Some files have content, others are stubs
│   ├── installation/
│   │   ├── install.md     # 🚧 STUB - Needs content from installation.md
│   │   └── [setup files ref setup/ directory]
│   ├── simulations/
│   │   ├── run-simulation.md        # 🚧 STUB - Needs content from usage.md
│   │   ├── grid-simulations.md      # 🚧 STUB - Needs content from usage.md
│   │   ├── remote-clusters.md       # 🚧 STUB - Needs content from usage.md
│   │   ├── archiving.md             # 🚧 STUB
│   │   ├── offline-chemistry.md     # 🚧 STUB
│   │   ├── synthetic-observations.md # 🚧 STUB
│   │   └── bayesian-inference.md    # 🚧 STUB - Needs content from inference.md
│   ├── development/
│   │   ├── contribute.md   # 🚧 STUB - Needs content from CONTRIBUTING.md
│   │   ├── write-tests.md  # 🚧 STUB - Needs content from test_infrastructure.md
│   │   └── run-tests.md    # 🚧 STUB - Needs content from test_infrastructure.md
│   └── troubleshoot.md     # 🚧 STUB - Needs content from troubleshooting.md
├── explanation/            # 🚧 Stub files exist, need content
│   ├── architecture.md     # 🚧 STUB - Needs content from model.md
│   ├── ecosystem.md        # 🚧 STUB
│   ├── design-decisions.md # 🚧 STUB - Needs content from model.md
│   └── scientific-background/
│       └── planetary-evolution.md # 🚧 STUB - Needs content from model.md
├── index.md               # Home page
├── model.md              # In docs root (referenced in old menu)
├── installation.md       # In docs root (referenced in old menu)
├── usage.md             # In docs root (referenced in old menu)
├── config.md            # In docs root (referenced in current menu)
├── troubleshooting.md   # In docs root (referenced in old menu)
├── inference.md         # In docs root (referenced in old menu)
├── CONTRIBUTING.md      # In docs root (referenced in old menu)
├── contact.md
├── funding.md
└── CODE_OF_CONDUCT.md
```

**Legend:**
- ✅ = Completed with extracted content
- 🚧 = Stub file exists, needs content extraction
- 📁 = Directory created

**Target Final Structure:**

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

**Status:** ✅ COMPLETED

### Phase 3: Create Stub Files and Extract Existing Content ⚠️

**Status:** 🚧 PARTIALLY COMPLETED

**Strategy:** Create stub files for each new document with comments indicating what content from `old_structure/` should be extracted and placed there. Extract existing sections from original docs instead of writing new material.

**✅ Completed extractions:**
- `setup/local-setup.md` - Extracted from `local_machine_guide.md` (40 lines)
- `setup/kapteyn-setup.md` - Extracted from `kapteyn_cluster_guide.md` (200+ lines)
- `setup/habrok-setup.md` - Extracted from `habrok_cluster_guide.md` (83 lines)
- `setup/snellius-setup.md` - Extracted from `snellius_cluster_guide.md` (45 lines)
- `reference/bibliography.md` - Extracted from `bibliography.md` (66 lines)
- `reference/data-formats.md` - Extracted from `data.md` (136 lines)
- `reference/test-structure.md` - Extracted from `test_infrastructure.md` (80 lines)
- `reference/test-config.md` - Extracted from `test_infrastructure.md` (155+ lines)
- `reference/ci-workflows.md` - Extracted from `test_infrastructure.md` (222 lines)

**🚧 Stub files needing content extraction:**

#### Tutorials (learning-oriented) 🚧
- `tutorials/getting-started.md` - **STUB**
  - *Extract from:* `usage.md` "Running PROTEUS from the terminal" section
  - *Also add from:* `installation.md` user install steps

- `tutorials/first-simulation.md` - **STUB**
  - *Extract from:* `usage.md` "Running PROTEUS from the terminal" section
  - *Also add from:* `config.md` basic parameter explanation

- `tutorials/understanding-results.md` - **STUB**
  - *Extract from:* `usage.md` "Output and results" section

#### How-to Guides (task-oriented) 🚧

**Installation how-tos:**
- `how-to/installation/install.md` - **STUB**
  - *Extract from:* `installation.md` "Setup a Python environment" and "User install" sections

**Simulation how-tos:**
- `how-to/simulations/run-simulation.md` - **STUB**
  - *Extract from:* `usage.md` sections on running and grids

- `how-to/simulations/grid-simulations.md` - **STUB**
  - *Extract from:* `usage.md` "Running grids of simulations" section

- `how-to/simulations/remote-clusters.md` - **STUB**
  - *Extract from:* `usage.md` "Running PROTEUS on remote machines" section

- `how-to/simulations/archiving.md` - **STUB**
  - *Extract from:* `usage.md` archiving sections

- `how-to/simulations/offline-chemistry.md` - **STUB**
  - *Extract from:* `usage.md` or create new content

- `how-to/simulations/synthetic-observations.md` - **STUB**
  - *Extract from:* `usage.md` or create new content

- `how-to/simulations/bayesian-inference.md` - **STUB**
  - *Extract from:* `inference.md` (entire content)

**Development how-tos:**
- `how-to/development/write-tests.md` - **STUB**
  - *Extract from:* `test_infrastructure.md` the writing tests sections

- `how-to/development/run-tests.md` - **STUB**
  - *Extract from:* `test_infrastructure.md` "Quick Start" section

- `how-to/development/contribute.md` - **STUB**
  - *Extract from:* `CONTRIBUTING.md` (entire content)

**General troubleshooting:**
- `how-to/troubleshoot.md` - **STUB**
  - *Extract from:* `troubleshooting.md` (entire content)

#### Reference (information-oriented, pure description) ✅

✅ `reference/bibliography.md` - **COMPLETED**
  - *Extracted from:* `bibliography.md` (66 lines)

✅ `reference/data-formats.md` - **COMPLETED**
  - *Extracted from:* `data.md` (136 lines)

✅ `reference/test-structure.md` - **COMPLETED**
  - *Extracted from:* `test_infrastructure.md` "Architecture Overview" section (80 lines)

✅ `reference/test-config.md` - **COMPLETED**
  - *Extracted from:* `test_infrastructure.md` pytest/coverage config (155+ lines)

✅ `reference/ci-workflows.md` - **COMPLETED**
  - *Extracted from:* `test_infrastructure.md` CI/CD Pipeline section (222 lines)

#### Explanation (understanding-oriented, conceptual) 🚧

- `explanation/architecture.md` - **STUB**
  - *Extract from:* `model.md` overview and schematic

- `explanation/ecosystem.md` - **STUB**
  - *Create new:* Overview of PROTEUS ecosystem modules

- `explanation/design-decisions.md` - **STUB**
  - *Extract from:* `model.md` philosophical discussions

- `explanation/scientific-background/planetary-evolution.md` - **STUB**
  - *Extract from:* `model.md` conceptual explanations

### Phase 4: Content Extraction and Migration 🚧

**Status:** 🚧 IN PROGRESS

**Completed migrations:**

1. **✅ Setup Guides** (Phase 4 complete for this category)
   - `setup/local-setup.md` ← `old_structure/local_machine_guide.md`
   - `setup/kapteyn-setup.md` ← `old_structure/kapteyn_cluster_guide.md`
   - `setup/habrok-setup.md` ← `old_structure/habrok_cluster_guide.md`
   - `setup/snellius-setup.md` ← `old_structure/snellius_cluster_guide.md`

2. **✅ Reference Documentation** (Phase 4 complete for this category)
   - `reference/bibliography.md` ← `old_structure/bibliography.md`
   - `reference/data-formats.md` ← `old_structure/data.md`
   - `reference/test-structure.md` ← `old_structure/test_infrastructure.md`
   - `reference/test-config.md` ← `old_structure/test_infrastructure.md`
   - `reference/ci-workflows.md` ← `old_structure/test_infrastructure.md`

3. **✅ Navigation Updates**
   - mkdocs.yml restructured with Diataxis framework (153 lines)
   - Old menu moved to bottom as deprecated section
   - Restructure plan moved to top of menu

4. **✅ Deprecation Infrastructure**
   - Created `old_structure/README.md` with migration guide
   - Added ⚠️ warning emoji to old menu items
   - Documented file mapping in README

**🚧 Remaining extractions needed:**

**From `installation.md` → `how-to/installation/install.md`**
- [ ] Extract: Installation procedures
- [ ] Extract: Command sequences
- [ ] Extract: Dependency installation steps
- [ ] Remove: Why certain dependencies → move to explanation/
- [ ] Remove: Troubleshooting → move to how-to/troubleshoot.md

**From `usage.md` → Split into multiple how-tos**
- [ ] Extract to `how-to/simulations/run-simulation.md`: Basic execution
- [ ] Extract to `how-to/simulations/grid-simulations.md`: Running parameter grids
- [ ] Extract to `how-to/simulations/remote-clusters.md`: Remote execution
- [ ] Extract to `how-to/simulations/archiving.md`: Archiving results
- [ ] Extract to `tutorials/getting-started.md`: Beginner walkthrough
- [ ] Remove: Conceptual explanations → move to explanation/
- [ ] Remove: Configuration details → move to reference/config.md

**From `inference.md` → `how-to/simulations/bayesian-inference.md`**
- [ ] Extract: Entire content (how-to style)

**From `troubleshooting.md` → `how-to/troubleshoot.md`**
- [ ] Extract: Entire content

**From `CONTRIBUTING.md` → `how-to/development/contribute.md`**
- [ ] Extract: Contribution workflow
- [ ] Remove: Code style guidelines → move to reference/code-style.md
- [ ] Remove: Licensing discussion → move to explanation/licensing.md

**From `model.md` → Split into explanation/**
- [ ] Extract to `explanation/architecture.md`: System overview and schematic
- [ ] Extract to `explanation/scientific-background/planetary-evolution.md`: Scientific concepts
- [ ] Extract to `explanation/design-decisions.md`: Design philosophy

### Phase 5: Navigation and Cross-Linking ✅

**Status:** ✅ COMPLETED

**Completed:**
- ✅ Updated mkdocs.yml with Diataxis-based navigation
- ✅ Created hierarchical menu structure
- ✅ Organized into: Tutorials → How-To → Explanation → Reference → Community
- ✅ Added "Old Menu (Deprecated)" section at bottom
- ✅ Moved restructure plan to top of menu

**Remaining:**
- [ ] Add cross-links between document types after content extraction
- [ ] Verify all internal links work
- [ ] Add "See also" sections in each document

### Phase 6: Review and Testing 🚧

**Status:** ⏳ NOT STARTED

**Tasks:**
- [ ] Build documentation with mkdocs
- [ ] Test all navigation links
- [ ] Review extracted content for completeness
- [ ] Verify Diataxis compliance
- [ ] User testing and feedback
- [ ] Final cleanup of old files

---

## Content Migration Progress Summary

**Files Migrated:** 9 of ~20 planned
**Lines Extracted:** ~1000+ lines
**Directories Created:** 4 (setup/, reference/, old_structure/, explanation/)
**Stub Files Created:** ~15

**Completion by Category:**
- ✅ **Reference:** 100% (5/5 files)
- ✅ **Setup Guides:** 100% (4/4 files)
- 🚧 **How-To Guides:** 10% (0/10 files with full content)
- 🚧 **Tutorials:** 0% (0/3 files)
- 🚧 **Explanation:** 0% (0/4 files)

**Overall Progress:** ~35% complete

---

## Content Extraction Mapping Table

| Source File | Target File(s) | Status | Content Type |
|------------|---------------|---------|--------------|
| `bibliography.md` | `reference/bibliography.md` | ✅ Done | Reference |
| `data.md` | `reference/data-formats.md` | ✅ Done | Reference |
| `test_infrastructure.md` | `reference/test-structure.md` | ✅ Done | Reference |
| `test_infrastructure.md` | `reference/test-config.md` | ✅ Done | Reference |
| `test_infrastructure.md` | `reference/ci-workflows.md` | ✅ Done | Reference |
| `local_machine_guide.md` | `setup/local-setup.md` | ✅ Done | How-To |
| `kapteyn_cluster_guide.md` | `setup/kapteyn-setup.md` | ✅ Done | How-To |
| `habrok_cluster_guide.md` | `setup/habrok-setup.md` | ✅ Done | How-To |
| `snellius_cluster_guide.md` | `setup/snellius-setup.md` | ✅ Done | How-To |
| `installation.md` | `how-to/installation/install.md` | 🚧 Stub | How-To |
| `usage.md` | `how-to/simulations/run-simulation.md` | 🚧 Stub | How-To |
| `usage.md` | `how-to/simulations/grid-simulations.md` | 🚧 Stub | How-To |
| `usage.md` | `how-to/simulations/remote-clusters.md` | 🚧 Stub | How-To |
| `usage.md` | `tutorials/getting-started.md` | 🚧 Stub | Tutorial |
| `inference.md` | `how-to/simulations/bayesian-inference.md` | 🚧 Stub | How-To |
| `troubleshooting.md` | `how-to/troubleshoot.md` | 🚧 Stub | How-To |
| `CONTRIBUTING.md` | `how-to/development/contribute.md` | 🚧 Stub | How-To |
| `model.md` | `explanation/architecture.md` | 🚧 Stub | Explanation |
| `model.md` | `explanation/design-decisions.md` | 🚧 Stub | Explanation |
| `model.md` | `explanation/scientific-background/planetary-evolution.md` | 🚧 Stub | Explanation |

---

## Next Immediate Steps

1. **Continue Phase 4 content extraction:**
   - Extract from `installation.md` → `how-to/installation/install.md`
   - Extract from `usage.md` → multiple how-to files
   - Extract from `inference.md` → `how-to/simulations/bayesian-inference.md`

2. **Extract tutorial content:**
   - Create beginner-friendly tutorial from `usage.md` sections

3. **Extract explanation content:**
   - Split `model.md` into conceptual explanation files

4. **Add cross-links:**
   - Link between related documents
   - Add "See also" sections

5. **Testing:**
   - Build and test documentation site
   - Verify all links work

---

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

**Phase 1 (Preserve):** 1 hour ✅ COMPLETED
**Phase 2 (Structure):** 1 hour ✅ COMPLETED
**Phase 3 (Stub files & extract):** 4-5 days 🚧 ~35% COMPLETE
**Phase 4 (Content extraction):** 2-3 days 🚧 IN PROGRESS
**Phase 5 (Navigation updates):** 1 day ✅ COMPLETED
**Phase 6 (Review & refine):** 1-2 days ⏳ NOT STARTED
**Phase 7 (Testing & validation):** 1-2 days ⏳ NOT STARTED

**Total estimated time:** 8-12 days
**Time elapsed:** ~3-4 days
**Estimated remaining:** 4-8 days

---

## Success Criteria

After restructure, documentation should:

✅ Follow Diataxis principles consistently
🚧 Help new users get started (tutorials) - *Stub files exist*
🚧 Help experienced users solve problems (how-tos) - *Partially complete*
✅ Provide authoritative technical information (reference) - *Complete*
🚧 Build understanding of concepts (explanation) - *Stub files exist*
✅ Maintain all information from original docs - *Preserved in old_structure/*
✅ Have clear, intuitive navigation - *mkdocs.yml updated*
⏳ Work correctly with mkdocs/Material theme - *Not yet tested*
⏳ All internal links functional - *Not yet verified*
⏳ Pass documentation review - *Pending*

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

**Document Status:** In Progress (~35% complete)
**Next Steps:** Continue Phase 4 content extraction from source files
**Maintained by:** Tim Lichtenberg
**Last updated:** January 4, 2026
