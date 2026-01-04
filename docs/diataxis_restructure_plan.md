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

### Phase 1: Preserve Existing Documentation

**Action:** Move all current docs to `docs/old_structure/`

```bash
mkdir -p docs/old_structure
# Original files will be moved here for reference
```

**Purpose:**
- Preserve all existing content
- Ensure no information is lost
- Allow comparison during restructure

### Phase 2: Create Directory Structure

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

### Phase 3: Create New Tutorial Content

**NEW FILES TO CREATE:**

#### `tutorials/getting-started.md`
**Purpose:** Complete beginner's first experience with PROTEUS

**Content approach (Diataxis principles):**
- Learning-oriented: focus on what the user learns, not what they produce
- Concrete steps: specific commands and expected outputs
- Meaningful progress: visible results at each step
- Minimal explanation: link to explanation docs, don't explain inline
- Encourage repetition: make it easy to go back and try again

**Suggested outline:**
1. Prerequisites check
2. Install PROTEUS (link to how-to/install.md)
3. Download required data
4. Run a minimal example simulation
5. Check the outputs
6. What you've learned

#### `tutorials/first-simulation.md`
**Purpose:** Learn to configure and run a custom simulation

**Content approach:**
- Build on getting-started tutorial
- Create a simple planet scenario
- Configure basic parameters
- Run and interpret results
- Modify and re-run

#### `tutorials/understanding-results.md`
**Purpose:** Learn to navigate and interpret PROTEUS outputs

**Content approach:**
- Explore output directory structure
- Load data from output files
- Create basic plots
- Identify successful vs. failed runs
- Understand key output metrics

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

### Phase 5: Organize Reference Material

**PURE DESCRIPTION ONLY - No instructions or explanations**

#### `config.md` → `reference/config-options.md`
**Transform to:**
- Alphabetical list of all configuration options
- Type, default value, constraints for each
- Brief factual description only
- Links to how-tos for usage examples
- Links to explanation for concepts

#### `data.md` → Split into:
- `reference/data-sources.md` - Where data comes from (Zenodo/OSF)
- `reference/data-formats.md` - File format specifications
- `reference/spectral-files.md` - Spectral data details

#### `bibliography.md` → `reference/bibliography.md`
**Keep as-is** (already pure reference)

#### NEW: `reference/test-structure.md`
**Content from test_infrastructure.md:**
- Required directory structure
- Naming conventions
- File organization rules

#### NEW: `reference/test-config.md`
**Content from test_infrastructure.md:**
- pytest configuration options
- Coverage settings
- Marker definitions

#### NEW: `reference/ci-workflows.md`
**Content from test_infrastructure.md:**
- Workflow descriptions
- Job definitions
- Environment variables
- Artifact specifications

### Phase 6: Create Explanation Content

**UNDERSTANDING-ORIENTED - Discussion and context**

#### From `model.md` → Split into:
- `explanation/architecture.md` - Overall system architecture
- `explanation/coupling.md` - How coupling works
- `explanation/scientific-background/` - Scientific concepts

#### NEW: `explanation/ecosystem.md`
**Content:**
- Overview of PROTEUS module ecosystem
- How modules relate to each other
- When to use which module
- Module interdependencies

#### NEW: `explanation/design-decisions/`
**Topics to cover:**
- Why modular architecture?
- Why Python 3.12+?
- Why Linux/MacOS only?
- Why Zenodo/OSF for data?
- Why this testing approach?

#### NEW: `explanation/testing-strategy.md`
**Content from test_infrastructure.md:**
- Philosophy behind testing approach
- Why mirror source structure?
- Coverage strategy rationale
- Diataxis testing guidelines context

#### NEW: Scientific background pages
**Extract from existing docs:**
- Atmospheric evolution concepts
- Interior dynamics theory
- Stellar evolution background
- Magma ocean physics
- Volatile cycling

### Phase 7: Update Home Page

#### Restructure `index.md`
**New structure:**
1. Welcome message
2. What is PROTEUS? (brief)
3. Documentation organized by Diataxis
   - **Tutorials** - If you're new, start here
   - **How-to Guides** - Recipes for specific tasks
   - **Reference** - Technical specifications
   - **Explanation** - Concepts and background
4. Quick links
5. Ecosystem modules
6. Getting help

### Phase 8: Update Navigation (mkdocs.yml)

**New navigation structure:**
```yaml
nav:
  - Home: index.md
  
  - Tutorials:
    - Getting Started: tutorials/getting-started.md
    - First Simulation: tutorials/first-simulation.md
    - Understanding Results: tutorials/understanding-results.md
  
  - How-to Guides:
    - Installation:
      - Install PROTEUS: how-to/installation/install.md
      - Local Machines: how-to/installation/local-setup.md
      - Kapteyn Cluster: how-to/installation/kapteyn-setup.md
      - Snellius Cluster: how-to/installation/snellius-setup.md
      - Habrok Cluster: how-to/installation/habrok-setup.md
    - Running Simulations:
      - Basic Simulation: how-to/simulations/run-simulation.md
      - Configure Planet: how-to/simulations/configure-planet.md
      - Parameter Grids: how-to/simulations/run-grids.md
      - Bayesian Inference: how-to/simulations/bayesian-inference.md
    - Development:
      - Setup Dev Environment: how-to/development/dev-setup.md
      - Write Tests: how-to/development/write-tests.md
      - Run Tests: how-to/development/run-tests.md
      - Contribute Code: how-to/development/contribute.md
      - Add Input Data: how-to/development/add-data.md
    - Troubleshooting: how-to/troubleshoot.md
  
  - Reference:
    - Configuration: reference/config-options.md
    - Data Sources: reference/data-sources.md
    - Data Formats: reference/data-formats.md
    - Testing:
      - Test Structure: reference/test-structure.md
      - Test Configuration: reference/test-config.md
      - CI Workflows: reference/ci-workflows.md
    - API Documentation: reference/api/
    - Bibliography: reference/bibliography.md
  
  - Explanation:
    - Architecture: explanation/architecture.md
    - Coupling Framework: explanation/coupling.md
    - Module Ecosystem: explanation/ecosystem.md
    - Scientific Background: explanation/scientific-background/
    - Design Decisions: explanation/design-decisions/
    - Testing Strategy: explanation/testing-strategy.md
  
  - Community:
    - Contact: contact.md
    - Code of Conduct: CODE_OF_CONDUCT.md
    - Funding: funding.md
  
  - External Links:
    - Source Code: https://github.com/FormingWorlds/PROTEUS
    - Issues: https://github.com/FormingWorlds/PROTEUS/issues
  
  - Ecosystem Modules:
    - 🔗 MORS: https://fwl-mors.readthedocs.io/
    - 🔗 JANUS: https://fwl-janus.readthedocs.io/
    - 🔗 CALLIOPE: https://fwl-calliope.readthedocs.io/
    # ... (other modules)
```

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

**Phase 1 (Preserve):** 1 hour
**Phase 2 (Structure):** 1 hour  
**Phase 3 (Tutorials):** 2-3 days
**Phase 4 (How-tos):** 3-4 days
**Phase 5 (Reference):** 2-3 days
**Phase 6 (Explanation):** 3-4 days
**Phase 7 (Index):** 1 day
**Phase 8 (Navigation):** 1 day

**Total estimated time:** 12-16 days

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
