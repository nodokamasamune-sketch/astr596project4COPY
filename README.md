# Project 4: Building a Universal Inference Engine

**ASTR 596: Modeling the Universe**  
**Instructor:** Dr. Anna Rosen  
**Due:** Friday, November 7, 2025, 11:59 PM

---

## ⚠️ Important Deadlines

### Code Submission: Friday, November 7, 2025 by 11:59 PM
**This repository will lock automatically at the deadline.**

Your code submission must include:
- All implementation code (`src/`, `scripts/`, `tests/`)
- All analysis outputs (`outputs/chains/`, `outputs/figures/`)
- **All plots and figures for your research memo**

**In other words:** Your analysis must be complete by the code deadline. Run your chains, generate all figures, and commit everything to this repository.

### Research & Growth Memos: Friday, November 7, 2025 by 11:59 PM
- **Normal submission:** Include your memos in this repository with your code
- **Late submission option:** If needed, you may submit finalized memos up to 24 hours late (by Saturday, November 8, 2025, 11:59 PM) by sending them **privately on Slack** to Dr. Rosen

All plots, data, and computational work must be completed by the code deadline.

---

## 📚 Project Documentation

**Refer to the course website for all project requirements and specifications:**

- **[Project Description](https://astrobytes-edu.github.io/astr596-modeling-universe/project4-description/)** - Complete implementation requirements, parts, extensions, and deliverables
- **[Scientific Background](https://astrobytes-edu.github.io/astr596-modeling-universe/project4-scientific-background/)** - Physics of Type Ia supernovae, cosmology, and the dark energy discovery

---

## 🎯 The Big Picture

You're building **general-purpose Bayesian inference machinery** to measure cosmological parameters from Type Ia supernova data—the same Nobel Prize-winning observations that revealed dark energy in 1998.

This project synthesizes everything from the course: statistical inference, MCMC sampling, Hamiltonian dynamics, forward/inverse problems, and convergence diagnostics. You're creating modular tools that work for any scientific inference problem, not just cosmology.

---

## 🗂️ Repository Structure

See the [Project Description](https://astrobytes-edu.github.io/astr596-modeling-universe/project4-description/) for the complete required repository structure.

---

## 🚀 Getting Started

```bash
# Clone your repository
git clone <your-github-classroom-repo-url>
cd astr596-project-04

# Run your analysis (see project website for details)
python scripts/run_mcmc.py
python scripts/run_hmc.py
python scripts/analyze_results.py
```

---

## ❓ Questions?

- Check the course website first for detailed specifications
- Post in the course Slack channel for clarifications
- Attend office hours for implementation help

Good luck measuring the universe!
