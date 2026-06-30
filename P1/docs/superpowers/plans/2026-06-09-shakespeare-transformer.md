# Shakespeare Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Jupyter notebook assignment submission for a 6-layer decoder-only Transformer trained on Shakespeare text.

**Architecture:** Put reusable PyTorch model and analysis utilities in `src/shakespeare_transformer.py`, validate them with `pytest`, and generate `shakespeare_transformer.ipynb` as the course-facing deliverable. Keep data and tokenizer setup scripts separate so the notebook can run with local assets or documented download fallbacks.

**Tech Stack:** Python virtual environment, PyTorch, transformers, sentencepiece, datasets/modelscope helpers where available, pytest, nbformat, matplotlib.

---

### Task 1: Environment And Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `scripts/download_assets.py`

- [ ] Create a project-local virtual environment at `.venv`.
- [ ] Add Python dependencies needed for the notebook, tests, tokenizer loading, and optional downloads.
- [ ] Add download helper for TinyShakespeare and Llama 2 tokenizer files.

### Task 2: Core Transformer Module

**Files:**
- Create: `src/shakespeare_transformer.py`
- Create: `src/__init__.py`
- Create: `tests/test_shakespeare_transformer.py`

- [ ] Write tests for RoPE, causal attention masking, model output shape, parameter counting, and tiny loss calculation.
- [ ] Run tests before implementation and confirm expected import failures.
- [ ] Implement the model and utilities.
- [ ] Run tests again and confirm they pass.

### Task 3: Notebook

**Files:**
- Create: `shakespeare_transformer.ipynb`

- [ ] Generate notebook with assignment sections: setup, data, tokenizer, model implementation overview, parameter/FLOPs analysis, training loop, loss visualization, generation, and discussion.
- [ ] Ensure notebook imports `src.shakespeare_transformer` and includes enough explanatory markdown for grading.
- [ ] Include fallback settings for quick smoke runs and longer training.

### Task 4: Asset Preparation And Verification

**Files:**
- Modify: local `data/` and `tokenizer/` directories if downloads succeed.

- [ ] Install dependencies into `.venv`.
- [ ] Run `scripts/download_assets.py`.
- [ ] Run `pytest`.
- [ ] Run a notebook smoke execution or a Python smoke script that instantiates tokenizer/model and computes one loss.
