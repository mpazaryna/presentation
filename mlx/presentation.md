---
marp: true
theme: default
style: |
  section {
    font-size: 28px;
  }
---

# MLX Framework

Machine Learning on Apple Silicon

---

# The Landscape

- ML frameworks optimized for NVIDIA GPUs
- Apple Silicon left behind
- Gap in the market

---

# Enter MLX

- Open-source from Apple
- Python-first design
- Unified memory architecture

---

# Why Apple Silicon?

- Billions of devices
- High-performance GPU
- Shared memory model
- Massive untapped potential

---

# Core Design Principles

- Simple and intuitive API
- Familiar NumPy-like syntax
- Lazy evaluation
- Composable functions

---

# Architecture Overview

- NumPy-like array interface
- Unified memory (CPU + GPU)
- Automatic differentiation

---

# Key Features

- Efficient inference on-device
- Fine-tuning and training
- Memory-optimized operations
- Cross-platform compatibility

---

# MLX vs Alternatives

- PyTorch: GPU-focused, larger overhead
- TensorFlow: Production-heavy, complex
- JAX: Functional, steep learning curve
- MLX: Apple Silicon native, lightweight

---

# Core Concepts

- **Arrays:** Lazy computation model
- **Operations:** GPU-accelerated when beneficial
- **Gradients:** Full autodiff support
- **Training:** Lightning-fast fine-tuning

---

# Training Pipeline: Data to Inference

**Step 1: Data Preparation**
- Extract and annotate domain-specific patterns
- Convert to optimized training tensors

**Step 2: Model Training**
- Train specialized architectures from scratch
- Metal-accelerated on Apple Silicon

**Step 3: Deploy & Inference**
- Package models (.npz or .safetensors)
- Load at startup, run in parallel

---

# What You Can Do

- Train specialized models from scratch
- Inference on M-series chips in <50ms
- Parallel model execution
- Bundle models directly in apps

---

# The Future

- On-device AI becomes standard
- Privacy-first applications
- Distributed ML at the edge

---

# Questions?

**MLX:** https://github.com/ml-explore/mlx
**HuggingFace:** https://huggingface.co/docs/hub/en/mlx