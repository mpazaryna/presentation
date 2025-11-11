# MLX Framework Documentation Index

Complete documentation for Apple's MLX machine learning framework for Apple Silicon.

## Overview Documents

### [MLX Framework Overview](mlx-framework-overview.md)
Introduction to MLX, its core design philosophy, key features, installation instructions, and platform requirements.

**Key Topics:**
- What is MLX
- Core design principles (function transformations, lazy evaluation, multi-device support, unified memory)
- Key features
- Installation (macOS, Linux with CUDA, Linux CPU-only)
- Platform requirements
- Available examples
- Project information and citations

### [Getting Started](getting-started.md)
Quick start guide for new users with basic array operations, lazy evaluation, function transformations, and simple examples.

**Key Topics:**
- Basic array operations
- Understanding lazy evaluation
- Function transformations (grad, vmap, composition)
- Simple training loop example
- Working with devices

## Core Concepts

### [Lazy Evaluation](lazy-evaluation.md)
Deep dive into MLX's lazy evaluation model, including benefits, when evaluation occurs, and best practices.

**Key Topics:**
- Core concept of lazy evaluation
- Graph transformation support
- Memory efficiency benefits
- Selective computation
- Explicit vs. implicit evaluation
- Best practices and performance tips

### [Array Indexing](array-indexing.md)
Complete guide to indexing arrays in MLX, including basic indexing, advanced techniques, and key differences from NumPy.

**Key Topics:**
- Basic indexing with integers and slices
- Multidimensional indexing with ellipsis
- Adding new axes
- Array indexing
- In-place updates
- Key differences from NumPy (no bounds checking, no boolean masking)
- Performance tips

### [Function Transforms](function-transforms.md)
Comprehensive documentation on automatic differentiation, automatic vectorization, and composing transformations.

**Key Topics:**
- Automatic differentiation (grad, higher-order derivatives)
- Gradient with respect to multiple arguments
- Efficient loss and gradient calculation
- Complex parameter structures
- Stopping gradient flow
- Automatic vectorization (vmap)
- Vector-Jacobian products (vjp)
- Jacobian-vector products (jvp)
- Composing transformations
- Practical neural network examples

## API Reference

### [Neural Networks](neural-networks.md)
Complete reference for the MLX neural networks module (`mlx.nn`).

**Key Topics:**
- Module class and parameters
- Parameter management (freezing, unfreezing)
- Module inspection and utilities
- Training control (train/eval modes)
- Layer types:
  - Activation functions (ReLU, GELU, SiLU, Sigmoid, Tanh)
  - Pooling layers (AvgPool, MaxPool in 1D/2D/3D)
  - Normalization (LayerNorm, GroupNorm, BatchNorm, RMSNorm)
  - Recurrent networks (LSTM, GRU, RNN)
  - Attention mechanisms (MultiHeadAttention, ALiBi, RoPE)
  - Specialized layers (Embedding, Dropout, Transformer)
- Training integration with value_and_grad
- Common patterns (Sequential, custom modules, fine-tuning)

### [Optimizers and Learning Rates](optimizers-and-learning-rates.md)
Complete guide to optimization algorithms and learning rate scheduling in MLX.

**Key Topics:**
- Core optimization workflow
- First-order methods (SGD, RMSprop, Adagrad, Adadelta)
- Adaptive methods (Adam, AdamW, Adamax, Adafactor)
- Modern optimizers (Lion, Muon)
- MultiOptimizer for different parameter groups
- Learning rate scheduling:
  - Cosine decay
  - Exponential decay
  - Linear schedule
  - Step decay
  - Joining schedules
- Gradient clipping
- Saving and loading optimizer state
- Complete training examples

### [Random Number Generation](random-numbers.md)
Comprehensive reference for MLX's random number generation functions.

**Key Topics:**
- PRNG design (global and explicit key-based)
- Basic usage and key splitting
- Distribution functions:
  - Uniform, Normal, Truncated Normal
  - Multivariate Normal
  - Bernoulli, Gumbel, Laplace
- Integer operations:
  - Random integers, permutations
  - Categorical sampling
- State management (seeds, explicit keys)
- Practical examples (reproducibility, augmentation, dropout)
- Performance considerations

## Examples and Tutorials

### [Examples and Tutorials](examples-tutorial.md)
Practical code examples demonstrating MLX functionality.

**Included Examples:**
1. Linear regression from scratch
2. Multi-layer perceptron for MNIST
3. Simple linear regression training loop
4. Automatic differentiation and derivatives
5. Automatic vectorization (vmap)
6. Random number generation
7. Neural network with data augmentation
8. Custom training with gradient clipping
9. Saving and loading models
10. Distributed training

## Advanced Topics

### [Advanced Topics](advanced-topics.md)
Advanced features and optimization techniques.

**Key Topics:**
- Distributed training:
  - MPI and Ring backends
  - Launching distributed programs
  - Gradient averaging
  - Backend selection
- FFT operations:
  - 1D, 2D, and n-dimensional transforms
  - Real-valued FFT variants
  - Utility functions
- Streams and asynchronous computation
- Memory optimization
- Performance profiling
- Custom gradient functions
- Model export and interoperability
- Debugging tips

## Documentation Structure

```
mlx/
├── INDEX.md                              # This file
├── mlx-framework-overview.md             # Framework introduction
├── getting-started.md                    # Quick start guide
├── lazy-evaluation.md                    # Lazy evaluation concepts
├── array-indexing.md                     # Array indexing guide
├── function-transforms.md                # Autodiff and vmap
├── neural-networks.md                    # nn module reference
├── optimizers-and-learning-rates.md      # Optimization guide
├── random-numbers.md                     # RNG documentation
├── examples-tutorial.md                  # Practical examples
└── advanced-topics.md                    # Advanced features
```

## Quick Navigation by Task

### I want to...

#### Get started quickly
1. Read: [Getting Started](getting-started.md)
2. Run: Examples from [Examples and Tutorials](examples-tutorial.md)

#### Build a neural network
1. Read: [Neural Networks](neural-networks.md)
2. Learn: [Function Transforms](function-transforms.md) for training
3. Train: Use [Optimizers and Learning Rates](optimizers-and-learning-rates.md)

#### Train a model efficiently
1. Understand: [Lazy Evaluation](lazy-evaluation.md)
2. Implement: Linear regression example from [Examples and Tutorials](examples-tutorial.md)
3. Optimize: Follow [Advanced Topics](advanced-topics.md) for performance

#### Use distributed training
1. Read: Distributed section of [Advanced Topics](advanced-topics.md)
2. Check: Gradient averaging patterns

#### Process arrays effectively
1. Learn: [Array Indexing](array-indexing.md)
2. Apply: [Function Transforms](function-transforms.md) for transformations

#### Generate random data
1. Reference: [Random Number Generation](random-numbers.md)
2. Examples: Practical examples section in same document

#### Optimize hyperparameters
1. Study: [Optimizers and Learning Rates](optimizers-and-learning-rates.md)
2. Implement: Learning rate schedules from examples

#### Debug training issues
1. Reference: Debugging tips in [Advanced Topics](advanced-topics.md)
2. Use: NaN detection and shape printing examples

## Key Concepts Summary

### Lazy Evaluation
Operations in MLX don't compute immediately; they build a computation graph evaluated on-demand. This enables efficient gradient computation and memory usage.

### Function Transformations
Composable transformations for automatic differentiation (`grad`), automatic vectorization (`vmap`), and other operations. They can be nested: `grad(vmap(grad(...)))`.

### Unified Memory
Arrays live in shared memory between CPU and GPU, eliminating unnecessary data copies and enabling seamless multi-device operations.

### Module System
Neural network modules are containers for arrays and other modules, supporting parameter management, freezing, and serialization.

### Optimizers
MLX provides multiple optimization algorithms (SGD, Adam, etc.) with learning rate scheduling and gradient clipping for training.

## Installation Commands

```bash
# macOS (recommended)
pip install mlx

# Linux with CUDA
pip install mlx[cuda]

# Linux CPU-only
pip install mlx[cpu]

# From source (all platforms)
git clone git@github.com:ml-explore/mlx.git mlx && cd mlx
pip install -e ".[dev]"
```

## Official Resources

- **Official Documentation**: https://ml-explore.github.io/mlx/
- **GitHub Repository**: https://github.com/ml-explore/mlx
- **Examples Repository**: https://github.com/ml-explore/mlx-examples
- **License**: MIT

## Citation

If you use MLX in your research, please cite:

```
Awni Hannun, Jagrit Digani, Angelos Katharopoulos, Ronan Collobert (2023)
MLX: An array framework for machine learning on Apple silicon
```

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: Consult these guides
- **Examples**: Review practical examples in the examples repository

## Contributing

MLX is open source and welcomes contributions. See the GitHub repository for contribution guidelines.

---

**Last Updated:** 2024
**MLX Version:** Latest
**Documentation Scope:** Core framework, neural networks, optimization, and advanced topics
