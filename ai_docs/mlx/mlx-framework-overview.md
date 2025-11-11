# MLX Framework Overview

MLX is a NumPy-like array framework designed for efficient and flexible machine learning on Apple silicon, brought to you by Apple machine learning research.

## What is MLX?

MLX is an array framework for machine learning optimized for Apple silicon, created by Apple's machine learning research team. The framework emphasizes user-friendly design while maintaining efficiency for training and deploying models.

## Core Design Philosophy

MLX distinguishes itself through several key design decisions:

1. **Function Transformations** - Supports automatic differentiation, vectorization, and graph optimization
2. **Lazy Evaluation** - Computations defer until arrays are actually needed
3. **Multi-Device Support** - Operations run on CPU, GPU, or other available devices
4. **Unified Memory Model** - "Arrays in MLX live in shared memory. Operations on MLX arrays can be performed on any of the supported device type without performing data copies."

## Key Features

The framework offers several distinguishing capabilities:

- **Familiar interfaces**: Python API mirroring NumPy, plus C++, C, and Swift implementations. Higher-level modules like `mlx.nn` and `mlx.optimizers` follow PyTorch conventions.

- **Function transformations**: Supports automatic differentiation, vectorization, and graph optimization through composable transformations.

- **Lazy evaluation**: "Arrays are only materialized when needed," enabling efficiency gains.

- **Dynamic graphs**: No compilation delays when argument shapes change; straightforward debugging.

- **Multi-device support**: Operations run on CPU or GPU seamlessly.

- **Unified memory**: A key distinction—arrays reside in shared memory, eliminating unnecessary data transfers between devices.

## Installation

Installation via pip varies by platform:

- **macOS**: `pip install mlx`
- **Linux with CUDA**: `pip install mlx[cuda]`
- **Linux CPU-only**: `pip install mlx[cpu]`

### Building from Source

**Prerequisites:**
- C++17 compatible compiler (Clang 5.0+)
- CMake 3.25+
- Xcode 15.0+ with macOS SDK 14.0+

**For Python development:**
```bash
git clone git@github.com:ml-explore/mlx.git mlx && cd mlx
pip install -e ".[dev]"
```

**For C++ library:**
```bash
mkdir -p build && cd build
cmake .. && make -j
make install
```

## Platform Requirements

### macOS (Apple Silicon)
- M series Apple silicon chip
- Native Python 3.10+
- macOS 14.0 or newer

### Linux CUDA Support
- Nvidia architecture SM 7.0+
- Driver version 550.54.14+
- CUDA 12.0+

### Linux CPU-Only
- glibc 2.35+

## Troubleshooting Installation

If pip can't find distributions despite meeting version requirements, verify you're running native Python:
```python
python -c "import platform; print(platform.processor())"
```

This should output "arm", not "i386".

For build failures on macOS, ensure your shell runs natively (test with `uname -p`—should show "arm").

## Available Examples

The MLX examples repository includes implementations for:

- Transformer language models
- LLaMA with LoRA fine-tuning
- Stable Diffusion
- OpenAI Whisper integration
- Linear regression
- Multi-layer perceptrons (MLPs)
- LLM inference

## Project Information

- **Repository**: https://github.com/ml-explore/mlx
- **Documentation**: https://ml-explore.github.io/mlx/
- **License**: MIT
- **Stars**: 22.8k+
- **Contributors**: 181+
- **Language composition**: C++ (66%), Python (22.3%), CUDA (6.6%), Metal (3.7%)

## Citation

The original developers were Awni Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert (2023).
