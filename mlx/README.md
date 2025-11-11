# MLX Framework Documentation

Complete reference documentation for **MLX** - Apple's machine learning framework optimized for Apple Silicon.

## About This Documentation

This is a comprehensive offline documentation set for the MLX framework, scraped from the official MLX documentation. It covers:

- Framework overview and architecture
- Getting started guide
- Core concepts (lazy evaluation, function transforms, indexing)
- Complete API reference for neural networks and optimizers
- Random number generation
- Practical examples and tutorials
- Advanced topics (distributed training, FFT, optimization)

## Quick Start

### Installation

```bash
# For macOS (M1/M2/M3/M4 Apple Silicon)
pip install mlx

# For Linux with CUDA
pip install mlx[cuda]

# For Linux CPU-only
pip install mlx[cpu]
```

### Your First MLX Program

```python
import mlx.core as mx

# Create arrays
a = mx.array([1, 2, 3, 4])
b = mx.array([5, 6, 7, 8])

# Perform operations (lazy evaluated)
c = a + b

# Force evaluation
result = mx.eval(c)
print(result)
```

### A Simple Training Loop

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Loss and optimizer
optimizer = optim.Adam(learning_rate=0.001)

def loss_fn(model, x, y):
    return mx.mean((model(x) - y) ** 2)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Training
for epoch in range(10):
    for x_batch, y_batch in data_loader:
        loss, grads = loss_and_grad_fn(model, x_batch, y_batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
```

## Documentation Index

### Start Here

1. **[MLX Framework Overview](mlx-framework-overview.md)** - What is MLX and why use it
2. **[Getting Started](getting-started.md)** - Your first MLX programs

### Core Concepts

3. **[Lazy Evaluation](lazy-evaluation.md)** - Understanding MLX's computation model
4. **[Array Indexing](array-indexing.md)** - Working with arrays
5. **[Function Transforms](function-transforms.md)** - Automatic differentiation and vectorization

### API Reference

6. **[Neural Networks](neural-networks.md)** - Building deep learning models
7. **[Optimizers and Learning Rates](optimizers-and-learning-rates.md)** - Training your models
8. **[Random Number Generation](random-numbers.md)** - Generating random data

### Learning by Example

9. **[Examples and Tutorials](examples-tutorial.md)** - 10 practical code examples
10. **[Advanced Topics](advanced-topics.md)** - Distributed training, FFT, performance tuning

### Navigation

11. **[Complete Index](INDEX.md)** - Detailed navigation and task-based guidance

## Key Features of MLX

### 1. NumPy-like Array Framework

Familiar API that closely follows NumPy conventions with full support for:

```python
import mlx.core as mx

x = mx.random.normal((100, 50))
y = mx.sin(x) * mx.exp(-x**2)
z = mx.sum(y, axis=0)
```

### 2. Automatic Differentiation

Compute gradients of any function:

```python
import mlx.core as mx

def f(x):
    return mx.sum(x ** 2)

grad_f = mx.grad(f)
x = mx.array([1.0, 2.0, 3.0])
gradients = grad_f(x)  # [2.0, 4.0, 6.0]
```

### 3. Automatic Vectorization

Automatically vectorize functions without explicit loops:

```python
import mlx.core as mx

def apply_fn(x):
    return mx.sin(x) + mx.cos(x)

# Vectorize over batch dimension
batch_fn = mx.vmap(apply_fn, in_axes=0)

x_batch = mx.random.normal((32, 10))
result = batch_fn(x_batch)  # Efficiently processes all 32 examples
```

### 4. Lazy Evaluation

Computations defer until needed, enabling efficient memory usage and graph optimization:

```python
import mlx.core as mx

# Operations are recorded but not computed
x = mx.random.normal((1000, 1000))
y = x @ x
z = mx.sin(y)

# Only evaluated when explicitly requested
result = mx.eval(z)
```

### 5. Unified Memory

Arrays live in shared memory - seamless CPU/GPU operations without data copies:

```python
import mlx.core as mx

# Operations automatically use available hardware efficiently
a = mx.random.normal((1000, 1000), device=mx.gpu)
b = mx.random.normal((1000, 1000), device=mx.cpu)
c = a + b  # No explicit data transfer needed
```

### 6. Composable Function Transformations

Combine transformations for powerful operations:

```python
import mlx.core as mx

# Compute Hessian (gradient of gradient)
f = lambda x: mx.sum(mx.sin(x))
hessian_fn = mx.grad(mx.grad(f))

# Vectorized gradient computation
batch_grad = mx.vmap(mx.grad(f), in_axes=0)
```

### 7. Comprehensive Neural Network Module

High-level abstractions for building deep learning models:

```python
import mlx.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

    def __call__(self, x):
        x = mx.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```

### 8. Multiple Optimization Algorithms

Choose from classic and modern optimizers:

```python
import mlx.optimizers as optim

# Classic optimizers
sgd = optim.SGD(learning_rate=0.01, momentum=0.9)
adam = optim.Adam(learning_rate=0.001)

# Modern optimizers
adamw = optim.AdamW(learning_rate=0.001, weight_decay=0.01)
lion = optim.Lion(learning_rate=0.001)
```

### 9. Learning Rate Scheduling

Adaptive learning rates during training:

```python
import mlx.optimizers as optim

def learning_rate_schedule(step, total_steps):
    return optim.cosine_decay(step, total_steps, 0.1, 0.0)

# Use in training loop
for step, (x, y) in enumerate(data_loader):
    optimizer.learning_rate = learning_rate_schedule(step, total_steps)
```

### 10. Distributed Training

Scale across multiple devices and machines:

```python
import mlx.core as mx
import mlx.nn as nn

# Initialize distributed training
world = mx.distributed.init()

# Training with gradient averaging
grads = mx.nn.average_gradients(grads)
optimizer.update(model, grads)
```

## Documentation File Structure

```
mlx/
├── README.md                             # This file
├── INDEX.md                              # Complete navigation index
├── mlx-framework-overview.md             # Framework introduction and basics
├── getting-started.md                    # Quick start guide
├── lazy-evaluation.md                    # Lazy evaluation concepts
├── array-indexing.md                     # Array indexing guide
├── function-transforms.md                # Autodiff and vectorization
├── neural-networks.md                    # Neural network modules
├── optimizers-and-learning-rates.md      # Optimization algorithms
├── random-numbers.md                     # Random number generation
├── examples-tutorial.md                  # Practical code examples
└── advanced-topics.md                    # Advanced features
```

## Use Cases

MLX is ideal for:

- **Research**: Test new algorithms on Apple Silicon
- **Production**: Deploy ML models efficiently on Apple devices
- **Education**: Learn machine learning with a clean, Pythonic API
- **Enterprise**: On-device inference for privacy-critical applications
- **Prototyping**: Fast iteration with integrated GPU acceleration

## Platform Support

### Primary Target
- **macOS** with Apple Silicon (M1/M2/M3/M4 and variants)
- Python 3.10+
- macOS 14.0+

### Linux Support
- **CUDA support** for NVIDIA GPUs
- **CPU-only** builds available

### Performance

MLX demonstrates exceptional performance on Apple Silicon:

- **Model loading**: ~34ms for multi-billion parameter models
- **Inference**: 2-14ms for typical forward passes
- **Training**: Efficient multi-device gradient computation
- **Memory**: Unified memory model reduces overhead

## Key Advantages Over Alternatives

1. **Native Apple Silicon Support** - Optimized for M1/M2/M3/M4 chips
2. **Unified Memory** - No data transfer overhead between CPU/GPU
3. **Lazy Evaluation** - Efficient graph optimization
4. **NumPy Compatibility** - Familiar API for easy adoption
5. **Composable Transforms** - Powerful function composition
6. **Production Ready** - Used in real applications

## Common Workflows

### Training a Classifier

See [Examples and Tutorials](examples-tutorial.md) for complete multi-layer perceptron example.

### Fine-tuning a Model

```python
import mlx.nn as nn

# Load pre-trained model
model = load_pretrained_model()

# Freeze encoder
model.encoder.freeze()

# Train only head
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
```

### Custom Training Loop

```python
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        loss, grads = loss_and_grad_fn(model, x_batch, y_batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
```

### Model Checkpointing

```python
import mlx.core as mx

# Save
checkpoint = {
    'model': dict(model.parameters()),
    'optimizer': dict(optimizer.state),
}
mx.save_safetensors('checkpoint.safetensors', checkpoint)

# Load
weights = mx.load('checkpoint.safetensors')
model.update(mx.tree_unflatten(weights['model']))
```

## Official Resources

- **Official Docs**: https://ml-explore.github.io/mlx/
- **GitHub Repo**: https://github.com/ml-explore/mlx
- **Examples**: https://github.com/ml-explore/mlx-examples
- **License**: MIT

## Citation

If you use MLX in your research:

```bibtex
@article{hannun2023mlx,
  title={MLX: An array framework for machine learning on Apple silicon},
  author={Hannun, Awni and Digani, Jagrit and Katharopoulos, Angelos and Collobert, Ronan},
  year={2023}
}
```

## Troubleshooting

### Installation Issues

If `pip install mlx` fails:

1. Verify you're using **native Python**: `python -c "import platform; print(platform.processor())"`
   - Should output "arm", not "i386"

2. Verify shell is native: `uname -p` should show "arm"

3. Ensure Python 3.10+: `python --version`

4. Update pip: `pip install --upgrade pip`

### Performance Issues

1. **Use lazy evaluation properly** - Call `mx.eval()` at batch level, not per-operation
2. **Monitor graph size** - Large graphs can slow training
3. **Profile your code** - Use timing utilities to identify bottlenecks
4. **Check device usage** - Verify operations run on intended device

### Numerical Issues

1. **NaN in loss** - Check gradient magnitudes and learning rate
2. **Gradient clipping** - Use `nn.clip_grad_norm()` if gradients explode
3. **Float precision** - Consider mixed-precision training

## Next Steps

1. **Install MLX** following the quick start above
2. **Run examples** from [Examples and Tutorials](examples-tutorial.md)
3. **Read core concepts** starting with [Lazy Evaluation](lazy-evaluation.md)
4. **Build your model** using [Neural Networks](neural-networks.md)
5. **Optimize training** with [Optimizers and Learning Rates](optimizers-and-learning-rates.md)

## Contributing

MLX is open source. Contributions are welcome! See the GitHub repository for:

- Bug reports
- Feature requests
- Pull requests
- Development guidelines

## Support

For questions and discussions:

- Check the [INDEX](INDEX.md) for topic-specific guidance
- Review examples in [Examples and Tutorials](examples-tutorial.md)
- Search GitHub Issues for similar problems
- File an issue with reproducible examples

---

**Documentation Version:** 2024
**MLX Framework:** Latest version
**Scope:** Core framework, neural networks, optimization, advanced features

Start with the [Getting Started](getting-started.md) guide or jump to the [INDEX](INDEX.md) for navigation by topic.
