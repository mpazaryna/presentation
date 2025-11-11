# MLX Getting Started Guide

## Quick Start

MLX is a machine learning framework with array operations and automatic differentiation. Here's what you need to know to get started.

### Basic Array Operations

Begin by importing the core module and creating arrays:

```python
import mlx.core as mx

a = mx.array([1, 2, 3, 4])
b = mx.array([1.0, 2.0, 3.0, 4.0])
```

Arrays support standard properties like `shape` and `dtype` for inspection:

```python
print(a.shape)  # (4,)
print(b.dtype)  # float32
```

## Lazy Evaluation Model

A key feature of MLX is that "operations in MLX are lazy. The outputs of MLX operations are not computed until they are needed." To force computation, use `mx.eval(c)`:

```python
c = a + b
mx.eval(c)  # Forces computation
```

Arrays automatically evaluate in specific scenarios:

- Extracting scalar values via `.item()`
- Printing arrays
- Converting to NumPy arrays
- Using `save()` functions

### Example: Creating and Evaluating Arrays

```python
import mlx.core as mx

# Create arrays
x = mx.array([1.0, 2.0, 3.0])
y = mx.array([4.0, 5.0, 6.0])

# Lazy operations
z = x + y
w = z * 2

# Explicit evaluation
result = mx.eval(w)
print(result)
```

## Function Transformations

MLX provides gradient computations and other transformations that compose freely:

### Basic Gradients

Compute derivatives of functions:

```python
# Simple derivative
grad_sin = mx.grad(mx.sin)
x = mx.array(0.0)
derivative = grad_sin(x)

# For loss functions
def loss_fn(w):
    return mx.sum(w ** 2)

grad_fn = mx.grad(loss_fn)
w = mx.array([1.0, 2.0, 3.0])
gradients = grad_fn(w)
```

### Chaining Transformations

Transformations compose freely, enabling sophisticated operations:

```python
# Higher-order derivatives
d2fdx2 = mx.grad(mx.grad(mx.sin))

# Complex compositions
complex_fn = mx.grad(mx.vmap(mx.grad(some_function)))
```

### Advanced Options

- **`vjp()`** - Vector-Jacobian products for efficient backpropagation
- **`jvp()`** - Jacobian-vector products for forward-mode differentiation
- **`value_and_grad()`** - Simultaneously compute function outputs and gradients

### Using value_and_grad()

For efficiency, use `value_and_grad()` to avoid redundant computation:

```python
def loss_fn(params):
    return mx.sum(params ** 2)

loss_and_grad_fn = mx.value_and_grad(loss_fn)
params = mx.array([1.0, 2.0, 3.0])

loss, grads = loss_and_grad_fn(params)
print(f"Loss: {loss.item()}")
print(f"Gradients: {grads}")
```

## Simple Training Loop Example

Here's a minimal example showing how to train a linear model:

```python
import mlx.core as mx

# Generate synthetic data
X = mx.random.normal((100, 10))
w_true = mx.random.normal((10,))
y = X @ w_true + 0.1 * mx.random.normal((100,))

# Define model and loss
def loss_fn(w):
    return mx.mean(mx.square(X @ w - y))

# Training loop
w = mx.zeros((10,))
learning_rate = 0.01

for epoch in range(100):
    loss, grads = mx.value_and_grad(loss_fn)(w)
    w = w - learning_rate * grads
    mx.eval(w)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
```

## Working with Devices

MLX supports operations on different devices:

```python
import mlx.core as mx

# Get available devices
cpu = mx.cpu
gpu = mx.gpu

# Create arrays on specific devices
a = mx.ones((100, 100), device=gpu)
b = mx.ones((100, 100), device=cpu)

# Operations work seamlessly across devices
c = a + b  # Automatic device handling
```

## Next Steps

- Learn about [lazy evaluation](lazy-evaluation.md) for performance optimization
- Explore [indexing arrays](array-indexing.md) for data manipulation
- Study [function transforms](function-transforms.md) for advanced gradient computations
- Review [neural networks](neural-networks.md) for building deep learning models
