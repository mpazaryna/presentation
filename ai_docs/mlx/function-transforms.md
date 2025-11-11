# MLX Function Transforms Documentation

## Overview

MLX implements **composable function transformations** for automatic differentiation, vectorization, and compute graph optimization. The key principle is that every transformation returns a function that can be further transformed.

## Automatic Differentiation

### Basic Gradient Computation

The `grad()` function computes derivatives of scalar-valued functions. By default, it differentiates with respect to the first argument:

```python
import mlx.core as mx

def loss_fn(w, x, y):
    return mx.mean(mx.square(w * x - y))

grad_fn = mx.grad(loss_fn)
dloss_dw = grad_fn(w, x, y)
```

### Higher-Order Derivatives

Composing `grad()` enables computing higher-order derivatives:

```python
import mlx.core as mx

# Second derivative
d2fdx2 = mx.grad(mx.grad(mx.sin))

# Third derivative
d3fdx3 = mx.grad(mx.grad(mx.grad(mx.sin)))

# Evaluate at a point
x = mx.array(0.0)
second_deriv = d2fdx2(x)
third_deriv = d3fdx3(x)
```

### Gradient with Respect to Multiple Arguments

Use the `argnums` parameter to target specific arguments:

```python
import mlx.core as mx

def loss_fn(w, x, y):
    return mx.mean(mx.square(w * x - y))

# Gradient w.r.t. second argument (x)
grad_fn = mx.grad(loss_fn, argnums=1)
dloss_dx = grad_fn(w, x, y)

# Gradient w.r.t. multiple arguments
grad_fn_multi = mx.grad(loss_fn, argnums=[0, 1])
dloss_dw, dloss_dx = grad_fn_multi(w, x, y)
```

### Efficient Loss and Gradient Calculation

`value_and_grad()` avoids redundant computation when you need both the loss and gradients:

```python
import mlx.core as mx

def loss_fn(w, x, y):
    return mx.mean(mx.square(w * x - y))

loss_and_grad_fn = mx.value_and_grad(loss_fn)
loss, gradient = loss_and_grad_fn(w, x, y)

print(f"Loss: {loss.item()}")
print(f"Gradient: {gradient}")
```

This is more efficient than calling `grad()` and the loss function separately.

### Complex Parameter Structures

Gradients preserve nested container structures (dictionaries, tuples, lists):

```python
import mlx.core as mx

def loss_fn(params, x, y):
    return mx.mean(mx.square(params["weight"] * x - y))

params = {"weight": mx.array(1.0), "bias": mx.array(0.0)}
grad_fn = mx.grad(loss_fn)
grads = grad_fn(params, x, y)

# Returns: {'weight': tensor(...), 'bias': None}
print(grads)
```

The gradient structure mirrors the input structure, with `None` for non-differentiated parameters.

### Stopping Gradient Flow

The `stop_gradient()` function prevents backpropagation through specific computation branches:

```python
import mlx.core as mx

def loss_fn(x, y):
    # y.grad will be None
    y_stopped = mx.stop_gradient(y)
    return mx.mean(mx.square(x - y_stopped))

grad_fn = mx.grad(loss_fn)
dx, dy = grad_fn(x, y)

print(f"dx: {dx}")  # Has gradients
print(f"dy: {dy}")  # None
```

**Use cases**:
- Fixing target values in RL
- Detaching encoder outputs in multi-task learning
- Implementing custom gradient schemes

## Automatic Vectorization

### Using vmap()

The `vmap()` function automatically vectorizes functions without manual loop optimization:

```python
import mlx.core as mx

# Simple function
def add(x, y):
    return x + y

# Vectorize over first dimension
vmap_add = mx.vmap(add, in_axes=(0, 1))

# Create batched inputs
x_batch = mx.random.normal((32, 10))
y_batch = mx.random.normal((10, 32))

# Apply vectorized function
result = vmap_add(x_batch, y_batch)
print(result.shape)  # (32, 10)
```

### Axis Specification

Fine-grained control over which axes to vectorize:

```python
import mlx.core as mx

def outer_product(x, y):
    return mx.outer(x, y)

# Vectorize over first axis of x, second axis of y
vmap_fn = mx.vmap(
    outer_product,
    in_axes=(0, 1),
    out_axes=0
)

x_batch = mx.random.normal((32, 10))
y_batch = mx.random.normal((20, 10))

result = vmap_fn(x_batch, y_batch)
print(result.shape)  # (32, 20, 10)
```

**Parameters**:

- **`in_axes`**: Specifies which dimensions to vectorize over in inputs (tuple of ints or None)
- **`out_axes`**: Specifies output axis placement for vectorized results

### Broadcasting with vmap()

When not all arguments need vectorization:

```python
import mlx.core as mx

def scale_and_add(x, scale, bias):
    return scale * x + bias

# Only vectorize x (in_axes[0]), broadcast scale and bias
vmap_fn = mx.vmap(
    scale_and_add,
    in_axes=(0, None, None)
)

x_batch = mx.random.normal((32, 10))
scale = mx.array(2.0)
bias = mx.array(1.0)

result = vmap_fn(x_batch, scale, bias)
```

### Performance Benefits

On Apple Silicon, vectorized operations demonstrate significant speedups. The documentation reports >200x improvement over naive loops for certain operations.

### Limitations

Some operations lack `vmap()` support. Unsupported functions generate:

```
ValueError: Primitive's vmap not implemented.
```

If you encounter this, you can:

1. File an issue requesting vmap support
2. Use an alternative approach
3. Manually implement the vectorized version

## Advanced Transformations

### Jacobian-Vector Products (jvp)

Forward-mode differentiation for efficient computation of Jacobian-vector products:

```python
import mlx.core as mx

def f(x):
    return mx.array([mx.sin(x[0]), mx.cos(x[1])])

x = mx.array([1.0, 2.0])
v = mx.array([1.0, 0.0])

# Compute Jacobian-vector product
result = mx.jvp(f, (x,), (v,))
```

**Use cases**:
- Forward-mode AD for functions with many inputs, few outputs
- Computing directional derivatives
- Efficient perturbation analysis

### Vector-Jacobian Products (vjp)

Reverse-mode differentiation (backpropagation) for the Jacobian:

```python
import mlx.core as mx

def f(x):
    return mx.array([mx.sum(x ** 2), mx.sum(mx.sin(x))])

x = mx.array([1.0, 2.0, 3.0])
v = mx.array([1.0, 0.0])

# Compute vector-Jacobian product
primals, vjp_fn = mx.vjp(f, x)
result = vjp_fn(v)
```

**Use cases**:
- Custom gradient implementations
- Multiple output functions
- Efficient backpropagation for complex architectures

## Composing Transformations

A powerful feature is composing multiple transformations:

```python
import mlx.core as mx

def batch_loss(model_params, x_batch, y_batch):
    # Loss is averaged over batch
    logits = model(model_params, x_batch)
    return mx.mean(mx.square(logits - y_batch))

# Compute batched gradients (grad over each sample)
grad_per_sample = mx.vmap(
    mx.grad(batch_loss),
    in_axes=(None, 0, 0)
)

# Compute Hessian-vector product
hessian_vp = mx.vmap(
    mx.jvp(mx.grad(loss_fn), (params,)),
    in_axes=(None, 0)
)
```

## Practical Example: Neural Network Training

Combining transformations for efficient neural network training:

```python
import mlx.core as mx
import mlx.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def __call__(self, x):
        x = mx.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet()

def loss_fn(model, x, y):
    pred = model(x)
    return mx.mean(mx.square(pred - y))

# Create a function that computes loss and gradients
loss_and_grad = mx.value_and_grad(model, loss_fn)

# Training loop
for epoch in range(100):
    for x_batch, y_batch in data_loader:
        loss, grads = loss_and_grad(model, x_batch, y_batch)

        # Update parameters
        for param in model.parameters():
            param -= 0.01 * grads[param]

        mx.eval(model.parameters())
```

## Summary

MLX function transformations provide:

1. **Automatic differentiation** - Compute gradients efficiently
2. **Automatic vectorization** - Scale to batched inputs
3. **Composability** - Combine transformations for complex operations
4. **Efficiency** - Avoid redundant computations
5. **Flexibility** - Target specific arguments and axes

These features make MLX powerful for machine learning workflows.
