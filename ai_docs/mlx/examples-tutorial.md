# MLX Examples and Tutorials

This document contains practical examples demonstrating core MLX functionality.

## Example 1: Linear Regression

### Complete Linear Regression Implementation

A classic machine learning task implemented with MLX:

```python
import mlx.core as mx

# Configuration
num_features = 100
num_examples = 1_000
num_iters = 10_000  # iterations of SGD
lr = 0.01  # learning rate for SGD

# Generate synthetic data
# True parameters
w_star = mx.random.normal((num_features,))

# Input examples (design matrix)
X = mx.random.normal((num_examples, num_features))

# Noisy labels
eps = 1e-2 * mx.random.normal((num_examples,))
y = X @ w_star + eps

# Define loss function
def loss_fn(w):
    return 0.5 * mx.mean(mx.square(X @ w - y))

# Create gradient function
grad_fn = mx.grad(loss_fn)

# Stochastic gradient descent optimization loop
w = 1e-2 * mx.random.normal((num_features,))

for iteration in range(num_iters):
    grad = grad_fn(w)
    w = w - lr * grad
    mx.eval(w)

# Evaluate final loss and error
loss = loss_fn(w)
error_norm = mx.sum(mx.square(w - w_star)).item() ** 0.5

print(f"Loss {loss.item():.5f}, |w-w*| = {error_norm:.5f}")
# Expected output: Loss 0.00005, |w-w*| = 0.00364
```

**Key concepts**:
- Using `grad()` to compute derivatives
- Lazy evaluation with explicit `eval()` in the training loop
- Evaluating convergence by comparing to ground truth

## Example 2: Multi-Layer Perceptron for MNIST Classification

### Complete MLP for Image Classification

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Define the model
class MLP(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = mx.maximum(l(x), 0.0)  # ReLU activation
        return self.layers[-1](x)

# Loss function
def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))

# Evaluation function
def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)

# Create model and optimizer
model = MLP(num_layers=2, input_dim=784, hidden_dim=32, output_dim=10)
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.SGD(learning_rate=0.1)

# Training loop
num_epochs = 10
batch_size = 256

for epoch in range(num_epochs):
    # Training phase
    total_loss = 0.0
    num_batches = 0

    for X, y in batch_iterate(batch_size, train_images, train_labels):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()
        num_batches += 1

    # Evaluation phase
    accuracy = eval_fn(model, test_images, test_labels)
    avg_loss = total_loss / num_batches

    print(f"Epoch {epoch}: Test accuracy {accuracy.item():.3f}, Loss {avg_loss:.4f}")

# Expected: Test accuracy ~95% within several epochs
```

**Key concepts**:
- Custom module definition with `nn.Module`
- Combining automatic differentiation with neural network modules
- Using `value_and_grad()` for efficient loss and gradient computation
- Training and evaluation loops

## Example 3: Simple Linear Regression Training Loop

### Minimal Hands-On Example

```python
import mlx.core as mx
import mlx.nn as nn

# Generate synthetic data
X = mx.random.normal((100, 10))
w_true = mx.random.normal((10,))
y = X @ w_true + 0.1 * mx.random.normal((100,))

# Define a simple model
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = mx.random.normal((10,))

    def __call__(self, x):
        return x @ self.w

model = LinearModel()

# Loss function
def loss_fn(model, x, y):
    pred = model(x)
    return mx.mean(mx.square(pred - y))

# Training
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
learning_rate = 0.01

for epoch in range(100):
    loss, grads = loss_and_grad_fn(model, X, y)

    # Manual parameter update
    model.w = model.w - learning_rate * grads['w']
    mx.eval(model.w)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
```

## Example 4: Function Transformations - Automatic Differentiation

### Computing Derivatives and Higher-Order Derivatives

```python
import mlx.core as mx

# Simple function
def f(x):
    return mx.sum(x ** 2)

# First derivative
df = mx.grad(f)
x = mx.array([1.0, 2.0, 3.0])
grad_f = df(x)
print(f"Gradient: {grad_f}")  # [2, 4, 6]

# Second derivative (Hessian)
d2f = mx.grad(mx.grad(f))
x_scalar = mx.array(2.0)
hessian = d2f(x_scalar)
print(f"Hessian at 2.0: {hessian}")  # 2.0

# Trigonometric function derivatives
sin_fn = mx.sin
cos_fn = mx.grad(sin_fn)  # cos is derivative of sin
tan_fn = mx.grad(cos_fn)  # -sin is derivative of cos

x = mx.array(0.0)
print(f"sin(0): {sin_fn(x)}")
print(f"cos(0): {cos_fn(x)}")
print(f"-sin(0): {tan_fn(x)}")
```

## Example 5: Automatic Vectorization (vmap)

### Efficiently Processing Batches

```python
import mlx.core as mx

# Simple function that operates on a single example
def predict(params, x):
    return params['w'] @ x + params['b']

# Vectorize over batch dimension
params = {
    'w': mx.random.normal((10,)),
    'b': mx.array(0.0)
}

# Create vectorized version
batch_predict = mx.vmap(
    lambda params, x: predict(params, x),
    in_axes=(None, 0)  # vectorize over second argument
)

# Single example
x_single = mx.random.normal((10,))
y_single = predict(params, x_single)

# Batch of examples
x_batch = mx.random.normal((32, 10))  # 32 examples
y_batch = batch_predict(params, x_batch)

print(f"Single prediction shape: {y_single.shape}")  # ()
print(f"Batch prediction shape: {y_batch.shape}")     # (32,)
```

## Example 6: Random Number Generation

### Using MLX's Random Functions

```python
import mlx.core as mx

# Basic random distributions
x1 = mx.random.normal((5, 5))           # Normal distribution
x2 = mx.random.uniform(0, 1, (5, 5))    # Uniform [0, 1)
x3 = mx.random.randint(0, 10, (5, 5))   # Integers [0, 10)

# Specific distributions
x4 = mx.random.bernoulli(p=0.5, shape=(100,))     # Binary
x5 = mx.random.laplace(scale=1.0, shape=(100,))   # Laplace
x6 = mx.random.gumbel(shape=(100,))               # Gumbel

# With explicit seed for reproducibility
key = mx.random.key(42)
x7 = mx.random.normal(shape=(5, 5), key=key)

# Permutations and sampling
x8 = mx.random.permutation(10)           # Random permutation of 0-9
x9 = mx.random.categorical(logits=mx.array([1, 2, 3, 4]))  # Categorical sampling
```

## Example 7: Neural Network with Data Augmentation

### Simple CNN for Image Classification

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def __call__(self, x):
        x = mx.relu(self.conv1(x))
        x = self.pool(x)
        x = mx.relu(self.conv2(x))
        x = self.pool(x)
        x = mx.flatten(x, 1)
        x = mx.relu(self.fc1(x))
        return self.fc2(x)

# Model setup
model = SimpleCNN()
optimizer = optim.Adam(learning_rate=0.001)

def loss_fn(model, x, y):
    return mx.mean(nn.losses.cross_entropy(model(x), y))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Training
for epoch in range(10):
    for x_batch, y_batch in train_loader:
        loss, grads = loss_and_grad_fn(model, x_batch, y_batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
```

## Example 8: Custom Training with Gradient Clipping

### Preventing Gradient Explosion

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

model = YourModel()
optimizer = optim.Adam(learning_rate=0.001)

def loss_fn(model, x, y):
    return mx.mean(mx.square(model(x) - y))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Training with gradient clipping
max_grad_norm = 1.0

for epoch in range(num_epochs):
    for x_batch, y_batch in data_loader:
        loss, grads = loss_and_grad_fn(model, x_batch, y_batch)

        # Clip gradients by norm
        clipped_grads = nn.clip_grad_norm(grads, max_norm=max_grad_norm)

        optimizer.update(model, clipped_grads)
        mx.eval(model.parameters(), optimizer.state)
```

## Example 9: Saving and Loading Models

### Checkpoint Management

```python
import mlx.core as mx
import mlx.nn as nn

model = YourModel()
optimizer = nn.optimizers.Adam(learning_rate=0.001)

# Save checkpoint
checkpoint = {
    'model': dict(model.parameters()),
    'optimizer_state': dict(optimizer.state),
    'epoch': epoch,
}
mx.save_safetensors('checkpoint.safetensors', checkpoint)

# Load checkpoint
weights = mx.load('checkpoint.safetensors')
model.update(mx.tree_unflatten(weights['model']))
optimizer.state = mx.tree_unflatten(weights['optimizer_state'])
epoch = weights['epoch']
```

## Example 10: Distributed Training

### Multi-GPU/Multi-Machine Training

```python
import mlx.core as mx
import mlx.nn as nn

# Initialize distributed group
world = mx.distributed.init()

# Model and optimizer
model = YourModel()
optimizer = nn.optimizers.SGD(learning_rate=0.01)

def loss_fn(model, x, y):
    return mx.mean(mx.square(model(x) - y))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Training with gradient averaging
for x_batch, y_batch in data_loader:
    loss, grads = loss_and_grad_fn(model, x_batch, y_batch)

    # Average gradients across all processes
    grads = mx.nn.average_gradients(grads)

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
```

## Summary

These examples demonstrate:

1. **Basic training loops** with manual gradient descent
2. **Neural network modules** for complex architectures
3. **Automatic differentiation** for computing derivatives
4. **Automatic vectorization** for efficient batch processing
5. **Distributed training** for large-scale applications
6. **Model checkpointing** for experiment management

Start with simpler examples and gradually explore more advanced features as you become comfortable with MLX.
