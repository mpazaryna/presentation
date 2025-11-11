# MLX Optimizers Documentation

## Overview

MLX provides a comprehensive optimizer module for training neural networks. These optimizers work seamlessly with both `mlx.nn` and pure `mlx.core` functions.

## Core Workflow

The typical optimization pattern involves three key steps:

### 1. Compute Gradients

Use `nn.value_and_grad()` to create a function that returns both loss and gradients:

```python
import mlx.core as mx
import mlx.nn as nn

def loss_fn(model, x, y):
    return mx.mean(mx.square(model(x) - y))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
```

### 2. Update Parameters

Call `Optimizer.update()` with the model and gradients:

```python
optimizer = nn.optimizers.Adam(learning_rate=0.001)

for x_batch, y_batch in data_loader:
    loss, grads = loss_and_grad_fn(model, x_batch, y_batch)
    optimizer.update(model, grads)
```

### 3. Evaluate State

Execute `mx.eval()` to compute both model parameters and optimizer state:

```python
mx.eval(model.parameters(), optimizer.state)
```

## Available Optimizers

### First-Order Methods

#### SGD (Stochastic Gradient Descent)

```python
import mlx.nn as nn

optimizer = nn.optimizers.SGD(learning_rate=0.01)

# With momentum
optimizer = nn.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# With dampening and nesterov
optimizer = nn.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,
    dampening=0.0,
    nesterov=True
)
```

#### RMSprop

```python
import mlx.nn as nn

optimizer = nn.optimizers.RMSprop(
    learning_rate=0.001,
    alpha=0.99,
    eps=1e-8
)
```

#### AdaGrad

```python
import mlx.nn as nn

optimizer = nn.optimizers.Adagrad(
    learning_rate=0.01,
    eps=1e-8
)
```

#### AdaDelta

```python
import mlx.nn as nn

optimizer = nn.optimizers.Adadelta(
    learning_rate=1.0,
    rho=0.9,
    eps=1e-8
)
```

### Adaptive Methods

#### Adam

```python
import mlx.nn as nn

optimizer = nn.optimizers.Adam(
    learning_rate=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

#### AdamW

AdamW is Adam with weight decay decoupling, often preferred for deep learning:

```python
import mlx.nn as nn

optimizer = nn.optimizers.AdamW(
    learning_rate=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

#### Adamax

A variant of Adam using L-infinity norm:

```python
import mlx.nn as nn

optimizer = nn.optimizers.Adamax(
    learning_rate=0.002,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

#### Adafactor

Memory-efficient optimizer useful for large models:

```python
import mlx.nn as nn

optimizer = nn.optimizers.Adafactor(
    learning_rate=None,  # Uses default schedule
    clipping_threshold=1.0,
    decay_rate=-0.8,
    step_size=1.0,
    min_step_size=1e-30
)
```

### Modern Optimizers

#### Lion

A newer optimizer combining momentum with sign updates:

```python
import mlx.nn as nn

optimizer = nn.optimizers.Lion(
    learning_rate=0.001,
    betas=(0.9, 0.99),
    weight_decay=0.0
)
```

#### Muon

Another modern optimizer:

```python
import mlx.nn as nn

optimizer = nn.optimizers.Muon(
    learning_rate=0.002,
    momentum=0.95,
    nesterov=False,
    weight_decay=0.0
)
```

### MultiOptimizer

For different parameter groups with different optimizers:

```python
import mlx.nn as nn

# Separate optimizers for different parameter groups
backbone_optimizer = nn.optimizers.SGD(learning_rate=0.01, momentum=0.9)
head_optimizer = nn.optimizers.Adam(learning_rate=0.001)

multi_optimizer = nn.optimizers.MultiOptimizer(
    [
        (["backbone"], backbone_optimizer),
        (["head"], head_optimizer)
    ]
)
```

## Learning Rate Scheduling

MLX includes several scheduler functions for adaptive learning rates:

### Cosine Decay

Cosine annealing schedule:

```python
import mlx.nn as nn

def learning_rate_schedule(step, lr_init=0.1, steps=1000):
    return nn.optimizers.cosine_decay(step, steps, lr_init, 0.0)

# Update learning rate each step
for step in range(total_steps):
    optimizer.learning_rate = learning_rate_schedule(step)
```

### Exponential Decay

Exponential learning rate decay:

```python
import mlx.nn as nn

def learning_rate_schedule(step, lr_init=0.1, decay_rate=0.1, decay_steps=1000):
    return nn.optimizers.exponential_decay(step, lr_init, decay_rate, decay_steps)
```

### Linear Schedule

Linear warmup followed by decay:

```python
import mlx.nn as nn

def learning_rate_schedule(step, lr_init=0.1, warmup_steps=1000, total_steps=10000):
    return nn.optimizers.linear_schedule(step, lr_init, warmup_steps, total_steps)
```

### Step Decay

Learning rate decay at fixed step intervals:

```python
import mlx.nn as nn

def learning_rate_schedule(step, lr_init=0.1, steps=[1000, 5000, 9000]):
    return nn.optimizers.step_decay(step, lr_init, steps, 0.1)
```

### Joining Schedules

Combine multiple schedules:

```python
import mlx.nn as nn

def combined_schedule(step):
    # Warmup for first 1000 steps
    warmup = nn.optimizers.linear_schedule(step, 0.0, 0.1, 1000)

    # Cosine decay after warmup
    if step < 1000:
        return warmup
    else:
        return nn.optimizers.cosine_decay(step - 1000, 9000, 0.1, 0.0)

# Alternative: use join_schedules
schedule = nn.optimizers.join_schedules([
    nn.optimizers.linear_schedule(0, 0.0, 0.1, 1000),
    nn.optimizers.cosine_decay(0, 9000, 0.1, 0.0)
], [1000])
```

## Training Loop with Learning Rate Schedule

```python
import mlx.core as mx
import mlx.nn as nn

model = YourModel()
optimizer = nn.optimizers.AdamW(learning_rate=0.001)

def loss_fn(model, x, y):
    return mx.mean(mx.square(model(x) - y))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

total_steps = len(data_loader) * num_epochs

for epoch in range(num_epochs):
    for step, (x_batch, y_batch) in enumerate(data_loader):
        global_step = epoch * len(data_loader) + step

        # Update learning rate
        lr = nn.optimizers.cosine_decay(
            global_step,
            total_steps,
            0.001,
            0.0
        )
        optimizer.learning_rate = lr

        # Training step
        loss, grads = loss_and_grad_fn(model, x_batch, y_batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
```

## Gradient Clipping

Control exploding gradients:

```python
import mlx.core as mx
import mlx.nn as nn

def clip_gradients(grads, max_norm=1.0):
    # Flatten all gradients
    flat_grads = mx.tree_flatten(grads)[0]

    # Compute total norm
    grad_norms = [mx.sum(g ** 2) for g in flat_grads if g is not None]
    total_norm = mx.sqrt(sum(grad_norms))

    # Clip if needed
    clip_coef = min(1.0, max_norm / (total_norm + 1e-6))

    def clip_grad(g):
        return g * clip_coef if g is not None else None

    return mx.tree_map(clip_grad, grads)

# In training loop
loss, grads = loss_and_grad_fn(model, x_batch, y_batch)
grads = clip_gradients(grads, max_norm=1.0)
optimizer.update(model, grads)
```

Alternatively, use the built-in utility:

```python
import mlx.nn as nn

# Clip gradients by norm
clipped_grads = nn.clip_grad_norm(grads, max_norm=1.0)
optimizer.update(model, clipped_grads)
```

## Persistence

### Saving Optimizer State

Extract optimizer state using `tree_flatten()` and save with `mx.save_safetensors()`:

```python
import mlx.core as mx
import mlx.nn as nn

# Save
state_dict = dict(model.parameters())
state_dict["optimizer"] = dict(optimizer.state)
mx.save_safetensors("checkpoint.safetensors", state_dict)
```

### Loading Optimizer State

Use `tree_unflatten()` to restore state to a new optimizer instance:

```python
import mlx.core as mx
import mlx.nn as nn

# Load
weights = mx.load("checkpoint.safetensors")
model.update(mx.tree_unflatten(weights["model"]))

# Restore optimizer state
optimizer = nn.optimizers.Adam(learning_rate=0.001)
optimizer.state = mx.tree_unflatten(weights["optimizer"])
```

**Important**: Only parameters that can be scheduled (like learning rate) are included in saved state.

## Complete Training Example

```python
import mlx.core as mx
import mlx.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def __call__(self, x):
        x = mx.relu(self.fc1(x))
        x = mx.relu(self.fc2(x))
        return self.fc3(x)

# Model and optimizer setup
model = SimpleNet()
optimizer = nn.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)

def loss_fn(model, x, y):
    pred = model(x)
    return mx.mean((pred - y) ** 2)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Training
num_epochs = 100
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0

    for x_batch, y_batch in data_loader:
        # Forward and backward
        loss, grads = loss_and_grad_fn(model, x_batch, y_batch)

        # Update parameters
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save model
mx.save_safetensors("model.safetensors", dict(model.parameters()))
```

## Summary

MLX optimizers provide:

1. **Multiple optimization algorithms** from classic to modern methods
2. **Learning rate scheduling** for adaptive learning rates
3. **Gradient clipping** to prevent explosions
4. **State persistence** for checkpoint/resume workflows
5. **Multi-optimizer support** for complex training scenarios

Choose the optimizer based on your specific task and model architecture.
