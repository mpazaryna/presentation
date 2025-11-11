# MLX Advanced Topics

This document covers advanced features and optimization techniques in MLX.

## Distributed Training

### Core Capabilities

MLX provides distributed computing support through two communication backends:

1. **MPI (Message Passing Interface)** - A mature, full-featured distributed communications library
2. **Ring Backend** - MLX's custom TCP socket implementation, optimized for Thunderbolt connections

### Basic Usage

The simplest distributed program initializes a group and performs collective operations:

```python
import mlx.core as mx

# Initialize distributed training
world = mx.distributed.init()

# Perform collective operations
x = mx.distributed.all_sum(mx.ones(10))
print(f"Rank {world.rank()}: {x}")

# All operations in mx.distributed are noops when the distributed group
# has a size of one, eliminating the need for conditional checks
```

**Key properties of the distributed group**:

```python
import mlx.core as mx

world = mx.distributed.init()

rank = world.rank()          # Current process rank (0-indexed)
size = world.size()          # Total number of processes
local_rank = world.local_rank()  # Rank within the node
```

### Launching Distributed Programs

Use the `mlx.launch` helper script to run programs across multiple processes:

```bash
# Local execution - 4 processes
mlx.launch -n 4 my_script.py

# Remote execution across machines
mlx.launch --hosts ip1,ip2,ip3,ip4 my_script.py

# With additional options
mlx.launch -n 8 --nprocs-per-host 2 my_script.py
```

### Training with Gradient Averaging

For data-parallel distributed training, leverage `mx.nn.average_gradients()`:

```python
import mlx.core as mx
import mlx.nn as nn

# Initialize distributed training
world = mx.distributed.init()

model = YourModel()
optimizer = nn.optimizers.SGD(learning_rate=0.01)

def loss_fn(model, x, y):
    return mx.mean(mx.square(model(x) - y))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Training with gradient averaging
def step(model, x, y):
    loss, grads = loss_and_grad_fn(model, x, y)

    # Average gradients across all processes
    grads = mx.nn.average_gradients(grads)

    optimizer.update(model, grads)
    return loss

# Main training loop
for epoch in range(num_epochs):
    for x_batch, y_batch in data_loader:
        loss = step(model, x_batch, y_batch)
        mx.eval(model.parameters(), optimizer.state)
```

This approach aggregates gradients efficiently with fewer communication steps than manual tree mapping.

### Backend Selection

Initialize specific backends via the `backend` parameter:

```python
import mlx.core as mx

# Force MPI backend
world = mx.distributed.init(backend="mpi")

# Force ring backend
world = mx.distributed.init(backend="ring")

# Default: Try ring first, fall back to MPI
world = mx.distributed.init(backend="any")
```

### Ring Topology Configuration

The ring backend connects nodes sequentially (rank 0↔1↔2↔3...), limiting peer-to-peer communication to adjacent neighbors. Define rings using JSON hostfiles:

```json
{
  "hosts": [
    {
      "hostname": "machine1.example.com",
      "local_ip": "192.168.1.10"
    },
    {
      "hostname": "machine2.example.com",
      "local_ip": "192.168.1.11"
    }
  ]
}
```

Use the hostfile with:

```bash
mlx.launch --hostfile hosts.json my_script.py
```

## FFT Operations

MLX provides comprehensive Fast Fourier Transform (FFT) capabilities through the `mlx.core.fft` module.

### One-Dimensional Transforms

```python
import mlx.core as mx

# Complex-valued FFT
x_complex = mx.array([1+2j, 3+4j, 5+6j])
fft_result = mx.fft.fft(x_complex)

# Inverse FFT
x_recovered = mx.fft.ifft(fft_result)

# Real-valued FFT (more efficient)
x_real = mx.array([1.0, 2.0, 3.0, 4.0])
rfft_result = mx.fft.rfft(x_real)

# Inverse real FFT
x_real_recovered = mx.fft.irfft(rfft_result)
```

### Multi-Dimensional Transforms

```python
import mlx.core as mx

# 2D FFT
x_2d = mx.random.normal((64, 64))
fft2d = mx.fft.fft2(x_2d)
ifft2d = mx.fft.ifft2(fft2d)

# Real-valued 2D FFT
x_real_2d = mx.random.normal((64, 64))
rfft2d = mx.fft.rfft2(x_real_2d)

# N-dimensional FFT
x_3d = mx.random.normal((32, 32, 32))
fftn = mx.fft.fftn(x_3d)

# Real-valued n-dimensional FFT
rfftn = mx.fft.rfftn(x_3d)
```

### Utility Functions

```python
import mlx.core as mx

# Shift zero-frequency component to center
x = mx.random.normal((64, 64))
fft_x = mx.fft.fft2(x)

# Center the spectrum
centered = mx.fft.fftshift(fft_x)

# Inverse shift
uncentered = mx.fft.ifftshift(centered)
```

## Specifying Output Sizes

Control FFT output dimensions:

```python
import mlx.core as mx

x = mx.random.normal((64,))

# Pad or truncate to specific size
fft_padded = mx.fft.rfft(x, n=128)
fft_truncated = mx.fft.rfft(x, n=32)
```

## Streams and Asynchronous Computation

MLX supports asynchronous computation through streams:

```python
import mlx.core as mx

# Get default stream
stream = mx.default_stream()

# Create custom stream
custom_stream = mx.stream(device=mx.gpu)

# Execute operations on specific stream
a = mx.ones((1000, 1000), device=mx.gpu)
b = mx.ones((1000, 1000), device=mx.gpu)

# These operations are async on the stream
c = a @ b
d = c + a

# Explicitly synchronize
mx.eval(c, d)  # Wait for computation to finish
```

## Custom Extensions and Kernels

### Using Streams with Operations

```python
import mlx.core as mx

stream = mx.stream(device=mx.gpu)

a = mx.random.normal((1000, 1000), device=mx.gpu)
b = mx.random.normal((1000, 1000), device=mx.gpu)

# Operations on stream
c = a @ b
d = mx.sin(c)

# Synchronization
mx.eval(d)
```

## Memory Optimization

### Smart Memory Usage with Lazy Evaluation

```python
import mlx.core as mx
import mlx.nn as nn

# Load large model without initializing all weights
model = nn.Sequential(
    nn.Linear(10000, 5000),
    nn.Linear(5000, 1000),
    nn.Linear(1000, 10)
)

# Convert to lower precision to reduce memory
for module in model.layers:
    if hasattr(module, 'weight'):
        module.weight = module.weight.astype(mx.float16)

# Actual memory only allocated when eval() is called
x = mx.random.normal((32, 10000))
y = model(x)
mx.eval(y)  # Only now is memory allocated
```

### In-Place Operations

Use in-place updates to reduce memory:

```python
import mlx.core as mx

a = mx.array([1.0, 2.0, 3.0])

# In-place update (memory efficient)
a[0] = 100.0

# Less efficient: creates new array
b = mx.array([100.0, 2.0, 3.0])
```

## Performance Profiling

### Timing Operations

```python
import mlx.core as mx
import time

# Create data
x = mx.random.normal((1000, 1000))
y = mx.random.normal((1000, 1000))

# Time a computation
start = time.perf_counter()
z = x @ y
mx.eval(z)
elapsed = time.perf_counter() - start

print(f"Matrix multiplication took {elapsed:.4f} seconds")
```

### Memory Profiling

```python
import mlx.core as mx

# Get current device memory usage
device = mx.gpu
memory_info = device.memory

print(f"Total memory: {memory_info.total}")
print(f"Used memory: {memory_info.used}")
```

## Advanced Gradient Techniques

### Custom Gradient Functions

```python
import mlx.core as mx

def custom_relu(x):
    return mx.maximum(x, 0)

# Define custom gradient
def relu_grad(x):
    return mx.where(x > 0, 1.0, 0.0)

# Use with grad
x = mx.array([-1.0, 0.5, 2.0])
grad_x = mx.grad(lambda x: mx.sum(custom_relu(x)))(x)
```

### Checkpointing for Memory Efficiency

```python
import mlx.core as mx
import mlx.nn as nn

def checkpoint_forward(layer, x):
    # Forward pass
    y = layer(x)
    return y

def checkpoint_backward(layer, x, grad_output):
    # Recompute forward
    y = layer(x)

    # Compute gradient
    grad_fn = mx.grad(lambda x: mx.sum(layer(x) * grad_output))
    return grad_fn(x)

# Use in deep networks
model = nn.Sequential(
    nn.Linear(100, 200),
    nn.Linear(200, 200),
    nn.Linear(200, 100)
)
```

## Model Export and Interoperability

### Saving Models in Different Formats

```python
import mlx.core as mx
import mlx.nn as nn

model = nn.Linear(10, 5)

# Save as safetensors (recommended)
mx.save_safetensors('model.safetensors', dict(model.parameters()))

# Save as NPZ
mx.save('model.npz', dict(model.parameters()))

# Load back
weights = mx.load('model.safetensors')
model.update(mx.tree_unflatten(weights))
```

### Converting Between Formats

```python
import mlx.core as mx
import numpy as np

# MLX to NumPy
mlx_array = mx.random.normal((10, 10))
numpy_array = np.array(mlx_array)

# NumPy to MLX
numpy_array = np.random.randn(10, 10)
mlx_array = mx.array(numpy_array)
```

## Debugging Tips

### Print Shapes During Forward Pass

```python
import mlx.core as mx
import mlx.nn as nn

def debug_forward(module, x):
    print(f"Input shape: {x.shape}")
    for i, layer in enumerate(module.layers):
        x = layer(x)
        print(f"After layer {i}: {x.shape}")
    return x

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

x = mx.random.normal((32, 10))
y = debug_forward(model, x)
```

### Debugging NaN Issues

```python
import mlx.core as mx

def check_for_nans(name, x):
    has_nan = mx.any(mx.isnan(x)).item()
    if has_nan:
        print(f"NaN found in {name}")
    return x

# Use in training loop
loss, grads = loss_and_grad_fn(model, x, y)
check_for_nans("loss", loss)
check_for_nans("gradients", grads)
```

## Summary

Advanced MLX topics include:

1. **Distributed training** - Scale across multiple devices/machines
2. **FFT operations** - Signal processing and spectral analysis
3. **Asynchronous computation** - Streams for performance
4. **Memory optimization** - Efficient use of device memory
5. **Custom gradients** - Implement specialized backpropagation
6. **Model export** - Save and load in standard formats
7. **Debugging** - Tools for troubleshooting training issues

These advanced features enable production-quality machine learning systems on Apple Silicon.
