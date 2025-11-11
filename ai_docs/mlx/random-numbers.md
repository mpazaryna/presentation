# MLX Random Number Generation Documentation

## Overview

MLX provides comprehensive random sampling functions accessible through `mlx.core.random`. The framework uses an implicit global PRNG state by default, though users can pass explicit keys for fine-grained control.

## Key Design Philosophy

The library follows "JAX's PRNG design" and implements a splittable version of Threefry, which is a counter-based pseudo-random number generator. This allows both implicit and explicit state management patterns.

## Basic Usage

### Using the Global PRNG

```python
import mlx.core as mx

# Generate random numbers using global state
x = mx.random.normal((10, 20))
y = mx.random.uniform(0, 1, (10, 20))
z = mx.random.randint(0, 100, (10, 20))
```

### Using Explicit Keys

For reproducibility and fine-grained control:

```python
import mlx.core as mx

# Create a key from a seed
key = mx.random.key(42)

# Use the key to generate random numbers
x = mx.random.normal(shape=(10, 20), key=key)
```

### Splitting Keys

Create independent random streams:

```python
import mlx.core as mx

# Initial key
key = mx.random.key(42)

# Split into two independent keys
key1, key2 = mx.random.split(key)

# Each key generates independent random numbers
x = mx.random.normal((10,), key=key1)
y = mx.random.normal((10,), key=key2)

# Multiple splits
key1, key2, key3, key4 = mx.random.split(key, num=4)
```

## Available Distribution Functions

### Uniform Distribution

Generate uniformly distributed numbers:

```python
import mlx.core as mx

# Default uniform [0, 1)
x1 = mx.random.uniform(shape=(100,))

# Custom range
x2 = mx.random.uniform(low=-1, high=1, shape=(100,))

# With explicit key
key = mx.random.key(0)
x3 = mx.random.uniform(shape=(100,), key=key)
```

### Normal Distribution

Generate normally distributed values:

```python
import mlx.core as mx

# Standard normal (mean=0, scale=1)
x1 = mx.random.normal((100,))

# Custom mean and scale
x2 = mx.random.normal((100,), loc=5.0, scale=2.0)

# Multi-dimensional
x3 = mx.random.normal((10, 20, 30))

# With explicit key
key = mx.random.key(0)
x4 = mx.random.normal((100,), key=key)
```

### Truncated Normal

Sample from a normal distribution within bounds:

```python
import mlx.core as mx

# Truncated normal between -2 and 2
x = mx.random.truncated_normal(low=-2, high=2, shape=(100,))

# With custom location and scale
x = mx.random.truncated_normal(
    low=-3,
    high=3,
    loc=0.0,
    scale=1.0,
    shape=(100,)
)
```

### Multivariate Normal

Generate correlated samples from a multivariate distribution:

```python
import mlx.core as mx
import numpy as np

# Define covariance matrix
cov = mx.array([
    [1.0, 0.5],
    [0.5, 2.0]
])

# Generate samples
samples = mx.random.multivariate_normal(shape=(1000, 2), cov=cov)
```

### Bernoulli Distribution

Generate binary random values:

```python
import mlx.core as mx

# Fair coin flips (p=0.5)
coin_flips = mx.random.bernoulli(shape=(100,))

# Biased coin (70% probability of 1)
biased_flips = mx.random.bernoulli(p=0.7, shape=(100,))

# Custom shape
matrix = mx.random.bernoulli(p=0.5, shape=(10, 20))
```

### Gumbel Distribution

Sample from the standard Gumbel distribution:

```python
import mlx.core as mx

# Standard Gumbel
x = mx.random.gumbel(shape=(100,))

# Useful for Gumbel-Softmax trick
logits = mx.array([1.0, 2.0, 3.0])
tau = 0.5
gumbel_softmax = mx.softmax((logits + mx.random.gumbel(shape=(3,))) / tau)
```

### Laplace Distribution

Generate values from a Laplace (double exponential) distribution:

```python
import mlx.core as mx

# Standard Laplace
x1 = mx.random.laplace(shape=(100,))

# Custom location and scale
x2 = mx.random.laplace(loc=0.0, scale=1.0, shape=(100,))
```

## Integer Operations

### Random Integers

Generate random integers within a specified interval:

```python
import mlx.core as mx

# Random integers in [0, 100)
x1 = mx.random.randint(0, 100, shape=(100,))

# Different ranges
x2 = mx.random.randint(low=-50, high=50, shape=(50,))

# Multi-dimensional
x3 = mx.random.randint(0, 10, shape=(5, 10, 10))
```

### Permutations

Create random permutations of arrays:

```python
import mlx.core as mx

# Random permutation of [0, 1, 2, ..., 9]
perm = mx.random.permutation(10)

# Permute an existing array
arr = mx.array([1, 2, 3, 4, 5])
shuffled = arr[mx.random.permutation(5)]
```

### Categorical Distribution

Sample from categorical distributions using logits:

```python
import mlx.core as mx

# Logits for 3 categories
logits = mx.array([1.0, 2.0, 0.5])

# Single sample
sample1 = mx.random.categorical(logits=logits)

# Multiple samples
samples = mx.random.categorical(logits=logits, shape=(1000,))

# Custom number of samples
probabilities = mx.softmax(logits)
samples = mx.random.categorical(logits=logits, num_samples=1000)
```

## State Management

### Setting the Global Seed

Initialize the global PRNG with a seed:

```python
import mlx.core as mx

# Set global seed for reproducibility
mx.random.seed(42)

# All subsequent random calls use this seed
x1 = mx.random.normal((10,))
x2 = mx.random.normal((10,))
```

### Creating Keys from Seeds

```python
import mlx.core as mx

# Create key from seed
key = mx.random.key(123)

# Use key for generation
x = mx.random.normal((100,), key=key)
```

## Practical Examples

### Reproducible Experiments

```python
import mlx.core as mx

# Setup reproducible randomness
mx.random.seed(42)

# Generate training data
X_train = mx.random.normal((1000, 10))
y_train = mx.random.normal((1000,))

# Generate test data (different key)
key = mx.random.key(43)
X_test = mx.random.normal((100, 10), key=key)
y_test = mx.random.normal((100,), key=key)
```

### Batch Processing with Different Seeds

```python
import mlx.core as mx

def create_batch_with_augmentation(base_key, batch_size=32):
    # Split key for multiple purposes
    key_image, key_noise = mx.random.split(base_key)

    # Generate augmented images
    images = mx.random.normal((batch_size, 28, 28), key=key_image)

    # Add noise
    noise = 0.1 * mx.random.normal((batch_size, 28, 28), key=key_noise)

    return images + noise

# Use in training
key = mx.random.key(0)
for i in range(num_batches):
    key, subkey = mx.random.split(key)
    batch = create_batch_with_augmentation(subkey)
```

### Dropout Implementation

```python
import mlx.core as mx

def dropout(x, p=0.5, training=True):
    if not training:
        return x

    # Generate binary mask
    mask = mx.random.bernoulli(p=1-p, shape=x.shape)

    # Scale and apply
    return x * mask / (1 - p)

# In neural network
x = mx.random.normal((32, 64))
x = dropout(x, p=0.5, training=True)
```

### Stochastic Gradient Descent with Noise Injection

```python
import mlx.core as mx

def noisy_sgd_step(params, grads, learning_rate=0.01, noise_scale=0.01):
    # Add Gaussian noise to gradients
    noise = noise_scale * mx.random.normal(grads.shape)

    # Update
    return params - learning_rate * (grads + noise)
```

### Monte Carlo Sampling

```python
import mlx.core as mx

def estimate_pi(num_samples=1000000):
    # Sample points in unit square
    x = mx.random.uniform(0, 1, (num_samples,))
    y = mx.random.uniform(0, 1, (num_samples,))

    # Check if inside unit circle
    distances = mx.sqrt(x**2 + y**2)
    inside_circle = distances <= 1.0

    # Estimate pi
    ratio = mx.mean(inside_circle)
    pi_estimate = 4.0 * ratio

    return pi_estimate

pi = estimate_pi()
print(f"Estimated pi: {pi:.4f}")
```

## Performance Considerations

### Vectorized Random Generation

Always prefer vectorized operations over loops:

```python
import mlx.core as mx

# Slow: loop-based
slow_samples = []
for i in range(10000):
    slow_samples.append(mx.random.normal((1,)))

# Fast: vectorized
fast_samples = mx.random.normal((10000,))
```

### Key Management

For distributed/parallel code, use explicit keys:

```python
import mlx.core as mx

def parallel_worker(worker_id, base_key):
    # Each worker gets independent key
    worker_key = mx.random.split(base_key, num=num_workers)[worker_id]

    # Generate random numbers
    data = mx.random.normal((1000,), key=worker_key)
    return data
```

## Summary

MLX random functions provide:

1. **Multiple distributions** - Normal, uniform, categorical, and more
2. **Flexible control** - Global or explicit key-based randomness
3. **Reproducibility** - Deterministic generation with seeds
4. **Efficiency** - Vectorized operations
5. **Distributed support** - Key splitting for parallel code

These functions are essential for data generation, augmentation, and stochastic training procedures.
