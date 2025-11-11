# MLX Array Indexing Guide

## Basic Indexing

MLX arrays support indexing similar to NumPy arrays using integers, slices, and ellipsis notation:

```python
import mlx.core as mx

arr = mx.arange(10)
arr[3]        # Returns element at index 3 → 3
arr[-2]       # Negative indexing supported → 8
arr[2:8:2]    # Slicing with start, stop, stride → [2, 4, 6]
```

## Multidimensional Indexing

For multidimensional arrays, you can use ellipsis (`...`) to represent multiple dimensions:

```python
import mlx.core as mx

# 3D array
arr_3d = mx.random.normal((5, 6, 7))

# These are equivalent:
arr_3d[..., 0]      # Get last dimension's first element
arr_3d[:, :, 0]     # Explicit slicing

# Mixed indexing
arr_3d[1, ..., 0]   # Index first and last dimensions
```

## Advanced Indexing Techniques

### Adding New Axes

Use `None` to add dimensions:

```python
import mlx.core as mx

arr = mx.array([1, 2, 3])
print(arr.shape)           # (3,)

arr_expanded = arr[None]
print(arr_expanded.shape)  # (1, 3)

arr_expanded_2 = arr[:, None, None]
print(arr_expanded_2.shape)  # (3, 1, 1)
```

### Array Indexing

Arrays can index other arrays:

```python
import mlx.core as mx

arr = mx.array([10, 20, 30, 40, 50])
indices = mx.array([0, 2, 4])

result = arr[indices]  # [10, 30, 50]
```

### Utility Functions

The `take()` and `take_along_axis()` functions provide additional indexing capabilities:

```python
import mlx.core as mx

arr = mx.random.normal((10, 20, 30))

# Take along a specific axis
indices = mx.array([0, 2, 5, 10, 15])
result = mx.take(arr, indices, axis=0)

# Take along axis with per-element indices
indices = mx.random.randint(0, 20, (10, 30))
result = mx.take_along_axis(arr, indices, axis=1)
```

## In-Place Updates

MLX supports direct assignment to indexed locations:

```python
import mlx.core as mx

a = mx.array([1, 2, 3])
a[2] = 0  # Valid in-place modification → [1, 2, 0]
```

### Important Caveat About Slicing

Unlike NumPy, "slicing an array creates a copy, not a view," so modifying sliced arrays doesn't affect the original:

```python
import mlx.core as mx

a = mx.array([1, 2, 3, 4, 5])
b = a[1:4]  # b is a copy, not a view
b[0] = 99

print(a)  # [1, 2, 3, 4, 5] - unchanged
print(b)  # [99, 3, 4]
```

### Nondeterministic Behavior with Multiple Assignments

"Updates to the same location are nondeterministic" when multiple assignments target identical indices:

```python
import mlx.core as mx

a = mx.array([1, 2, 3, 4, 5])

# Avoid this pattern - result is undefined
indices = mx.array([1, 1, 2])
values = mx.array([10, 20, 30])
a[indices] = values  # Behavior undefined for duplicate indices
```

## Key Differences from NumPy

MLX has two significant limitations compared to NumPy:

### 1. No Bounds Checking

Out-of-bounds indexing produces undefined behavior rather than errors. This is because "GPU exceptions cannot propagate":

```python
import mlx.core as mx

arr = mx.array([1, 2, 3])
result = arr[10]  # Undefined behavior - no error raised
```

**Best practice**: Always validate indices before using them.

### 2. No Boolean Masking

"Boolean mask based indexing is not yet supported." MLX generally doesn't support operations where output shapes depend on input data:

```python
import mlx.core as mx

arr = mx.array([1, 2, 3, 4, 5])
mask = arr > 2

# This is NOT supported:
result = arr[mask]  # Error!

# Use alternative approaches instead:
indices = mx.where(mask)
```

## Indexing Performance Tips

### 1. Use Vectorized Operations

Prefer vectorized operations over manual indexing loops:

```python
import mlx.core as mx

# Slower: loop-based indexing
arr = mx.random.normal((1000, 1000))
result = []
for i in range(100):
    result.append(arr[i:i+10].sum())

# Faster: vectorized operations
arr = mx.random.normal((1000, 1000))
result = mx.sum(arr[:100], axis=1)
```

### 2. Batch Indexing with take()

For performance-critical code, use `mx.take()`:

```python
import mlx.core as mx

arr = mx.random.normal((1000, 100))
indices = mx.random.randint(0, 1000, (1000,))

# Efficient batch indexing
result = mx.take(arr, indices, axis=0)
```

### 3. Avoid Scalar Array Control Flow

Don't use scalar array indexing in tight loops:

```python
import mlx.core as mx

arr = mx.random.normal((1000, 1000))

# Avoid in loops:
for i in range(1000):
    if arr[i, i] > 0:  # Triggers evaluation each time
        pass

# Better: extract all diagonals at once
diag = mx.diagonal(arr)
```

## Summary

MLX indexing is similar to NumPy with some key differences:

- Supports integers, slices, ellipsis, and array indexing
- Slicing creates copies, not views
- No boolean masking support
- No bounds checking
- Use `mx.take()` for efficient batch indexing
- Avoid scalar indexing in performance-critical code
