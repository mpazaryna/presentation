# Lazy Evaluation in MLX

## Core Concept

MLX implements lazy evaluation where operations don't compute immediately. Instead, the framework records a compute graph, executing it only when explicitly triggered via `eval()` or implicitly through specific actions.

## Key Benefits

### Graph Transformation Support

"Lazy evaluation lets us record a compute graph without actually doing any computations. This is useful for function transformations like `grad()` and `vmap()` and graph optimizations."

This enables powerful features like:

- **Automatic differentiation** - Build gradient computation graphs
- **Vectorization** - Generate efficient vectorized implementations
- **Graph optimization** - Optimize operations before execution

### Memory Efficiency

You can load large models without initializing all weights upfront. The typical workflow:

1. Load a model architecture
2. Update weights to lower precision
3. Actual memory consumption happens only during evaluation

This reduces peak memory usage substantially.

### Selective Computation

Unused outputs in branching code paths don't execute. If a function returns two values but you only use one, expensive operations computing the unused output still build their graph (with associated costs) but skip execution.

## When Evaluation Occurs

### Explicit Evaluation

Call `mx.eval()` directly on arrays or model parameters:

```python
import mlx.core as mx

# Create lazy computation
x = mx.array([1.0, 2.0, 3.0])
y = mx.array([4.0, 5.0, 6.0])
z = x + y
w = z * 2

# Explicitly force evaluation
result = mx.eval(w)
print(result)  # Computation happens here
```

### Implicit Evaluation Triggers

Operations that automatically trigger evaluation include:

- **Printing arrays**: `print(array)`
- **Converting to NumPy arrays**: `array.numpy()`
- **Accessing memory via**: `memoryview(array)`
- **Saving arrays**: `mx.save()`, `mx.save_safetensors()`
- **Calling `.item()` on scalars**: `mx.array(5.0).item()`
- **Using scalar arrays in Python control flow**: `if mx.array(True): ...`

### Example: Implicit Evaluation

```python
import mlx.core as mx

a = mx.random.normal((100, 100))
b = mx.random.normal((100, 100))
c = a @ b

# These trigger evaluation automatically
print(c)                    # Printing
numpy_array = c.numpy()    # NumPy conversion
item_value = c[0, 0].item()  # Scalar extraction
```

## Best Practices

### Optimal Timing for eval()

Place `mx.eval()` at outer loop iterations—typically once per batch during training loops. This balances evaluation overhead against excessive graph growth:

```python
import mlx.core as mx
import mlx.nn as nn

model = nn.Linear(10, 5)
optimizer = mx.optimizers.SGD(learning_rate=0.01)

# Training loop - good placement of eval()
for epoch in range(100):
    for X_batch, y_batch in data_loader:
        loss, grads = mx.value_and_grad(loss_fn)(model, X_batch, y_batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)  # Once per batch
```

### Graph Size Management

Compute graphs ranging from "a few tens of operations to many thousands of operations per evaluation should be okay" with MLX.

Monitor graph size in performance-critical applications. If you notice slowdowns, consider evaluating more frequently to reduce accumulated computation graphs.

### Avoiding Excessive Evaluations

"Using scalar arrays for control-flow will cause an evaluation." This pattern works but can become inefficient if evaluations trigger too frequently:

```python
# Avoid this pattern in tight loops
for i in range(1000):
    if mx.array(some_condition):  # Each triggers evaluation!
        # do something

# Better approach: use NumPy for control flow decisions
for i in range(1000):
    condition_value = mx.array(some_condition).item()
    if condition_value:
        # do something
```

## Advanced Usage

### Custom Evaluation Control

For fine-grained control over evaluation, combine explicit `eval()` calls with graph construction:

```python
import mlx.core as mx

# Build computation graph
x = mx.array([1.0, 2.0, 3.0])
y = mx.sin(x)
z = mx.cos(x)
w = y + z

# Evaluate only specific outputs
result_y, result_z = mx.eval(y, z)

# Later, evaluate combined result
result_w = mx.eval(w)
```

### Lazy Evaluation with Neural Networks

Lazy evaluation integrates seamlessly with neural network training:

```python
import mlx.core as mx
import mlx.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def __call__(self, x):
        x = self.linear1(x)
        x = mx.relu(x)
        x = self.linear2(x)
        return x

model = SimpleModel()

def loss_fn(model, x, y):
    return mx.mean(mx.square(model(x) - y))

# All model computations are lazy until eval()
loss_and_grad_fn = mx.value_and_grad(model, loss_fn)
```

## Summary

Lazy evaluation is a fundamental feature of MLX that enables:

1. **Efficient gradient computation** through graph recording
2. **Lower memory usage** for large models
3. **Better performance** through automatic optimizations
4. **Flexible control** over computation timing

Mastering lazy evaluation patterns is key to writing efficient MLX code.
