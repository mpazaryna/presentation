# MLX Neural Networks Module Documentation

## Overview

The MLX neural networks module (`mlx.nn`) provides a comprehensive framework for building neural network architectures. The documentation states: "Writing arbitrarily complex neural networks in MLX can be done using only `mlx.core.array` and `mlx.core.value_and_grad()`."

However, `mlx.nn` provides high-level abstractions for convenience and clarity.

## Core Components

### Module Class

The foundational building block serving as "a container of `mlx.core.array` or `Module` instances." Its primary functions include recursively accessing and updating parameters across nested submodules.

```python
import mlx.core as mx
import mlx.nn as nn

class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.norm = nn.LayerNorm(5)

    def __call__(self, x):
        x = self.linear(x)
        return self.norm(x)

model = SimpleModule()
```

### Parameters

Any public member of type `mlx.core.array` (names without leading underscores) qualifies as a parameter. The system supports arbitrary nesting within other modules, lists, and dictionaries.

```python
import mlx.core as mx
import mlx.nn as nn

class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.random.normal((10, 5))  # Parameter
        self._private = mx.array([1, 2, 3])      # Not a parameter
        self.layers = [
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        ]  # Nested modules

# Access all parameters
all_params = model.parameters()

# Access only trainable (unfrozen) parameters
trainable = model.trainable_parameters()

# Freeze specific parameters
model.freeze()
model.linear.unfreeze()
```

## Key Functionality

### Parameter Management

```python
import mlx.core as mx
import mlx.nn as nn

model = nn.Linear(10, 5)

# Access parameters
params = model.parameters()

# Freeze/unfreeze parameters
model.freeze()           # All parameters frozen
model.unfreeze()        # All parameters trainable
model.linear.freeze()   # Freeze specific submodule

# Check if parameters frozen
if model.linear.frozen:
    print("Linear layer is frozen")

# Get trainable parameters
trainable = model.trainable_parameters()
```

### Module Inspection

```python
import mlx.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# Print architecture
print(model)

# Tree mapping for shape analysis
def print_shapes(arr):
    if isinstance(arr, mx.array):
        return arr.shape
    return None

shapes = mx.tree_map(print_shapes, model.parameters())

# Parameter counting
num_params = sum(p.size for p in mx.tree_flatten(model.parameters())[0])
print(f"Total parameters: {num_params}")
```

### Training Control

```python
import mlx.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.BatchNorm(20, momentum=0.9),
    nn.Linear(20, 5)
)

# Set training mode
model.train()   # Enable dropout, batch norm statistics computation

# Set evaluation mode
model.eval()    # Disable dropout, use running statistics
```

### Weight Loading and Saving

```python
import mlx.core as mx
import mlx.nn as nn

model = nn.Linear(10, 5)

# Save model parameters
mx.save_safetensors("model.safetensors", dict(model.parameters()))

# Load model parameters
weights = mx.load("model.safetensors")
model.update(mx.tree_unflatten(weights))
```

## Layer Types

### Activation Functions

MLX provides standard activation functions:

```python
import mlx.core as mx
import mlx.nn as nn

x = mx.random.normal((10, 20))

# ReLU variants
y1 = nn.relu(x)
y2 = nn.gelu(x)
y3 = nn.gelu_approx(x)

# Other activations
y4 = nn.silu(x)
y5 = nn.sigmoid(x)
y6 = nn.tanh(x)
```

### Pooling Layers

```python
import mlx.core as mx
import mlx.nn as nn

x = mx.random.normal((2, 32, 32, 3))  # (batch, height, width, channels)

# Average pooling
pool = nn.AvgPool2d(kernel_size=2, stride=2)
y1 = pool(x)

# Max pooling
pool = nn.MaxPool2d(kernel_size=2, stride=2)
y2 = pool(x)

# 1D pooling
x_1d = mx.random.normal((2, 100, 3))
pool_1d = nn.AvgPool1d(kernel_size=3)
y3 = pool_1d(x_1d)
```

### Normalization Layers

```python
import mlx.core as mx
import mlx.nn as nn

x = mx.random.normal((32, 256))

# Layer normalization
ln = nn.LayerNorm(256)
y1 = ln(x)

# Group normalization
gn = nn.GroupNorm(num_groups=8, dims=256)
y2 = gn(x)

# Batch normalization
bn = nn.BatchNorm(256)
y3 = bn(x)

# RMSNorm (root mean square normalization)
rmsnorm = nn.RMSNorm(256)
y4 = rmsnorm(x)
```

### Recurrent Networks

```python
import mlx.core as mx
import mlx.nn as nn

batch_size, seq_len, input_size = 32, 10, 64
hidden_size = 128

x = mx.random.normal((batch_size, seq_len, input_size))

# LSTM
lstm = nn.LSTM(input_size, hidden_size, num_layers=2)
y, (h, c) = lstm(x)

# GRU
gru = nn.GRU(input_size, hidden_size)
y, h = gru(x)

# Basic RNN
rnn = nn.RNN(input_size, hidden_size)
y, h = rnn(x)
```

### Attention Mechanisms

```python
import mlx.core as mx
import mlx.nn as nn

batch_size, seq_len, d_model = 32, 10, 64
num_heads = 4

x = mx.random.normal((batch_size, seq_len, d_model))

# Multi-head attention
attn = nn.MultiHeadAttention(dims=d_model, num_heads=num_heads)
y, attention_scores = attn(x, x, x)

# Attention with ALiBi (Attention with Linear Biases)
attn_alibi = nn.MultiHeadAttention(
    dims=d_model,
    num_heads=num_heads,
    bias=nn.ALiBi()
)

# Attention with RoPE (Rotary Position Embedding)
attn_rope = nn.MultiHeadAttention(
    dims=d_model,
    num_heads=num_heads,
    bias=nn.RoPE()
)
```

### Specialized Layers

```python
import mlx.core as mx
import mlx.nn as nn

batch_size, seq_len, vocab_size, embed_dim = 32, 10, 1000, 64

# Embedding
embedding = nn.Embedding(num_embeddings=vocab_size, dims=embed_dim)
indices = mx.random.randint(0, vocab_size, (batch_size, seq_len))
embedded = embedding(indices)

# Dropout
dropout = nn.Dropout(p=0.5)
x = mx.random.normal((32, 64))
y_train = dropout(x)  # Applies dropout during training
model.eval()
y_eval = dropout(x)   # No dropout during evaluation

# Spatial dropout variants
dropout_2d = nn.Dropout2d(p=0.5)
dropout_3d = nn.Dropout3d(p=0.5)

# Transformer block
transformer = nn.TransformerEncoderLayer(
    dims=64,
    num_heads=4,
    mlp_dims=256,
    activation=nn.gelu,
    dropout=0.1,
    layer_norm_first=False
)
y = transformer(x)
```

### Linear and Convolutional Layers

```python
import mlx.core as mx
import mlx.nn as nn

# Linear layer
linear = nn.Linear(in_features=10, out_features=5)
x = mx.random.normal((batch_size, 10))
y = linear(x)

# 2D Convolution
conv2d = nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    stride=1,
    padding=1
)
x = mx.random.normal((batch_size, 32, 32, 3))
y = conv2d(x)

# 1D Convolution
conv1d = nn.Conv1d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=1,
    padding=1
)
x = mx.random.normal((batch_size, 100, 64))
y = conv1d(x)

# 3D Convolution
conv3d = nn.Conv3d(
    in_channels=1,
    out_channels=32,
    kernel_size=3
)
```

## Training Integration

The `nn.value_and_grad()` function wraps loss functions to simultaneously compute values and gradients relative to trainable parameters:

```python
import mlx.core as mx
import mlx.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def __call__(self, x):
        x = mx.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet()

def loss_fn(model, x, y):
    logits = model(x)
    return mx.mean(mx.square(logits - y))

# Create loss and gradient function
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Training loop
optimizer = nn.optimizers.Adam(learning_rate=0.001)

for epoch in range(100):
    for x_batch, y_batch in data_loader:
        loss, grads = loss_and_grad_fn(model, x_batch, y_batch)

        # Update parameters
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
```

## Common Patterns

### Sequential Model

```python
import mlx.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

x = mx.random.normal((32, 10))
y = model(x)
```

### Custom Module with Parameter Freezing

```python
import mlx.core as mx
import mlx.nn as nn

class FineTuneModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Pre-trained encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        # Fine-tune head
        self.head = nn.Linear(128, 10)

    def __call__(self, x):
        features = self.encoder(x)
        return self.head(features)

model = FineTuneModel()

# Freeze encoder, train only head
model.encoder.freeze()

# Now only model.head parameters will receive gradients
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
```

## Summary

The MLX neural networks module provides:

1. **Flexible module system** for building complex architectures
2. **Comprehensive layer types** covering standard deep learning operations
3. **Training utilities** for efficient gradient computation
4. **Parameter management** including freezing and serialization
5. **Integration with function transforms** for advanced optimization

This makes MLX suitable for a wide range of machine learning applications.
