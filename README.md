# NeuroCore-CPP

A minimalist automatic differentiation engine and neural network framework implemented from scratch in C++. This project demonstrates the fundamental mathematics behind modern deep learning systems without relying on external machine learning libraries.

## Overview

NeuroCore-CPP implements the core components required for training neural networks:

- **Automatic Differentiation Engine**: Computes gradients automatically using backpropagation
- **Value Class**: Scalar values with gradient tracking for computational graph construction
- **Neural Network Primitives**: Neurons, layers, and multi-layer perceptrons
- **Training Loop**: Complete gradient descent optimization

## Technical Architecture

### Engine Core (`engine.hpp`)

The foundation is the `Value` class, which wraps scalar values and maintains gradient information for automatic differentiation. Each operation (addition, multiplication, ReLU) constructs a computational graph that enables reverse-mode automatic differentiation.

Key features:
- Lazy gradient computation via lambda closures
- Topological sorting for correct gradient flow
- Support for arbitrary computational graphs

### Neural Network Module (`nn.hpp`)

Implements the standard building blocks of neural networks:

- **Neuron**: Single computational unit with weights, bias, and optional ReLU activation
- **Layer**: Collection of neurons operating in parallel
- **MLP**: Multi-layer perceptron combining multiple layers

Parameters are initialized with uniform random distribution in the range [-1, 1] using the Mersenne Twister engine for quality randomness.

### Training Implementation (`main.cpp`)

Demonstrates a complete training loop solving a non-linear classification problem using:
- Forward propagation through the network
- Loss computation (Mean Squared Error)
- Backpropagation for gradient calculation
- Stochastic Gradient Descent for parameter updates

## Build and Run

Requirements:
- C++11 or later
- g++ compiler

Compilation:
```bash
g++ main.cpp -o neurocore
```

Execution:
```bash
./neurocore
```

Expected output shows decreasing loss over epochs, demonstrating successful learning.

## Implementation Details

This implementation prioritizes clarity and educational value over performance. The automatic differentiation mechanism uses shared pointers for memory management and lambda functions for gradient computation closures.

The backward pass performs topological sorting to ensure gradients propagate in the correct order through the computational graph, mirroring techniques used in frameworks like PyTorch and TensorFlow.

## Mathematical Foundation

The core operation is the chain rule of calculus, applied recursively:
```
∂L/∂x = ∂L/∂y × ∂y/∂x
```

Each operation implements both forward computation and gradient propagation rules. For example, multiplication:
```cpp
// Forward: out = a * b
// Backward: a.grad += b.data * out.grad
//           b.grad += a.data * out.grad
```

The ReLU activation function applies:
```cpp
// Forward: out = max(0, x)
// Backward: x.grad += (out > 0) ? out.grad : 0
```

## Training Results

The network successfully learns to approximate target values through gradient descent. Example output from a 3-4-4-1 architecture:
```
Epoch 0  | Loss: 3.45483
Epoch 90 | Loss: 2.00012

Predictions:
Input [2, 3, -1]    -> Pred: 1.00071  | Target: 1
Input [3, -1, 0.5]  -> Pred: 0.00000  | Target: -1
Input [0.5, 1, 1]   -> Pred: 0.00000  | Target: -1
Input [1, 1, -1]    -> Pred: 0.99543  | Target: 1
```

## Limitations and Future Work

Current limitations:
- Single-threaded execution
- No GPU acceleration
- Limited to dense feedforward architectures

Potential extensions:
- Convolutional and recurrent layers
- Additional activation functions (tanh, sigmoid, softmax)
- Optimizers beyond vanilla SGD (Adam, RMSprop)
- Model serialization and loading
- Batch processing for improved performance

## License

This project is provided for educational purposes
