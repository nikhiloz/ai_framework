# AI Framework: From Scratch to Transformers

A pedagogical implementation of neural network architectures written in pure C. This project serves as a deep-dive into the mathematical foundations of modern AI, tracing the evolution from simple linear models to the complex Transformer architecture used in Large Language Models (LLMs).

## 🚀 Project Goal
The objective is to build a functional, from-scratch framework that implements the "alphabet" of AI:
**Weights $ightarrow$ Forward Pass $ightarrow$ Loss $ightarrow$ Backpropagation $ightarrow$ Update**.

The journey moves from basic vector-to-vector mappings (MLPs) to sequence-aware attention mechanisms (Transformers).

## 🛠️ Technical Stack
- **Language**: C (C11 standard)
- **Build System**: GNU Make
- **Dependencies**: `libc` (math library `-lm`)

## 🧩 Implemented Features

### 1. Core Engine
- **Matrix Engine**: A custom linear algebra library for matrix multiplication, transposition, and element-wise operations.
- **Optimization**: Implementation of SGD, Momentum, and **Adam** optimizers.

### 2. Multi-Layer Perceptron (MLP)
- **Forward Pass**: Support for multiple layers with various activation functions (Sigmoid, ReLU, Tanh, Softmax).
- **Backpropagation**: Full implementation of the chain rule for gradient descent.
- **Loss Functions**: MSE, Binary Cross-Entropy (BCE), and Categorical Cross-Entropy (CCE).

### 3. Transformer Foundations (Stage 1 & 2)
- **Tokenizer**: A character-level tokenizer for converting raw text into token IDs.
- **Embedding Layers**: Mapping discrete tokens to high-dimensional continuous vector spaces.
- **Positional Encoding**: Sinusoidal encoding to provide the model with sequence order information.
- **Self-Attention**: The core QKV (Query, Key, Value) mechanism.
- **Multi-Head Attention (MHA)**: Parallel attention heads for diverse feature extraction.
- **Layer Normalization**: Numerical stability for deep networks.
- **Transformer Block**: An integrated block combining MHA, Feed-Forward Networks, Residual Connections, and LayerNorm.

## 📖 How to Run

### Build the Project
```bash
make
```

### Run Tutorials
The project includes a series of tutorials in `bin/` that demonstrate each component:
- `./bin/tut_perceptron`: The simplest neural unit.
- `./bin/tut_linear_regression`: Basic linear mapping.
- `./bin/tut_mlp_deep_dive`: Solving the non-linear XOR problem.
- `./bin/tut_tokenizer`: Text $\leftrightarrow$ Token ID conversion.
- `./bin/tut_embedding`: Token $ightarrow$ Vector mapping.
- `./bin/tut_positional`: Injecting sequence order.
- `./bin/tut_attention`: Core self-attention mechanism.
- `./bin/tut_mha`: Multi-head attention processing.
- `./bin/tut_transformer`: The full integrated Transformer block.

## 🗺️ Roadmap
See [ROADMAP.md](ROADMAP.md) for the detailed path towards a conversational LLM.
